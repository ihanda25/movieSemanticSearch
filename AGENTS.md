# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

A quick UI where you describe a scene you remember and get back the Disney movie
it came from.

**Current state: end-to-end search works, in two stages.** `POST /api/search`
scans the 826 × 384 matrix in `database/disney_vector_db.npz` with the
bi-encoder, takes the top 50, rescores those with a cross-encoder, and returns
the top 5 with a match percentage; the frontend renders them as a list.
Retrieval quality is still the open problem, not plumbing — see
[Retrieval quality](#retrieval-quality-the-open-problem).

Search is local embeddings (sentence-transformers) with brute-force cosine
similarity over a numpy array, deliberately **not** a vector database. At ~826
movies the whole matrix is a few MB and an exhaustive scan takes well under a
millisecond, so an approximate index like Chroma's HNSW would add a dependency
and inexact results to avoid a cost that isn't there. Revisit around 100k
vectors.

## Run

Dependencies live in a virtualenv at `.venv/` (gitignored). Set it up once:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then, to run:

```bash
.venv/bin/python backend/app.py   # http://127.0.0.1:8000
```

Calling `.venv/bin/python` directly works without activating; `source
.venv/bin/activate` first if you prefer a bare `python`. Either way, use the
venv rather than system `python3`. The system interpreter happens to have Flask,
requests, and python-dotenv installed too, so `python3 backend/app.py` will
appear to work while silently using different package versions.

There is no separate frontend server and no build step. Everything in `tests/`
is a standalone script you run directly (`.venv/bin/python tests/test_tmdb.py`),
not a suite — there is no pytest and no runner.

Port 8000 is a deliberate choice: macOS AirPlay Receiver (`ControlCenter`) holds
port 5000, and the user wants AirPlay left running. If you change the port, also
update the URL in `README.md`.

## Architecture

Flask serves both the API and the static frontend from the same origin
(`backend/app.py` routes `/` and `/<path:filename>` into `frontend/`). This is
what lets `app.js` call `fetch("/api/search")` with no CORS configuration —
splitting the frontend onto its own dev server would break that assumption.

- `frontend/` — plain HTML/CSS/JS, no framework and no dependencies. Theming is
  CSS custom properties on `:root` with a `prefers-color-scheme: light` override.
- `backend/app.py` — the whole backend. Calls `search_movies_reranked()` and
  shapes the response as `{results: [{title, year, overview, match}]}`.
  `to_percent()` turns the cross-encoder logit into 0–100 (see
  [Score display](#score-display)). Both models are warmed at startup so the
  first search does not pay for the load; with `debug=True` the reloader does
  that twice, once per process.
- `database/` — `build_catalog.py` pulls 826 movies from TMDB into
  `catalog.json` (checked in) plus a `genres.json` int→name map. Regenerating is
  idempotent and takes ~3s: `.venv/bin/python database/build_catalog.py`.
  Credentials live in `.env` (gitignored) as `tmdb_token` / `tmdb_api`; the
  script uses the bearer token. Depends on `requests` and `python-dotenv`.
  `disney_vector_db.npz` also lands here but is gitignored — regenerate it
  rather than expecting a clone to have it.
- `disney_overview_search/embed.py` — embeds the catalog with
  sentence-transformers `all-MiniLM-L6-v2` (384 dims, runs locally, no API key)
  into `database/disney_vector_db.npz`. Run it with
  `.venv/bin/python disney_overview_search/embed.py`; the first run downloads
  ~90MB of model weights, after which it is cached in `~/.cache/huggingface`.
  The embedded text is `f"Title: {title}, Overview: {overview}"`, one row per
  movie in catalog order, L2-normalized so a plain dot product is cosine
  similarity. Paths are anchored to the repo root the same way
  `build_catalog.py` does it, so it runs from any directory.
- `disney_overview_search/search_disney.py` — stage 1. `search_movies()` is the
  bi-encoder scan; it caches the model and the `.npz` in a module-level global
  so the cost is paid once, not per request. Returns raw cosine scores — the
  percentage conversion is the caller's job, so evaluation work can use the
  unmodified numbers. The backend no longer calls it directly, but it is
  deliberately left unchanged: it is both the stage-1 entry point and the REPL
  for seeing what the bi-encoder does on its own, which is how you tell the two
  stages apart.
- `disney_overview_search/disney_cross_encode.py` — stage 2.
  `search_movies_reranked(query, top_k=5, candidate_k=50)` is what the backend
  calls: it asks `search_movies()` for `CANDIDATE_K` candidates, then rescores
  those pairs with `cross-encoder/ms-marco-MiniLM-L-6-v2` in one batched
  `predict()`. Each result keeps the `bi_score` and `bi_rank` it arrived with,
  so the two stages can be compared. The pair text is
  `f"Title: {title}, Overview: {overview}"` — deliberately identical to what
  `embed.py` vectorizes, so a ranking difference is the two models disagreeing
  rather than the inputs differing. `CANDIDATE_K` is a hard ceiling on recall:
  a film the bi-encoder ranks 51st can never be returned, however well the
  cross-encoder would have scored it. The cross-encoder is loaded lazily and
  cached separately from the bi-encoder, so a `search_disney.py`-only caller
  never pays to load it. Running the file directly gives a REPL that prints
  both rankings side by side with the elapsed time.
- `tests/test_tmdb.py` — checks that the TMDB credentials in `.env` work.
- `tests/check_vectors.py` — sanity-checks the `.npz`: shape, unit norms,
  distinct rows, row↔movie alignment, and the nearest neighbours of a known
  film. It reuses the stored vectors and never loads the model, so it says
  nothing about the query-encoding path that `search_movies()` uses.

`requirements.txt` lists direct dependencies only, pinned, with a comment saying
what each is for. Add new ones there rather than relying on whatever happens to
be installed in `.venv/`.

`build_catalog.py` queries seven TMDB company IDs (see `STUDIOS`): the Disney
labels — Pixar 3, Walt Disney Animation Studios 6125, Walt Disney Pictures 2,
Walt Disney Productions 3166 — plus the Disney-owned subsidiaries Marvel Studios
420, Lucasfilm 1, Lucasfilm Animation 108270. They are near-disjoint, not
nested: TMDB does not roll subsidiaries into a parent, so dropping any one
silently loses films (without 6125 there is no *Encanto*, without 3166 no
pre-1986 classics). Order matters — `studio` holds a single value and the loop
skips already-claimed IDs, so specific labels come first and the subsidiaries
last.

Shorts are excluded by a `--min-runtime` floor of 40 minutes; most TMDB entries
under these companies are shorts (Pixar alone: 141 titles, 35 features). Note
this also drops films whose runtime TMDB doesn't know.

Known limitation of the corpus: TMDB overviews are marketing blurbs describing
*premises*, median 42 words, and 7 entries have no overview at all (written as
`"N/A"`). The product premise is "describe a scene you remember", but the Lion
King overview never mentions the stampede and Up's never mentions the balloons.
This is now confirmed as the dominant source of bad results, and the fix is
enriching the embedded text, not a better index — see
[Retrieval quality](#retrieval-quality-the-open-problem).

Two encoding details that bite silently. The query must be encoded with the same
model as the documents and normalized the same way, or the dot product stops
being cosine similarity — and nothing errors, because two different models both
produce valid 384-dim unit vectors that dot-product fine and mean nothing. The
model name is currently duplicated in `embed.py` and `search_disney.py` and the
two MUST match; worth importing from one place before changing either. Separately,
`metadata` in the `.npz` is a 0-d object array holding the catalog dict, so
reading it back requires `allow_pickle=True` and `.item()`. Storing it as a JSON
string instead would avoid unpickling at server startup.

## Score display

`to_percent()` in `backend/app.py` is `sigmoid(logit) * 100`. The ms-marco
cross-encoder is trained with a binary relevance objective, so its raw output is
a logit (roughly ±11, centred near 0) and the sigmoid is the model's own
probability that the film answers the query. There is nothing to tune: the
hand-measured cosine band the bi-encoder needed is gone, exactly as the old note
here predicted would happen once a reranker landed. The map is still monotonic,
so it never reorders results.

These numbers are meaningfully ABSOLUTE, unlike the ordinal cosine scores they
replaced, so a fixed cutoff now behaves consistently across queries. Measured:
"a girl who lives on an island and goes on an adventure with a demigod" gives
87/67/64/7/2, while "a robot and a kid who are superheros" gives 8/1/0/0/0.

The known wart is that second example. Big Hero 6 is the correct answer and
reads 0%, because its overview says "high-tech heroes" and never "superhero" —
the model is being honest about thin source text, but a user reads 0% as a
broken search. Two ways out, undecided: show a "no strong match" banner when the
top result is under ~20% (only possible because these scores are calibrated), or
rescale an observed logit band the way the old code rescaled cosine. Enriching
the corpus (item 1 below) attacks the cause rather than the display.

## Retrieval quality: the open problem

`findings.txt` (repo root) is the running log of how search actually behaves,
with measured evidence. Two failures are documented there, and they have
different causes and different fixes:

- **Scene-level queries fail** because the corpus describes premises. Up's
  overview never contains "balloons". No model or index change retrieves
  information that was never encoded. This is the one that matters most — the
  product premise is "describe a scene you remember".
- **Filler in the query dominates the result** — addressed by the cross-encoder.
  "A movie where there is a robot and a kid who are superheros" ranked Big Hero
  6 18th on the bi-encoder alone; deleting the first five words moved it to 5th.
  Mean pooling makes an average, so proportion is what counts, not presence —
  half that query encodes "generic English sentence". Reranking the top 50 puts
  it at 5th with the filler left in.

Premise-level queries already work well ("lion cub whose father dies" → The Lion
King at 89%), so do not read a broad failure into it.

### Order of work

**Build the evaluation set first.** Everything below helps some queries and hurts
others, and without measurement "did that work" is unanswerable — the current
loop is typing a query and eyeballing it. 30–50 hand-written
`{query, expected_tmdb_id}` pairs in a JSON file, no framework.

This is now overdue rather than merely first: reranking shipped on the strength
of two eyeballed queries, so nothing here knows whether it helps on average or
what `candidate_k` should be. It also cannot be measured retroactively — the
bi-encoder-only baseline it should be compared against is still reproducible
today via `search_disney.py`'s REPL, and will stop being so the moment the
embedding model or the corpus changes.

Two rules that decide whether it is valid. Write queries **from memory of the
film, never by reading the overview** — otherwise it tests whether the model can
match text to a paraphrase of itself. And **label each row `scene` or `premise`**
and score them separately; premise queries already pass, so mixing them dilutes
exactly the signal being chased. Score Recall@5 (what users feel, and what the UI
shows) and MRR (sensitive to movement inside the top 5, which Recall@5 cannot
see). Record the baseline before changing anything — once the model changes the
old numbers are unrecoverable.

**Done — cross-encoder reranking**, in `disney_cross_encode.py` and wired into
the backend. Measured on CPU: ~0.15s for the full 50-pair rerank, well under the
~0.5s at k=25 estimated before building it, so k=50 is comfortable and there is
headroom to raise it. It does what it was predicted to do about filler — Big
Hero 6 on the robot/kid query moved 18 → 5 with the filler left in. Two caveats
carried forward: it is unmeasured over any query set, and `candidate_k = 50` was
picked by hand rather than from stage-1 Recall@k.

1. **Enrich the embedded text** — the fix for scene-level queries, and the only
   one that raises the recall ceiling rather than reordering within it.
   `/movie/{id}?append_to_response=keywords` returns dense concept terms
   ("robot", "superhero", "san fransokyo") plus the tagline in one call per film;
   826 calls, a couple of minutes. Note this is a `build_catalog.py` change, not
   just an `embed.py` one — `catalog.json` has no keywords field today. Cache
   into `catalog.json` so `embed.py` stays offline. Also rescues the 7 films
   whose overview is `"N/A"` and which no query can currently reach. Concatenating
   keywords onto the overview lengthens the document and by the mean-pooling
   logic dilutes it; embedding keywords as a **separate row** and taking each
   movie's max score avoids that entirely, at 1652 × 384 — still trivial to scan.
   Note there are now two consumers of the document text: whatever keywords land
   in `embed.py` must also reach `document_text()` in `disney_cross_encode.py`,
   or stage 2 will keep reranking on the thin overview that stage 1 has moved
   past. The mean-pooling dilution argument does not apply to the cross-encoder,
   so the separate-row trick is a stage-1 concern only.
2. **Swap the embedding model** — `multi-qa-MiniLM-L6-cos-v1` is trained on
   question/answer pairs (asymmetric: short query vs longer passage) rather than
   sentence-similarity (symmetric), which is the shape this project actually has.
   Drop-in: same architecture, 384 dims, same pooling and normalization; re-run
   `embed.py` after. Now that reranking has landed this is squarely a stage-1
   recall question, not a precision one — the reranker handles precision, but it
   only ever sees the `candidate_k` films stage 1 hands it, so anything stage 1
   misses is unrecoverable. Worth benchmarking `BAAI/bge-small-en-v1.5` in the
   same harness while set up for it.

Deliberately not doing: bolting an LLM onto the results. It hides the retrieval
problem rather than solving it, adds latency and non-determinism, and leaves no
way to say why a result ranked where it did. A cross-encoder is the principled
version of the same idea and is measurable with the same harness.

Useful trick once the eval set exists: run the cross-encoder over all 826 for
each eval query to get the true full-corpus ranking, then compare the two-stage
result against it. That measures exactly what the `candidate_k` cutoff costs
instead of guessing at it. Cheaper than this note used to assume — the measured
~0.15s for 50 pairs extrapolates to a few seconds per query and a couple of
minutes for a 40-query set, not the 20s/query estimated before the thing
existed. Worth doing as part of the eval harness rather than as a special
exercise.
