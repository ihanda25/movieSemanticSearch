# movieSemanticSearch

Describe a movie plot or scene you remember and get five suggested Disney,
Pixar, Marvel, or Lucasfilm movies. Search runs locally using pretrained
embedding and reranking models; it does not call an LLM to generate answers.

General plot lookup currently works better than scene lookup: the catalog's
short descriptions and keywords still omit many specific events.

## How a search works

1. **Prepare the catalog offline.** Cache movie descriptions, keywords, and
   taglines from TMDB. `embed.py` converts their text into numeric vectors using
   `all-MiniLM-L6-v2` and saves them in `database/disney_vector_db.npz`.
2. **Retrieve candidates.** Encode the user's query with the same model, compare
   it with every stored vector using cosine similarity, and keep 50 unique
   movies. The current index contains 1,546 vectors for 826 movies. A NumPy
   scan is enough at this size; there is no separate vector database service.
3. **Rerank candidates.** `cross-encoder/ms-marco-MiniLM-L-6-v2` reads each
   query/movie-text pair together and scores its relevance. This second model
   can compare the wording more closely than the initial vector search.
4. **Display five results.** Flask returns the five highest reranker scores to
   the frontend. A movie outside the initial 50 candidates cannot be recovered
   by the reranker.

Both models are pretrained. Local feedback collection is implemented; fine-tuning
is future work, described in the plan below.

## Project history: what we built, tried, and learned

### 1. Build a local movie catalog and the first semantic search

The first approach collected TMDB titles and short overviews for 826 feature
movies across Disney, Pixar, Marvel, and Lucasfilm. Discovery uses seven studio
IDs because TMDB does not automatically include subsidiaries under a parent
company. A 40-minute runtime floor excludes shorts (and also excludes movies
whose runtime TMDB does not know).

Each movie originally had one passage, `Title: ..., Overview: ...`, embedded
with `all-MiniLM-L6-v2` into a 384-dimensional vector. Both movie and query
vectors were normalized to unit length, making their dot product cosine
similarity. Search scanned the entire NumPy matrix and returned the nearest
movies. This established the **bi-encoder-only approach**: encode the query and
documents separately, then compare vectors.

We deliberately used an exact scan rather than adding a vector database or
approximate index: this catalog is small enough that scanning it is inexpensive.
The vectors are generated locally and cached on disk; TMDB is a data source,
not a service called for every search.

### 2. Connect search to the UI and investigate bad results

Flask was wired to serve both the static frontend and `/api/search`, so the UI
could submit a description and display movie results. Models and vectors are
cached in the backend rather than loaded per request. The server uses port 8000
because macOS AirPlay occupies port 5000.

Manual testing exposed two different problems:

- **Missing information:** Up's overview did not mention balloons or the flying
  house. A query could describe the movie correctly without matching the
  information we had indexed. Seven catalog entries had no overview at all.
- **Sensitivity to phrasing:** “A movie where there is a robot and a kid who
  are superheros” put Big Hero 6 at rank 18. Removing the opening filler moved
  it to rank 5; “inflatable robot and a boy genius” put it first. Correcting
  “superheros” to “superheroes” did not materially change that recorded example.

These observations showed that information coverage and ranking were separate
issues. Mean-pooled embeddings summarize the whole text, which is a plausible
contributor to wording sensitivity; the manual comparisons do not isolate the
model's internal cause or establish that filler always hurts.

### 3. Add cross-encoder reranking

We kept the original embedding model and added
`cross-encoder/ms-marco-MiniLM-L-6-v2`. The bi-encoder now retrieves 50 candidates;
the cross-encoder jointly reads the query and each candidate's description and
returns the best five. The original bi-encoder search remains available for
comparison. Each reranked result retains its original score and rank for analysis.

On the robot/kid example, reranking moved Big Hero 6 from rank 18 to rank 5.
That was encouraging, but the first decision was based on individual queries,
not a full evaluation set. The 50-candidate cutoff was initially chosen by hand.
Reranking cannot rescue a movie the first stage never retrieves, nor guarantee
that missing plot details are inferred correctly.

### 4. Replace eyeballing with a reproducible evaluation

We added 50 manually written premise queries with expected TMDB movie IDs and
an evaluation script that measures both stages on identical queries. Scene
queries were kept separately so easier premise searches would not hide scene
retrieval failures. The saved overview-only results established this baseline:

| Metric on 50 premise queries | Bi-encoder only | With reranking |
| --- | ---: | ---: |
| Correct movie first | 26/50 (52%) | 35/50 (70%) |
| Correct movie in top five | 42/50 (84%) | 42/50 (84%) |
| Mean reciprocal rank (MRR) | 0.660 | 0.762 |

MRR averages `1 / rank` for the correct movie; higher is better. In this harness,
the bi-encoder uses the full catalog ranking, while the reranker gives zero
when the correct movie is outside its candidates. Top-five recall measures
whether the correct movie is among the five suggestions the user sees.

Reranking improved placement overall but did not increase the number of top-five
hits: Bambi moved from 15 to 5, while Peter Pan moved from 4 to 7. Guardians of
the Galaxy, Up, and Star Wars never reached the original 50 candidates. This
gave us concrete failures to investigate instead of only a few good examples.

### 5. Enrich both stages with TMDB keywords and taglines

We fetched keywords and taglines for the same 826 movies, preserving the original
fields and catalog order so the experiment did not also change the movie set.
Of those records, 692 have nonempty keywords and 521 have nonempty taglines.
Empty TMDB fields remain empty; we did not generate replacement facts.

The original overview passage is retained, with an optional separate
keywords/tagline passage. This produces 1,546 vectors. Retrieval takes each
movie's best passage score and deduplicates movies before selecting candidates.
The reranker receives all fields together. Shared formatting and the embedding
model constant now live in `documents.py` so the two stages stay consistent.

The original `evaluation/baseline.json` and `evaluation/results.csv` were kept
untouched. A separate enriched run improved first-place accuracy to **82%** and
top-five recall to **96%**. No final rank worsened on those 50 queries. Vector
alignment/normalization, unique-movie retrieval, legacy-index loading, and an
API smoke test were checked. Detailed results appear in the evaluation section
below and `evaluation/enrichment_comparison.md`.

### 6. Test fresh wording in the UI and identify the next limitation

A fresh query, “A movie about a soldier who takes a serum and becomes
superhuman,” returned Civil War first and The First Avenger third. Repeating
the query directly against the enriched pipeline reproduced the UI result.
The First Avenger's stored text includes “super soldier” but not the serum;
that is a remaining information gap, not proof of the model's exact reason
for choosing Civil War.

The UI also displayed only 2% and 1% matches. The current backend turns raw
cross-encoder scores into percentages with `sigmoid(score) * 100`. This is
monotonic, so it preserves ranking, but **has not been calibrated to the chance
that a user intended a movie**. Earlier notes described these numbers as
reliable absolute probabilities; the observed results do not support that
claim. Display calibration and retrieval accuracy are distinct problems.

### Where the project stands now

End-to-end enriched search works locally with two pretrained models. The saved
development set has 48/50 top-five hits; Guardians of the Galaxy and Star Wars
remain misses. Fresh phrasings can still rank incorrectly, scene performance
has not been established by this enrichment run, and displayed percentages are
uncalibrated. Feedback collection now works in the UI and reranking terminal REPL;
there is no training job or fine-tuned model yet. A 377-query agent-authored
trial set (see [Agent-generated search trials](#agent-generated-search-trials))
exists to seed that eventual training data, but it is unreviewed and does not
change this picture on its own.

Alternative embedding models, full-catalog reranker comparisons, fuller plot
summaries, confidence calibration, and feedback-driven fine-tuning have been
discussed but are **not completed experiments**. The plan at the end of this
README describes the next phase. `findings.txt` preserves chronological notes;
its older proposals and statements describe earlier project states.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Rebuilding the movie catalog also needs a TMDB API token in a `.env` file at the
project root:

```
tmdb_token=<your TMDB API read access token>
```

## Run locally

```bash
.venv/bin/python backend/app.py
```

Then open http://127.0.0.1:8000

Alternatively, activate the virtual environment with `source .venv/bin/activate`
and run `python backend/app.py`. There is no separate frontend server or build.
On a fresh checkout, first run `.venv/bin/python disney_overview_search/embed.py`:
the catalog is checked in, but the vector index is not. Initial model loading
downloads weights; subsequent runs can use the locally cached models.

## Layout

- `frontend/` — static page (HTML/CSS/JS, no build step)
- `backend/app.py` — Flask server; serves the frontend and exposes
  `POST /api/search`, which retrieves and reranks movies, then returns the top
  five with title, year, overview, and a displayed match score.
- `database/` — `build_catalog.py` pulls 826 Disney, Pixar, Marvel, and
  Lucasfilm movies from TMDB into `catalog.json`. Already checked in; re-run
  with `.venv/bin/python database/build_catalog.py` to refresh.

## Enriched search

Cache TMDB keywords and taglines for the existing catalog (resumes after an
interruption), then rebuild the local index:

```bash
.venv/bin/python database/build_catalog.py --enrich-only
.venv/bin/python disney_overview_search/embed.py
```

A normal catalog rebuild also fetches these fields. Movies without keywords or
taglines retain empty fields; embedding and search need no TMDB connection.
The index stores the original overview plus an optional separate keyword/tagline
passage. Retrieval takes the maximum passage score per movie before choosing 50
unique candidates. The reranker sees the overview, keywords, and tagline together.
Restart an already-running backend after rebuilding so it reloads its cached index.

### Where enrichment comes from and how it is weighted

Keywords and taglines come directly from each movie's TMDB entry; they are not
generated by a model. `enrich_catalog()` in `database/build_catalog.py` requests
`/movie/{id}?append_to_response=keywords` and caches the returned keyword names
and tagline in `database/catalog.json`. The original overview stays unchanged.

`disney_overview_search/documents.py` formats these fields for both stages:

| Stage | How the fields affect matching |
| --- | --- |
| Bi-encoder retrieval | `embedding_texts()` produces an overview passage and, when available, a separate keywords/tagline passage, each including the title. `embed.py` embeds them separately. `search_disney.py` uses each movie's highest passage cosine score and selects 50 unique movies. It does not average the two scores. |
| Cross-encoder reranking | `document_text()` combines the title, overview, keywords, and tagline into one passage. `disney_cross_encode.py` pairs that passage with the query and produces one relevance score per movie. |

The combined reranker passage has this form (placeholders shown):

```text
Title: <title>, Overview: <overview>, Keywords: <keyword list>, Tagline: <tagline>
```

There is **no explicit field weighting**, such as 70% overview and 30% keywords.
The cross-encoder's learned model scores the query and combined passage jointly;
keywords can affect its interpretation of the whole passage. Attention is not
a fixed or reported percentage contribution for each field.

The bi-encoder score only selects candidates: it is **not blended with the
cross-encoder score**. Final ordering uses the cross-encoder score alone. The
backend applies a sigmoid for the displayed match percentage; that display is
not a validated probability that the user intended this movie.

### Evaluation

Keep the original evaluation files when comparing enrichment:

```bash
.venv/bin/python evaluation/evaluate.py --csv evaluation/enriched.csv --save evaluation/enriched.json
.venv/bin/python tests/check_enrichment.py
.venv/bin/python tests/check_vectors.py
```

`evaluation/baseline.json` and `evaluation/results.csv` are the original
overview-only baseline. The enriched outputs use the same models, 50 candidates,
and the same 50 premise queries.

On that set, enrichment improved first-place accuracy from 35/50 (70%) to
41/50 (82%), top-5 recall from 42/50 (84%) to 48/50 (96%), and MRR from 0.762
to 0.878. See `evaluation/enrichment_comparison.md` for changed ranks and
remaining misses. This measures enrichment in both stages together; separate
experiments are needed to attribute gains to retrieval versus reranking, or to
keywords versus taglines. These are development-set results, not a held-out
estimate, and do not measure scene-query performance.

## Collect feedback locally

Feedback is stored in `database/feedback.sqlite3`, created automatically on the
first search. SQLite ships with Python, so there is no extra dependency or
separate database server. It handles writes from the UI and terminal more safely
than appending rows to a shared CSV. The database and its sidecar files are
ignored by Git; they contain your local query history and labels. Back it up if
you want to preserve collected examples across machines or fresh checkouts.

In the UI, search normally and select **This is it**, **None of these**, or use
**Find title** to identify the correct catalog movie outside the five suggestions.
Selecting again updates the label for that search. No selection leaves an
unlabeled search; “none” records a rejection without inventing a correct answer.
A title absent from the catalog cannot be positively labeled yet.

For the same two-stage search and feedback flow in the terminal:

```bash
.venv/bin/python disney_overview_search/disney_cross_encode.py
```

After each result list, enter `1`–`5` to confirm a movie, `n` for none, `t` to
look up the correct title, or Enter to skip. The bi-encoder-only diagnostic REPL
is unchanged; use the cross-encoder REPL above to collect feedback.

The `searches` table contains immutable query/result snapshots. The `feedback`
table holds one current label per search ID, with an update timestamp. Repeating
a query creates a new search; deduplicate related queries when preparing training
splits. Only the five displayed results are saved, not all 50 internal candidates.
No model weights, rankings, or displayed percentages change when you submit a label.

View all local searches and their current labels without changing the database:

```bash
.venv/bin/python database/view_feedback.py
.venv/bin/python database/view_feedback.py --table feedback
.venv/bin/python database/view_feedback.py --table searches --details
```

The default view joins searches to labels and resolves selected movie titles.
`--table feedback` shows every feedback row; `--table searches --details` shows
all search fields and full result snapshots. There is no row limit. Run again
for fresh data. Successful UI selections show **Saved ✓** on the clicked button;
errors appear beside that button. Refresh the browser to load UI changes.

Export labeled searches for inspection (nested result details are in a JSON column):

```bash
.venv/bin/python -m disney_overview_search.feedback --export evaluation/feedback.csv
.venv/bin/python tests/check_feedback.py
```

The export includes all labeled outcomes, including “none”; it is not a ready-made
training dataset. Review ambiguous labels and construct negatives before training.
Restart the backend after installing these changes so the new API routes load.

### Agent-generated search trials

Agent-assisted dataset collection must exercise the real search and feedback
API, rather than merely inventing successful results. Agent queries and proposed
movie labels remain synthetic and need human review, even when the expected
movie appears first. Search agreement does not prove a label is correct.

Store these trials separately in `database/agent_feedback.sqlite3` (gitignored),
with an agent source marker. Never insert them into human feedback or count them
as independently confirmed user selections. You can inspect the separate store:

```bash
.venv/bin/python database/view_feedback.py --db database/agent_feedback.sqlite3
```

`evaluation/synthetic_seed.json` now holds 377 queries across 277 movies, built
up in three batches with different goals:

| Batch | Queries | Movies | Bias |
| --- | --- | --- | --- |
| 1 | 100 | 50 | premise + scene pair per movie, roughly even split |
| 2 | 100 | 50 | premise + scene pair per movie, deliberately scene-majority (60/40) |
| 3 | 177 | 177 | one query per movie, prioritizing new-movie breadth over per-movie depth |

Combined recall on this agent-authored set (not the hand-written development
set above — see the caveats below before reading too much into it): 212/377
top-1 (56%), 300/377 top-5 (80%); split by kind, premise is 84/122 top-1 (69%)
vs. 114/122 top-5 (93%), scene is 128/255 top-1 (50%) vs. 186/255 top-5 (73%).
The gap tracks the same premise-vs-scene split documented in `findings.txt`.

The actual API trial output is `evaluation/agent_search_run.json`, including
returned ranks and scores for every trial. Validate the seed with
`.venv/bin/python evaluation/check_synthetic_seed.py` — it hard-codes the
current row count, so bump that assertion whenever the seed grows.
`evaluation/collect_agent_feedback.py` runs the batch through Flask's real
route handlers without opening a browser. It now appends rather than refusing
outright: it diffs the seed against already-recorded normalized query text and
processes only the rows it hasn't seen yet, so re-running after growing the
seed file never replays or duplicates an earlier batch into
`database/agent_feedback.sqlite3` or the run report.

Human review of this set is starting with the `outcome=other` rows (the
correct movie missing from the top 5 entirely) — those are the highest-value
rows for eventual fine-tuning and the most likely to contain an agent
misremembering a plot detail, since nothing has independently checked them yet.

Write queries before inspecting the movie's indexed text. Keep related queries
and movies grouped when constructing dataset splits, and check overlap with the
existing development evaluation. Generated examples are candidate training data,
not a new held-out test set or a substitute for real user feedback.

## Plan: user feedback and periodic fine-tuning

Feedback collection (step 1) is implemented. Training and deployment below remain
a proposed next phase, not an implemented training system. The goal is
to learn which movie a user intended from explicit feedback while keeping the
existing retrieval architecture. This is supervised learning from feedback,
not an online reinforcement-learning loop or an LLM wrapper.

### 1. Collect explicit selections

The UI now provides a **“This is it”** action on each suggestion, plus **“None of these”** and
a title lookup so the user can identify a movie outside the returned five.
SQLite stores the query, displayed movie IDs and order, stage scores, selected
movie ID, timestamps, source (UI or terminal), model names, and index SHA-256.
It also stores the displayed movie passages so later catalog edits do not erase
that evidence. Manually authored versus real-user provenance is not separately
collected yet; distinguish those sources when preparing a training dataset.

An ordinary click is not necessarily confirmation. “None of these” alone does
not identify a positive example. Remakes and vague franchise queries can have
multiple acceptable answers, so unselected results must not automatically be
treated as definitely wrong. Corrections and duplicate submissions should be
resolved before training.

### 2. Turn feedback into training examples

Start with the cross-encoder. Pair a query with the confirmed movie and a few
plausible incorrect candidates, called **hard negatives**. For example:

```text
Query: A soldier takes a serum and becomes superhuman.
Preferred movie: Captain America: The First Avenger
Incorrect alternative: Captain America: Civil War
```

Fine-tuning starts from pretrained model weights and adjusts them so the
confirmed movie scores above incorrect alternatives. A pairwise ranking
objective is one approach to test. The model still jointly reads the query and
movie text; it does not need to generate a reasoning explanation or answer.

Missing facts remain a separate issue: if neither the overview nor keywords
mention the serum, improving the source description may help more than training.
Use feedback immediately to find such gaps, even before enough labels exist
for a useful fine-tuning experiment.

### 3. Accumulate enough varied queries

Use these as rough planning ranges, not guaranteed training thresholds:

| Distinct labeled queries | Intended use |
| --- | --- |
| 50–100 | Check feedback quality and build evaluation cases. |
| A few hundred | Run an exploratory fine-tuning experiment; watch closely for overfitting. |
| Roughly 1,000–3,000 | Aim for a more informative training/evaluation comparison across varied movies and phrasing. |
| Several thousand more | Expand coverage and investigate performance on different query types. |

One query with one positive and four negatives creates five pairs, but remains
only **one independent query**. Repeated clicks or near-identical paraphrases
do not provide the coverage of different user memories. These totals must also
leave room for validation and test examples, rather than all going into training.

Collect successful searches and failures, short and detailed descriptions, and
both scene and premise queries. Keep scene and premise metrics separate.
The practical first milestone is a few hundred usable queries for an exploratory
run, then approximately 1,000 for a broader assessment—not a promise of improvement.

The 377-query agent-authored set in
[Agent-generated search trials](#agent-generated-search-trials) sits in the "a
few hundred" row above by count alone, but it does not yet satisfy this
section: it is single-source (one agent's memory, not varied real users),
unreviewed, and has not been split into training/validation/test. Human
review — starting with the `outcome=other` rows — comes before any of that
counts toward an exploratory run.

### 4. Train separately, evaluate, then decide whether to deploy

1. Freeze a versioned snapshot of feedback and the catalog text used for training.
2. Split it into training, validation, and untouched test sets. Keep duplicate
   or closely paraphrased queries in the same split to prevent answer leakage.
   The existing 50 queries are development cases because they already guided
   changes; create additional held-out cases for judging generalization.
3. Run a separate training job on the selected pretrained checkpoint. Record
   its data version, settings, and output checkpoint. Training does not happen
   inside the Flask request handler or after each click.
4. Use validation results to choose training settings, then compare the candidate
   with the current system on held-out queries. Measure first-place accuracy,
   Recall@5, MRR, candidate Recall@50, and search latency. Review regressions as
   well as improvements; MRR rewards moving the correct movie nearer the top.
5. Deploy only if the comparison supports improvement. Point the backend at
   the accepted reranker checkpoint and restart it; retain the previous version
   for rollback. Fine-tuning only the reranker does not require new embeddings.

Initially trigger this manually after a meaningful batch of new labels. A weekly
or monthly schedule is only useful when enough new data has accumulated.
Automate collection, dataset preparation, training, and reports later; do not
automatically promote every newly trained checkpoint.

### 5. Improve the stage responsible for the failure

| Failure | Next experiment |
| --- | --- |
| Correct movie reaches the 50 candidates but ranks too low | Improve descriptions or fine-tune the cross-encoder with confirmed positives and hard negatives. |
| Correct movie never reaches the 50 candidates | Improve descriptions, evaluate the candidate cutoff, or fine-tune the bi-encoder. Changing the bi-encoder requires rebuilding all document vectors with the matching model. |
| Correct results display tiny match percentages | Treat this as score calibration/display work, separate from ranking quality. |

For future confidence calibration, collect incorrect and no-match searches as
well as correct selections, and validate any score-to-probability mapping on
separate data. Rescaling numbers alone does not establish confidence or fix
the ordering. The current percentage display remains uncalibrated.
