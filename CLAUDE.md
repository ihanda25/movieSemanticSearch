# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A quick UI where you describe a scene you remember and get back the Disney movie
it came from.

**Current state: the search does not exist yet.** `POST /api/search` receives the
user's description, logs it, and echoes it back so the round trip is visible in
the UI. The dataset exists (see `database/`) and it has been embedded
(`disney_overview_search/embed.py` writes an 826 × 384 matrix to
`database/disney_vector_db.npz`), but nothing is wired into the backend yet —
`search()` still just echoes.

The plan is local embeddings (sentence-transformers) with brute-force cosine
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
- `backend/app.py` — the whole backend. Flask 3 is its only dependency.
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
- `tests/test_tmdb.py` — checks that the TMDB credentials in `.env` work.
- `tests/check_vectors.py` — sanity-checks the `.npz`: shape, unit norms,
  distinct rows, row↔movie alignment, and the nearest neighbours of a known
  film. It reuses the stored vectors and never loads the model, so it says
  nothing about the query-encoding path that `search()` will need.

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
If scene-level recall is poor, the fix is enriching the embedded text — TMDB
`/movie/{id}/keywords` and taglines are the cheap sources — not a better index.

The `search()` handler is the seam for future work: swap the echo for real
search there and the frontend contract (`{query} -> {received}`) is the only
thing that needs to change alongside it. Two things to know before wiring it up:
the query has to be encoded with the same model and normalized the same way, or
the dot product stops being cosine similarity; and `metadata` in the `.npz` is a
0-d object array holding the catalog dict, so reading it back requires
`allow_pickle=True` and `.item()`. Storing it as a JSON string instead would
avoid unpickling at server startup.
