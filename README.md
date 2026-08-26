# movieSemanticSearch

Quick UI where you can find a disney movie based on a specific scene

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

## Layout

- `frontend/` — static page (HTML/CSS/JS, no build step)
- `backend/app.py` — Flask server; serves the frontend and exposes
  `POST /api/search`, which currently just receives the user's description
  and echoes it back. The search itself goes here next.
- `database/` — `build_catalog.py` pulls 826 Disney, Pixar, Marvel, and
  Lucasfilm movies from TMDB into `catalog.json`. Already checked in; re-run
  with `.venv/bin/python database/build_catalog.py` to refresh.
