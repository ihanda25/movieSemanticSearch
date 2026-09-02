"""Minimal backend for movieSemanticSearch.

Serves the frontend and runs a description through the semantic search in
`disney_overview_search/search_disney.py`.
"""

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

# Running this file directly puts backend/ on sys.path, not the repo root, so
# the sibling package needs the root added before it can be imported.
sys.path.insert(0, str(ROOT))

from disney_overview_search.search_disney import search_movies  # noqa: E402

TOP_K = 5

# Cosine scores from all-MiniLM-L6-v2 occupy a narrow band rather than the full
# 0..1 range: measured across a spread of queries, genuine matches land
# 0.45-0.55, a query whose answer is not in the corpus tops out near 0.29, and
# off-domain nonsense near 0.15. Multiplying by 100 would label a correct hit
# "49% match", so rescale that observed band onto 0-100 instead. The map is
# linear and monotonic, so it never reorders results, and a query with no good
# answer still reads as uniformly low instead of being stretched up to 100%.
SCORE_FLOOR = 0.15
SCORE_CEILING = 0.60

app = Flask(__name__, static_folder=None)


def to_percent(score):
    """Rescales a raw cosine score onto a 0-100 'match' figure."""

    ratio = (score - SCORE_FLOOR) / (SCORE_CEILING - SCORE_FLOOR)
    return round(max(0.0, min(1.0, ratio)) * 100)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.post("/api/search")
def search():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Please describe a movie."}), 400

    app.logger.info("Received description: %s", query)

    results = []
    for hit in search_movies(query, top_k=TOP_K):
        movie = hit["movie"]
        # release_date is "YYYY-MM-DD", or missing/empty for a few entries.
        release_date = movie.get("release_date") or ""
        results.append({
            "title": movie.get("title", "Unknown title"),
            "year": release_date[:4],
            "overview": movie.get("overview", ""),
            "match": to_percent(hit["score"]),
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    # Load the model and vectors up front so the first real search is not the
    # one that pays for it. With debug=True the reloader runs this twice, once
    # in the parent and once in the child it spawns.
    print("Loading model and vector database...")
    search_movies("warmup")
    print("Ready on http://127.0.0.1:8000")

    app.run(host="127.0.0.1", port=8000, debug=True)
