"""Minimal backend for movieSemanticSearch.

Serves the frontend and runs a description through the semantic search in
`disney_overview_search/search_disney.py`.
"""

import math
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

# Running this file directly puts backend/ on sys.path, not the repo root, so
# the sibling package needs the root added before it can be imported.
sys.path.insert(0, str(ROOT))

from disney_overview_search.disney_cross_encode import search_movies_reranked  # noqa: E402

TOP_K = 5

app = Flask(__name__, static_folder=None)


def to_percent(score):
    """Turns a cross-encoder logit into a 0-100 'match' figure.

    The ms-marco cross-encoder is trained with a binary relevance objective, so
    its raw output is a logit and sigmoid(logit) is the model's own probability
    that the document answers the query. That replaces the hand-tuned rescaling
    the bi-encoder needed: cosine scores sat in a narrow band whose endpoints
    had to be measured by hand, whereas these numbers mean something absolute
    on their own. Still monotonic, so it never reorders results.

    Consequence worth knowing: a query the model has no good answer for now
    reads in the single digits rather than a comfortable-looking 40%.
    """

    # Clamped only so math.exp cannot overflow; real logits from this model sit
    # within about +/-11.
    score = max(-30.0, min(30.0, score))
    return round(100 / (1 + math.exp(-score)))


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
    for hit in search_movies_reranked(query, top_k=TOP_K):
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
    # Load both models and the vectors up front so the first real search is
    # not the one that pays for it. With debug=True the reloader runs this
    # twice, once in the parent and once in the child it spawns.
    print("Loading models and vector database...")
    search_movies_reranked("warmup")
    print("Ready on http://127.0.0.1:8000")

    app.run(host="127.0.0.1", port=8000, debug=True)
