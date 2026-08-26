"""Minimal backend for movieSemanticSearch.

Serves the frontend and accepts a movie description from the user. For now the
only job of /api/search is to receive that prompt — the actual search comes
later.
"""

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = Flask(__name__, static_folder=None)


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

    # This is where the semantic search will go. For now, just confirm receipt.
    app.logger.info("Received description: %s", query)

    return jsonify({"received": query})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
