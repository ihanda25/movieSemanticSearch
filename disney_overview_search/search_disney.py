"""Semantic search over the embedded Disney catalog.

`search_movies(query)` is the entry point: it loads the model and the vector
database once, caches them, and returns the closest movies to a description.
"""

from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "database" / "disney_vector_db.npz"
sys.path.insert(0, str(ROOT))
from disney_overview_search.documents import EMBEDDING_MODEL

# Loading the model and the .npz costs a few seconds, so do it once and reuse it
# across queries rather than on every request.
_cache = None


def load_database(filepath):
    """Returns (embeddings, movies) where row i of embeddings <-> movies[i]."""

    with np.load(filepath, allow_pickle=True) as data:
        if "embedding_model" in data and data["embedding_model"].item() != EMBEDDING_MODEL:
            raise ValueError("Index model differs from query model; rebuild with embed.py")
        movies = list(data["metadata"].item().values())
        indices = data["movie_indices"] if "movie_indices" in data else np.arange(len(movies))
        embeddings = data["embeddings"]
        if len(indices) != len(embeddings):
            raise ValueError("Vector rows and movie mapping differ in length")
        return embeddings, [movies[int(i)] for i in indices]


def search(query, model, db_embeddings, db_metadata, top_k=3):
    # 1. Tokenize, Embed, and Normalize the user's prompt
    # It must be the exact same model and normalization used during ingestion
    query_vector = model.encode(query, normalize_embeddings=True)

    # 2. The Math: Dot Product
    # This multiplies the query vector against every movie vector simultaneously
    # Returns an array of similarity scores (e.g., [0.12, 0.85, 0.04, ...])
    scores = np.dot(db_embeddings, query_vector)

    # 3. Sort and extract the top K results
    # np.argsort sorts the scores from lowest to highest.
    # [::-1] reverses it to highest to lowest. [:top_k] grabs the top ones.
    top_indicies = np.argsort(scores)[::-1]

    # 4. Fetch the original JSON metadata for those top scores
    results = []
    seen = set()
    if top_k <= 0:
        return results
    for idx in top_indicies:
        movie_id = db_metadata[idx]["id"]
        if movie_id in seen:
            continue
        seen.add(movie_id)
        results.append({
            # float() because np.float32 is not JSON-serializable
            "score": float(scores[idx]),
            "movie": db_metadata[idx]
        })
        if len(results) == top_k:
            break
    return results


def search_movies(query, top_k=3):
    """Searches the catalog for `query`. Safe to call per request."""

    global _cache
    if _cache is None:
        if not DB_FILE.exists():
            raise FileNotFoundError(
                f"{DB_FILE} not found -- run disney_overview_search/embed.py first"
            )
        embeddings, movies = load_database(DB_FILE)
        _cache = (SentenceTransformer(EMBEDDING_MODEL), embeddings, movies)

    model, embeddings, movies = _cache
    return search(query, model, embeddings, movies, top_k=top_k)


def main():
    """Interactive prompt for trying out searches from the terminal. For testing purposes"""

    print("Loading model and vector database...")
    search_movies("warmup")  # fills the cache so the first real query is instant
    print("Ready. Describe a scene you remember. Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not query:
            continue

        for result in search_movies(query, top_k=3):
            movie = result["movie"]
            print(f"  {result['score']:.3f}  {movie['title']}")
            print(f"         {movie['overview'][:100]}")
        print()


if __name__ == "__main__":
    main()
