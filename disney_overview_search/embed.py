"""Build offline overview and enrichment vectors, mapped back to unique movies."""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from disney_overview_search.documents import EMBEDDING_MODEL, embedding_texts

JSON_PATH = ROOT / "database" / "catalog.json"
OUTPUT_FILE = ROOT / "database" / "disney_vector_db.npz"


def build_index():
    movies = json.loads(JSON_PATH.read_text())
    texts, movie_indices = [], []
    for index, movie in enumerate(movies.values()):
        passages = embedding_texts(movie)
        texts.extend(passages)
        movie_indices.extend([index] * len(passages))
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    temporary = OUTPUT_FILE.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, embeddings=embeddings,
                        metadata=np.array(movies, dtype=object),
                        movie_indices=np.array(movie_indices, dtype=np.int64),
                        embedding_model=np.array(EMBEDDING_MODEL))
    temporary.replace(OUTPUT_FILE)
    print(f"Wrote {len(texts)} vectors for {len(movies)} movies")


if __name__ == "__main__":
    build_index()
