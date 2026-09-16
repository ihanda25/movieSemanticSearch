"""Offline regression checks for multiple passages per movie and old indexes."""
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from disney_overview_search.documents import embedding_texts, document_text
from disney_overview_search.search_disney import load_database, search


class QueryModel:
    def encode(self, query, normalize_embeddings):
        return np.array([1., 0.])


a = {"id": 1, "title": "A", "overview": "Original", "keywords": ["robot"], "tagline": "Hello"}
b = {"id": 2, "title": "B", "overview": "Other"}
assert len(embedding_texts(a)) == 2
assert len(embedding_texts(b)) == 1
assert all(word in document_text(a) for word in ["Original", "robot", "Hello"])
vectors = np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2]])
hits = search("query", QueryModel(), vectors, [a, a, b], top_k=2)
assert [h["movie"]["id"] for h in hits] == [1, 2]
assert hits[0]["score"] == 0.9
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "index.npz"
    np.savez(path, embeddings=vectors[:2], metadata=np.array({"1": a, "2": b}, dtype=object))
    _, movies = load_database(path)
    assert [m["id"] for m in movies] == [1, 2]
    np.savez(path, embeddings=vectors, metadata=np.array({"1": a, "2": b}, dtype=object),
             movie_indices=np.array([0, 0, 1]))
    _, movies = load_database(path)
    assert [m["id"] for m in movies] == [1, 1, 2]
print("Enrichment regression checks passed")
