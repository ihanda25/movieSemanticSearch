"""Sanity-check the embedded catalog written by disney_overview_search/embed.py."""

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = ROOT / "database" / "disney_vector_db.npz"

if not NPZ_PATH.exists():
    print(f"Error: {NPZ_PATH.relative_to(ROOT)} not found -- run embed.py first")
    exit(1)

d = np.load(NPZ_PATH, allow_pickle=True)
emb = d["embeddings"]
meta = d["metadata"].item()      # 0-d object array holding the dict
movies = list(meta.values())     # row i of emb <-> movies[i]

print("shape:      ", emb.shape, emb.dtype)
print("movies:     ", len(movies))
print("aligned:    ", emb.shape[0] == len(movies))

norms = np.linalg.norm(emb, axis=1)
print(f"norms:       min {norms.min():.4f}  max {norms.max():.4f}  (want ~1.0)")

uniq = len(np.unique(emb, axis=0))
print(f"unique rows: {uniq} of {emb.shape[0]} (want all distinct)")

print("\nfirst 3:", [m["title"] for m in movies[:3]])

# Nearest neighbours of a known film -- the real test that the vectors mean something.
probe = next(i for i, m in enumerate(movies) if m["title"] == "The Lion King")
sims = emb @ emb[probe]
print(f"\nclosest to {movies[probe]['title']!r}:")
for i in np.argsort(-sims)[:6]:
    print(f"  {sims[i]:.3f}  {movies[i]['title']}")
