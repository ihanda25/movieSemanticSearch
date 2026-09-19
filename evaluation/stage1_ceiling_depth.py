"""How far outside CANDIDATE_K=50 do the true stage-1 ceiling misses actually
sit? Cheap, no training: if most are just past the cutoff, raising
CANDIDATE_K alone could rescue several before touching training again.

Run: .venv/bin/python evaluation/stage1_ceiling_depth.py
Reads the 36 "stage-1 ceiling miss" rows from evaluation/stage1_ceiling.csv
(built by evaluation/diagnose_stage1_ceiling.py) and re-retrieves each with no
cutoff to find the confirmed movie's true bi-encoder rank.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disney_overview_search.search_disney import search_movies  # noqa: E402

FULL_CORPUS = 1_000_000


def main():
    rows = list(csv.DictReader((ROOT / "evaluation/stage1_ceiling.csv").open()))
    misses = [r for r in rows if r["classification"] == "stage-1 ceiling miss"]
    print(f"Re-retrieving true rank for {len(misses)} stage-1 ceiling misses...")

    results = []
    for r in misses:
        hits = search_movies(r["query"], top_k=FULL_CORPUS)
        rank = next((i for i, h in enumerate(hits, start=1) if h["movie"]["id"] == int(r["expected_tmdb_id"])), None)
        results.append((rank, r["split"], r["expected_title"], r["query"]))

    results.sort(key=lambda x: (x[0] is None, x[0]))
    buckets = {"51-75": 0, "76-100": 0, "101-200": 0, "201-400": 0, "401+": 0, "not found": 0}
    for rank, split, title, query in results:
        print(f"  {rank!s:>6}  [{split:<5}]  {title}")
        if rank is None:
            buckets["not found"] += 1
        elif rank <= 75:
            buckets["51-75"] += 1
        elif rank <= 100:
            buckets["76-100"] += 1
        elif rank <= 200:
            buckets["101-200"] += 1
        elif rank <= 400:
            buckets["201-400"] += 1
        else:
            buckets["401+"] += 1

    print("\nDistribution:")
    for label, count in buckets.items():
        print(f"  {label:<12} {count}")

    for threshold in (75, 100, 150, 200):
        recoverable = sum(1 for rank, *_ in results if rank and rank <= threshold)
        print(f"Raising CANDIDATE_K to {threshold} would put {recoverable}/{len(misses)} "
              f"of these back within reach of the reranker.")


if __name__ == "__main__":
    main()
