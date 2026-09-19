"""Cache the full CANDIDATE_K=50 bi-encoder pool + production cross-encoder
scores for every query, once, so the v2 sweep and its evaluation don't each
re-retrieve and re-score from scratch.

Run: .venv/bin/python evaluation/build_candidate_pool.py
Writes evaluation/full_candidate_pool.json.

This also fixes a methodology gap from v1: v1's val/test evaluation
(finetune_cross_encoder.py) reranked only each query's already-stored top-5
candidates, which is an easier task than the real one (reranking among the
full 50 CANDIDATE_K the backend actually hands the reranker). Everything
downstream of this cache -- training-negative mining and val/test evaluation
alike -- uses the full 50, so v1's checkpoint can be re-scored fairly on the
same basis as v2.
"""
import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disney_overview_search.disney_cross_encode import CANDIDATE_K, CROSS_ENCODER_MODEL  # noqa: E402
from disney_overview_search.documents import document_text  # noqa: E402
from disney_overview_search.search_disney import search_movies  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--out", type=Path,
                         default=ROOT / "evaluation/full_candidate_pool.json")
    args = parser.parse_args()

    trials = json.loads((ROOT / "evaluation/agent_search_run.json").read_text())["trials"]
    split_of_group = json.loads((ROOT / "evaluation/splits.json").read_text())["split_of_group"]

    print("Loading production cross-encoder...")
    model = CrossEncoder(CROSS_ENCODER_MODEL)

    pool = []
    print(f"Retrieving the full {args.candidate_k}-candidate pool and scoring for {len(trials)} queries...")
    for i, t in enumerate(trials, start=1):
        hits = search_movies(t["query"], top_k=args.candidate_k)
        pairs = [(t["query"], document_text(hit["movie"])) for hit in hits]
        scores = model.predict(pairs, show_progress_bar=False)
        candidates = [
            {"movie_id": hit["movie"]["id"], "title": hit["movie"]["title"],
             "bi_rank": rank, "bi_score": hit["score"], "production_score": float(score)}
            for rank, (hit, score) in enumerate(zip(hits, scores), start=1)
        ]
        pool.append({
            "query": t["query"], "kind": t["kind"], "split_group": t["split_group"],
            "split": split_of_group[t["split_group"]],
            "expected_tmdb_id": t["expected_tmdb_id"], "expected_title": t["expected_title"],
            "candidates": candidates,
        })
        if i % 50 == 0:
            print(f"  {i}/{len(trials)}")

    args.out.write_text(json.dumps(pool, indent=2) + "\n")
    print(f"Wrote {args.out} ({len(pool)} queries)")


if __name__ == "__main__":
    main()
