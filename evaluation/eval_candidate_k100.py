"""Does raising CANDIDATE_K from 50 to 100 actually recover any of the 36
stage-1 ceiling misses -- measured, not just "back in reach"?

Run: .venv/bin/python evaluation/eval_candidate_k100.py
Requires evaluation/full_candidate_pool_k100.json (build_candidate_pool.py
--candidate-k 100). Uses the best available reranker checkpoint (v1, which
scored identically to neg_1 under fair evaluation -- see findings.txt).

Production candidates' scores are already cached in the k100 pool (from
build_candidate_pool.py), so only the fine-tuned checkpoint needs a fresh
predict() pass over the wider 100-candidate pool.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disney_overview_search.documents import document_text  # noqa: E402
from disney_overview_search.disney_cross_encode import TOP_K  # noqa: E402

CHECKPOINT = ROOT / "models" / "cross-encoder-finetuned-v1" / "final"


def rank_of(ordered_ids, expected_id, k=None):
    ids = ordered_ids[:k] if k else ordered_ids
    for rank, movie_id in enumerate(ids, start=1):
        if movie_id == expected_id:
            return rank
    return None


def main():
    catalog = json.loads((ROOT / "database/catalog.json").read_text())
    pool = json.loads((ROOT / "evaluation/full_candidate_pool_k100.json").read_text())

    print(f"Loading fine-tuned checkpoint {CHECKPOINT.relative_to(ROOT)}...")
    model = CrossEncoder(str(CHECKPOINT))

    classification = Counter()
    per_split = {"train": [], "val": [], "test": []}

    print(f"Scoring {len(pool)} queries' full 100-candidate pool with the fine-tuned checkpoint...")
    for i, e in enumerate(pool, start=1):
        expected_id = e["expected_tmdb_id"]
        candidates = sorted(e["candidates"], key=lambda c: c["bi_rank"])
        production_order = [c["movie_id"] for c in sorted(candidates, key=lambda c: -c["production_score"])]

        pairs = [(e["query"], document_text(catalog[str(c["movie_id"])])) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        finetuned_order = [c["movie_id"] for c, _ in sorted(zip(candidates, scores), key=lambda p: -p[1])]

        prod_rank = rank_of(production_order, expected_id, k=TOP_K)
        ft_rank = rank_of(finetuned_order, expected_id, k=TOP_K)
        in_pool = any(c["movie_id"] == expected_id for c in candidates)

        if not in_pool:
            cls = "stage-1 ceiling miss (still beyond 100)"
        elif prod_rank and ft_rank:
            cls = "hit before and after"
        elif not prod_rank and ft_rank:
            cls = "fixed by fine-tune"
        elif prod_rank and not ft_rank:
            cls = "regressed"
        else:
            cls = "still missed"
        classification[cls] += 1
        per_split[e["split"]].append((cls, e["expected_title"], prod_rank, ft_rank))

        if i % 50 == 0:
            print(f"  {i}/{len(pool)}")

    print("\nClassification at CANDIDATE_K=100 (all 377 queries):")
    for label, count in classification.most_common():
        print(f"  {label:<45} {count}")

    print("\nBy split:")
    for split, rows in per_split.items():
        c = Counter(r[0] for r in rows)
        print(f"  {split}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))

    out_path = ROOT / "evaluation/candidate_k100_results.json"
    out_path.write_text(json.dumps({
        "classification": dict(classification),
        "by_split": {split: [{"classification": c, "title": t, "prod_rank": pr, "ft_rank": fr}
                              for c, t, pr, fr in rows] for split, rows in per_split.items()},
    }, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
