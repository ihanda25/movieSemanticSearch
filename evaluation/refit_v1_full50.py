"""Re-score the existing v1 checkpoint under the same full-50 evaluation used
for the v2 sweep, for a fair three-way comparison: v1 (1 hard negative mined
from each query's stored top-5) vs. neg_1/neg_4 (1 or 4 hard negatives mined
from the full CANDIDATE_K=50 pool). No retraining -- v1's checkpoint is fixed;
only how it's measured changes.

Run: .venv/bin/python evaluation/refit_v1_full50.py
Merges a "v1" entry into evaluation/finetune_v2_results.json.
"""
import json
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from disney_overview_search.disney_cross_encode import TOP_K  # noqa: E402
from finetune_v2 import build_reranking_samples, load_pool  # noqa: E402

V1_CHECKPOINT = ROOT / "models" / "cross-encoder-finetuned-v1" / "final"
RESULTS_PATH = ROOT / "evaluation/finetune_v2_results.json"
BEFORE_PATH = ROOT / "evaluation/finetune_v2_before.json"


def main():
    catalog, by_split = load_pool()
    val_samples = build_reranking_samples(catalog, by_split["val"])
    test_samples = build_reranking_samples(catalog, by_split["test"])
    val_evaluator = CrossEncoderRerankingEvaluator(val_samples, at_k=TOP_K, name="val", show_progress_bar=False)
    test_evaluator = CrossEncoderRerankingEvaluator(test_samples, at_k=TOP_K, name="test", show_progress_bar=False)

    before = json.loads(BEFORE_PATH.read_text())

    print(f"Loading v1 checkpoint from {V1_CHECKPOINT.relative_to(ROOT)}...")
    v1_model = CrossEncoder(str(V1_CHECKPOINT))
    after = {"val": val_evaluator(v1_model), "test": test_evaluator(v1_model)}

    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    results["v1"] = {
        "num_negatives": 1, "negative_source": "stored top-5 (not the full 50)",
        "checkpoint": str(V1_CHECKPOINT.relative_to(ROOT)),
        "before": before, "after": after,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote v1 entry to {RESULTS_PATH.relative_to(ROOT)}")

    for split in ("val", "test"):
        b, a = before[split], after[split]
        print(f"  {split}: production mrr@{TOP_K}={b[f'{split}_mrr@{TOP_K}']:.3f} "
              f"ndcg@{TOP_K}={b[f'{split}_ndcg@{TOP_K}']:.3f}  |  "
              f"v1 mrr@{TOP_K}={a[f'{split}_mrr@{TOP_K}']:.3f} "
              f"ndcg@{TOP_K}={a[f'{split}_ndcg@{TOP_K}']:.3f}")


if __name__ == "__main__":
    main()
