"""Add train-split before/after numbers to evaluation/finetune_results.json.

finetune_cross_encoder.py only scores val (during training) and test (after),
deliberately -- train-split reranking numbers reflect memorization, not
generalization, so they're not part of the deploy decision. This script exists
purely to surface that memorization number alongside the others for a
complete before/after table across all three splits.

Run: .venv/bin/python evaluation/eval_train_split.py
Requires models/cross-encoder-finetuned-v1/final to already exist (run
finetune_cross_encoder.py first).
"""
import json
from pathlib import Path

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

from finetune_cross_encoder import CROSS_ENCODER_MODEL, MODEL_DIR, TOP_K, build_reranking_samples, load_inputs

ROOT = Path(__file__).resolve().parent.parent


def main():
    catalog, by_split = load_inputs()
    train_samples = build_reranking_samples(catalog, by_split["train"])
    evaluator = CrossEncoderRerankingEvaluator(train_samples, at_k=TOP_K, name="train", show_progress_bar=False)

    print(f"Evaluating base and fine-tuned models on the {len(train_samples)} training queries...")
    base_model = CrossEncoder(CROSS_ENCODER_MODEL)
    before = evaluator(base_model)
    del base_model

    finetuned_model = CrossEncoder(str(MODEL_DIR / "final"))
    after = evaluator(finetuned_model)

    results_path = ROOT / "evaluation/finetune_results.json"
    results = json.loads(results_path.read_text())
    results["before"]["train"] = before
    results["after"]["train"] = after
    results["metadata"]["train_queries"] = len(train_samples)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Updated {results_path.relative_to(ROOT)} with train-split numbers.")
    print(f"  base (bi-encoder order):      mrr@{TOP_K}={before[f'train_base_mrr@{TOP_K}']:.3f}  "
          f"ndcg@{TOP_K}={before[f'train_base_ndcg@{TOP_K}']:.3f}")
    print(f"  current production reranker:  mrr@{TOP_K}={before[f'train_mrr@{TOP_K}']:.3f}  "
          f"ndcg@{TOP_K}={before[f'train_ndcg@{TOP_K}']:.3f}")
    print(f"  fine-tuned reranker:           mrr@{TOP_K}={after[f'train_mrr@{TOP_K}']:.3f}  "
          f"ndcg@{TOP_K}={after[f'train_ndcg@{TOP_K}']:.3f}")


if __name__ == "__main__":
    main()
