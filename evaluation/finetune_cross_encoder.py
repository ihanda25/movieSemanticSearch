"""Fine-tune the cross-encoder reranker on the agent-authored trial set.

Run: .venv/bin/python evaluation/finetune_cross_encoder.py
Requires evaluation/splits.json (build with evaluation/build_split.py).
Writes evaluation/finetune_results.json and a checkpoint under
models/cross-encoder-finetuned-v1/ (gitignored, regenerate rather than commit).

Objective: MultipleNegativesRankingLoss, a pairwise/contrastive objective --
for each query, push the confirmed movie's score above one explicit hard
negative (the wrong candidate this exact pretrained model currently scores
highest) plus the other positives in the batch as free in-batch negatives.
This targets ranking (what Recall@5/MRR measure) rather than the absolute
sigmoid percentage, which CLAUDE.md already documents as uncalibrated and out
of scope here. See CLAUDE.md "Retrieval quality" and README.md's fine-tuning
plan for the reasoning this follows.

Both the pre-fine-tune ("before") and post-fine-tune ("after") numbers are
measured with the SAME evaluator on the SAME val/test samples, sliced from
evaluation/splits.json, so the comparison is apples-to-apples on movies the
model never trained on. "before" is the actual current production model
(cross-encoder/ms-marco-MiniLM-L-6-v2, unmodified), not a re-derived estimate.
"""
import json
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import MultipleNegativesRankingLoss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disney_overview_search.documents import document_text  # noqa: E402
from disney_overview_search.disney_cross_encode import PRETRAINED_CROSS_ENCODER_MODEL as CROSS_ENCODER_MODEL, TOP_K  # noqa: E402

SEED = 42
MODEL_DIR = ROOT / "models" / "cross-encoder-finetuned-v1"


def load_inputs():
    catalog = json.loads((ROOT / "database/catalog.json").read_text())
    trials = json.loads((ROOT / "evaluation/agent_search_run.json").read_text())["trials"]
    split_of_group = json.loads((ROOT / "evaluation/splits.json").read_text())["split_of_group"]
    by_split = {"train": [], "val": [], "test": []}
    for trial in trials:
        by_split[split_of_group[trial["split_group"]]].append(trial)
    return catalog, by_split


def doc_for(catalog, movie_id):
    return document_text(catalog[str(movie_id)])


def build_train_dataset(catalog, trials):
    queries, positives, negatives = [], [], []
    skipped = 0
    for t in trials:
        wrong = [c for c in t["results"] if c["movie_id"] != t["expected_tmdb_id"]]
        if not wrong:
            skipped += 1
            continue
        hardest = max(wrong, key=lambda c: c["score"])
        queries.append(t["query"])
        positives.append(doc_for(catalog, t["expected_tmdb_id"]))
        negatives.append(doc_for(catalog, hardest["movie_id"]))
    if skipped:
        print(f"Skipped {skipped} training rows with no wrong candidate to use as a hard negative.")
    return Dataset.from_dict({"query": queries, "positive": positives, "negative": negatives})


def build_reranking_samples(catalog, trials):
    samples = []
    for t in trials:
        ordered = sorted(t["results"], key=lambda c: c["bi_rank"])
        samples.append({
            "query": t["query"],
            "positive": [doc_for(catalog, t["expected_tmdb_id"])],
            "documents": [doc_for(catalog, c["movie_id"]) for c in ordered],
        })
    return samples


def main():
    catalog, by_split = load_inputs()
    print({name: len(trials) for name, trials in by_split.items()})

    train_dataset = build_train_dataset(catalog, by_split["train"])
    val_samples = build_reranking_samples(catalog, by_split["val"])
    test_samples = build_reranking_samples(catalog, by_split["test"])

    val_evaluator = CrossEncoderRerankingEvaluator(val_samples, at_k=TOP_K, name="val", show_progress_bar=False)
    test_evaluator = CrossEncoderRerankingEvaluator(test_samples, at_k=TOP_K, name="test", show_progress_bar=False)

    print("Evaluating the current production model (before fine-tuning)...")
    base_model = CrossEncoder(CROSS_ENCODER_MODEL)
    before = {"val": val_evaluator(base_model), "test": test_evaluator(base_model)}
    del base_model

    print(f"Fine-tuning on {len(train_dataset)} training pairs...")
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    loss = MultipleNegativesRankingLoss(model)
    args = CrossEncoderTrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=0.1,  # fraction of training, not a step count -- see Transformers v5 warmup_ratio deprecation
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=val_evaluator.primary_metric,
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=SEED,
    )
    trainer = CrossEncoderTrainer(
        model=model, args=args, train_dataset=train_dataset, evaluator=val_evaluator, loss=loss,
    )
    trainer.train()

    print("Evaluating the fine-tuned model (after)...")
    after = {"val": val_evaluator(model), "test": test_evaluator(model)}

    final_dir = MODEL_DIR / "final"
    model.save_pretrained(str(final_dir))
    print(f"Saved best checkpoint to {final_dir.relative_to(ROOT)}")

    results = {
        "metadata": {
            "base_model": CROSS_ENCODER_MODEL,
            "seed": SEED,
            "train_pairs": len(train_dataset),
            "val_queries": len(val_samples),
            "test_queries": len(test_samples),
            "loss": "MultipleNegativesRankingLoss",
            "checkpoint": str(final_dir.relative_to(ROOT)),
        },
        "before": before,
        "after": after,
    }
    out_path = ROOT / "evaluation/finetune_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out_path.relative_to(ROOT)}")

    for split in ("val", "test"):
        b, a = before[split], after[split]
        print(f"\n{split} (n={len(val_samples) if split == 'val' else len(test_samples)}):")
        print(f"  base (bi-encoder order):      mrr@{TOP_K}={b[f'{split}_base_mrr@{TOP_K}']:.3f}  "
              f"ndcg@{TOP_K}={b[f'{split}_base_ndcg@{TOP_K}']:.3f}")
        print(f"  current production reranker:  mrr@{TOP_K}={b[f'{split}_mrr@{TOP_K}']:.3f}  "
              f"ndcg@{TOP_K}={b[f'{split}_ndcg@{TOP_K}']:.3f}")
        print(f"  fine-tuned reranker:           mrr@{TOP_K}={a[f'{split}_mrr@{TOP_K}']:.3f}  "
              f"ndcg@{TOP_K}={a[f'{split}_ndcg@{TOP_K}']:.3f}")


if __name__ == "__main__":
    main()
