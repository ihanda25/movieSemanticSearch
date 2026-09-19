"""v2 fine-tune sweep: how many hard negatives per training query, mined from
the full CANDIDATE_K=50 pool instead of v1's stored top-5.

Run: .venv/bin/python evaluation/finetune_v2.py --num-negatives 1
     .venv/bin/python evaluation/finetune_v2.py --num-negatives 4
     .venv/bin/python evaluation/finetune_v2.py --num-negatives 8

Requires evaluation/full_candidate_pool.json (build_candidate_pool.py).
Diagnosis this responds to (findings.txt's second 2026-09-17 entry): 22 of
33 stage-2 "still missed" queries were in v1's OWN training split and it
still didn't learn them, because v1 mined exactly 1 hard negative per query
from each query's stored top-5 (<=4 wrong candidates) rather than the real
50-candidate pool.

Also fixes a methodology gap: v1's val/test evaluation reranked only each
query's stored top-5, an easier task than reranking the real 50 the backend
hands the reranker. This script's val/test evaluation uses the full 50 for
both the "before" (production) and "after" (this config) numbers, so v1's
checkpoint is re-scored the same way for a fair three-way comparison -- see
compare_v2_sweep.py.

Writes:
  models/cross-encoder-finetuned-v2-neg{N}/final   -- checkpoint
  evaluation/finetune_v2_before.json                -- shared baseline
    (production model, full-50 val/test), computed once and reused so the
    sweep doesn't reload/rescore with the production model 3 times
  evaluation/finetune_v2_results.json               -- one entry per config
"""
import argparse
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

from disney_overview_search.disney_cross_encode import CROSS_ENCODER_MODEL, TOP_K  # noqa: E402
from disney_overview_search.documents import document_text  # noqa: E402

SEED = 42
BEFORE_PATH = ROOT / "evaluation/finetune_v2_before.json"
RESULTS_PATH = ROOT / "evaluation/finetune_v2_results.json"


def load_pool():
    catalog = json.loads((ROOT / "database/catalog.json").read_text())
    pool = json.loads((ROOT / "evaluation/full_candidate_pool.json").read_text())
    by_split = {"train": [], "val": [], "test": []}
    for entry in pool:
        by_split[entry["split"]].append(entry)
    return catalog, by_split


def doc_for(catalog, movie_id):
    return document_text(catalog[str(movie_id)])


def build_train_dataset(catalog, entries, num_negatives):
    queries, positives = [], []
    negative_cols = [[] for _ in range(num_negatives)]
    skipped = 0
    for e in entries:
        wrong = sorted(
            (c for c in e["candidates"] if c["movie_id"] != e["expected_tmdb_id"]),
            key=lambda c: c["production_score"], reverse=True,
        )[:num_negatives]
        if len(wrong) < num_negatives:
            skipped += 1
            continue
        queries.append(e["query"])
        positives.append(doc_for(catalog, e["expected_tmdb_id"]))
        for col, c in zip(negative_cols, wrong):
            col.append(doc_for(catalog, c["movie_id"]))
    if skipped:
        print(f"Skipped {skipped} rows with fewer than {num_negatives} candidates available.")
    data = {"query": queries, "positive": positives}
    for i, col in enumerate(negative_cols, start=1):
        data[f"negative_{i}"] = col
    return Dataset.from_dict(data)


def build_reranking_samples(catalog, entries):
    samples = []
    for e in entries:
        ordered = sorted(e["candidates"], key=lambda c: c["bi_rank"])
        samples.append({
            "query": e["query"],
            "positive": [doc_for(catalog, e["expected_tmdb_id"])],
            "documents": [doc_for(catalog, c["movie_id"]) for c in ordered],
        })
    return samples


def get_before(catalog, by_split, val_evaluator, test_evaluator):
    if BEFORE_PATH.exists():
        print(f"Reusing cached baseline from {BEFORE_PATH.relative_to(ROOT)}")
        return json.loads(BEFORE_PATH.read_text())
    print("Computing shared production-model baseline on full-50 val/test (first sweep run only)...")
    base_model = CrossEncoder(CROSS_ENCODER_MODEL)
    before = {"val": val_evaluator(base_model), "test": test_evaluator(base_model)}
    BEFORE_PATH.write_text(json.dumps(before, indent=2) + "\n")
    return before


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-negatives", type=int, required=True)
    args = parser.parse_args()
    tag = f"neg_{args.num_negatives}"
    model_dir = ROOT / "models" / f"cross-encoder-finetuned-v2-{tag}"

    catalog, by_split = load_pool()
    print({name: len(entries) for name, entries in by_split.items()})

    train_dataset = build_train_dataset(catalog, by_split["train"], args.num_negatives)
    val_samples = build_reranking_samples(catalog, by_split["val"])
    test_samples = build_reranking_samples(catalog, by_split["test"])
    val_evaluator = CrossEncoderRerankingEvaluator(val_samples, at_k=TOP_K, name="val", show_progress_bar=False)
    test_evaluator = CrossEncoderRerankingEvaluator(test_samples, at_k=TOP_K, name="test", show_progress_bar=False)

    before = get_before(catalog, by_split, val_evaluator, test_evaluator)

    print(f"[{tag}] Fine-tuning on {len(train_dataset)} training queries, "
          f"{args.num_negatives} explicit hard negatives each...")
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    loss = MultipleNegativesRankingLoss(model)
    train_args = CrossEncoderTrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # Not val_evaluator.primary_metric: it's "ndcg@5" until the evaluator has
        # actually been called once (a mutable-attribute side effect), which
        # get_before() skips on a cached baseline -- hardcode the real key instead.
        metric_for_best_model=f"val_ndcg@{TOP_K}",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=SEED,
    )
    trainer = CrossEncoderTrainer(
        model=model, args=train_args, train_dataset=train_dataset, evaluator=val_evaluator, loss=loss,
    )
    trainer.train()

    print(f"[{tag}] Evaluating on full-50 val/test...")
    after = {"val": val_evaluator(model), "test": test_evaluator(model)}

    final_dir = model_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"[{tag}] Saved to {final_dir.relative_to(ROOT)}")

    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    results[tag] = {
        "num_negatives": args.num_negatives,
        "train_queries": len(train_dataset),
        "val_queries": len(val_samples),
        "test_queries": len(test_samples),
        "checkpoint": str(final_dir.relative_to(ROOT)),
        "before": before,
        "after": after,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[{tag}] Wrote results to {RESULTS_PATH.relative_to(ROOT)}")

    for split in ("val", "test"):
        b, a = before[split], after[split]
        print(f"  {split}: production mrr@{TOP_K}={b[f'{split}_mrr@{TOP_K}']:.3f} "
              f"ndcg@{TOP_K}={b[f'{split}_ndcg@{TOP_K}']:.3f}  |  "
              f"{tag} mrr@{TOP_K}={a[f'{split}_mrr@{TOP_K}']:.3f} "
              f"ndcg@{TOP_K}={a[f'{split}_ndcg@{TOP_K}']:.3f}")


if __name__ == "__main__":
    main()
