"""Per-query before/after: the production reranker vs. the fine-tuned checkpoint.

Run: .venv/bin/python evaluation/compare_checkpoints.py
Requires models/cross-encoder-finetuned-v1/final (run finetune_cross_encoder.py
first). Writes:

  evaluation/finetune_comparison.csv  -- one row per query (all 377), rank of
    the confirmed movie among that query's already-retrieved top-5 candidates
    under each model, so every query is inspectable, not just the aggregates.
  evaluation/finetune_comparison.md   -- aggregate metrics table (train/val/
    test, from finetune_results.json) plus the biggest rank changes and any
    regressions, same shape as evaluation/enrichment_comparison.md.

Rank here is 1-5 within the candidates search already retrieved for that
query -- not a fresh retrieval -- so this isolates what the reranker fine-tune
changed from what stage-1 retrieval changed. A query whose confirmed movie
never reached the candidates in the first place is marked "not retrieved" for
both models; no amount of reranker fine-tuning can fix that, only stage 1 can.
"""
import csv
import json
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from finetune_cross_encoder import CROSS_ENCODER_MODEL, MODEL_DIR, doc_for, load_inputs  # noqa: E402


def rank_under(model, query, candidates, catalog, expected_id):
    pairs = [(query, doc_for(catalog, c["movie_id"])) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    ordered = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    top1 = ordered[0][0]["title"]
    for rank, (c, _) in enumerate(ordered, start=1):
        if c["movie_id"] == expected_id:
            return rank, top1
    return None, top1


def main():
    catalog, by_split = load_inputs()
    split_of_group = json.loads((ROOT / "evaluation/splits.json").read_text())["split_of_group"]

    print("Loading models...")
    base_model = CrossEncoder(CROSS_ENCODER_MODEL)
    finetuned_model = CrossEncoder(str(MODEL_DIR / "final"))

    rows = []
    all_trials = [t for trials in by_split.values() for t in trials]
    print(f"Scoring {len(all_trials)} queries under both models...")
    for i, t in enumerate(all_trials, start=1):
        candidates = sorted(t["results"], key=lambda c: c["bi_rank"])
        expected_id = t["expected_tmdb_id"]
        before_rank, before_top1 = rank_under(base_model, t["query"], candidates, catalog, expected_id)
        after_rank, after_top1 = rank_under(finetuned_model, t["query"], candidates, catalog, expected_id)
        rows.append({
            "split": split_of_group[t["split_group"]],
            "kind": t["kind"],
            "query": t["query"],
            "expected_title": t["expected_title"],
            "expected_tmdb_id": expected_id,
            "before_rank": before_rank or "",
            "before_top1": before_top1,
            "after_rank": after_rank or "",
            "after_top1": after_top1,
            "rank_change": (before_rank - after_rank) if (before_rank and after_rank) else "",
        })
        if i % 50 == 0:
            print(f"  {i}/{len(all_trials)}")

    csv_path = ROOT / "evaluation/finetune_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path.relative_to(ROOT)} ({len(rows)} rows)")

    results = json.loads((ROOT / "evaluation/finetune_results.json").read_text())
    lines = ["# Cross-encoder fine-tune comparison", "",
             "Production `cross-encoder/ms-marco-MiniLM-L-6-v2` vs. "
             "`models/cross-encoder-finetuned-v1/final`, scored on the same "
             "already-retrieved candidates per query -- see "
             "`evaluation/finetune_cross_encoder.py` and `findings.txt`'s "
             "2026-09-17 entry for how the checkpoint was produced. Full "
             "per-query rows: `evaluation/finetune_comparison.csv`.", "",
             "## Aggregate metrics (MRR@5 / NDCG@5)", "",
             "| Split | n | Base (bi-encoder) | Production reranker | Fine-tuned reranker |",
             "|---|---:|---:|---:|---:|"]
    for split in ("train", "val", "test"):
        b, a = results["before"][split], results["after"][split]
        n = results["metadata"].get(f"{split}_queries", "")
        lines.append(
            f"| {split} | {n} "
            f"| {b[f'{split}_base_mrr@5']:.3f} / {b[f'{split}_base_ndcg@5']:.3f} "
            f"| {b[f'{split}_mrr@5']:.3f} / {b[f'{split}_ndcg@5']:.3f} "
            f"| **{a[f'{split}_mrr@5']:.3f} / {a[f'{split}_ndcg@5']:.3f}** |"
        )

    changed = [r for r in rows if r["rank_change"] not in ("", 0)]
    improved = sorted((r for r in changed if r["rank_change"] > 0),
                       key=lambda r: r["rank_change"], reverse=True)
    regressed = sorted((r for r in changed if r["rank_change"] < 0), key=lambda r: r["rank_change"])

    lines += ["", f"## Biggest improvements ({len(improved)} queries moved up)", "",
              "| Split | Movie | Before rank | After rank |", "|---|---|---:|---:|"]
    for r in improved[:20]:
        lines.append(f"| {r['split']} | {r['expected_title']} | {r['before_rank']} | {r['after_rank']} |")

    lines += ["", f"## Regressions ({len(regressed)} queries moved down)", "",
              "| Split | Movie | Before rank | After rank |", "|---|---|---:|---:|"]
    if regressed:
        for r in regressed:
            lines.append(f"| {r['split']} | {r['expected_title']} | {r['before_rank']} | {r['after_rank']} |")
    else:
        lines.append("None.")

    never_retrieved = [r for r in rows if r["before_rank"] == "" and r["after_rank"] == ""]
    lines += ["", f"## Not among the 5 stored candidates ({len(never_retrieved)} queries)", "",
              "This only checks the 5 candidates already stored from collection time, so it "
              "cannot tell a true stage-1 ceiling miss (confirmed movie beyond the bi-encoder's "
              "full 50, unreachable by any reranker) apart from a stage-2 demotion (movie WAS in "
              "the 50, a reranker pushed it below rank 5, and a better reranker could in principle "
              "recover it). See `evaluation/diagnose_stage1_ceiling.py`, which re-retrieves the "
              "full 50 and separates the two -- run it for the current breakdown; "
              "`evaluation/stage1_ceiling.csv` has the up-to-date numbers.", ""]

    md_path = ROOT / "evaluation/finetune_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(f"\n{len(improved)} improved, {len(regressed)} regressed, "
          f"{len(never_retrieved)} never retrieved by either model.")


if __name__ == "__main__":
    main()
