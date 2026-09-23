"""Separate true stage-1 misses from stage-2 demotions, and test whether the
fine-tuned reranker recovers any of the latter.

evaluation/compare_checkpoints.py only compared models over each query's
already-stored 5 candidates -- for outcome=other queries the confirmed movie
isn't one of those 5 by definition, so that comparison couldn't say anything
about them. This script re-retrieves the FULL CANDIDATE_K=50 bi-encoder pool
fresh for every query, which is what actually decides whether a reranker
(production or fine-tuned) could ever have seen the confirmed movie at all.

Run: .venv/bin/python evaluation/diagnose_stage1_ceiling.py
Requires models/cross-encoder-finetuned-v1/final. Writes
evaluation/stage1_ceiling.csv and prints a classification summary:

  stage-1 ceiling miss   -- confirmed movie beyond bi-encoder top 50. Neither
                             model could ever rank it in the top 5; only a
                             better bi-encoder or richer corpus text fixes this.
  stage-2, fixed          -- was in the 50, production reranker missed top 5,
                             fine-tuned reranker now hits top 5.
  stage-2, still missed    -- was in the 50, both rerankers still miss top 5.
  hit before and after     -- both rerankers already had it in the top 5.
"""
import csv
import json
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disney_overview_search.documents import document_text  # noqa: E402
from disney_overview_search.disney_cross_encode import CANDIDATE_K, PRETRAINED_CROSS_ENCODER_MODEL as CROSS_ENCODER_MODEL, TOP_K  # noqa: E402
from disney_overview_search.search_disney import search_movies  # noqa: E402

MODEL_DIR = ROOT / "models" / "cross-encoder-finetuned-v1" / "final"


def rank_and_top1(model, query, candidates, expected_id):
    pairs = [(query, document_text(c["movie"])) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    ordered = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    top1 = ordered[0][0]["movie"]["title"]
    for rank, (c, _) in enumerate(ordered, start=1):
        if c["movie"]["id"] == expected_id:
            return rank, top1
    return None, top1


def main():
    trials = json.loads((ROOT / "evaluation/agent_search_run.json").read_text())["trials"]
    split_of_group = json.loads((ROOT / "evaluation/splits.json").read_text())["split_of_group"]

    print("Loading models...")
    base_model = CrossEncoder(CROSS_ENCODER_MODEL)
    finetuned_model = CrossEncoder(str(MODEL_DIR))

    rows = []
    counts = {"stage-1 ceiling miss": 0, "stage-2, fixed": 0,
              "stage-2, still missed": 0, "hit before and after": 0}

    print(f"Re-retrieving the full {CANDIDATE_K}-candidate bi-encoder pool for {len(trials)} queries...")
    for i, t in enumerate(trials, start=1):
        expected_id = t["expected_tmdb_id"]
        candidates = search_movies(t["query"], top_k=CANDIDATE_K)
        bi_rank = next((r for r, c in enumerate(candidates, start=1) if c["movie"]["id"] == expected_id), None)

        row = {"split": split_of_group[t["split_group"]], "kind": t["kind"], "query": t["query"],
               "expected_title": t["expected_title"], "expected_tmdb_id": expected_id,
               "bi_rank_of_50": bi_rank or ""}

        if bi_rank is None:
            row["classification"] = "stage-1 ceiling miss"
            row["before_rank"] = row["after_rank"] = ""
        else:
            before_rank, _ = rank_and_top1(base_model, t["query"], candidates, expected_id)
            after_rank, _ = rank_and_top1(finetuned_model, t["query"], candidates, expected_id)
            before_hit = bool(before_rank and before_rank <= TOP_K)
            after_hit = bool(after_rank and after_rank <= TOP_K)
            row["before_rank"] = before_rank or ""
            row["after_rank"] = after_rank or ""
            if before_hit and after_hit:
                row["classification"] = "hit before and after"
            elif not before_hit and after_hit:
                row["classification"] = "stage-2, fixed"
            elif not before_hit and not after_hit:
                row["classification"] = "stage-2, still missed"
            else:
                row["classification"] = "stage-2, regressed"
                counts.setdefault("stage-2, regressed", 0)

        counts[row["classification"]] += 1
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(trials)}")

    csv_path = ROOT / "evaluation/stage1_ceiling.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path.relative_to(ROOT)} ({len(rows)} rows)")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
