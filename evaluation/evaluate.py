"""Scores retrieval quality over the hand-written query set in queries.json.

Run it directly; there is no framework and no runner:

    .venv/bin/python evaluation/evaluate.py

Every query is scored twice against the same expected film -- once on the
bi-encoder alone and once through the full two-stage pipeline the backend
serves -- so the table says what reranking actually bought rather than only
what the current system scores. Keeping the bi-encoder column is the point:
once the embedding model or the corpus changes, the pre-rerank baseline stops
being reproducible.

Two metrics, per the plan in CLAUDE.md:

  Recall@5  fraction of queries with the right film in the top 5. This is what
            a user feels, and the UI shows exactly 5 results.
  MRR       mean of 1/rank, which sees a film moving from 5th to 1st where
            Recall@5 cannot.

Both are reported separately per `kind` when a set mixes kinds. queries.json
is premise-only for now: scene queries are parked in queries_scene.json,
because the corpus describes premises and scene retrieval cannot improve until
the embedded text is enriched. Point `--queries` at that file to measure
whether enrichment worked.

The number to watch is stage-1 Recall@CANDIDATE_K: it is a hard ceiling on the
whole system. Anything the bi-encoder misses at that cutoff cannot be recovered
by reranking, so if it is below 100% the fix is stage 1 (a better embedding
model, or a richer corpus), not a better reranker.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# backend/ is not a package, so import app.py the way app.py imports its own
# sibling: by path. to_percent is imported rather than reimplemented so the
# percentages in the table are literally the ones the UI renders.
sys.path.insert(0, str(ROOT / "backend"))

from app import to_percent  # noqa: E402
from disney_overview_search.disney_cross_encode import (  # noqa: E402
    CANDIDATE_K,
    CROSS_ENCODER_MODEL,
    search_movies_reranked,
)
from disney_overview_search.search_disney import EMBEDDING_MODEL, search_movies  # noqa: E402

QUERY_FILE = Path(__file__).resolve().parent / "queries.json"
CUTOFF = 5

# Large enough to rank the whole catalog: the bi-encoder needs a full ranking so
# a film at rank 300 still contributes 1/300 to MRR instead of vanishing.
FULL_CORPUS = 1_000_000


def rank_of(results, expected_id):
    """1-based rank of the expected film, or None if it is not in `results`."""

    for rank, hit in enumerate(results, start=1):
        if hit["movie"]["id"] == expected_id:
            return rank
    return None


def summarize(ranks):
    """Recall@CUTOFF and MRR over a list of ranks, where None means 'not found'."""

    if not ranks:
        return {"n": 0, "recall": 0.0, "mrr": 0.0, "hits": 0}
    hits = sum(1 for rank in ranks if rank and rank <= CUTOFF)
    mrr = sum(1.0 / rank for rank in ranks if rank) / len(ranks)
    return {"n": len(ranks), "recall": hits / len(ranks), "mrr": mrr, "hits": hits}


def truncate(text, width):
    return text if len(text) <= width else text[: width - 1] + "…"


def miss_cause(result, candidate_k):
    """Which stage lost a miss -- the column worth filtering the CSV on.

    A film stage 1 ranked past candidate_k was never shown to the reranker, so
    the fix is stage 1 (richer corpus, better embedding model). A film that
    arrived in the candidates and still missed was demoted by the reranker, and
    no amount of extra candidates would help.
    """

    if result["xe_rank"] and result["xe_rank"] <= CUTOFF:
        return ""
    if not result["bi_rank"] or result["bi_rank"] > candidate_k:
        return "stage-1: never reached the reranker"
    return "stage-2: reranker demoted it"


CSV_COLUMNS = [
    "n", "kind", "query", "intended", "intended_tmdb_id", "bi_rank", "xe_rank",
    "rank_change", "hit_at_5", "miss_cause", "returned_first",
    "returned_first_percent", "top_5",
]


def write_csv(path, results, candidate_k):
    """One row per query, with both rankings and what actually came back.

    `rank_change` is bi_rank - xe_rank, so positive means the reranker helped;
    sorting a spreadsheet on it shows what stage 2 is worth per query.
    """

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, r in enumerate(results, start=1):
            change = (r["bi_rank"] - r["xe_rank"]
                      if r["bi_rank"] and r["xe_rank"] else "")
            writer.writerow({
                "n": i,
                "kind": r["kind"],
                "query": r["query"],
                "intended": r["expected_title"],
                "intended_tmdb_id": r["expected_tmdb_id"],
                # Blank rather than a sentinel: "not found" is not a rank, and a
                # 999 here would quietly poison any average taken in a spreadsheet.
                "bi_rank": r["bi_rank"] or "",
                "xe_rank": r["xe_rank"] or "",
                "rank_change": change,
                "hit_at_5": "yes" if r["xe_rank"] and r["xe_rank"] <= CUTOFF else "no",
                "miss_cause": miss_cause(r, candidate_k),
                "returned_first": r["xe_top1"],
                "returned_first_percent": r["xe_top1_percent"],
                "top_5": " | ".join(r["xe_top5"]),
            })


def evaluate(rows, candidate_k):
    """Scores every row, returning one result dict per query."""

    results = []
    for row in rows:
        query, expected_id = row["query"], row["expected_tmdb_id"]

        started = time.perf_counter()
        # Stage 1 alone, over the whole catalog.
        bi_results = search_movies(query, top_k=FULL_CORPUS)
        # The production path, asked for every candidate so the expected film's
        # post-rerank position is visible even when it lands outside the top 5.
        xe_results = search_movies_reranked(
            query, top_k=candidate_k, candidate_k=candidate_k
        )
        elapsed = time.perf_counter() - started

        results.append({
            "query": query,
            "kind": row["kind"],
            "expected_tmdb_id": expected_id,
            "expected_title": row["expected_title"],
            "bi_rank": rank_of(bi_results, expected_id),
            "xe_rank": rank_of(xe_results, expected_id),
            "bi_top1": bi_results[0]["movie"]["title"],
            "xe_top1": xe_results[0]["movie"]["title"],
            "xe_top1_percent": to_percent(xe_results[0]["score"]),
            "xe_top5": [hit["movie"]["title"] for hit in xe_results[:CUTOFF]],
            "seconds": elapsed,
        })
    return results


def print_table(results):
    header = (f"{'#':>3}  {'kind':<7}  {'intended':<30}  {'bi':>3}  {'xe':>3}  "
              f"{'hit':>3}  {'returned first':<30}  {'%':>3}")
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, start=1):
        hit = "yes" if r["xe_rank"] and r["xe_rank"] <= CUTOFF else "NO"
        print(f"{i:>3}  {r['kind']:<7}  {truncate(r['expected_title'], 30):<30}  "
              f"{r['bi_rank'] or '-':>3}  {r['xe_rank'] or '-':>3}  {hit:>3}  "
              f"{truncate(r['xe_top1'], 30):<30}  {r['xe_top1_percent']:>3}")


def print_misses(results):
    misses = [r for r in results
              if not (r["xe_rank"] and r["xe_rank"] <= CUTOFF)]
    if not misses:
        print("\nNo misses.")
        return

    print(f"\nMisses ({len(misses)}/{len(results)}) -- full query text, and what came "
          f"back instead:")
    for r in misses:
        # Where the film sat before reranking says which stage lost it: a
        # bi_rank beyond candidate_k means stage 1 never handed it over, so no
        # reranker could have saved it.
        where = (f"rerank put it at {r['xe_rank']}" if r["xe_rank"]
                 else f"never reached stage 2 (bi-encoder rank {r['bi_rank']})")
        print(f"\n  [{r['kind']}] {r['query']}")
        print(f"      wanted: {r['expected_title']}  ({where})")
        print(f"      got:    {', '.join(truncate(t, 28) for t in r['xe_top5'])}")


def print_summary(results, candidate_k):
    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r)

    print(f"\n{'':<10}  {'n':>3}  {'bi Recall@5':>11}  {'bi MRR':>7}  "
          f"{'xe Recall@5':>11}  {'xe MRR':>7}")
    print("-" * 62)
    breakdown = [("overall", results)]
    if len(by_kind) > 1:
        breakdown += [(kind, by_kind[kind]) for kind in sorted(by_kind)]
    for label, rows in breakdown:
        if not rows:
            continue
        bi = summarize([r["bi_rank"] for r in rows])
        xe = summarize([r["xe_rank"] for r in rows])
        print(f"{label:<10}  {bi['n']:>3}  {bi['recall']:>10.0%}  {bi['mrr']:>7.3f}  "
              f"{xe['recall']:>10.0%}  {xe['mrr']:>7.3f}")

    # The ceiling: how often stage 1 puts the answer in reach of stage 2 at all.
    reached = sum(1 for r in results if r["bi_rank"] and r["bi_rank"] <= candidate_k)
    print(f"\nStage-1 Recall@{candidate_k} (ceiling on the two-stage system): "
          f"{reached}/{len(results)} = {reached / len(results):.0%}")
    lost = [r for r in results if r["bi_rank"] and r["bi_rank"] > candidate_k]
    if lost:
        worst = max(r["bi_rank"] for r in lost)
        print(f"  {len(lost)} film(s) ranked past {candidate_k} by the bi-encoder "
              f"(worst: {worst}) and are unreachable no matter how good the reranker is.")

    total = sum(r["seconds"] for r in results)
    print(f"\n{len(results)} queries in {total:.1f}s "
          f"({total / len(results):.2f}s each, both systems)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K,
                        help=f"candidates stage 1 hands stage 2 (default {CANDIDATE_K})")
    parser.add_argument("--queries", type=Path, default=QUERY_FILE,
                        help=f"query set to score (default {QUERY_FILE.name})")
    parser.add_argument("--limit", type=int,
                        help="score only the first N queries, for a quick check")
    parser.add_argument("--csv", type=Path,
                        default=Path(__file__).resolve().parent / "results.csv",
                        help="where to write the per-query CSV (default evaluation/results.csv)")
    parser.add_argument("--save", type=Path,
                        help="also write the full per-query results to this JSON file")
    args = parser.parse_args()

    spec = json.loads(args.queries.read_text())
    rows = spec["queries"][: args.limit] if args.limit else spec["queries"]

    print(f"Scoring {len(rows)} queries from {args.queries.name}. "
          f"candidate_k={args.candidate_k}")
    print(f"  stage 1: {EMBEDDING_MODEL}")
    print(f"  stage 2: {CROSS_ENCODER_MODEL}")
    print("Loading models...\n")
    search_movies_reranked("warmup")  # keep model loading out of the timings

    results = evaluate(rows, args.candidate_k)

    print_table(results)
    print_summary(results, args.candidate_k)
    print_misses(results)

    write_csv(args.csv, results, args.candidate_k)
    print(f"\nWrote {args.csv}")

    if args.save:
        args.save.write_text(json.dumps({
            "embedding_model": EMBEDDING_MODEL,
            "cross_encoder_model": CROSS_ENCODER_MODEL,
            "candidate_k": args.candidate_k,
            "cutoff": CUTOFF,
            "results": results,
        }, indent=2, ensure_ascii=False) + "\n")
        print(f"\nWrote {args.save}")


if __name__ == "__main__":
    main()
