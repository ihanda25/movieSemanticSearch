"""Split the agent-authored trial set into train/val/test by movie.

Run: .venv/bin/python evaluation/build_split.py
Writes evaluation/splits.json. Validate the result with
.venv/bin/python evaluation/check_splits.py.

The split unit is `split_group` (one movie), never an individual query, so a
movie's premise and scene queries can never land in different splits -- see
CLAUDE.md and README.md's "Agent-generated search trials" section for why.
Groups are stratified by (group size, outcome composition) before assignment
so train/val/test each get a representative mix of 1- and 2-query movies and
of already-selected vs. already-missed cases, rather than a plain shuffle that
could dump most of the hard "other" cases into one split by chance.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def bucket_for(group):
    outcomes = {t["feedback_outcome"] for t in group}
    if len(group) == 1:
        return f"1-{'selected' if outcomes == {'selected'} else 'other'}"
    if outcomes == {"selected"}:
        return "2-all_selected"
    if outcomes == {"other"}:
        return "2-all_other"
    return "2-mixed"


def allocate(group_ids, ratios, rng):
    """Largest-remainder rounding so small buckets still split close to ratio."""
    ordered = sorted(group_ids)
    rng.shuffle(ordered)
    n = len(ordered)
    raw = {name: n * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = n - sum(counts.values())
    # Give leftover slots to the splits with the largest fractional remainder.
    fractions = sorted(ratios, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in fractions[:remainder]:
        counts[name] += 1
    assignment = {}
    i = 0
    for name in ratios:
        for group_id in ordered[i:i + counts[name]]:
            assignment[group_id] = name
        i += counts[name]
    return assignment


def main():
    trials = json.loads((ROOT / "evaluation/agent_search_run.json").read_text())["trials"]
    groups = defaultdict(list)
    for t in trials:
        groups[t["split_group"]].append(t)

    by_bucket = defaultdict(list)
    for group_id, group in groups.items():
        by_bucket[bucket_for(group)].append(group_id)

    rng = random.Random(SEED)
    split_of = {}
    for bucket, group_ids in by_bucket.items():
        split_of.update(allocate(group_ids, RATIOS, rng))

    summary = {name: {"movies": 0, "queries": 0, "premise": 0, "scene": 0,
                       "selected": 0, "other": 0} for name in RATIOS}
    for group_id, group in groups.items():
        split = split_of[group_id]
        s = summary[split]
        s["movies"] += 1
        for t in group:
            s["queries"] += 1
            s[t["kind"]] += 1
            s[t["feedback_outcome"]] += 1

    out = {
        "metadata": {
            "seed": SEED,
            "ratios": RATIOS,
            "unit": "split_group (one movie; a movie's queries always share a split)",
            "bucket_counts": {b: len(ids) for b, ids in by_bucket.items()},
            "summary": summary,
        },
        "split_of_group": split_of,
    }
    out_path = ROOT / "evaluation/splits.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path.relative_to(ROOT)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
