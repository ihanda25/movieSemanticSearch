"""Validate evaluation/splits.json against the current trial set.

Run: .venv/bin/python evaluation/check_splits.py
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def validate():
    trials = json.loads((ROOT / "evaluation/agent_search_run.json").read_text())["trials"]
    splits = json.loads((ROOT / "evaluation/splits.json").read_text())
    split_of = splits["split_of_group"]

    groups = defaultdict(list)
    for t in trials:
        groups[t["split_group"]].append(t)

    assert set(groups) == set(split_of), (
        f"splits.json is stale relative to agent_search_run.json: "
        f"missing {set(groups) - set(split_of)}, extra {set(split_of) - set(groups)}. "
        f"Re-run evaluation/build_split.py."
    )

    for group_id, group in groups.items():
        queries = {t["query"] for t in group}
        assert len(queries) == len(group), f"Duplicate query text within group {group_id}"

    counts = defaultdict(int)
    for split in split_of.values():
        counts[split] += 1
    total = len(split_of)
    ratios = {name: round(n / total, 3) for name, n in counts.items()}

    report = {"total_movies": total, "counts": dict(counts), "ratios": ratios,
              "summary": splits["metadata"]["summary"]}
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
