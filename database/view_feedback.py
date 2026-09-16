"""Read-only terminal viewer for local search history and labels.

    .venv/bin/python database/view_feedback.py
    .venv/bin/python database/view_feedback.py --table feedback
    .venv/bin/python database/view_feedback.py --table searches --details
"""
import argparse
import json
import sqlite3
import textwrap
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def print_table(columns, rows):
    if not rows:
        print("No rows yet.")
        return
    values = [["—" if value is None else str(value) for value in row] for row in rows]
    widths = [min(48, max(len(name), *(len(row[i]) for row in values)))
              for i, name in enumerate(columns)]
    separator = "+".join("-" * (width + 2) for width in widths)
    print(separator)
    print(" | ".join(name.ljust(width) for name, width in zip(columns, widths)))
    print(separator)
    for row in values:
        wrapped = [textwrap.wrap(value, width=width) or [""] for value, width in zip(row, widths)]
        for line in range(max(map(len, wrapped))):
            print(" | ".join((part[line] if line < len(part) else "").ljust(width)
                             for part, width in zip(wrapped, widths)))
        print(separator)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=ROOT / "database" / "feedback.sqlite3")
    parser.add_argument("--table", choices=["combined", "searches", "feedback"], default="combined")
    parser.add_argument("--details", action="store_true", help="print full stored result snapshots after the table")
    args = parser.parse_args()
    if not args.db.exists():
        print(f"No feedback database yet: {args.db}\nRun a search in the UI or reranking terminal first.")
        return
    with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        searches = db.execute("SELECT * FROM searches ORDER BY created_at DESC").fetchall()
        labels = db.execute("SELECT * FROM feedback ORDER BY updated_at DESC").fetchall()
        print(f"Database: {args.db.resolve()}\nSearches: {len(searches)} | Labeled: {len(labels)} | Unlabeled: {len(searches)-len(labels)}\n")
        if args.table == "feedback":
            columns = ["search_id", "updated_at", "outcome", "movie_id"]
            rows = [tuple(row[c] for c in columns) for row in labels]
        elif args.table == "searches":
            columns = ["id", "created_at", "source", "query", "snapshot"]
            rows = [(r['id'], r['created_at'], r['source'], r['query'],
                     "Use --details for full JSON") for r in searches]
        else:
            catalog = json.loads((ROOT / "database" / "catalog.json").read_text())
            columns = ["search_id", "source", "query", "outcome", "selected movie", "updated_at"]
            by_id = {r['search_id']: r for r in labels}
            rows = []
            for search in searches:
                label = by_id.get(search['id'])
                movie_id = label['movie_id'] if label else None
                movie = catalog.get(str(movie_id), {})
                title = f"{movie.get('title', 'Unknown')} [{movie_id}]" if movie_id is not None else None
                rows.append((search['id'], search['source'], search['query'],
                             label['outcome'] if label else 'unlabeled', title,
                             label['updated_at'] if label else None))
        print_table(columns, rows)
        if args.details:
            for search in searches:
                print(f"\nSearch {search['id']} — {search['query']}")
                print(json.dumps(json.loads(search['snapshot']), indent=2, ensure_ascii=False))
        print("\nRead-only snapshot. Run again to see new feedback.")


if __name__ == "__main__":
    main()
