"""Local feedback storage; no training or model updates happen here."""
import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "database" / "feedback.sqlite3"


@contextmanager
def connect():
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.execute("""CREATE TABLE IF NOT EXISTS searches (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL,
        query TEXT NOT NULL, snapshot TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS feedback (
        search_id TEXT PRIMARY KEY REFERENCES searches(id), updated_at TEXT NOT NULL,
        outcome TEXT NOT NULL, movie_id INTEGER)""")
    try:
        with db:
            yield db
    finally:
        db.close()


@lru_cache(maxsize=4)
def fingerprint(path, size, modified):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def version(path):
    stat = path.stat()
    return fingerprint(str(path), stat.st_size, stat.st_mtime_ns)


def record_search(query, hits, source):
    from disney_overview_search.documents import EMBEDDING_MODEL, document_text
    from disney_overview_search.disney_cross_encode import CROSS_ENCODER_MODEL, CANDIDATE_K
    from disney_overview_search.search_disney import DB_FILE
    snapshot = {
        "schema_version": 1, "embedding_model": EMBEDDING_MODEL,
        "cross_encoder_model": CROSS_ENCODER_MODEL, "candidate_k": CANDIDATE_K,
        "index_sha256": version(DB_FILE),
        "results": [{"rank": i, "movie_id": h["movie"]["id"],
                     "title": h["movie"]["title"], "document": document_text(h["movie"]),
                     "score": h["score"], "bi_score": h["bi_score"], "bi_rank": h["bi_rank"]}
                    for i, h in enumerate(hits, 1)]}
    search_id = str(uuid.uuid4())
    with connect() as db:
        db.execute("INSERT INTO searches VALUES (?, ?, ?, ?, ?)",
                   (search_id, datetime.now(timezone.utc).isoformat(), source, query, json.dumps(snapshot)))
    return search_id


def lookup_movies(title):
    catalog = json.loads((ROOT / "database" / "catalog.json").read_text())
    return [{"id": m["id"], "title": m["title"], "year": (m.get("release_date") or "")[:4]}
            for m in catalog.values() if title.casefold() in m["title"].casefold()][:30]


def save_feedback(search_id, outcome, movie_id=None):
    if outcome not in {"selected", "none", "other"}:
        raise ValueError("Invalid feedback outcome")
    if outcome == "none":
        if movie_id is not None:
            raise ValueError("No-match feedback cannot select a movie")
    elif type(movie_id) is not int:
        raise ValueError("Choose a movie")
    with connect() as db:
        row = db.execute("SELECT snapshot FROM searches WHERE id = ?", (search_id,)).fetchone()
        if row is None:
            raise ValueError("Search not found; search again")
        displayed = {r["movie_id"] for r in json.loads(row[0])["results"]}
        if outcome == "selected" and movie_id not in displayed:
            raise ValueError("Movie was not among the displayed results")
        if outcome == "other":
            catalog = json.loads((ROOT / "database" / "catalog.json").read_text())
            if str(movie_id) not in catalog:
                raise ValueError("Movie not found in the catalog")
        db.execute("""INSERT INTO feedback VALUES (?, ?, ?, ?)
            ON CONFLICT(search_id) DO UPDATE SET updated_at=excluded.updated_at,
            outcome=excluded.outcome, movie_id=excluded.movie_id""",
                   (search_id, datetime.now(timezone.utc).isoformat(), outcome, movie_id))


def terminal_feedback(query, hits):
    search_id = record_search(query, hits, "terminal")
    print("Feedback: 1–5 = correct result, n = none, t = find correct title, Enter = skip")
    while True:
        choice = input("feedback> ").strip().lower()
        if not choice:
            return
        if choice == "n":
            save_feedback(search_id, "none")
            print("Saved: none of these. No correct movie label supplied.")
            return
        if choice.isdigit() and 1 <= int(choice) <= len(hits):
            save_feedback(search_id, "selected", hits[int(choice)-1]["movie"]["id"])
            print("Feedback saved locally.")
            return
        if choice == "t":
            title = input("Movie title: ").strip()
            matches = lookup_movies(title) if title else []
            for i, m in enumerate(matches, 1):
                print(f"  {i}. {m['title']} ({m['year']}) [TMDB {m['id']}]")
            selection = input("Choose title number (Enter to go back): ").strip()
            if selection.isdigit() and 1 <= int(selection) <= len(matches):
                save_feedback(search_id, "other", matches[int(selection)-1]["id"])
                print("Feedback saved locally.")
                return
        print("Choose a listed result, n, t, or Enter.")


def main():
    parser = argparse.ArgumentParser(description="Export locally collected feedback to CSV")
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    with connect() as db, args.export.open("w", newline="") as handle:
        rows = db.execute("""SELECT s.id, s.created_at, s.source, s.query, f.updated_at,
            f.outcome, f.movie_id, s.snapshot FROM searches s JOIN feedback f ON s.id=f.search_id
            ORDER BY s.created_at""")
        writer = csv.writer(handle)
        writer.writerow(["search_id", "searched_at", "source", "query", "labeled_at",
                         "outcome", "selected_tmdb_id", "snapshot_json"])
        writer.writerows(rows)
    print(f"Exported labeled searches to {args.export}")


if __name__ == "__main__":
    main()
