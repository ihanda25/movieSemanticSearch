"""Pull the Disney/Pixar movie catalog from TMDB into database/catalog.json.

Run from anywhere:  python3 database/build_catalog.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BASE = "https://api.themoviedb.org/3"

# Ordered most-specific first: a film credited to several of these keeps the
# first studio it matches, so Pixar/WDAS win over the generic Disney label.
STUDIOS = [
    (3, "Pixar"),
    (6125, "Walt Disney Animation Studios"),
    (2, "Walt Disney Pictures"),
    (3166, "Walt Disney Productions"),  # pre-1986 name; holds the classics
    # Disney-owned subsidiaries. Listed after the Disney labels so a
    # co-production keeps its Disney studio rather than flipping to these.
    (420, "Marvel Studios"),
    (1, "Lucasfilm"),
    (108270, "Lucasfilm Animation"),
]


def get(session, path, **params):
    """GET with a retry on TMDB's rate limiter."""
    for attempt in range(4):
        r = session.get(f"{BASE}{path}", params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 1)) + 1)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {path} after repeated 429s")


def discover(session, company_id, min_runtime):
    """Yield every movie credited to one company, walking all pages."""
    params = {
        "with_companies": company_id,
        "include_adult": "false",
        "language": "en-US",
        # Stable sort: popularity.desc reshuffles between requests, which
        # drops and duplicates rows across page boundaries.
        "sort_by": "primary_release_date.asc",
    }
    if min_runtime:
        params["with_runtime.gte"] = min_runtime

    page = 1
    while True:
        data = get(session, "/discover/movie", page=page, **params)
        yield from data["results"]
        if page >= data["total_pages"] or page >= 500:
            return
        page += 1


def save_catalog(path, catalog):
    """Atomic checkpoints allow interrupted enrichment to resume."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def enrich_catalog(session, catalog, path):
    failures = []
    for i, movie in enumerate(catalog.values(), 1):
        if "keywords" in movie and "tagline" in movie:
            continue
        try:
            details = get(session, f"/movie/{movie['id']}",
                          append_to_response="keywords", language="en-US")
            keywords = details["keywords"]["keywords"]
            movie["keywords"] = list(dict.fromkeys(k["name"] for k in keywords))
            movie["tagline"] = details.get("tagline") or ""
        except requests.ConnectionError:
            save_catalog(path, catalog)
            raise RuntimeError("TMDB connection failed; progress saved, rerun to resume") from None
        except (requests.RequestException, RuntimeError, KeyError) as exc:
            failures.append(movie["id"])
            print(f"Could not enrich {movie['id']}: {type(exc).__name__}", flush=True)
        if i % 25 == 0:
            save_catalog(path, catalog)
            print(f"Enriched {i}/{len(catalog)}", flush=True)
    save_catalog(path, catalog)
    if failures:
        raise RuntimeError(f"Enrichment incomplete for {failures}; rerun to retry")
    print(f"Enrichment complete: {len(catalog)} movies", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-runtime",
        type=int,
        default=40,
        help="drop anything shorter, in minutes (0 keeps shorts). Default 40.",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "database" / "catalog.json")
    ap.add_argument("--enrich-only", action="store_true",
                    help="enrich the existing catalog without rediscovering movies; resumable")
    args = ap.parse_args()

    token = os.getenv("tmdb_token")
    if not token:
        sys.exit("Error: tmdb_token not found in .env")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    if args.enrich_only:
        enrich_catalog(session, json.loads(args.out.read_text()), args.out)
        return

    catalog = {}
    for company_id, studio_name in STUDIOS:
        added = 0
        for m in discover(session, company_id, args.min_runtime):
            if m["id"] in catalog:  # already claimed by a more specific studio
                continue
            catalog[m["id"]] = {
                "id": m["id"],
                "title": m["title"],
                "overview": m["overview"] or "N/A",
                "release_date": m.get("release_date"),
                "genre_ids": m.get("genre_ids"),
                "poster_path": m.get("poster_path"),
                "studio": studio_name,
            }
            added += 1
        print(f"{studio_name:<30} +{added:>4} new  (total {len(catalog)})")

    # genre_ids are bare integers; without this map the field is unusable.
    genres = {str(g["id"]): g["name"] for g in get(session, "/genre/movie/list")["genres"]}
    (args.out.parent / "genres.json").write_text(json.dumps(genres, indent=2) + "\n")

    ordered = dict(sorted(catalog.items(), key=lambda kv: (kv[1]["release_date"] or "", kv[0])))
    args.out.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")

    enrich_catalog(session, ordered, args.out)

    missing = sum(1 for m in ordered.values() if m["overview"] == "N/A")
    print(f"\nWrote {len(ordered)} movies to {args.out.relative_to(ROOT)}")
    print(f'{missing} had no overview and were written as "N/A".')


if __name__ == "__main__":
    main()
