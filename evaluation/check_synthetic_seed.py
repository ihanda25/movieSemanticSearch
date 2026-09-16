"""Validate synthetic seed identity/provenance; this cannot verify film facts."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(query):
    return re.sub(r"[^\w]+", " ", query.casefold()).strip()


def validate():
    data = json.loads((ROOT / 'evaluation/synthetic_seed.json').read_text())
    catalog = json.loads((ROOT / 'database/catalog.json').read_text())
    rows = data['queries']
    seen = set()
    for row in rows:
        movie = catalog[str(row['expected_tmdb_id'])]
        assert row['expected_title'] == f"{movie['title']} ({movie['release_date'][:4]})"
        assert row['kind'] in {'premise', 'scene'}
        assert row['source'] == 'agent_authored'
        assert row['label_status'] == 'needs_human_review'
        assert row['split_group'] == f"tmdb:{movie['id']}"
        text = normalize(row['query'])
        assert text and text not in seen, f'Duplicate query: {text}'
        seen.add(text)
    ids = {r['expected_tmdb_id'] for r in rows}
    report = {'queries': len(rows), 'movies': len(ids), 'kinds': dict(Counter(r['kind'] for r in rows)), 'overlap': {}}
    for path in sorted((ROOT / 'evaluation').glob('queries*.json')):
        existing = json.loads(path.read_text())['queries']
        duplicates = seen & {normalize(r['query']) for r in existing}
        assert not duplicates, f'Existing query duplicated: {duplicates}'
        overlap = ids & {r['expected_tmdb_id'] for r in existing}
        report['overlap'][path.name] = {'exact_normalized_queries': 0, 'shared_movie_count': len(overlap), 'shared_movie_ids': sorted(overlap)}
    assert len(rows) == 377 and len(ids) >= 277
    report['warning'] = 'Same-film and semantically similar queries are not independent held-out data. Human factual/ambiguity review is still required.'
    return report


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
