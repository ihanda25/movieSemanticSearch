"""Exercise actual local API with synthetic queries, isolating all agent labels.

Run: .venv/bin/python evaluation/collect_agent_feedback.py

Appends into the single shared evaluation/agent_search_run.json report and the
single shared database/agent_feedback.sqlite3 database. A row already present
in the report (matched by normalized query text, same rule check_synthetic_seed
uses for duplicate detection) is never replayed through the API -- growing
evaluation/synthetic_seed.json with a new batch and re-running this script
processes only the new rows and appends their trials, leaving every previously
recorded search/feedback row untouched. No training and no writes to
database/feedback.sqlite3.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from evaluation.check_synthetic_seed import validate, normalize


def main():
    validation = validate()
    seed_path = ROOT / 'evaluation/synthetic_seed.json'
    output = ROOT / 'evaluation/agent_search_run.json'
    agent_db = ROOT / 'database/agent_feedback.sqlite3'
    if output.exists() != agent_db.exists():
        raise SystemExit('Report and database are out of sync (one exists without the other); '
                          'resolve manually before appending another batch.')
    # Set the module-level database before importing/using API functions.
    from disney_overview_search import feedback as storage
    storage.DB_PATH = agent_db
    from backend.app import app
    app.config['TESTING'] = True
    client = app.test_client()
    rows = json.loads(seed_path.read_text())['queries']

    if output.exists():
        report = json.loads(output.read_text())
        already_processed = {normalize(t['query']) for t in report['trials']}
        report['metadata']['seed_sha256'] = hashlib.sha256(seed_path.read_bytes()).hexdigest()
        report['metadata']['validation'] = validation
        report['metadata'].pop('completed_at', None)
    else:
        report = {'metadata': {'source': 'agent_authored', 'label_status': 'needs_human_review',
                  'purpose': 'Synthetic local API collection trial, not human feedback or held-out evaluation',
                  'started_at': datetime.now(timezone.utc).isoformat(),
                  'seed_sha256': hashlib.sha256(seed_path.read_bytes()).hexdigest(),
                  'database': str(agent_db.relative_to(ROOT)), 'training_performed': False,
                  'validation': validation}, 'trials': []}
        already_processed = set()

    new_rows = [row for row in rows if normalize(row['query']) not in already_processed]
    print(f'{len(already_processed)} rows already recorded; {len(new_rows)} new rows to process '
          f'(seed file has {len(rows)} total).', flush=True)

    for i, row in enumerate(new_rows, 1):
        response = client.post('/api/search', json={'query': row['query']})
        if response.status_code != 200:
            raise RuntimeError(response.get_data(as_text=True))
        result = response.get_json()
        search_id = result['search_id']
        # API identifies browser calls as ui; explicitly preserve agent origin.
        with storage.connect() as db:
            db.execute('UPDATE searches SET source=? WHERE id=?', ('agent_authored', search_id))
            snapshot = json.loads(db.execute('SELECT snapshot FROM searches WHERE id=?', (search_id,)).fetchone()[0])
            snapshot['label_provenance'] = {'source': 'agent_authored', 'label_status': 'needs_human_review', 'split_group': row['split_group']}
            db.execute('UPDATE searches SET snapshot=? WHERE id=?', (json.dumps(snapshot), search_id))
        ranks = [n for n, hit in enumerate(result['results'], 1) if hit['id'] == row['expected_tmdb_id']]
        outcome = 'selected' if ranks else 'other'
        if not ranks:
            title = row['expected_title'].rsplit(' (', 1)[0]
            lookup = client.get('/api/movies', query_string={'q': title})
            if lookup.status_code != 200 or row['expected_tmdb_id'] not in {m['id'] for m in lookup.get_json()['results']}:
                raise RuntimeError(f'Title lookup failed for {title}')
        feedback = client.post('/api/feedback', json={'search_id': search_id, 'outcome': outcome, 'movie_id': row['expected_tmdb_id']})
        if feedback.status_code != 200 or feedback.get_json() != {'saved': True}:
            raise RuntimeError(feedback.get_data(as_text=True))
        trial = {**row, 'search_id': search_id, 'intended_rank_in_top5': ranks[0] if ranks else None,
                 'feedback_outcome': outcome, 'results': [{k: hit[k] for k in ('rank','movie_id','title','score','bi_score','bi_rank')} for hit in snapshot['results']]}
        report['trials'].append(trial)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        print(f"{i:3}/{len(new_rows)} {row['expected_title']}: intended rank {ranks[0] if ranks else 'outside five'}; {outcome} saved", flush=True)

    report['metadata']['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['summary'] = {}
    for kind in ('all', 'premise', 'scene'):
        subset = [r for r in report['trials'] if kind == 'all' or r['kind'] == kind]
        report['summary'][kind] = {'count': len(subset), 'top1': sum(r['intended_rank_in_top5'] == 1 for r in subset), 'top5': sum(r['intended_rank_in_top5'] is not None for r in subset)}
    with storage.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM searches').fetchone()[0] == len(report['trials'])
        assert db.execute('SELECT COUNT(*) FROM feedback').fetchone()[0] == len(report['trials'])
        assert db.execute("SELECT COUNT(*) FROM searches WHERE source!='agent_authored'").fetchone()[0] == 0
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
