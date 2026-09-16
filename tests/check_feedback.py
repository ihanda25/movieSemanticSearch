"""Exercise feedback persistence and API validation without loading models."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from disney_overview_search import feedback
from backend.app import app

with tempfile.TemporaryDirectory() as directory:
    feedback.DB_PATH = Path(directory) / "feedback.sqlite3"
    movie = {"id": 1771, "title": "Captain America: The First Avenger", "overview": "A soldier", "release_date": "2011-01-01"}
    hits = [{"movie": movie, "score": -2., "bi_score": .5, "bi_rank": 3}]
    with app.test_client() as client, patch('backend.app.search_movies_reranked', return_value=hits):
        data = client.post('/api/search', json={"query": "soldier serum"}).get_json()
        sid = data['search_id']
        assert data['results'][0]['id'] == 1771
        assert client.post('/api/feedback', json={"search_id":sid,"outcome":"selected","movie_id":1771}).status_code == 200
        assert client.post('/api/feedback', json={"search_id":sid,"outcome":"selected","movie_id":11}).status_code == 400
        assert client.post('/api/feedback', json={"search_id":"missing","outcome":"none"}).status_code == 400
        assert client.post('/api/feedback', json=[]).status_code == 400
        assert client.post('/api/feedback', json={"search_id":sid,"outcome":"none"}).status_code == 200
        assert client.post('/api/feedback', json={"search_id":sid,"outcome":"other","movie_id":11}).status_code == 200
        assert client.get('/api/movies?q=First%20Avenger').get_json()['results'][0]['id'] == 1771
    with feedback.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM feedback').fetchone()[0] == 1
        assert db.execute('SELECT outcome,movie_id FROM feedback').fetchone() == ('other',11)
        snapshot = json.loads(db.execute('SELECT snapshot FROM searches').fetchone()[0])
        assert snapshot['results'][0]['bi_rank'] == 3
        assert snapshot['index_sha256']
    with patch('builtins.input', side_effect=['1']):
        feedback.terminal_feedback('soldier serum', hits)
    with feedback.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM searches WHERE source='terminal'").fetchone()[0] == 1
    export = Path(directory) / 'export.csv'
    with patch('sys.argv', ['feedback', '--export', str(export)]):
        feedback.main()
    import csv
    assert len(list(csv.DictReader(export.open()))) == 2
print('Feedback API, corrections, terminal selection, persistence and CSV export passed')
