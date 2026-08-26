import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "database" / "catalog.json"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
OUTPUT_FILE = ROOT / "database" / "disney_vector_db.npz"


def load_json(filepath):
    """Loads the JSON file and returns the list of movie dictionaries."""

    with open(filepath, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    print(f"loaded {len(movies)} from {filepath}")
    return movies

def build_index():
    movies = load_json(JSON_PATH)

    texts = []

    for movie in movies.values():
        title = movie.get('title', 'Unknown Title')
        overview = movie.get('overview', 'No Overview')
        text_chunk = f"Title: {title}, Overview: {overview}"
        texts.append(text_chunk)

    
    print("Model is being embedded...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    #Generate embeddings
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print("Done!")

    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
        metadata=np.array(movies, dtype=object)
    )


if __name__ == '__main__':
    build_index()












