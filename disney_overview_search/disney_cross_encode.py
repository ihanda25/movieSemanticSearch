"""Two-stage search: bi-encoder retrieval, then cross-encoder reranking.

`search_movies_reranked(query)` is the entry point. Stage 1 is the existing
bi-encoder scan in `search_disney.py`, which pulls CANDIDATE_K candidates out of
the whole catalog. Stage 2 rescores just those with a cross-encoder.

The difference between the two: a bi-encoder embeds query and document
separately and compares two pre-computed summaries, so mean pooling makes
"A movie where there is a robot and a kid" an average in which the filler counts
as much as the content. A cross-encoder concatenates the pair and runs attention
across both, so every query token can attend to every document token and the
model can learn to discount "A movie where there is".

The cost of that is why stage 1 exists: a cross-encoder score depends on the
pair jointly, so it cannot be precomputed, and at ~1.75 GFLOPs per pair scoring
all 826 films would take ~20s on CPU. Scoring 50 takes about a second.

The tradeoff to watch is that CANDIDATE_K is a hard ceiling on the whole system:
a film the bi-encoder ranks 51st can never be returned, however well the
cross-encoder would have scored it.

Run this file directly for a terminal REPL that shows both rankings side by
side, which is the point of testing it here first:

    .venv/bin/python disney_overview_search/disney_cross_encode.py
"""

import sys
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

# Running this file directly puts disney_overview_search/ on sys.path rather
# than the repo root, so add the root before importing the sibling module. This
# keeps the import working both ways -- as a script and as a package module
# imported from backend/app.py, which does the same thing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disney_overview_search.search_disney import search_movies  # noqa: E402

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates stage 1 hands to stage 2. Reranking cost is linear in this
# number, and it caps recall -- see the ceiling note in the module docstring.
CANDIDATE_K = 50
TOP_K = 5

# Loaded lazily and cached, the same way search_disney.py caches the bi-encoder:
# the first run downloads ~90MB of weights into ~/.cache/huggingface.
_cross_encoder = None


def load_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def document_text(movie):
    """The passage the cross-encoder scores the query against.

    Deliberately the same fields embed.py puts into the vector, so both stages
    are ranking the same text and a difference in results is the models
    disagreeing rather than the inputs differing.
    """

    overview = movie.get("overview") or ""
    return f"Title: {movie.get('title', '')}, Overview: {overview}"


def search_movies_reranked(query, top_k=TOP_K, candidate_k=CANDIDATE_K):
    """Retrieves with the bi-encoder, then reorders the candidates by cross-encoder.

    Each result carries the cross-encoder `score` it was ranked by, plus the
    `bi_score` and `bi_rank` it came in with, so the two stages can be compared.

    NOTE: `score` here is a cross-encoder logit, roughly -11..+11 and centred
    near 0, NOT a cosine similarity. `to_percent()` in backend/app.py rescales
    the 0.15-0.60 cosine band and would clamp almost every one of these to 0 or
    100, so that mapping needs replacing before this feeds the UI.
    """

    candidates = search_movies(query, top_k=candidate_k)

    model = load_cross_encoder()
    # One batched predict() rather than a call per pair -- the per-call overhead
    # dominates at this size.
    pairs = [(query, document_text(hit["movie"])) for hit in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    reranked = []
    for bi_rank, (hit, score) in enumerate(zip(candidates, scores), start=1):
        reranked.append({
            "score": float(score),
            "bi_score": hit["score"],
            "bi_rank": bi_rank,
            "movie": hit["movie"],
        })

    reranked.sort(key=lambda result: result["score"], reverse=True)
    return reranked[:top_k]


def main():
    """Interactive prompt for comparing the reranked order against stage 1."""

    print("Loading models and vector database...")
    search_movies_reranked("warmup")  # fills both caches
    print(f"Ready. Retrieving {CANDIDATE_K} candidates, reranking to {TOP_K}.")
    print("Describe a scene you remember. Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not query:
            continue

        started = time.perf_counter()
        results = search_movies_reranked(query)
        elapsed = time.perf_counter() - started

        for rank, result in enumerate(results, start=1):
            movie = result["movie"]
            # "was 18" is the interesting column: it says what the reranking
            # actually changed, which a single ordered list cannot show.
            print(f"  {rank}. {result['score']:+7.3f}  {movie['title']}"
                  f"  (bi-encoder rank {result['bi_rank']}, {result['bi_score']:.3f})")
            print(f"         {movie['overview'][:100]}")
        print(f"  [{elapsed:.2f}s]\n")


if __name__ == "__main__":
    main()
