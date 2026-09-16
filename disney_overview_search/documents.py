"""Shared model configuration and offline retrieval text."""

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def overview_text(movie):
    return f"Title: {movie.get('title', '')}, Overview: {movie.get('overview') or ''}"


def extra_text(movie):
    parts = []
    if movie.get("keywords"):
        parts.append("Keywords: " + ", ".join(movie["keywords"]))
    if movie.get("tagline"):
        parts.append("Tagline: " + movie["tagline"])
    return ", ".join(parts)


def embedding_texts(movie):
    """Keep the original overview vector; add a separate concept vector."""
    texts = [overview_text(movie)]
    extra = extra_text(movie)
    if extra:
        texts.append(f"Title: {movie.get('title', '')}, {extra}")
    return texts


def document_text(movie):
    extra = extra_text(movie)
    return overview_text(movie) + (", " + extra if extra else "")
