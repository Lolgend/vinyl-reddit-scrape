"""Vector embedding generation using local CPU FastEmbed."""

from __future__ import annotations

from collections.abc import Sequence

from fastembed import TextEmbedding

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_model_instance: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    """Retrieve or lazily initialize the singleton fastembed TextEmbedding model."""
    global _model_instance
    if _model_instance is None:
        _model_instance = TextEmbedding(model_name=DEFAULT_EMBEDDING_MODEL)
    return _model_instance


def compute_release_text(
    combined_artist: str,
    album_title: str,
    variant: str | None = None,
    retailer: str | None = None,
) -> str:
    """Build the canonical text representation for release vector embedding.

    Format: "{combined_artist} - {album_title}" plus optional " ({variant})" and " [{retailer}]".
    """
    clean_artist = combined_artist.strip() if combined_artist else ""
    clean_album = album_title.strip() if album_title else ""
    text = f"{clean_artist} - {clean_album}"

    if variant and variant.strip():
        text += f" ({variant.strip()})"

    if retailer and retailer.strip():
        text += f" [{retailer.strip()}]"

    return text


def compute_artist_text(artist_name: str) -> str:
    """Build the canonical text representation for artist vector embedding."""
    return artist_name.strip() if artist_name else ""


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Generate 384-dimensional dense vector embeddings for a sequence of texts."""
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.embed(texts)
    result: list[list[float]] = []
    for vector in embeddings:
        if hasattr(vector, "tolist"):
            result.append([float(x) for x in vector.tolist()])
        else:
            result.append([float(x) for x in vector])
    return result


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "compute_artist_text",
    "compute_release_text",
    "embed_texts",
    "get_embedding_model",
]
