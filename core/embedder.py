"""
embedder.py - Sentence-level embeddings via sentence-transformers.

Maintains a single module-level model instance to avoid reloading from
disk on every call. All vectors are L2-normalised so that inner-product
search in FAISS is equivalent to cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required: pip install sentence-transformers"
    ) from exc

from core.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Return the shared SentenceTransformer instance, loading it on first call.

    Returns:
        Loaded and cached SentenceTransformer model.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into a normalised float32 embedding matrix.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        NumPy array of shape (len(texts), embedding_dim), dtype float32.

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    embeddings = get_model().encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string into a normalised float32 vector.

    Args:
        query: Search or question string.

    Returns:
        NumPy array of shape (embedding_dim,), dtype float32.

    Raises:
        ValueError: If query is blank.
    """
    if not query.strip():
        raise ValueError("Query string must not be blank.")
    return embed_texts([query])[0]


def get_embedding_dim() -> int:
    """
    Return the output dimensionality of the loaded embedding model.

    Returns:
        Integer dimension (e.g. 384 for all-MiniLM-L6-v2).
    """
    return get_model().get_sentence_embedding_dimension()
