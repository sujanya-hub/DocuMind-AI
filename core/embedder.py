"""
embedder.py - Memory-efficient sentence embeddings for DocuMind.

The model is imported and loaded lazily so Streamlit startup stays light on
Render. Embeddings are returned as normalised float32 arrays for efficient
FAISS cosine-similarity search.
"""

from __future__ import annotations

from typing import List

import numpy as np

from core.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL

_model = None


def get_model():
    """Return the shared embedding model, loading it only when needed."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into a normalized float32 embedding matrix.

    Raises:
        ValueError: If texts is empty.
        RuntimeError: If the embedding model cannot be loaded.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    cleaned_texts = [str(text).strip() for text in texts if str(text).strip()]
    if not cleaned_texts:
        raise ValueError("Cannot embed empty text content.")

    try:
        model = get_model()
    except Exception as exc:
        raise RuntimeError("Embedding unavailable") from exc

    embeddings = model.encode(
        cleaned_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string into a normalized float32 vector."""
    if not query.strip():
        raise ValueError("Query string must not be blank.")
    return embed_texts([query])[0]


def get_embedding_dim() -> int:
    """
    Return the configured embedding dimension without forcing model load.

    This keeps VectorDB initialization lightweight under tight memory limits.
    """
    return EMBEDDING_DIM
