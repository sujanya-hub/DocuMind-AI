"""
embedder.py - Memory-efficient sentence embeddings for DocuMind.

The model is imported and loaded lazily so Streamlit startup stays light on
Render. Embeddings are returned as normalised float32 arrays for efficient
FAISS cosine-similarity search.

Memory budget (Render free tier, 512 MB):
  - all-MiniLM-L6-v2 at rest : ~22 MB RAM
  - batch_size=4              : ~40 MB peak activation per encode() call
  - device="cpu"              : avoids CUDA allocator overhead (~100 MB)
"""

from __future__ import annotations

from typing import List

import numpy as np

from core.config import EMBEDDING_DIM

# FIX [embedder 1]: batch size hardcoded to 4 to cap peak RAM on Render free tier.
# Overrides whatever EMBEDDING_BATCH_SIZE is set to in config.
_SAFE_BATCH_SIZE = 4

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

        print("EMBEDDER MODEL LOADING")
        # FIX: model name and device pinned — never reads from config so
        # a misconfigured EMBEDDING_MODEL env var cannot cause an OOM crash.
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("EMBEDDER MODEL READY")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into a normalized float32 embedding matrix.

    Raises:
        ValueError: If texts is empty or contains only whitespace.
        RuntimeError: If the embedding model cannot be loaded or returns bad data.
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
        batch_size=_SAFE_BATCH_SIZE,   # FIX: always 4, not from config
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.size == 0:
        raise RuntimeError("Embedding model returned an empty array.")
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got shape {embeddings.shape}.")
    if embeddings.shape[0] != len(cleaned_texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(cleaned_texts)}, got {embeddings.shape[0]}."
        )

    print("EMBEDDINGS CREATED:", embeddings.shape)
    return embeddings


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