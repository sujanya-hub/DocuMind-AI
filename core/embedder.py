"""
embedder.py - Memory-efficient sentence embeddings for DocuMind.

The model is imported and loaded lazily so Streamlit startup stays light on
Render. Embeddings are returned as normalised float32 arrays for efficient
FAISS cosine-similarity search.

Memory budget (Render free tier, 512 MB):
  - paraphrase-MiniLM-L3-v2 at rest : ~12 MB RAM  (vs ~22 MB for L6)
  - batch_size=2                     : ~20 MB peak activation per encode() call
  - device="cpu"                     : avoids CUDA allocator overhead (~100 MB)
"""

from __future__ import annotations

from typing import List

import numpy as np

from core.config import EMBEDDING_DIM

# Batch size of 2 keeps peak activation memory below 25 MB on Render free tier.
# Do not raise this without profiling — the free tier OOMs silently above ~40 MB
# peak per encode() call when combined with FAISS index memory.
_SAFE_BATCH_SIZE = 2

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
        # paraphrase-MiniLM-L3-v2: 3-layer model, ~12 MB at rest.
        # Chosen over all-MiniLM-L6-v2 (~22 MB) specifically to stay within
        # the 512 MB Render free tier limit. Retrieval quality is marginally
        # lower but acceptable for document Q&A workloads.
        # device="cpu" is pinned — never reads from config so a misconfigured
        # EMBEDDING_MODEL env var cannot trigger a CUDA OOM crash.
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
        print("EMBEDDER MODEL READY")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into a normalized float32 embedding matrix.

    Raises:
        ValueError:   If texts is empty or contains only whitespace.
        RuntimeError: If the model cannot be loaded or encode() fails.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    cleaned_texts = [str(text).strip() for text in texts if str(text).strip()]
    if not cleaned_texts:
        raise ValueError("Cannot embed empty text content.")

    try:
        model = get_model()
    except Exception as exc:
        raise RuntimeError(f"Embedding model could not be loaded: {exc}") from exc

    try:
        embeddings = model.encode(
            cleaned_texts,
            batch_size=_SAFE_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    except MemoryError as exc:
        raise RuntimeError(
            f"Out of memory during embedding (batch_size={_SAFE_BATCH_SIZE})"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Embedding encode() failed for {len(cleaned_texts)} texts: {exc}"
        ) from exc

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.size == 0:
        raise RuntimeError("Embedding model returned an empty array.")
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got shape {embeddings.shape}.")
    if embeddings.shape[0] != len(cleaned_texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(cleaned_texts)}, "
            f"got {embeddings.shape[0]}."
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

    Keeps VectorDB initialisation lightweight under tight memory limits.
    paraphrase-MiniLM-L3-v2 outputs 384-dimensional vectors, identical to
    all-MiniLM-L6-v2, so EMBEDDING_DIM in config requires no change.
    """
    return EMBEDDING_DIM
