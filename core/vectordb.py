"""
vectordb.py - FAISS vector index with chunk metadata storage.

VectorDB pairs a FAISS IndexFlatIP index (cosine similarity via inner
product on normalised vectors) with a parallel list of chunk metadata
dicts. The index can be persisted to and restored from disk.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss-cpu is required: pip install faiss-cpu"
    ) from exc

from core.config import TOP_K_RESULTS
from core.embedder import embed_query, embed_texts, get_embedding_dim


class VectorDB:
    """
    In-memory FAISS index paired with chunk metadata.

    Attributes:
        dim    (int):              Embedding vector dimensionality.
        index  (faiss.IndexFlatIP): FAISS cosine-similarity index.
        chunks (list[dict]):       Parallel metadata for each indexed vector.
    """

    def __init__(self) -> None:
        self.dim: int = get_embedding_dim()
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dim)
        self.chunks: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embed and add chunk dicts to the index.

        Each dict must contain a ``text`` key. All keys are preserved
        and returned verbatim during retrieval.

        Args:
            chunks: List of chunk dicts from ``chunker.chunk_pages()``.

        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        # Drop exact duplicate chunk texts before embedding so we do not pay
        # twice for redundant work on repetitive PDFs.
        unique_chunks: List[Dict[str, Any]] = []
        seen_texts: set[str] = set()

        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            fingerprint = " ".join(text.split())
            if fingerprint in seen_texts:
                continue
            seen_texts.add(fingerprint)
            unique_chunks.append({**chunk, "text": text})

        if not unique_chunks:
            raise ValueError("No non-empty chunks were available for indexing.")

        texts = [c["text"] for c in unique_chunks]
        vectors = embed_texts(texts)
        self.index.add(vectors)
        self.chunks.extend(unique_chunks)

    def reset(self) -> None:
        """Drop all indexed vectors and associated metadata."""
        self.index  = faiss.IndexFlatIP(self.dim)
        self.chunks = []

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query string.

        Args:
            query: Natural-language question or search phrase.
            top_k: Number of results to return.

        Returns:
            List of chunk dicts (copies), each augmented with a
            ``score`` key (float, cosine similarity in [0, 1]).

        Raises:
            RuntimeError: If the index is empty.
            ValueError:   If query is blank.
        """
        if self.index.ntotal == 0:
            raise RuntimeError(
                "VectorDB is empty. Call add_chunks() before searching."
            )
        if not query.strip():
            raise ValueError("Query must not be blank.")

        top_k  = min(top_k, self.index.ntotal)
        q_vec  = embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(q_vec, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result          = dict(self.chunks[idx])
            result["score"] = float(score)
            results.append(result)

        return results

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Write the FAISS index and chunk metadata to directory.

        Creates ``faiss.index`` and ``chunks.pkl`` inside the directory.

        Args:
            directory: Writable directory path (created if absent).
        """
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(
            self.index, os.path.join(directory, "faiss.index")
        )
        with open(os.path.join(directory, "chunks.pkl"), "wb") as fh:
            pickle.dump(self.chunks, fh)

    def load(self, directory: str) -> None:
        """
        Restore index and metadata from a previously saved directory.

        Args:
            directory: Path containing ``faiss.index`` and ``chunks.pkl``.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        index_path  = os.path.join(directory, "faiss.index")
        chunks_path = os.path.join(directory, "chunks.pkl")

        for path in (index_path, chunks_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index file not found: {path}")

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as fh:
            self.chunks = pickle.load(fh)

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        """Number of vectors currently held in the index."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return f"VectorDB(dim={self.dim}, total_chunks={self.total_chunks})"
