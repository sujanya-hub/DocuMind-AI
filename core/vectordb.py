"""
vectordb.py - FAISS vector index with chunk metadata storage.

VectorDB pairs a FAISS IndexFlatIP index (cosine similarity via inner
product on normalised vectors) with a parallel list of compact chunk metadata
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
    raise ImportError("faiss-cpu is required: pip install faiss-cpu") from exc

from core.config import TOP_K_RESULTS
from core.embedder import embed_query, embed_texts, get_embedding_dim


class VectorDB:
    """In-memory FAISS index paired with compact chunk metadata."""

    def __init__(self) -> None:
        self.dim: int = get_embedding_dim()
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dim)
        self.chunks: List[Dict[str, Any]] = []
        self.last_embeddings: np.ndarray | None = None
        print("VECTOR DB INITIALIZED:", self.dim)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Embed and add chunk dicts to the index."""
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

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

            # Keep stored metadata compact to reduce memory pressure on Render.
            unique_chunks.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": text,
                    "page_number": chunk.get("page_number"),
                    "source": chunk.get("source"),
                    "char_start": chunk.get("char_start"),
                    "char_end": chunk.get("char_end"),
                }
            )

        if not unique_chunks:
            raise ValueError("No non-empty chunks were available for indexing.")

        texts = [chunk["text"] for chunk in unique_chunks]
        vectors = np.ascontiguousarray(embed_texts(texts), dtype=np.float32)
        if vectors.shape[0] == 0:
            raise ValueError("No vectors were generated for indexing.")

        self.index.add(vectors)
        self.chunks.extend(unique_chunks)
        self.last_embeddings = vectors
        print("VECTOR DB BUILT:", self.index.ntotal)
        return vectors

    def reset(self) -> None:
        """Drop all indexed vectors and associated metadata."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most similar chunks for a query string."""
        if self.index.ntotal == 0:
            raise RuntimeError("VectorDB is empty. Call add_chunks() before searching.")
        if not query.strip():
            raise ValueError("Query must not be blank.")

        top_k = min(top_k, self.index.ntotal)
        q_vec = np.ascontiguousarray(embed_query(query).reshape(1, -1), dtype=np.float32)
        scores, indices = self.index.search(q_vec, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.chunks[idx])
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self, directory: str) -> None:
        """Write the FAISS index and chunk metadata to directory."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "chunks.pkl"), "wb") as fh:
            pickle.dump(self.chunks, fh)

    def load(self, directory: str) -> None:
        """Restore index and metadata from a previously saved directory."""
        index_path = os.path.join(directory, "faiss.index")
        chunks_path = os.path.join(directory, "chunks.pkl")

        for path in (index_path, chunks_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index file not found: {path}")

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as fh:
            self.chunks = pickle.load(fh)

    @property
    def total_chunks(self) -> int:
        """Number of vectors currently held in the index."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return f"VectorDB(dim={self.dim}, total_chunks={self.total_chunks})"
