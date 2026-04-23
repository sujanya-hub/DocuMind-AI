"""
core/qa_engine.py - Retrieval-Augmented Generation Q&A engine.

Retrieves the top-k most relevant chunks from VectorDB and passes them
as grounded context to the LLM via the centralised ai_engine connector.
No Groq client is constructed here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.ai_engine import generate_response
from core.config import MAX_TOKENS, TEMPERATURE, TOP_K_RESULTS
from core.vectordb import VectorDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CHUNK_CHARS = 30    # chunks shorter than this are treated as noise
_MAX_CONTEXT_CHARS = 12_000

_SYSTEM_PROMPT = """\
You are DocuMind Analyst, a precision document analysis assistant.
Answer the user's question using ONLY the numbered context excerpts below.

STRICT RULES:
- Ground every claim directly in the provided context.
- Quote exact figures, names, or dates from the text when they answer the question.
- Cite page numbers inline, e.g. (p. 3), whenever referencing a specific passage.
- If the answer is not present in the context, respond with exactly:
  "I could not find relevant information in the uploaded document for this question."
- Do NOT use outside knowledge, speculation, or assumptions.
- Write in clear, professional paragraphs. Be concise.
- End your response with a "Sources: Page X, Page Y" line listing pages referenced.
"""


class QAEngine:
    """
    Retrieval-augmented Q&A backed by Groq via ai_engine.

    Args:
        vector_db: A populated VectorDB instance.
    """

    def __init__(self, vector_db: VectorDB) -> None:
        self.db: VectorDB = vector_db

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: int = TOP_K_RESULTS,
    ) -> Dict[str, Any]:
        """
        Answer a question grounded in the indexed document.

        Args:
            question: Natural-language question from the user.
            top_k:    Number of chunks to retrieve as context.
                      Caller should pass the speed-mode value from app.py.

        Returns:
            Dict with keys:
                question (str)        - original question
                answer   (str)        - LLM-generated grounded answer
                sources  (list[dict]) - retrieved, deduplicated chunks

        Raises:
            ValueError:   If question is blank.
            RuntimeError: If the VectorDB is empty or the API call fails.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be blank.")

        # Retrieve, deduplicate, and filter noise
        raw_chunks = self._safe_search(question, top_k=top_k)
        chunks     = _deduplicate_chunks(_filter_short_chunks(raw_chunks))

        if not chunks:
            return {
                "question": question,
                "answer":   (
                    "I could not find relevant information in the uploaded document "
                    "for this question."
                ),
                "sources":  [],
            }

        context     = _build_context(chunks)
        user_prompt = (
            f"Context excerpts from the document:\n\n"
            f"{context}\n\n"
            f"Question: {question}"
        )

        try:
            answer_text = generate_response(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
        except Exception as exc:
            logger.error("[QAEngine] LLM call failed: %s", exc)
            answer_text = (
                "The document was indexed successfully, but the LLM request "
                "could not be completed. Confirm GROQ_API_KEY is set in the "
                f"environment. Details: {type(exc).__name__}: {exc}"
            )

        if not answer_text or not answer_text.strip():
            answer_text = (
                "I could not find relevant information in the uploaded document "
                "for this question."
            )

        return {
            "question": question,
            "answer":   answer_text.strip(),
            "sources":  chunks,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _safe_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Call self.db.search() with the given top_k, trying common signatures.
        Returns an empty list on any failure rather than raising.
        """
        for args, kwargs in [
            ((query,),      {"top_k": top_k}),
            ((query, top_k), {}),
        ]:
            try:
                result = self.db.search(*args, **kwargs)
                if result is not None:
                    return list(result) if not isinstance(result, list) else result
            except TypeError:
                continue
            except Exception as exc:
                logger.error("[QAEngine] VectorDB search failed: %s", exc)
                return []
        return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _filter_short_chunks(chunks: List[Any]) -> List[Any]:
    """Remove chunks that are too short to carry meaningful information."""
    result: List[Any] = []
    for chunk in chunks:
        text = _get_text(chunk)
        if len(text) >= _MIN_CHUNK_CHARS:
            result.append(chunk)
    return result


def _deduplicate_chunks(chunks: List[Any]) -> List[Any]:
    """
    Remove duplicate chunks by normalised text fingerprint.
    Preserves order; keeps the first occurrence of each unique text.
    """
    seen: set[str] = set()
    result: List[Any] = []
    for chunk in chunks:
        text = _get_text(chunk)
        key  = " ".join(text.lower().split())[:200]   # normalised 200-char fingerprint
        if key and key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def _get_text(chunk: Any) -> str:
    """Extract text string from a chunk dict or object."""
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("content") or chunk.get("page_content") or "").strip()
    return str(
        getattr(chunk, "text", None)
        or getattr(chunk, "content", None)
        or getattr(chunk, "page_content", None)
        or ""
    ).strip()


def _build_context(chunks: List[Any], max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """
    Format retrieved chunks into a numbered, page-labelled context block.
    Truncates gracefully at max_chars.
    """
    parts: List[str] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            page = chunk.get("page_number") or chunk.get("page") or "?"
            text = _get_text(chunk)
        else:
            page = getattr(chunk, "page_number", None) or getattr(chunk, "page", "?")
            text = _get_text(chunk)

        entry = f"[{i}] (Page {page})\n{text}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(entry[:remaining] + "\n...[truncated]")
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)
