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

_MIN_CHUNK_CHARS = 30
_MAX_CONTEXT_CHARS = 8000

_NOT_FOUND = (
    "I could not find relevant information in the uploaded document for this question."
)

_SYSTEM_PROMPT = f"""\
You are DocuMind Analyst, a precision document analysis assistant.
Answer the user's question using ONLY the numbered context excerpts below.

RULES:
- Ground every claim in the provided context. Do not use outside knowledge or speculation.
- Quote exact figures, names, or dates from the text when they answer the question.
- Cite page numbers inline, e.g. (p. 3), whenever referencing a specific passage.
- If the excerpts answer the question only partly, give what they do say and state plainly what is not covered.
- Only if the excerpts contain nothing relevant to the question, reply with exactly this sentence and nothing else (no Sources line): "{_NOT_FOUND}"
- Write in clear, professional paragraphs. Be concise.
- Use plain text only. Do NOT use markdown: no asterisks, no bold, no headings, no bullet symbols.
- When you did answer, end with one final plain line in this exact form: Sources: Page X, Page Y
"""


class QAEngine:
    """
    Retrieval-augmented Q&A backed by Groq via ai_engine.

    Args:
        vector_db: A populated VectorDB instance.
    """

    def __init__(self, vector_db: VectorDB) -> None:
        if vector_db is None:
            raise ValueError("QAEngine requires a populated VectorDB instance.")
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

        Returns:
            Dict with keys: question, answer, sources (retrieved, deduplicated chunks).

        Raises:
            ValueError: If question is blank.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be blank.")

        raw_chunks = self._safe_search(question, top_k=top_k)
        chunks = _deduplicate_chunks(_filter_short_chunks(raw_chunks))

        if not chunks:
            return {"question": question, "answer": _NOT_FOUND, "sources": []}

        context = _build_context(chunks)
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
                "could not be completed. Confirm GROQ_API_KEY is set in "
                f"Streamlit secrets. Details: {type(exc).__name__}: {exc}"
            )

        if not answer_text or not answer_text.strip() or _is_not_found(answer_text):
            # A "not found" reply must not carry sources or a match score:
            # they would suggest the document answered the question.
            return {
                "question": question,
                "answer": _NOT_FOUND,
                "sources": [],
                "retrieved": chunks,
            }

        return {
            "question": question,
            "answer": answer_text.strip(),
            "sources": chunks,
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
            ((query,), {"top_k": top_k}),
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

def _is_not_found(text: str) -> bool:
    """True when the model refused (short reply containing the not-found phrase)."""
    lowered = text.lower()
    return "could not find relevant information" in lowered and len(text) < 400


def _filter_short_chunks(chunks: List[Any]) -> List[Any]:
    """Remove chunks that are too short to carry meaningful information."""
    return [chunk for chunk in chunks if len(_get_text(chunk)) >= _MIN_CHUNK_CHARS]


def _deduplicate_chunks(chunks: List[Any]) -> List[Any]:
    """Remove duplicate chunks by normalised text fingerprint, keeping order."""
    seen: set[str] = set()
    result: List[Any] = []
    for chunk in chunks:
        text = _get_text(chunk)
        key = " ".join(text.lower().split())[:200]
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
    """Format retrieved chunks into a numbered, page-labelled context block."""
    parts: List[str] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            page = chunk.get("page_number") or chunk.get("page") or "?"
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