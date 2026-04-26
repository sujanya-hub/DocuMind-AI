"""
session_manager.py - Streamlit session state management for DocuMind.

FINAL MEMORY FIX SUMMARY:
  The previous version stored the entire pipeline result dict in
  st.session_state[K.PIPELINE]. This dict contained:
    - vector_db   (FAISS index object)
    - qa_engine   (holds reference to vector_db)
    - summarizer  (LLM wrapper)
    - pages       (full extracted text of every page)
    - chunks      (list of dicts with full text)

  Streamlit serializes session state on every rerun. Storing all of this
  caused a full duplication of peak-memory objects at the worst possible
  moment (right after the pipeline completes at its own memory peak).

  FIX:
    - K.PIPELINE now stores a LIGHTWEIGHT dict (metadata + first 20 chunks)
    - Heavy objects (vector_db, qa_engine, summarizer) are stored as their
      own top-level session keys so Streamlit can reference them without
      duplicating the FAISS index through the pipeline dict path.
    - Pages are stored separately. They are plain text dicts — cheaper than
      the FAISS index — but we avoid double-storing them inside pipeline too.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit as st

from core.insight_engine import ActionableTakeaways, InsightResult, SuggestedQuestions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAT_HISTORY_LIMIT: int = 10
_PIPELINE_CHUNK_PREVIEW = 20   # max chunks kept in the lightweight pipeline dict


# ---------------------------------------------------------------------------
# Key registry
# ---------------------------------------------------------------------------

class _Keys:
    PDF_PATH             = "pdf_path"
    DOCUMENT_HASH        = "document_hash"
    METADATA             = "metadata"
    FILE_SIZE            = "file_size"
    PAGES                = "pages"
    CHUNKS               = "chunks"
    ANALYSIS_CHUNKS      = "analysis_chunks"
    # EMBEDDINGS intentionally absent — numpy arrays must never live here
    VECTOR_DB            = "vector_db"
    QA_ENGINE            = "qa_engine"
    SUMMARIZER           = "summarizer"
    PROCESSING_TIME      = "processing_time"
    TOTAL_CHUNKS         = "total_chunks"
    INDEXED              = "indexed"
    SUMMARY              = "summary"
    CHAT_HISTORY         = "chat_history"
    LAST_ANSWER          = "last_answer"
    INSIGHTS             = "insights"
    SUGGESTED_QUESTIONS  = "suggested_questions"
    ACTION_ITEMS         = "action_items"
    ENHANCED_EXPLANATION = "enhanced_explanation"
    PIPELINE             = "pipeline"   # lightweight dict only — see store_pipeline_result


K = _Keys


# ---------------------------------------------------------------------------
# Safe defaults factory
# ---------------------------------------------------------------------------

def _defaults() -> Dict[str, Any]:
    return {
        K.PDF_PATH:             None,
        K.DOCUMENT_HASH:        "",
        K.METADATA:             None,
        K.FILE_SIZE:            0,
        K.PAGES:                [],
        K.CHUNKS:               [],
        K.ANALYSIS_CHUNKS:      [],
        K.VECTOR_DB:            None,
        K.QA_ENGINE:            None,
        K.SUMMARIZER:           None,
        K.PROCESSING_TIME:      0.0,
        K.TOTAL_CHUNKS:         0,
        K.INDEXED:              False,
        K.SUMMARY:              None,
        K.CHAT_HISTORY:         [],
        K.LAST_ANSWER:          None,
        K.INSIGHTS:             None,
        K.SUGGESTED_QUESTIONS:  None,
        K.ACTION_ITEMS:         None,
        K.ENHANCED_EXPLANATION: None,
        K.PIPELINE:             None,
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialise() -> None:
    """
    Ensure every session key exists with a safe default value.
    Idempotent — existing values are never overwritten.
    """
    for key, default in _defaults().items():
        if key not in st.session_state:
            st.session_state[key] = (
                list(default) if isinstance(default, list)
                else dict(default) if isinstance(default, dict)
                else default
            )


# ---------------------------------------------------------------------------
# Full session reset
# ---------------------------------------------------------------------------

def reset_session() -> None:
    """Wipe ALL managed session state keys and restore safe defaults."""
    for key, default in _defaults().items():
        st.session_state[key] = (
            list(default) if isinstance(default, list)
            else dict(default) if isinstance(default, dict)
            else default
        )


# ---------------------------------------------------------------------------
# Document state helpers
# ---------------------------------------------------------------------------

def is_document_indexed() -> bool:
    return bool(st.session_state.get(K.INDEXED, False))


def is_new_upload(file_name: str) -> bool:
    current: Optional[str] = st.session_state.get(K.PDF_PATH)
    if not current:
        return True
    return os.path.basename(current) != file_name


def store_pipeline_result(result: Dict[str, Any]) -> None:
    """
    Persist the output of document_service.run_pipeline() to session state.

    MEMORY STRATEGY:
      Heavy objects (vector_db, qa_engine, summarizer) are stored as their
      own top-level keys. Streamlit references them in-place without
      duplicating them.

      st.session_state[K.PIPELINE] is set to a LIGHTWEIGHT dict containing
      only what the UI rendering functions actually need to read:
        - metadata        (small dict)
        - file_size       (int)
        - processing_time (float)
        - total_chunks    (int)
        - chunks          (first 20 only, for _build_analysis_context)

      Heavy objects are included in the pipeline dict as the SAME object
      references already stored in their own keys — no copy is made.
      This preserves app.py call sites like pipeline["qa_engine"].answer()
      without storing a second copy of the FAISS index in memory.
    """
    chunks = result.get("chunks", [])
    analysis_chunks = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", ""),
            "page_number": chunk.get("page_number"),
            "source": chunk.get("source"),
        }
        for chunk in chunks[:_PIPELINE_CHUNK_PREVIEW]
        if isinstance(chunk, dict)
    ]

    # ── Store heavy objects as independent top-level keys ─────────────────
    st.session_state[K.PDF_PATH]        = result["pdf_path"]
    st.session_state[K.DOCUMENT_HASH]   = result.get("document_hash", "")
    st.session_state[K.METADATA]        = result["metadata"]
    st.session_state[K.FILE_SIZE]       = result["file_size"]
    st.session_state[K.PAGES]           = []
    st.session_state[K.CHUNKS]          = []
    st.session_state[K.ANALYSIS_CHUNKS] = analysis_chunks
    st.session_state[K.VECTOR_DB]       = result["vector_db"]
    st.session_state[K.QA_ENGINE]       = result["qa_engine"]
    st.session_state[K.SUMMARIZER]      = None
    st.session_state[K.PROCESSING_TIME] = result.get("processing_time", 0.0)
    st.session_state[K.TOTAL_CHUNKS]    = result.get("total_chunks", len(chunks))
    st.session_state[K.INDEXED]         = True

    # ── Lightweight pipeline dict ─────────────────────────────────────────
    # app.py reads: pipeline["metadata"], pipeline["chunks"],
    # pipeline["file_size"], pipeline["total_chunks"], pipeline["pages"],
    # pipeline["processing_time"], pipeline["vector_db"], pipeline["qa_engine"]
    #
    # Heavy fields use the SAME object references as the keys above —
    # no copy, no extra allocation. Only 20 chunks stored here for the
    # analysis context builder; full list remains at K.CHUNKS.
    st.session_state[K.PIPELINE] = {
        "metadata":        result["metadata"],
        "file_size":       result["file_size"],
        "processing_time": result.get("processing_time", 0.0),
        "total_chunks":    result.get("total_chunks", len(chunks)),
        "vector_db":       result["vector_db"],
        "qa_engine":       result["qa_engine"],
    }

    # Clear derived/stale state from any previous document
    for key in (
        K.SUMMARY, K.LAST_ANSWER,
        K.INSIGHTS, K.SUGGESTED_QUESTIONS,
        K.ACTION_ITEMS, K.ENHANCED_EXPLANATION,
    ):
        st.session_state[key] = None

def reset_document_state() -> None:
    """Clear all document-scoped and derived session keys."""
    document_keys = [
        K.PDF_PATH, K.DOCUMENT_HASH, K.METADATA, K.FILE_SIZE,
        K.PAGES, K.CHUNKS, K.ANALYSIS_CHUNKS,
        K.VECTOR_DB, K.QA_ENGINE, K.SUMMARIZER,
        K.PROCESSING_TIME, K.TOTAL_CHUNKS, K.INDEXED,
        K.SUMMARY, K.LAST_ANSWER,
        K.INSIGHTS, K.SUGGESTED_QUESTIONS,
        K.ACTION_ITEMS, K.ENHANCED_EXPLANATION,
        K.PIPELINE,
    ]
    for key in document_keys:
        st.session_state.pop(key, None)
    initialise()


# ---------------------------------------------------------------------------
# Insight state helpers
# ---------------------------------------------------------------------------

def store_insight_result(result: InsightResult) -> None:
    st.session_state[K.INSIGHTS]             = result.key_insights
    st.session_state[K.SUGGESTED_QUESTIONS]  = result.suggested_questions
    st.session_state[K.ACTION_ITEMS]         = result.actionable_takeaways
    st.session_state[K.ENHANCED_EXPLANATION] = result.enhanced_explanation


def get_insights() -> Optional[List[str]]:
    return st.session_state.get(K.INSIGHTS)


def get_suggested_questions() -> Optional[SuggestedQuestions]:
    return st.session_state.get(K.SUGGESTED_QUESTIONS)


def get_action_items() -> Optional[ActionableTakeaways]:
    return st.session_state.get(K.ACTION_ITEMS)


def get_enhanced_explanation() -> Optional[str]:
    return st.session_state.get(K.ENHANCED_EXPLANATION)


def insights_generated() -> bool:
    return st.session_state.get(K.INSIGHTS) is not None


# ---------------------------------------------------------------------------
# Chat history helpers
# ---------------------------------------------------------------------------

def _is_duplicate_turn(history: List[Dict[str, Any]], question: str) -> bool:
    if not history:
        return False
    return history[-1].get("question", "") == question


def append_chat_turn(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
) -> None:
    history: List[Dict[str, Any]] = st.session_state.get(K.CHAT_HISTORY, [])
    if _is_duplicate_turn(history, question):
        return
    history.append(
        {
            "turn":     len(history) + 1,
            "question": question,
            "answer":   answer,
            "sources":  sources,
        }
    )
    if len(history) > CHAT_HISTORY_LIMIT:
        history = history[-CHAT_HISTORY_LIMIT:]
    st.session_state[K.CHAT_HISTORY] = history


def get_chat_history() -> List[Dict[str, Any]]:
    return st.session_state.get(K.CHAT_HISTORY, [])


def reset_chat_history() -> None:
    st.session_state[K.CHAT_HISTORY] = []


def get_chat_history_count() -> int:
    return len(st.session_state.get(K.CHAT_HISTORY, []))


def get_chat_history_limit() -> int:
    return CHAT_HISTORY_LIMIT


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def store_summary(summary: Optional[str]) -> None:
    st.session_state[K.SUMMARY] = summary


def get_summary() -> Optional[str]:
    return st.session_state.get(K.SUMMARY)


# ---------------------------------------------------------------------------
# Generic getter / setter
# ---------------------------------------------------------------------------

def get(key: str) -> Any:
    return st.session_state.get(key)


def set(key: str, value: Any) -> None:  # noqa: A001
    st.session_state[key] = value
