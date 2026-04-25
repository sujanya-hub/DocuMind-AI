"""
session_manager.py - Streamlit session state management for DocuMind.

Centralises all session state initialisation, access, and mutation.
No module outside this one reads or writes st.session_state keys using
raw string literals. All keys are declared on the _Keys registry class.
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
    EMBEDDINGS           = "embeddings"
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
    # THE FIX: app.py reads st.session_state.pipeline as a nested dict.
    # session_manager must own and populate this key.
    PIPELINE             = "pipeline"


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
        K.EMBEDDINGS:           None,
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
        # pipeline starts as None — only set after a successful run_pipeline()
        K.PIPELINE:             None,
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialise() -> None:
    """
    Ensure every session key exists with a safe default value.

    Idempotent: existing values are never overwritten. Safe to call on
    every Streamlit rerun — it only writes keys that are genuinely absent.
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
    """Return True when a document has been fully processed and indexed."""
    return bool(st.session_state.get(K.INDEXED, False))


def is_new_upload(file_name: str) -> bool:
    """Return True when the supplied file name differs from the current document."""
    current: Optional[str] = st.session_state.get(K.PDF_PATH)
    if not current:
        return True
    return os.path.basename(current) != file_name


def store_pipeline_result(result: Dict[str, Any]) -> None:
    """
    Persist the output of document_service.run_pipeline() to session state.

    THE FIX: also writes the entire result dict to st.session_state.pipeline
    so that app.py can read pipeline["vector_db"], pipeline["chunks"], etc.
    without a KeyError or None guard failing silently.
    """
    st.session_state[K.PDF_PATH]        = result["pdf_path"]
    st.session_state[K.DOCUMENT_HASH]   = result.get("document_hash", "")
    st.session_state[K.METADATA]        = result["metadata"]
    st.session_state[K.FILE_SIZE]       = result["file_size"]
    st.session_state[K.PAGES]           = result["pages"]
    st.session_state[K.CHUNKS]          = result["chunks"]
    st.session_state[K.EMBEDDINGS]      = result.get("embeddings")
    st.session_state[K.VECTOR_DB]       = result["vector_db"]
    st.session_state[K.QA_ENGINE]       = result["qa_engine"]
    st.session_state[K.SUMMARIZER]      = result["summarizer"]
    st.session_state[K.PROCESSING_TIME] = result.get("processing_time", 0.0)
    st.session_state[K.TOTAL_CHUNKS]    = result.get("total_chunks", len(result["chunks"]))
    st.session_state[K.INDEXED]         = True

    # THE FIX: populate st.session_state.pipeline as the nested dict that
    # app.py expects. app.py reads pipeline["vector_db"], pipeline["chunks"],
    # pipeline["qa_engine"], pipeline["metadata"], pipeline["file_size"], etc.
    # Without this line, st.session_state.pipeline is always None after rerun.
    st.session_state[K.PIPELINE] = result

    for key in (
        K.SUMMARY, K.LAST_ANSWER,
        K.INSIGHTS, K.SUGGESTED_QUESTIONS,
        K.ACTION_ITEMS, K.ENHANCED_EXPLANATION,
    ):
        st.session_state[key] = None

    print("SESSION: pipeline stored, total_chunks =", result.get("total_chunks"))
    print("SESSION: pipeline key present =", st.session_state.get(K.PIPELINE) is not None)


def reset_document_state() -> None:
    """
    Clear all document-scoped and derived session keys and restore safe
    defaults. Chat history is preserved.
    """
    document_keys = [
        K.PDF_PATH, K.DOCUMENT_HASH, K.METADATA, K.FILE_SIZE, K.PAGES, K.CHUNKS,
        K.EMBEDDINGS, K.VECTOR_DB, K.QA_ENGINE, K.SUMMARIZER,
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