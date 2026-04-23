from __future__ import annotations

import logging
import re
from typing import Any

import streamlit as st

from core.ai_engine import generate_response
from core.config import ANALYSIS_CONTEXT_CHARS, UPLOAD_DIR, has_api_key
from core.document_service import run_pipeline
from utils.helpers import file_sha256, format_file_size

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DocuMind",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    .status-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: rgba(248, 249, 251, 0.6);
        margin-bottom: 1rem;
    }
    .metric-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    .metric-pill {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.85rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


_STATE_DEFAULTS = {
    "document_hash": "",
    "document_name": "",
    "pipeline": None,
    "analysis_cache": None,
    "qa_cache": {},
    "question_input": "",
}


_ANALYSIS_PROMPT = """
You are DocuMind, a grounded document analyst.
Use only the provided document context and return the following sections exactly.

[SUMMARY]
A concise executive summary in 5-7 sentences.

[INSIGHTS]
5 bullet points with the most important findings.

[RISKS]
3 bullet points covering limitations, gaps, or caveats.

[QUESTIONS]
6 follow-up questions, one per line, each starting with a hyphen.
""".strip()


def _initialise_state() -> None:
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default.copy() if isinstance(default, dict) else default


# Centralising resets keeps rerun behavior predictable and avoids stale UI.
def _clear_document() -> None:
    for key, default in _STATE_DEFAULTS.items():
        st.session_state[key] = default.copy() if isinstance(default, dict) else default


# Reusing precomputed chunks is much faster than rebuilding document context on each rerun.
def _build_analysis_context(chunks: list[dict[str, Any]], max_chars: int = ANALYSIS_CONTEXT_CHARS) -> str:
    selected: list[str] = []
    total_chars = 0

    # Prefer longer unique chunks to maximize signal per token.
    for chunk in sorted(chunks, key=lambda item: len(item.get("text", "")), reverse=True):
        text = str(chunk.get("text", "")).strip()
        page = chunk.get("page_number", "?")
        if not text:
            continue

        entry = f"[Page {page}]\n{text}"
        if entry in selected:
            continue
        if total_chars + len(entry) > max_chars:
            continue

        selected.append(entry)
        total_chars += len(entry)

        if total_chars >= max_chars:
            break

    return "\n\n".join(selected)


def _extract_section(text: str, tag: str) -> str:
    pattern = rf"\[{re.escape(tag)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_bullets(raw_text: str) -> list[str]:
    bullets: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^[-*\d.)\s]+", "", line).strip()
        if cleaned:
            bullets.append(cleaned)
    return bullets


def _ensure_analysis() -> None:
    if st.session_state.analysis_cache is not None:
        return

    pipeline = st.session_state.pipeline
    if pipeline is None or not has_api_key():
        return

    context = _build_analysis_context(pipeline["chunks"])
    if not context.strip():
        st.session_state.analysis_cache = {
            "summary": "No usable text was available for document analysis.",
            "insights": [],
            "risks": [],
            "questions": [],
        }
        return

    try:
        raw = generate_response(
            system_prompt=_ANALYSIS_PROMPT,
            user_prompt=f"DOCUMENT CONTEXT:\n\n{context}",
            max_tokens=900,
            temperature=0.1,
        )
    except Exception as exc:
        logger.warning("Document analysis failed: %s", exc)
        st.session_state.analysis_cache = {
            "summary": f"Analysis is unavailable until the LLM is configured correctly. {exc}",
            "insights": [],
            "risks": [],
            "questions": [],
        }
        return

    st.session_state.analysis_cache = {
        "summary": _extract_section(raw, "SUMMARY") or "Summary unavailable.",
        "insights": _parse_bullets(_extract_section(raw, "INSIGHTS")),
        "risks": _parse_bullets(_extract_section(raw, "RISKS")),
        "questions": _parse_bullets(_extract_section(raw, "QUESTIONS")),
    }


def _answer_question(question: str) -> dict[str, Any]:
    pipeline = st.session_state.pipeline
    if pipeline is None:
        return {"answer": "Upload a document first.", "sources": []}

    cache_key = f"{st.session_state.document_hash}:{question.strip().lower()}"
    cache = st.session_state.qa_cache
    if cache_key in cache:
        return cache[cache_key]

    try:
        result = pipeline["qa_engine"].answer(question)
    except Exception as exc:
        result = {
            "answer": (
                "The question could not be answered right now. "
                f"Details: {type(exc).__name__}: {exc}"
            ),
            "sources": [],
        }

    cache[cache_key] = result
    st.session_state.qa_cache = cache
    return result


_initialise_state()

with st.sidebar:
    st.title("DocuMind")
    st.caption("Streamlit document intelligence app")

    if has_api_key():
        st.success("LLM configuration detected")
    else:
        st.warning(
            "`GROQ_API_KEY` is not set. Document indexing still works, but AI analysis and Q&A will stay disabled until you add the environment variable on Render."
        )

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        incoming_hash = file_sha256(uploaded_file)
        if incoming_hash != st.session_state.document_hash:
            try:
                with st.spinner("Indexing document..."):
                    result = run_pipeline(uploaded_file, UPLOAD_DIR)
            except Exception as exc:
                st.error(f"Document processing failed: {exc}")
            else:
                # Store the full pipeline result once so reruns do not recompute it.
                st.session_state.document_hash = result["document_hash"]
                st.session_state.document_name = uploaded_file.name
                st.session_state.pipeline = result
                st.session_state.analysis_cache = None
                st.session_state.qa_cache = {}
                st.session_state.question_input = ""
                st.rerun()

    if st.session_state.pipeline is not None:
        pipeline = st.session_state.pipeline
        metadata = pipeline["metadata"]
        st.markdown("### Current Document")
        st.write(st.session_state.document_name)
        st.caption(
            f"{metadata.get('page_count', len(pipeline['pages']))} pages | "
            f"{pipeline['total_chunks']} chunks | "
            f"{format_file_size(pipeline['file_size'])} | "
            f"{pipeline['processing_time']}s"
        )
        if st.button("Clear document", use_container_width=True):
            _clear_document()
            st.rerun()

st.title("Document Intelligence")
st.caption("Production-safe config, faster ingestion, and simpler Streamlit flow.")

if st.session_state.pipeline is None:
    st.info("Upload a PDF from the sidebar to start indexing and analysis.")
    st.stop()

pipeline = st.session_state.pipeline
metadata = pipeline["metadata"]

st.markdown(
    f"""
    <div class="status-card">
        <strong>{metadata.get('file_name', st.session_state.document_name)}</strong><br/>
        Indexed document ready for retrieval and analysis.
        <div class="metric-row">
            <span class="metric-pill">Pages: {metadata.get('page_count', len(pipeline['pages']))}</span>
            <span class="metric-pill">Chunks: {pipeline['total_chunks']}</span>
            <span class="metric-pill">Processing time: {pipeline['processing_time']}s</span>
            <span class="metric-pill">Upload size: {format_file_size(pipeline['file_size'])}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not has_api_key():
    st.info(
        "For Render, add `GROQ_API_KEY` in the service Environment settings. "
        "This deployment expects environment variables only."
    )

_ensure_analysis()
analysis = st.session_state.analysis_cache or {"summary": "", "insights": [], "risks": [], "questions": []}

overview_tab, qa_tab = st.tabs(["Overview", "Ask Questions"])

with overview_tab:
    st.subheader("Executive Summary")
    st.write(analysis["summary"] or "Summary unavailable.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Insights")
        if analysis["insights"]:
            for item in analysis["insights"]:
                st.markdown(f"- {item}")
        else:
            st.write("Insights will appear here once analysis is available.")

    with col2:
        st.subheader("Risks And Caveats")
        if analysis["risks"]:
            for item in analysis["risks"]:
                st.markdown(f"- {item}")
        else:
            st.write("No risk analysis available yet.")

    st.subheader("Suggested Follow-Up Questions")
    if analysis["questions"]:
        for question in analysis["questions"]:
            if st.button(question, key=f"suggested::{question}"):
                st.session_state.question_input = question
                st.rerun()
    else:
        st.write("Suggested questions will appear here once analysis is available.")

with qa_tab:
    st.subheader("Grounded Q&A")
    question = st.text_input(
        "Ask about the uploaded document",
        key="question_input",
        placeholder="What are the main conclusions?",
        disabled=not has_api_key(),
    )

    if question and has_api_key():
        with st.spinner("Searching the document and drafting an answer..."):
            response = _answer_question(question)

        st.write(response.get("answer", "No answer available."))

        sources = response.get("sources", [])
        if sources:
            st.subheader("Sources")
            for source in sources:
                page = source.get("page_number", "?")
                score = source.get("score")
                text = str(source.get("text", "")).strip()
                preview = text[:300] + ("..." if len(text) > 300 else "")
                score_text = f" | score {score:.3f}" if isinstance(score, float) else ""
                st.caption(f"Page {page}{score_text}")
                st.write(preview)
                st.divider()
    elif question and not has_api_key():
        st.warning("Set `GROQ_API_KEY` to enable AI-powered Q&A.")
