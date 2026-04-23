from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from typing import Any

import streamlit as st

from core.ai_engine import generate_response
from core.config import ANALYSIS_CONTEXT_CHARS, UPLOAD_DIR, has_api_key
from core.document_service import run_pipeline
from utils.helpers import format_file_size

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DocuMind",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)


STATE_DEFAULTS = {
    "pipeline": None,
    "document_name": "",
    "analysis_cache": None,
    "qa_cache": {},
    "chat_history": [],
    "pending_prompt": "",
    "processing_error": "",
    "processing_success": "",
    "processed_upload_token": "",
    "uploader_key": 0,
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


def _init_state() -> None:
    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default.copy() if isinstance(default, (dict, list)) else default


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-top: #0f172a;
            --bg-mid: #111827;
            --bg-bottom: #020617;
            --text-main: #f8fafc;
            --text-muted: #cbd5e1;
        }

        html, body, [class*="css"] {
            color: var(--text-main);
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        }

        body {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.22), transparent 32%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.18), transparent 24%),
                linear-gradient(160deg, #0f172a 0%, #111827 48%, #020617 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(2, 6, 23, 0.92), rgba(15, 23, 42, 0.82));
            border-right: 1px solid rgba(148, 163, 184, 0.16);
            backdrop-filter: blur(22px);
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: var(--text-main);
            letter-spacing: -0.02em;
        }

        p, label, .stCaption, .stMarkdown, .stTextInput, .stChatInput {
            color: var(--text-main);
        }

        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px dashed rgba(148, 163, 184, 0.32);
            border-radius: 20px;
        }

        div[data-testid="stVerticalBlock"] div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 22px;
            backdrop-filter: blur(16px);
            box-shadow: 0 16px 40px rgba(2, 6, 23, 0.18);
            padding: 0.35rem 0.25rem;
        }

        .status-card,
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 18px 50px rgba(2, 6, 23, 0.24);
        }

        .hero-card {
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #7dd3fc;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
        }

        .hero-subtitle {
            margin: 0.55rem 0 0;
            color: var(--text-muted);
            max-width: 760px;
            line-height: 1.65;
        }

        .metric-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .metric-pill {
            background: rgba(255, 255, 255, 0.08);
            color: white;
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-size: 0.86rem;
        }

        .assistant-copy {
            color: var(--text-main);
            line-height: 1.75;
            font-size: 0.98rem;
        }

        .source-inline {
            display: inline-block;
            margin: 0 0.08rem;
            padding: 0.08rem 0.45rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.16);
            border: 1px solid rgba(56, 189, 248, 0.32);
            color: #bae6fd;
            font-weight: 700;
        }

        .sources-line {
            margin-top: 0.85rem;
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #e2e8f0;
        }

        .sources-line strong {
            color: #7dd3fc;
        }

        .source-card {
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 0.75rem;
        }

        .source-page {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            background: rgba(52, 211, 153, 0.14);
            border: 1px solid rgba(52, 211, 153, 0.28);
            color: #a7f3d0;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .source-text {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.65;
        }

        .source-score {
            color: #7dd3fc;
            font-size: 0.78rem;
            margin-left: 0.45rem;
        }

        .empty-state {
            padding: 1.2rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
        }

        div.stButton > button,
        div.stDownloadButton > button,
        button[kind="primary"] {
            border-radius: 999px;
            border: 1px solid rgba(125, 211, 252, 0.22);
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.24), rgba(59, 130, 246, 0.18));
            color: #f8fafc;
            font-weight: 700;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        button[kind="primary"]:hover {
            border-color: rgba(125, 211, 252, 0.4);
            color: white;
        }

        [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            color: var(--text-muted);
            padding: 0.5rem 1rem;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            color: white;
            background: rgba(56, 189, 248, 0.16);
            border-color: rgba(56, 189, 248, 0.26);
        }

        .stAlert {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        /* Regenerate button subtle styling */
        button[data-testid^="regen_"] {
            margin-top: 0.5rem;
            font-size: 0.82rem;
            opacity: 0.78;
        }

        button[data-testid^="regen_"]:hover {
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clear_document() -> None:
    for key, default in STATE_DEFAULTS.items():
        st.session_state[key] = default.copy() if isinstance(default, (dict, list)) else default
    st.session_state.uploader_key += 1


def _build_analysis_context(chunks: list[dict[str, Any]], max_chars: int = ANALYSIS_CONTEXT_CHARS) -> str:
    selected: list[str] = []
    total_chars = 0

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


@st.cache_data(show_spinner=False)
def _cached_analysis(context: str) -> str:
    """
    Cache LLM analysis by context string.
    Identical context (same document, same chunks) returns the cached result
    immediately without an API round-trip.
    """
    return generate_response(
        system_prompt=_ANALYSIS_PROMPT,
        user_prompt=f"DOCUMENT CONTEXT:\n\n{context}",
        max_tokens=900,
        temperature=0.1,
    )


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
        raw = _cached_analysis(context)
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


def _answer_question(question: str, force_refresh: bool = False) -> dict[str, Any]:
    pipeline = st.session_state.pipeline
    if pipeline is None:
        return {"answer": "Upload a document first.", "sources": []}

    cache_key = f"{st.session_state.document_name}:{question.strip().lower()}"
    cache = st.session_state.qa_cache

    # On regenerate, bust the cache for this question
    if force_refresh and cache_key in cache:
        del cache[cache_key]
        st.session_state.qa_cache = cache

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


def _process_uploaded_file(uploaded_file: Any) -> None:
    if not hasattr(uploaded_file, "_cached_bytes"):
        uploaded_file._cached_bytes = uploaded_file.getvalue()

    file_bytes = uploaded_file._cached_bytes
    upload_token = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.processed_upload_token == upload_token:
        return

    try:
        with st.spinner("Indexing document..."):
            result = run_pipeline(uploaded_file, UPLOAD_DIR)

        st.session_state.pipeline = result
        st.session_state.document_name = uploaded_file.name
        st.session_state.analysis_cache = None
        st.session_state.qa_cache = {}
        st.session_state.chat_history = []
        st.session_state.pending_prompt = ""
        st.session_state.processing_error = ""
        st.session_state.processing_success = "Document indexed successfully!"
        st.session_state.processed_upload_token = upload_token

        st.rerun()

    except Exception as exc:
        st.session_state.processing_error = str(exc)
        st.session_state.processing_success = ""


def _format_answer_html(text: str) -> str:
    safe_text = html.escape((text or "No answer available.").strip())

    safe_text = re.sub(
        r"(?im)^Sources:\s*(.+)$",
        lambda match: f"<div class='sources-line'><strong>Sources:</strong> {match.group(1)}</div>",
        safe_text,
    )
    safe_text = re.sub(
        r"\(p\.\s*(\d+)\)",
        lambda match: f"<span class='source-inline'>(p. {match.group(1)})</span>",
        safe_text,
    )
    safe_text = re.sub(
        r"\bPage\s+(\d+)\b",
        lambda match: f"<span class='source-inline'>Page {match.group(1)}</span>",
        safe_text,
    )
    safe_text = safe_text.replace("\n", "<br/>")
    return f"<div class='assistant-copy'>{safe_text}</div>"


def stream_text(text: str) -> None:
    placeholder = st.empty()

    if not text:
        placeholder.markdown(_format_answer_html("No answer available."), unsafe_allow_html=True)
        return

    chunk_size = 12
    delay = 0.015
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[: i + chunk_size])
        # Typing cursor shown during streaming
        placeholder.markdown(_format_answer_html(chunk + " ▌"), unsafe_allow_html=True)
        time.sleep(delay)

    # Final render — no cursor
    placeholder.markdown(_format_answer_html(text), unsafe_allow_html=True)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No explicit source chunks were retrieved for this answer.")
        return

    st.markdown("### 📌 Sources")
    for src in sources:
        page = src.get("page_number", "?")
        text = re.sub(r"\s+", " ", str(src.get("text", "")).strip())
        preview = html.escape(text[:200] + ("..." if len(text) > 200 else ""))
        score = src.get("score")
        score_html = ""
        if isinstance(score, (float, int)):
            score_html = f"<span class='source-score'>score {float(score):.3f}</span>"

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-page">Page {page}{score_html}</div>
                <div class="source-text">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("DocuMind")
        st.caption("AI document intelligence app")

        if has_api_key():
            st.success("LLM configuration detected")
        else:
            st.warning(
                "`GROQ_API_KEY` is not set. Indexing still works, but AI analysis and Q&A stay disabled until the environment variable is available."
            )

        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            key=f"uploaded_file_{st.session_state.uploader_key}",
        )

        if uploaded_file is not None:
            _process_uploaded_file(uploaded_file)

        if st.session_state.processing_success:
            st.success(st.session_state.processing_success)

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


def _render_hero() -> None:
    if st.session_state.pipeline is None:
        body = """
        <div class="status-card hero-card">
            <div class="eyebrow">Document Intelligence</div>
            <h1 class="hero-title">Premium RAG for dense PDFs</h1>
            <p class="hero-subtitle">
                Upload a document to generate grounded analysis, search with citations,
                and chat against your indexed content in a modern glassmorphism workspace.
            </p>
        </div>
        """
        st.markdown(body, unsafe_allow_html=True)
        return

    pipeline = st.session_state.pipeline
    metadata = pipeline["metadata"]
    st.markdown(
        f"""
        <div class="status-card hero-card">
            <div class="eyebrow">Indexed And Ready</div>
            <h1 class="hero-title">{html.escape(metadata.get('file_name', st.session_state.document_name))}</h1>
            <p class="hero-subtitle">
                Retrieval, analysis, and grounded chat are ready. Every answer stays anchored to the indexed document.
            </p>
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


def _render_overview_tab() -> None:
    if st.session_state.pipeline is None:
        st.markdown(
            """
            <div class="empty-state">
                Upload a PDF from the sidebar to unlock document analysis, suggested prompts, and grounded Q&A.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    _ensure_analysis()
    analysis = st.session_state.analysis_cache or {
        "summary": "",
        "insights": [],
        "risks": [],
        "questions": [],
    }

    st.subheader("Executive Summary")
    st.markdown(
        f"<div class='glass-card'>{html.escape(analysis['summary'] or 'Summary unavailable.')}</div>",
        unsafe_allow_html=True,
    )

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
            if st.button(question, key=f"suggested::{question}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()
    else:
        st.write("Suggested questions will appear here once analysis is available.")


def _render_chat_history() -> None:
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                Ask a question to start the conversation. Answers stream live and keep source evidence attached.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(_format_answer_html(turn["answer"]), unsafe_allow_html=True)
            _render_sources(turn.get("sources", []))


def _append_chat_turn(question: str, answer: str, sources: list[dict[str, Any]]) -> None:
    st.session_state.chat_history.append(
        {
            "turn": len(st.session_state.chat_history) + 1,
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    )


def _handle_prompt(prompt: str, force_refresh: bool = False) -> None:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return

    with st.chat_message("user"):
        st.markdown(cleaned_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = _answer_question(cleaned_prompt, force_refresh=force_refresh)
        answer = result.get("answer") or "No answer available."
        sources = result.get("sources", [])

        stream_text(answer)
        _render_sources(sources)

        # Regenerate button — key is stable per turn index (pre-append)
        regen_key = f"regen_{len(st.session_state.chat_history)}"
        if st.button("🔄 Regenerate", key=regen_key):
            # Remove last turn so history stays clean after re-run
            if st.session_state.chat_history:
                st.session_state.chat_history.pop()
            st.session_state.pending_prompt = cleaned_prompt
            st.session_state["_regen_force"] = True
            st.rerun()

    _append_chat_turn(cleaned_prompt, answer, sources)


def _render_chat_tab() -> None:
    st.subheader("Grounded Q&A")

    clear_col, info_col = st.columns([0.22, 0.78])
    with clear_col:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.pending_prompt = ""
            st.rerun()
    with info_col:
        if st.session_state.pipeline is None:
            st.caption("Upload a document to enable chat.")
        elif not has_api_key():
            st.caption("Add `GROQ_API_KEY` to enable AI-powered Q&A.")
        else:
            st.caption("Answers are grounded in retrieved document chunks and stream live.")

    _render_chat_history()
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    prompt = st.chat_input(
        "Ask about the uploaded document...",
        disabled=st.session_state.pipeline is None or not has_api_key(),
    )

    incoming_prompt = None
    force_refresh = False

    # Check if a regenerate was triggered
    if st.session_state.get("_regen_force"):
        force_refresh = True
        st.session_state["_regen_force"] = False

    if st.session_state.pending_prompt:
        incoming_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
    elif prompt:
        incoming_prompt = prompt.strip()

    if incoming_prompt and st.session_state.pipeline and has_api_key():
        _handle_prompt(incoming_prompt, force_refresh=force_refresh)


# ── Bootstrap ────────────────────────────────────────────────────────────────

_init_state()
_inject_styles()
_render_sidebar()

st.markdown(
    """
    <div class="eyebrow">DocuMind</div>
    """,
    unsafe_allow_html=True,
)
_render_hero()

if st.session_state.processing_error:
    st.error(f"Processing failed: {st.session_state.processing_error}")

if st.session_state.pipeline is not None and not has_api_key():
    st.info(
        "For Render, add `GROQ_API_KEY` in the service environment settings. Document indexing remains available without it."
    )

overview_tab, qa_tab = st.tabs(["Overview", "Ask Questions"])

with overview_tab:
    _render_overview_tab()

with qa_tab:
    _render_chat_tab()