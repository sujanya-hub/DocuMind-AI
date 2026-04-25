from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import time
from typing import Any

import streamlit as st

from core.ai_engine import generate_response
from core.config import ANALYSIS_CONTEXT_CHARS, has_api_key
from core.document_service import run_pipeline
from core.session_manager import (
    initialise,
    is_document_indexed,
    is_new_upload,
    reset_document_state,
    store_pipeline_result,
)
from utils.helpers import format_file_size

logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="DocuMind",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# initialise() runs before _init_state() so session_manager owns pipeline first
initialise()


# pipeline is intentionally excluded — session_manager is the sole owner
STATE_DEFAULTS = {
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
    """Initialise session state keys that are not owned by session_manager.

    pipeline is deliberately omitted — only session_manager.store_pipeline_result()
    may write to st.session_state.pipeline. Touching it here would reset it to None
    on every Streamlit rerun, which is the root cause being fixed.
    """
    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default.copy() if isinstance(default, (dict, list)) else default


def _hydrate_pipeline_from_session() -> None:
    # Pipeline is stored directly in st.session_state.pipeline — no reconstruction needed.
    return


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

        :root {
            --bg-top: #0a0f1e;
            --bg-mid: #0d1526;
            --bg-bottom: #050810;
            --text-main: #f0f4ff;
            --text-muted: #8fa4c8;
            --accent-blue: #38bdf8;
            --accent-blue-dim: rgba(56, 189, 248, 0.18);
            --accent-green: #34d399;
            --accent-green-dim: rgba(52, 211, 153, 0.14);
            --border-subtle: rgba(148, 163, 184, 0.12);
            --glass-bg: rgba(255, 255, 255, 0.04);
            --glass-border: rgba(255, 255, 255, 0.09);
        }

        html, body, [class*="css"] {
            color: var(--text-main);
            font-family: "DM Sans", "Segoe UI", sans-serif;
        }

        body {
            background: var(--bg-top);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse 80% 50% at -10% -5%, rgba(56, 189, 248, 0.14) 0%, transparent 55%),
                radial-gradient(ellipse 60% 40% at 110% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
                radial-gradient(ellipse 40% 30% at 50% 100%, rgba(52, 211, 153, 0.06) 0%, transparent 50%),
                linear-gradient(170deg, #0a0f1e 0%, #0d1526 55%, #050810 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,
                rgba(5, 8, 16, 0.96) 0%,
                rgba(10, 15, 30, 0.90) 100%);
            border-right: 1px solid var(--border-subtle);
            backdrop-filter: blur(28px);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.75rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--text-main);
            font-family: "Syne", sans-serif;
            letter-spacing: -0.025em;
        }

        p, label, .stCaption, .stMarkdown, .stTextInput, .stChatInput {
            color: var(--text-main);
        }

        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploaderDropzone"] {
            background: var(--glass-bg);
            border: 1.5px dashed rgba(56, 189, 248, 0.25);
            border-radius: 18px;
            transition: border-color 0.2s ease;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: rgba(56, 189, 248, 0.5);
        }

        div[data-testid="stVerticalBlock"] div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            backdrop-filter: blur(14px);
            box-shadow: 0 12px 40px rgba(2, 6, 23, 0.22);
            padding: 0.4rem 0.3rem;
        }

        .status-card,
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            border-radius: 22px;
            padding: 1.35rem 1.5rem;
            border: 1px solid var(--glass-border);
            box-shadow:
                0 0 0 1px rgba(56, 189, 248, 0.04),
                0 20px 60px rgba(2, 6, 23, 0.28);
        }

        .hero-card {
            padding: 1.6rem 1.75rem;
            margin-bottom: 1.5rem;
            background:
                linear-gradient(135deg, rgba(56, 189, 248, 0.06) 0%, rgba(99, 102, 241, 0.04) 100%),
                var(--glass-bg);
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--accent-blue);
            margin-bottom: 0.65rem;
        }

        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin: 0;
            font-family: "Syne", sans-serif;
            background: linear-gradient(135deg, #f0f4ff 30%, #7dd3fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-subtitle {
            margin: 0.6rem 0 0;
            color: var(--text-muted);
            max-width: 820px;
            line-height: 1.7;
            font-size: 0.98rem;
        }

        .metric-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1.1rem;
        }

        .metric-pill {
            background: rgba(255, 255, 255, 0.07);
            color: var(--text-main);
            border-radius: 999px;
            padding: 0.4rem 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.11);
            font-size: 0.83rem;
            font-weight: 500;
            letter-spacing: 0.01em;
        }

        .metric-pill.ready {
            background: rgba(52, 211, 153, 0.12);
            border-color: rgba(52, 211, 153, 0.3);
            color: #a7f3d0;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin: 1.25rem 0;
        }

        .stat-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            text-align: center;
        }

        .stat-value {
            font-family: "Syne", sans-serif;
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--accent-blue);
            line-height: 1;
        }

        .stat-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-top: 0.35rem;
        }

        .assistant-copy {
            color: var(--text-main);
            line-height: 1.8;
            font-size: 0.97rem;
        }

        .source-inline {
            display: inline-block;
            margin: 0 0.07rem;
            padding: 0.07rem 0.42rem;
            border-radius: 999px;
            background: var(--accent-blue-dim);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #bae6fd;
            font-weight: 700;
            font-size: 0.85em;
        }

        .sources-line {
            margin-top: 0.9rem;
            padding: 0.75rem 1rem;
            border-radius: 14px;
            background: rgba(148, 163, 184, 0.07);
            border: 1px solid rgba(148, 163, 184, 0.15);
            color: #e2e8f0;
        }
        .sources-line strong { color: var(--accent-blue); }

        .source-card {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid var(--glass-border);
            margin-bottom: 0.7rem;
            transition: border-color 0.15s ease;
        }
        .source-card:hover { border-color: rgba(56, 189, 248, 0.25); }

        .source-page {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: var(--accent-green-dim);
            border: 1px solid rgba(52, 211, 153, 0.26);
            color: #a7f3d0;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .source-text {
            color: var(--text-muted);
            font-size: 0.91rem;
            line-height: 1.65;
        }

        .source-score {
            color: var(--accent-blue);
            font-size: 0.76rem;
            margin-left: 0.45rem;
        }

        .empty-state {
            padding: 2rem 1.5rem;
            border-radius: 20px;
            background: var(--glass-bg);
            border: 1.5px dashed var(--border-subtle);
            color: var(--text-muted);
            text-align: center;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        div.stButton > button,
        div.stDownloadButton > button,
        button[kind="primary"] {
            border-radius: 999px;
            border: 1px solid rgba(125, 211, 252, 0.2);
            background: linear-gradient(135deg,
                rgba(14, 165, 233, 0.22) 0%,
                rgba(59, 130, 246, 0.16) 100%);
            color: #f0f4ff;
            font-weight: 600;
            font-family: "DM Sans", sans-serif;
            letter-spacing: 0.01em;
            transition: all 0.18s ease;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            border-color: rgba(125, 211, 252, 0.42);
            background: linear-gradient(135deg,
                rgba(14, 165, 233, 0.34) 0%,
                rgba(59, 130, 246, 0.26) 100%);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.18);
        }

        [data-baseweb="tab-list"] { gap: 0.5rem; }
        [data-baseweb="tab"] {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 999px;
            color: var(--text-muted);
            padding: 0.48rem 1.1rem;
            font-family: "DM Sans", sans-serif;
            font-weight: 500;
            transition: all 0.15s ease;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            color: white;
            background: rgba(56, 189, 248, 0.15);
            border-color: rgba(56, 189, 248, 0.3);
        }

        .stAlert {
            background: rgba(10, 15, 30, 0.82);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
        }

        .chat-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.5rem;
        }
        .chat-header h3 {
            margin: 0;
            font-size: 1.25rem;
        }

        .section-divider {
            height: 1px;
            background: linear-gradient(90deg,
                transparent 0%, var(--border-subtle) 30%,
                var(--border-subtle) 70%, transparent 100%);
            margin: 1.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clear_document() -> None:
    reset_document_state()
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
    return generate_response(
        system_prompt=_ANALYSIS_PROMPT,
        user_prompt=f"DOCUMENT CONTEXT:\n\n{context}",
        max_tokens=900,
        temperature=0.1,
    )


def _ensure_analysis() -> None:
    if st.session_state.analysis_cache is not None:
        return

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        return

    if not has_api_key():
        st.session_state.analysis_cache = {
            "summary": "LLM not configured (missing API key).",
            "insights": [],
            "risks": [],
            "questions": [],
        }
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
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        return {"answer": "Upload a document first.", "sources": []}

    cache_key = f"{st.session_state.document_name}:{question.strip().lower()}"
    cache = st.session_state.qa_cache

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
    file_bytes = uploaded_file.getvalue()
    upload_token = hashlib.md5(file_bytes).hexdigest()

    if (
        not is_new_upload(uploaded_file.name)
        and st.session_state.processed_upload_token == upload_token
        and is_document_indexed()
    ):
        return

    try:
        with st.spinner("Processing document with AI..."):
            result = run_pipeline(uploaded_file, UPLOAD_DIR)

        if not isinstance(result, dict):
            raise TypeError(f"run_pipeline() must return a dict, got {type(result).__name__}")

        required_keys = {"pages", "chunks", "qa_engine", "metadata"}
        missing = required_keys - set(result.keys())
        if missing:
            raise ValueError(f"Pipeline missing keys: {missing}")
        if not result["chunks"]:
            raise ValueError("No chunks generated")
        if result["qa_engine"] is None:
            raise ValueError("QA engine not initialized")

        if result.get("vector_db") is None:
            raise ValueError("vector_db missing after pipeline execution")

        # session_manager is the single source of truth for pipeline
        store_pipeline_result(result)

        st.session_state.document_name = uploaded_file.name
        st.session_state.analysis_cache = None
        st.session_state.qa_cache = {}
        st.session_state.chat_history = []
        st.session_state.pending_prompt = ""
        st.session_state.processing_error = ""
        st.session_state.processing_success = "Document indexed successfully!"
        st.session_state.processed_upload_token = upload_token

        st.toast("Document indexed successfully")
        st.rerun()

    except Exception as exc:
        st.session_state.processing_error = str(exc)
        st.session_state.processing_success = ""
        logger.error("Pipeline failed: %s", exc)
        st.error(f"Processing failed: {exc}")


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
        placeholder.markdown(_format_answer_html(chunk + " |"), unsafe_allow_html=True)
        time.sleep(delay)

    placeholder.markdown(_format_answer_html(text), unsafe_allow_html=True)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No explicit source chunks were retrieved for this answer.")
        return

    st.markdown("### Sources")
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
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;"
            "background:linear-gradient(135deg,#f0f4ff,#7dd3fc);-webkit-background-clip:text;"
            "-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.1rem'>DocuMind</div>",
            unsafe_allow_html=True,
        )
        st.caption("AI document intelligence")
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        if has_api_key():
            st.success("LLM configuration detected")
        else:
            st.warning(
                "`GROQ_API_KEY` is not set. Indexing still works, "
                "but AI analysis and Q&A require the environment variable."
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

        pipeline = st.session_state.get("pipeline")
        if pipeline is not None:
            metadata = pipeline["metadata"]
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("**Current Document**")
            st.write(st.session_state.document_name)
            st.caption(
                f"{metadata.get('page_count', len(pipeline['pages']))} pages · "
                f"{pipeline.get('total_chunks', len(pipeline['chunks']))} chunks · "
                f"{format_file_size(pipeline['file_size'])} · "
                f"{pipeline.get('processing_time', 0)}s"
            )
            if st.button("Clear document", use_container_width=True):
                _clear_document()
                st.rerun()

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        debug_mode = st.toggle("Debug mode", value=False)
        if debug_mode and st.session_state.get("pipeline") is not None:
            st.write("**Pipeline state:**")
            safe_debug = {
                k: v for k, v in st.session_state.pipeline.items()
                if k not in ("pages", "chunks", "qa_engine", "summarizer", "vector_db")
            }
            safe_debug["pages_count"] = len(st.session_state.pipeline.get("pages", []))
            safe_debug["chunks_count"] = len(st.session_state.pipeline.get("chunks", []))
            safe_debug["vector_db_present"] = st.session_state.pipeline.get("vector_db") is not None
            st.json(safe_debug)


def _render_hero() -> None:
    pipeline = st.session_state.get("pipeline")

    if pipeline is None:
        st.markdown(
            """
            <div class="status-card hero-card">
                <div class="eyebrow">Document Intelligence</div>
                <h1 class="hero-title">Premium RAG for dense PDFs</h1>
                <p class="hero-subtitle">
                    Upload a document to generate grounded analysis, search with citations,
                    and chat against your indexed content in a modern glassmorphism workspace.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    metadata = pipeline["metadata"]
    vector_db = pipeline.get("vector_db")
    vector_ready = vector_db is not None
    vector_count = vector_db.total_chunks if vector_ready else 0

    st.markdown(
        f"""
        <div class="status-card hero-card">
            <div class="eyebrow">Indexed And Ready</div>
            <h1 class="hero-title">{html.escape(metadata.get('file_name', st.session_state.document_name))}</h1>
            <p class="hero-subtitle">
                Retrieval, analysis, and grounded chat are live.
                Every answer stays anchored to your indexed document.
            </p>
            <div class="metric-row">
                <span class="metric-pill">Pages: {metadata.get('page_count', len(pipeline['pages']))}</span>
                <span class="metric-pill">Chunks: {pipeline.get('total_chunks', len(pipeline['chunks']))}</span>
                <span class="metric-pill">{pipeline.get('processing_time', 0)}s processing</span>
                <span class="metric-pill">{format_file_size(pipeline['file_size'])}</span>
                <span class="metric-pill {'ready' if vector_ready else ''}">
                    Vector DB: {'Ready' if vector_ready else 'Missing'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{metadata.get('page_count', len(pipeline['pages']))}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pipeline.get('total_chunks', len(pipeline['chunks']))}</div>
                <div class="stat-label">Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{vector_count}</div>
                <div class="stat-label">Vectors</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview_tab() -> None:
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.markdown(
            "<div class='empty-state'>"
            "Upload a PDF and unlock AI-powered insights, summaries, and chat."
            "</div>",
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

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Insights")
        if analysis["insights"]:
            for item in analysis["insights"]:
                st.markdown(f"- {item}")
        else:
            st.markdown(
                "<div class='empty-state'>Insights will appear once analysis is available.</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("Risks & Caveats")
        if analysis["risks"]:
            for item in analysis["risks"]:
                st.markdown(f"- {item}")
        else:
            st.markdown(
                "<div class='empty-state'>No risk analysis available yet.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Suggested Follow-Up Questions")
    if analysis["questions"]:
        for question in analysis["questions"]:
            if st.button(question, key=f"suggested::{question}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()
    else:
        st.markdown(
            "<div class='empty-state'>Suggested questions will appear once analysis is available.</div>",
            unsafe_allow_html=True,
        )


def _render_chat_history() -> None:
    if not st.session_state.chat_history:
        st.markdown(
            "<div class='empty-state'>"
            "Ask a question to start the conversation. "
            "Answers stream live and keep source evidence attached."
            "</div>",
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
        with st.spinner("Generating answer..."):
            result = _answer_question(cleaned_prompt, force_refresh=force_refresh)
        answer = result.get("answer") or "No answer available."
        sources = result.get("sources", [])

        stream_text(answer)

        if sources:
            valid_scores = [s.get("score") for s in sources if isinstance(s.get("score"), (int, float))]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                confidence_pct = min(int(avg_score * 100), 100)
                color = "#34d399" if confidence_pct >= 70 else "#fbbf24" if confidence_pct >= 45 else "#f87171"
                st.markdown(
                    f"<div style='margin-top:0.5rem'>"
                    f"<span style='display:inline-flex;align-items:center;gap:0.4rem;"
                    f"padding:0.25rem 0.75rem;border-radius:999px;"
                    f"background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);"
                    f"font-size:0.78rem;font-weight:600;color:{color}'>"
                    f"Confidence {confidence_pct}%"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

        _render_sources(sources)

        regen_key = f"regen_{len(st.session_state.chat_history)}"
        if st.button("Regenerate", key=regen_key):
            if st.session_state.chat_history:
                st.session_state.chat_history.pop()
            st.session_state.pending_prompt = cleaned_prompt
            st.session_state["_regen_force"] = True
            st.rerun()

    _append_chat_turn(cleaned_prompt, answer, sources)


def _render_chat_tab() -> None:
    st.markdown(
        "<div class='chat-header'>"
        "<h3>Ask Anything About Your Document</h3>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Answers are grounded in document chunks with page-level citations.")

    clear_col, info_col = st.columns([0.22, 0.78])
    with clear_col:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.pending_prompt = ""
            st.rerun()
    with info_col:
        pipeline = st.session_state.get("pipeline")
        if pipeline is None:
            st.caption("Upload a document to enable chat.")
        elif not has_api_key():
            st.caption("Add `GROQ_API_KEY` to enable AI-powered Q&A.")
        else:
            st.caption("Live streaming · Citation-backed · Grounded in your document")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _render_chat_history()
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    pipeline = st.session_state.get("pipeline")
    prompt = st.chat_input(
        "Ask about the uploaded document...",
        disabled=pipeline is None or not has_api_key(),
    )

    incoming_prompt = None
    force_refresh = False

    if st.session_state.get("_regen_force"):
        force_refresh = True
        st.session_state["_regen_force"] = False

    if st.session_state.pending_prompt:
        incoming_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
    elif prompt:
        incoming_prompt = prompt.strip()

    if incoming_prompt and pipeline is not None and has_api_key():
        _handle_prompt(incoming_prompt, force_refresh=force_refresh)


# Bootstrap

_init_state()
_hydrate_pipeline_from_session()
_inject_styles()
_render_sidebar()

st.markdown("<div class='eyebrow'>DocuMind</div>", unsafe_allow_html=True)
_render_hero()

# Temporary debug line — remove once session state is confirmed stable
st.write("PIPELINE EXISTS:", st.session_state.get("pipeline") is not None)

if st.session_state.processing_error:
    st.error(f"Processing failed: {st.session_state.processing_error}")
    st.markdown(
        "<div class='empty-state'>"
        "Something went wrong while indexing your document. "
        "Try re-uploading or check the error above for details."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

pipeline = st.session_state.get("pipeline")

if pipeline is None:
    st.markdown(
        "<div class='empty-state'>"
        "Upload a PDF and unlock AI-powered insights, summaries, and chat."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if not has_api_key():
    st.info(
        "Add `GROQ_API_KEY` in your environment settings to enable AI analysis and Q&A. "
        "Document indexing is already active."
    )

overview_tab, qa_tab = st.tabs(["Overview", "Ask Questions"])

with overview_tab:
    _render_overview_tab()

with qa_tab:
    _render_chat_tab()