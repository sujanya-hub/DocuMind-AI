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
from core.config import ANALYSIS_CONTEXT_CHARS, ANALYSIS_MAX_TOKENS, has_api_key
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

initialise()

STATE_DEFAULTS = {
    "document_name": "",
    "analysis_cache": None,
    "analysis_chunks": [],
    "qa_cache": {},
    "chat_history": [],
    "pending_prompt": "",
    "processing_error": "",
    "processing_success": "",
    "processed_upload_token": "",
    "is_processing": False,
    "uploader_key": 0,
    "_regen_force": False,
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
6 questions a reader could ask, one per line, each starting with a hyphen.
Each question must be answerable directly from a specific passage of the document:
ask about results, methods, numbers, definitions or conclusions the text actually states.
Do not ask hypotheticals, comparisons with other universities or populations, or
questions that combine two variables unless the document reports that combination.
""".strip()


def _init_state() -> None:
    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default.copy() if isinstance(default, (dict, list)) else default


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600&family=Public+Sans:wght@400;500;600;700&display=swap');

        :root {
            --paper: #ffffff;
            --panel: #f4f5f7;
            --ink: #17191c;
            --graphite: #586070;
            --faint: #8a92a0;
            --rule: #e2e4e8;
            --rule-strong: #c9cdd5;
            --action: #1f3fa3;
            --action-dark: #17307d;
            --marker: #ffeea6;
            --marker-edge: #f0d25a;
            --rust: #a4381c;
            --sans: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
            --serif: "Literata", Georgia, "Times New Roman", serif;
            /* names still used by inline styles in the app */
            --text-main: var(--ink);
            --text-muted: var(--graphite);
            --text-dim: var(--faint);
        }

        html, body, [class*="css"] {
            font-family: var(--sans);
            font-size: 16px;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] { background: var(--paper); }
        [data-testid="stHeader"] { background: transparent; }
        footer { visibility: hidden; }

        [data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--rule);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.75rem; }

        .block-container {
            max-width: 46rem;
            padding: 3rem 2rem 6rem;
        }
        @media (max-width: 640px) {
            .block-container { padding: 2rem 1.25rem 4rem; }
            [data-testid="stMarkdownContainer"] h1.hero-title { font-size: 1.5rem; }
        }

        :focus-visible { outline: 2px solid var(--action); outline-offset: 2px; }

        /* ─── Wordmark ─── */
        .wordmark {
            font-family: var(--sans);
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }

        .divider {
            height: 1px;
            background: var(--rule);
            margin: 1.6rem 0;
        }

        /* ─── Document header ─── */
        [data-testid="stMarkdownContainer"] h1.hero-title {
            font-family: var(--serif);
            font-size: 2rem;
            font-weight: 600;
            line-height: 1.2;
            letter-spacing: -0.01em;
            color: var(--ink);
            margin: 0 0 0.6rem;
            padding: 0;
            overflow-wrap: anywhere;
        }
        .hero-subtitle {
            font-family: var(--serif);
            font-size: 1.05rem;
            line-height: 1.7;
            color: var(--graphite);
            margin: 0;
            max-width: 38rem;
        }
        .doc-meta {
            font-size: 0.9rem;
            color: var(--graphite);
            margin: 0 0 0.25rem;
        }
        .doc-meta .warn { color: var(--rust); }

        /* ─── Section headings and reading text ─── */
        .section-title {
            font-family: var(--sans);
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--ink);
            margin: 2.2rem 0 0.6rem;
            padding-top: 1rem;
            border-top: 1px solid var(--rule);
        }
        .section-title.first { margin-top: 1.6rem; border-top: none; padding-top: 0; }
        .prose {
            font-family: var(--serif);
            font-size: 1.0625rem;
            line-height: 1.75;
            color: var(--ink);
            max-width: 38rem;
        }
        .bullet-item {
            display: flex;
            align-items: flex-start;
            gap: 0.8rem;
            padding: 0.35rem 0;
            font-family: var(--serif);
            font-size: 1.0625rem;
            line-height: 1.65;
            color: var(--ink);
            max-width: 38rem;
        }
        .bullet-dot {
            flex-shrink: 0;
            width: 0.7rem;
            height: 1px;
            margin-top: 0.95rem;
            background: var(--faint);
        }
        .dot-amber { background: var(--rust); }
        .empty-note {
            font-size: 0.95rem;
            line-height: 1.6;
            color: var(--graphite);
            max-width: 34rem;
        }

        /* ─── Conversation ─── */
        .turn-q {
            font-family: var(--sans);
            font-size: 1.05rem;
            font-weight: 600;
            line-height: 1.45;
            color: var(--ink);
            margin: 2rem 0 0.7rem;
            padding-top: 1.4rem;
            border-top: 1px solid var(--rule);
        }
        .assistant-copy {
            font-family: var(--serif);
            font-size: 1.0625rem;
            line-height: 1.75;
            color: var(--ink);
            max-width: 38rem;
        }
        .source-inline {
            font-family: var(--sans);
            font-size: 0.78em;
            font-weight: 600;
            white-space: nowrap;
            background: var(--marker);
            color: var(--ink);
            padding: 0.05em 0.35em;
            border-radius: 2px;
        }
        .sources-line .source-inline { font-size: 0.95em; }
        .sources-line {
            margin-top: 0.9rem;
            font-family: var(--sans);
            font-size: 0.85rem;
            color: var(--graphite);
        }
        .confidence {
            font-size: 0.85rem;
            color: var(--graphite);
            margin: 0.7rem 0 0.2rem;
        }
        .confidence span { font-weight: 600; }

        /* ─── Cited passages ─── */
        .source-heading {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--graphite);
            margin: 0.4rem 0 0.2rem;
        }
        .source-card {
            border-left: 3px solid var(--marker-edge);
            padding: 0.1rem 0 0.1rem 0.9rem;
            margin: 0.9rem 0;
        }
        .source-meta {
            display: flex;
            gap: 0.9rem;
            align-items: baseline;
            margin-bottom: 0.25rem;
            font-size: 0.82rem;
            color: var(--graphite);
        }
        .badge-page { font-weight: 600; color: var(--ink); }
        .badge-score { color: var(--faint); }
        .source-text {
            font-family: var(--serif);
            font-size: 0.95rem;
            line-height: 1.65;
            color: var(--graphite);
        }

        /* ─── Suggested questions (plain list) ─── */
        .suggestion-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--graphite);
            margin: 0.4rem 0 0.2rem;
        }
        [class*="st-key-sugg_chat"] button {
            justify-content: flex-start;
            text-align: left;
            width: 100%;
            background: transparent;
            border: none;
            border-bottom: 1px solid var(--rule);
            border-radius: 0;
            padding: 0.75rem 0;
            color: var(--ink);
            font-weight: 400;
            font-size: 0.98rem;
            line-height: 1.45;
            box-shadow: none;
        }
        [class*="st-key-sugg_chat"] button:hover {
            background: transparent;
            border-bottom-color: var(--action);
            color: var(--action);
        }
        [class*="st-key-sugg_chat"] button p { text-align: left; }

        /* ─── Tabs ─── */
        [data-baseweb="tab-list"] {
            gap: 1.6rem;
            background: transparent !important;
            border-bottom: 1px solid var(--rule);
        }
        [data-baseweb="tab"] {
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0;
            color: var(--graphite);
            padding: 0.6rem 0;
            font-family: var(--sans);
            font-weight: 500;
            font-size: 0.95rem;
            margin-bottom: -1px;
        }
        [data-baseweb="tab"]:hover { color: var(--ink); background: transparent; }
        [aria-selected="true"][data-baseweb="tab"] {
            color: var(--ink);
            border-bottom-color: var(--ink);
            font-weight: 600;
        }
        [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

        /* ─── Buttons ─── */
        div.stButton > button {
            border-radius: 6px;
            border: 1px solid var(--rule-strong);
            background: var(--paper);
            color: var(--ink);
            font-family: var(--sans);
            font-weight: 500;
            font-size: 0.88rem;
            padding: 0.4rem 0.9rem;
            box-shadow: none;
        }
        div.stButton > button:hover {
            border-color: var(--action);
            color: var(--action);
            background: var(--paper);
        }
        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="stBaseButton-primary"] {
            background: var(--action);
            border-color: var(--action);
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:hover,
        div.stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: var(--action-dark);
            border-color: var(--action-dark);
            color: #ffffff;
        }

        /* ─── File uploader ─── */
        div[data-testid="stFileUploader"] section {
            background: var(--paper);
            border: 1px solid var(--rule-strong);
            border-radius: 6px;
        }
        div[data-testid="stFileUploader"] section:hover { border-color: var(--action); }

        /* ─── Expanders and alerts ─── */
        [data-testid="stExpander"] {
            background: transparent;
            border: none;
            border-top: 1px solid var(--rule);
            border-radius: 0;
            margin-top: 0.8rem;
        }
        [data-testid="stExpander"] details { border: none; }
        [data-testid="stExpander"] summary { padding-left: 0; font-size: 0.9rem; color: var(--graphite); }
        [data-testid="stAlert"] {
            background: var(--panel) !important;
            border: 1px solid var(--rule) !important;
            border-radius: 4px !important;
            color: var(--ink) !important;
        }

        /* ─── Chat input ─── */
        [data-testid="stChatInput"] {
            background: var(--paper) !important;
            border: 1px solid var(--rule-strong) !important;
            border-radius: 6px !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--action) !important;
            box-shadow: none !important;
        }

        p, label, .stMarkdown { color: var(--ink); }
        .stCaption, [data-testid="stCaptionContainer"] { color: var(--graphite) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clear_document() -> None:
    reset_document_state()
    for key, default in STATE_DEFAULTS.items():
        st.session_state[key] = default.copy() if isinstance(default, (dict, list)) else default
    st.session_state.uploader_key += 1


# ─── Analysis ────────────────────────────────────────────────────────────────


def _page_sort_key(chunk: dict[str, Any]) -> int:
    page = chunk.get("page_number")
    return page if isinstance(page, int) else 0


def _build_analysis_context(chunks: list[dict[str, Any]], max_chars: int = ANALYSIS_CONTEXT_CHARS) -> str:
    """
    Build the context for the summary.

    Chunks are kept in page order and sampled evenly across the whole
    document, so the summary covers the full paper instead of the first
    few pages.
    """
    valid = [c for c in chunks if isinstance(c, dict)]
    try:
        valid = sorted(valid, key=_page_sort_key)  # stable: keeps order within a page
    except Exception:
        pass

    entries: list[str] = []
    seen: set[str] = set()
    for chunk in valid:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        entry = f"[Page {chunk.get('page_number', '?')}]\n{text}"
        if entry in seen:
            continue
        seen.add(entry)
        entries.append(entry)

    if not entries:
        return ""

    separator_cost = 2  # the "\n\n" between entries
    total = sum(len(e) + separator_cost for e in entries)
    if total <= max_chars:
        return "\n\n".join(entries)

    avg_len = total / len(entries)
    target = max(1, int(max_chars // avg_len))

    if target >= len(entries):
        picks = entries
    elif target == 1:
        picks = [entries[0]]
    else:
        last = len(entries) - 1
        indices = sorted({round(i * last / (target - 1)) for i in range(target)})
        picks = [entries[i] for i in indices]

    selected: list[str] = []
    used = 0
    for entry in picks:
        cost = len(entry) + separator_cost
        if used + cost > max_chars:
            continue
        selected.append(entry)
        used += cost
    return "\n\n".join(selected)


def _extract_section(text: str, tag: str) -> str:
    pattern = rf"\[{re.escape(tag)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


# A list marker is a dash, asterisk or bullet followed by a space, or a
# number followed by "." or ")" and a space. A bare leading number such as
# "13% of students..." is content and must be kept.
_BULLET_MARKER = re.compile(r"^\s*(?:[-*\u2022]\s+|\d+[.)]\s+)")


def _parse_bullets(raw_text: str) -> list[str]:
    bullets: list[str] = []
    for line in raw_text.splitlines():
        cleaned = _BULLET_MARKER.sub("", line).strip()
        if cleaned:
            bullets.append(cleaned)
    return bullets


def _generate_analysis(context: str) -> str:
    return generate_response(
        system_prompt=_ANALYSIS_PROMPT,
        user_prompt=f"DOCUMENT CONTEXT:\n\n{context}",
        max_tokens=ANALYSIS_MAX_TOKENS,
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
    chunks = st.session_state.get("analysis_chunks") or pipeline.get("chunks", [])
    context = _build_analysis_context(chunks)
    if not context.strip():
        st.session_state.analysis_cache = {
            "summary": "No usable text was available for document analysis.",
            "insights": [],
            "risks": [],
            "questions": [],
        }
        return
    try:
        with st.spinner("Analyzing document..."):
            raw = _generate_analysis(context)
    except Exception as exc:
        logger.warning("Document analysis failed: %s", exc)
        st.session_state.analysis_cache = {
            "summary": f"Analysis unavailable: {exc}",
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


# ─── Question answering ──────────────────────────────────────────────────────


def _answer_question(question: str, force_refresh: bool = False) -> dict[str, Any]:
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        return {"answer": "Upload a document first.", "sources": []}
    qa_engine = pipeline.get("qa_engine")
    if qa_engine is None:
        return {"answer": "The document is still processing. Please wait a moment.", "sources": []}
    cache_key = f"{st.session_state.document_name}:{question.strip().lower()}"
    cache = st.session_state.qa_cache
    if force_refresh and cache_key in cache:
        del cache[cache_key]
        st.session_state.qa_cache = cache
    if cache_key in cache:
        return cache[cache_key]
    try:
        result = qa_engine.answer(question)
    except Exception as exc:
        result = {
            "answer": f"Could not answer right now. {type(exc).__name__}: {exc}",
            "sources": [],
        }
    cache[cache_key] = result
    st.session_state.qa_cache = cache
    return result


def _process_uploaded_file(uploaded_file: Any) -> None:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.getvalue()
    upload_token = hashlib.md5(file_bytes).hexdigest()
    uploaded_file.seek(0)

    if (
        not is_new_upload(uploaded_file.name)
        and st.session_state.processed_upload_token == upload_token
        and is_document_indexed()
    ):
        return

    st.session_state.is_processing = True
    st.session_state.processing_error = ""
    st.session_state.processing_success = ""

    try:
        with st.spinner("Processing document..."):
            uploaded_file.seek(0)
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
    finally:
        st.session_state.is_processing = False


# ─── Rendering helpers ───────────────────────────────────────────────────────


def _format_answer_html(text: str) -> str:
    safe_text = html.escape((text or "No answer available.").strip())
    # The model sometimes returns markdown bold (**Sources:**). Strip it so the
    # "Sources:" line below can be recognised.
    safe_text = safe_text.replace("**", "")
    safe_text = re.sub(
        r"(?im)\s*^Sources:\s*(.+)$",
        lambda m: f"<div class='sources-line'>Sources: {m.group(1)}</div>",
        safe_text,
    )
    safe_text = re.sub(
        r"\(p\.\s*(\d+)\)",
        lambda m: f"<span class='source-inline'>p. {m.group(1)}</span>",
        safe_text,
    )
    safe_text = re.sub(
        r"\bPage\s+(\d+)\b",
        lambda m: f"<span class='source-inline'>p. {m.group(1)}</span>",
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
    delay = 0.005
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[: i + chunk_size])
        placeholder.markdown(_format_answer_html(chunk + " ▍"), unsafe_allow_html=True)
        time.sleep(delay)
    placeholder.markdown(_format_answer_html(text), unsafe_allow_html=True)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No source chunks retrieved for this answer.")
        return
    for src in sources:
        page = src.get("page_number", "?")
        text = re.sub(r"\s+", " ", str(src.get("text", "")).strip())
        preview = html.escape(text[:220] + ("…" if len(text) > 220 else ""))
        score = src.get("score")
        score_html = ""
        if isinstance(score, (float, int)):
            score_html = f"<span class='badge-score'>match {float(score):.2f}</span>"
        st.markdown(
            f"""<div class="source-card">
                    <div class="source-meta">
                        <span class="badge-page">Page {page}</span>
                        {score_html}
                    </div>
                    <div class="source-text">{preview}</div>
                </div>""",
            unsafe_allow_html=True,
        )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="wordmark">DocuMind</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        if not has_api_key():
            st.warning(
                "`GROQ_API_KEY` is missing. Indexing works but AI analysis "
                "and Q&A are disabled until you add it to Streamlit secrets."
            )

        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            key=f"uploaded_file_{st.session_state.uploader_key}",
        )

        if uploaded_file is not None:
            _process_uploaded_file(uploaded_file)

        if st.session_state.is_processing:
            st.info("Processing document… please wait.")

        if st.session_state.processing_success:
            st.success(st.session_state.processing_success)

        pipeline = st.session_state.get("pipeline")
        if pipeline is not None:
            metadata = pipeline["metadata"]
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("**Current document**")
            st.markdown(
                f"<span style='color:var(--text-muted);font-size:0.88rem'>"
                f"{html.escape(st.session_state.document_name or metadata.get('file_name', 'Uploaded PDF'))}"
                f"</span>",
                unsafe_allow_html=True,
            )
            if st.button("Clear document", use_container_width=True):
                _clear_document()
                st.rerun()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        debug_mode = st.toggle("Debug mode", value=False)
        if debug_mode and st.session_state.get("pipeline") is not None:
            st.markdown("**Pipeline state:**")
            safe_debug = {
                k: v for k, v in st.session_state.pipeline.items()
                if k not in ("qa_engine", "vector_db")
            }
            safe_debug["analysis_chunks_count"] = len(st.session_state.get("analysis_chunks", []))
            safe_debug["vector_db_present"] = st.session_state.pipeline.get("vector_db") is not None
            st.json(safe_debug)


def _render_hero() -> None:
    pipeline = st.session_state.get("pipeline")

    if pipeline is None:
        st.markdown(
            '<h1 class="hero-title">Ask questions about a PDF</h1>'
            '<p class="hero-subtitle">Upload a document in the sidebar. DocuMind summarizes it, '
            "then answers your questions and tells you which pages it used.</p>",
            unsafe_allow_html=True,
        )
        return

    metadata = pipeline["metadata"]
    file_name = html.escape(metadata.get("file_name", st.session_state.document_name))
    index_note = (
        ""
        if pipeline.get("vector_db") is not None
        else ' <span class="warn">The search index is missing, so questions will not work.</span>'
    )
    total_pages = metadata.get("page_count", 0)
    indexed_pages = pipeline.get("indexed_pages") or total_pages
    pages_text = (
        f"{total_pages} pages"
        if indexed_pages >= total_pages
        else f'{total_pages} pages (first {indexed_pages} indexed)'
    )
    st.markdown(
        f'<h1 class="hero-title">{file_name}</h1>'
        f'<p class="doc-meta">{pages_text}, '
        f'{pipeline.get("total_chunks", 0)} passages, {format_file_size(pipeline["file_size"])}. '
        f'Indexed in {pipeline.get("processing_time", 0)} seconds.{index_note}</p>',
        unsafe_allow_html=True,
    )


def _render_overview_tab() -> None:
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.markdown(
            "<div class='empty-note'>Upload a PDF to see its summary.</div>",
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

    st.markdown("<div class='section-title first'>Summary</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='prose'>{html.escape(analysis['summary'] or 'Summary unavailable.')}</div>",
        unsafe_allow_html=True,
    )

    def _bullets(items: list[str], dot_class: str, empty_text: str) -> str:
        if not items:
            return f"<div class='empty-note'>{empty_text}</div>"
        return "".join(
            f"<div class='bullet-item'><span class='bullet-dot {dot_class}'></span>"
            f"<span>{html.escape(item)}</span></div>"
            for item in items
        )

    st.markdown("<div class='section-title'>Main points</div>", unsafe_allow_html=True)
    st.markdown(
        _bullets(analysis["insights"], "", "No main points yet."),
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Caveats</div>", unsafe_allow_html=True)
    st.markdown(
        _bullets(analysis["risks"], "dot-amber", "No caveats found."),
        unsafe_allow_html=True,
    )


# ─── Chat ────────────────────────────────────────────────────────────────────


def _regenerate_last() -> None:
    """Button callback: drop the last turn and ask the same question again."""
    history = st.session_state.chat_history
    if not history:
        return
    last = history.pop()
    st.session_state.pending_prompt = last["question"]
    st.session_state["_regen_force"] = True


def _render_chat_history(show_regenerate: bool = True) -> None:
    history = st.session_state.chat_history
    if not history:
        return

    for index, turn in enumerate(history):
        st.markdown(
            f'<div class="turn-q">{html.escape(turn["question"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_format_answer_html(turn["answer"]), unsafe_allow_html=True)

        sources = turn.get("sources", [])
        if sources:
            _render_confidence(sources)
            label = f"{len(sources)} source{'s' if len(sources) != 1 else ''}"
            with st.expander(f"View {label}", expanded=False):
                _render_sources(sources)

        is_last = index == len(history) - 1
        if is_last and show_regenerate:
            st.button(
                "Regenerate answer",
                key=f"regen_{turn.get('turn', index + 1)}",
                on_click=_regenerate_last,
            )


def _append_chat_turn(question: str, answer: str, sources: list[dict[str, Any]]) -> None:
    st.session_state.chat_history.append(
        {
            "turn": len(st.session_state.chat_history) + 1,
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    )


def _render_confidence(sources: list[dict[str, Any]]) -> None:
    """Show how closely the retrieved passages matched the question.

    This is the average similarity of the retrieved passages, not a measure of
    whether the answer is correct, so it is labelled accordingly.
    """
    valid_scores = [s.get("score") for s in sources if isinstance(s.get("score"), (int, float))]
    if not valid_scores:
        return
    avg_score = sum(valid_scores) / len(valid_scores)
    pct = max(0, min(int(avg_score * 100), 100))
    color = "#15803d" if pct >= 70 else "#b45309" if pct >= 45 else "#b91c1c"
    st.markdown(
        f"<div class='confidence'>Retrieval match <span style='color:{color}'>{pct}%</span></div>",
        unsafe_allow_html=True,
    )


def _handle_prompt(prompt: str, force_refresh: bool = False) -> None:
    cleaned = prompt.strip()
    if not cleaned:
        return

    st.markdown(
        f'<div class="turn-q">{html.escape(cleaned)}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Finding the answer…"):
        result = _answer_question(cleaned, force_refresh=force_refresh)

    answer = result.get("answer") or "No answer available."
    sources = result.get("sources", [])

    stream_text(answer)

    if sources:
        _render_confidence(sources)
        with st.expander(
            f"View {len(sources)} source{'s' if len(sources) != 1 else ''}",
            expanded=False,
        ):
            _render_sources(sources)

    _append_chat_turn(cleaned, answer, sources)

    st.button(
        "Regenerate answer",
        key=f"regen_{len(st.session_state.chat_history)}",
        on_click=_regenerate_last,
    )


def _render_suggestions_in_chat(questions: list[str]) -> None:
    if not questions:
        return
    st.markdown(
        "<div class='suggestion-label'>Suggested questions</div>",
        unsafe_allow_html=True,
    )
    for question in questions:
        if st.button(
            question,
            key=f"sugg_chat::{question}",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = question
            st.rerun()
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


def _render_chat_tab() -> None:
    pipeline = st.session_state.get("pipeline")
    ready = pipeline is not None and has_api_key()

    clear_col, info_col = st.columns([0.22, 0.78])
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.pending_prompt = ""
            st.rerun()
    with info_col:
        if pipeline is None:
            st.caption("Upload a document to enable chat.")
        elif not has_api_key():
            st.caption("Add `GROQ_API_KEY` to Streamlit secrets to turn on answers.")
        else:
            st.caption("Answers cite the pages they come from.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # The conversation goes in a container created BEFORE the chat input, so
    # questions and answers appear above the input box instead of below it.
    conversation = st.container()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    prompt = st.chat_input(
        "Ask anything about your document…",
        disabled=not ready,
    )

    force_refresh = bool(st.session_state.get("_regen_force", False))
    st.session_state["_regen_force"] = False

    incoming_prompt = None
    if st.session_state.pending_prompt:
        incoming_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
    elif prompt:
        incoming_prompt = prompt.strip()

    with conversation:
        if ready:
            _ensure_analysis()
            questions = (st.session_state.analysis_cache or {}).get("questions", [])
            # Suggestions only on an empty chat, and not while a question is being answered.
            if questions and not st.session_state.chat_history and not incoming_prompt:
                _render_suggestions_in_chat(questions)

        if st.session_state.chat_history:
            _render_chat_history(show_regenerate=not incoming_prompt)
        elif pipeline is None:
            st.markdown(
                "<div class='empty-note'>Upload a PDF to start asking questions.</div>",
                unsafe_allow_html=True,
            )
        elif not has_api_key():
            st.markdown(
                "<div class='empty-note'>Add your <code>GROQ_API_KEY</code> to Streamlit "
                "secrets to turn on answers.</div>",
                unsafe_allow_html=True,
            )

        if incoming_prompt and ready:
            _handle_prompt(incoming_prompt, force_refresh=force_refresh)


# ─── Bootstrap ───────────────────────────────────────────────────────────────

_init_state()
_inject_styles()
_render_sidebar()

_render_hero()

if st.session_state.processing_error:
    st.error(f"Processing failed: {st.session_state.processing_error}")
    st.markdown(
        "<div class='empty-note'>Indexing failed. Upload the file again, "
        "or check the error above.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

pipeline = st.session_state.get("pipeline")

if pipeline is None:
    if st.session_state.get("is_processing", False):
        st.info("Processing document… please wait.")
    else:
        st.markdown(
            "<div class='empty-note'>No document loaded yet.</div>",
            unsafe_allow_html=True,
        )
    st.stop()

if not has_api_key():
    st.info(
        "Add `GROQ_API_KEY` in Streamlit secrets to enable AI analysis and Q&A. "
        "Document indexing is already active."
    )

overview_tab, qa_tab = st.tabs(["Summary", "Ask questions"])

with overview_tab:
    _render_overview_tab()

with qa_tab:
    _render_chat_tab()