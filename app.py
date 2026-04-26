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
    "active_tab": "overview",
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


def _hydrate_pipeline_from_session() -> None:
    return


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Syne:wght@700;800;900&display=swap');

        :root {
            --bg-deep: #060a14;
            --bg-mid: #0b1022;
            --bg-surface: #0f1630;
            --text-main: #eef2ff;
            --text-muted: #7c92b8;
            --text-dim: #4a607f;
            --accent-blue: #38bdf8;
            --accent-blue-glow: rgba(56, 189, 248, 0.22);
            --accent-blue-dim: rgba(56, 189, 248, 0.12);
            --accent-violet: #818cf8;
            --accent-violet-dim: rgba(129, 140, 248, 0.12);
            --accent-emerald: #34d399;
            --accent-emerald-dim: rgba(52, 211, 153, 0.12);
            --accent-amber: #fbbf24;
            --border-subtle: rgba(148, 163, 184, 0.10);
            --border-medium: rgba(148, 163, 184, 0.18);
            --glass-bg: rgba(255, 255, 255, 0.032);
            --glass-border: rgba(255, 255, 255, 0.075);
            --glass-hover: rgba(255, 255, 255, 0.06);
            --shadow-deep: 0 24px 80px rgba(2, 6, 23, 0.55);
            --shadow-card: 0 8px 32px rgba(2, 6, 23, 0.35);
            --radius-xl: 24px;
            --radius-lg: 18px;
            --radius-md: 14px;
            --radius-sm: 10px;
            --radius-pill: 999px;
        }

        html, body, [class*="css"] {
            color: var(--text-main);
            font-family: "DM Sans", system-ui, sans-serif;
            font-size: 15px;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse 90% 60% at -15% -8%, rgba(56,189,248,0.13) 0%, transparent 55%),
                radial-gradient(ellipse 70% 50% at 115% 5%, rgba(129,140,248,0.11) 0%, transparent 50%),
                radial-gradient(ellipse 50% 35% at 50% 108%, rgba(52,211,153,0.07) 0%, transparent 55%),
                radial-gradient(ellipse 40% 30% at 80% 60%, rgba(56,189,248,0.04) 0%, transparent 40%),
                linear-gradient(168deg, #060a14 0%, #0b1022 50%, #060c18 100%);
            min-height: 100vh;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(4,6,14,0.97) 0%, rgba(8,12,24,0.94) 100%);
            border-right: 1px solid var(--border-subtle);
            backdrop-filter: blur(32px);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        h1, h2, h3, h4 {
            font-family: "Syne", sans-serif;
            color: var(--text-main);
            letter-spacing: -0.03em;
        }

        /* ─── SIDEBAR BRAND ─── */
        .sb-brand {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.25rem;
        }
        .sb-brand-icon {
            width: 34px; height: 34px;
            border-radius: 10px;
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            display: flex; align-items: center; justify-content: center;
            font-size: 1rem; font-weight: 900; color: #fff;
            font-family: "Syne", sans-serif;
            box-shadow: 0 4px 14px rgba(14,165,233,0.35);
            flex-shrink: 0;
        }
        .sb-brand-name {
            font-family: "Syne", sans-serif;
            font-size: 1.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f0f6ff 0%, #7dd3fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* ─── SECTION DIVIDER ─── */
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, var(--border-subtle) 25%, var(--border-medium) 50%, var(--border-subtle) 75%, transparent 100%);
            margin: 1.2rem 0;
        }

        /* ─── EYEBROW LABEL ─── */
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--accent-blue);
            margin-bottom: 0.5rem;
        }
        .eyebrow::before {
            content: '';
            display: inline-block;
            width: 18px; height: 2px;
            background: var(--accent-blue);
            border-radius: 2px;
            opacity: 0.7;
        }

        /* ─── HERO ─── */
        .hero-wrap {
            padding: 2rem 2.2rem 1.8rem;
            margin-bottom: 1.5rem;
            background:
                linear-gradient(135deg, rgba(56,189,248,0.055) 0%, rgba(129,140,248,0.04) 60%, transparent 100%),
                var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-card);
            position: relative;
            overflow: hidden;
        }
        .hero-wrap::before {
            content: '';
            position: absolute;
            top: -1px; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent 5%, rgba(56,189,248,0.45) 40%, rgba(129,140,248,0.35) 65%, transparent 95%);
        }
        .hero-title {
            font-family: "Syne", sans-serif;
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            font-weight: 900;
            margin: 0 0 0.5rem;
            line-height: 1.1;
            background: linear-gradient(140deg, #f0f6ff 0%, #bfdbfe 45%, #7dd3fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.75;
            margin: 0;
            max-width: 780px;
        }
        .hero-divider {
            height: 1px;
            background: var(--border-subtle);
            margin: 1.3rem 0;
        }

        /* ─── STATS GRID ─── */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
            margin-top: 1.2rem;
        }
        .stat-card {
            background: rgba(255,255,255,0.038);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            padding: 1rem 1.2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: border-color 0.2s ease;
        }
        .stat-card:hover { border-color: rgba(56,189,248,0.22); }
        .stat-value {
            font-family: "Syne", sans-serif;
            font-size: 2.1rem;
            font-weight: 900;
            line-height: 1;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-label {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--text-dim);
            margin-top: 0.3rem;
        }

        /* ─── METRIC PILLS ─── */
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }
        .pill {
            padding: 0.28rem 0.85rem;
            border-radius: var(--radius-pill);
            border: 1px solid var(--border-medium);
            background: rgba(255,255,255,0.055);
            color: var(--text-muted);
            font-size: 0.78rem;
            font-weight: 500;
        }
        .pill.green {
            background: var(--accent-emerald-dim);
            border-color: rgba(52,211,153,0.28);
            color: #a7f3d0;
        }
        .pill.blue {
            background: var(--accent-blue-dim);
            border-color: rgba(56,189,248,0.28);
            color: #bae6fd;
        }

        /* ─── GLASS CARD ─── */
        .glass-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            padding: 1.4rem 1.6rem;
            box-shadow: var(--shadow-card);
            color: var(--text-main);
            line-height: 1.75;
            font-size: 0.96rem;
        }

        /* ─── SECTION CARDS (insights / risks) ─── */
        .section-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            padding: 1.4rem 1.5rem;
            height: 100%;
            box-shadow: var(--shadow-card);
        }
        .section-card-title {
            font-family: "Syne", sans-serif;
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .section-card-title span.icon {
            width: 26px; height: 26px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
        }
        .icon-blue { background: var(--accent-blue-dim); }
        .icon-amber { background: rgba(251,191,36,0.14); }

        .bullet-item {
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
            padding: 0.55rem 0;
            border-bottom: 1px solid var(--border-subtle);
            font-size: 0.91rem;
            color: var(--text-muted);
            line-height: 1.6;
        }
        .bullet-item:last-child { border-bottom: none; }
        .bullet-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            flex-shrink: 0;
            margin-top: 0.52rem;
        }
        .dot-blue { background: var(--accent-blue); }
        .dot-amber { background: var(--accent-amber); }

        /* ─── EMPTY STATE ─── */
        .empty-state {
            padding: 2.5rem 2rem;
            border-radius: var(--radius-lg);
            background: var(--glass-bg);
            border: 1.5px dashed var(--border-subtle);
            color: var(--text-dim);
            text-align: center;
            font-size: 0.92rem;
            line-height: 1.7;
        }
        .empty-state-icon {
            font-size: 2.2rem;
            margin-bottom: 0.8rem;
            display: block;
            opacity: 0.6;
        }

        /* ─── CHAT BUBBLES ─── */
        .chat-outer-user {
            display: flex;
            justify-content: flex-end;
            margin: 0.6rem 0;
        }
        .chat-outer-ai {
            display: flex;
            justify-content: flex-start;
            margin: 0.6rem 0;
        }
        .chat-bubble-user {
            max-width: 78%;
            padding: 0.85rem 1.2rem;
            border-radius: 20px 20px 6px 20px;
            background: linear-gradient(135deg, rgba(14,165,233,0.28) 0%, rgba(99,102,241,0.22) 100%);
            border: 1px solid rgba(56,189,248,0.22);
            color: #e0f2fe;
            font-size: 0.94rem;
            line-height: 1.65;
            box-shadow: 0 4px 18px rgba(14,165,233,0.12);
        }
        .chat-bubble-ai {
            max-width: 84%;
            padding: 1rem 1.25rem;
            border-radius: 20px 20px 20px 6px;
            background: rgba(255,255,255,0.042);
            border: 1px solid var(--glass-border);
            color: var(--text-main);
            font-size: 0.94rem;
            line-height: 1.75;
            box-shadow: var(--shadow-card);
        }
        .chat-avatar {
            width: 30px; height: 30px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-top: 0.15rem;
        }
        .avatar-ai {
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            color: #fff;
            box-shadow: 0 2px 10px rgba(14,165,233,0.3);
        }
        .avatar-user {
            background: rgba(255,255,255,0.1);
            border: 1px solid var(--border-medium);
            color: var(--text-muted);
        }
        .chat-turn {
            display: flex;
            gap: 0.65rem;
            align-items: flex-start;
            width: 100%;
        }
        .chat-turn-user {
            flex-direction: row-reverse;
        }

        /* ─── ASSISTANT TEXT ─── */
        .assistant-copy {
            color: var(--text-main);
            line-height: 1.8;
            font-size: 0.94rem;
        }

        /* ─── SOURCE CARDS ─── */
        .source-card {
            padding: 0.9rem 1.1rem;
            border-radius: var(--radius-md);
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--glass-border);
            margin-bottom: 0.6rem;
            transition: border-color 0.15s;
        }
        .source-card:hover { border-color: rgba(56,189,248,0.22); }
        .source-meta {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.45rem;
        }
        .badge-page {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.6rem;
            border-radius: var(--radius-pill);
            background: var(--accent-emerald-dim);
            border: 1px solid rgba(52,211,153,0.25);
            color: #a7f3d0;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .badge-score {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.6rem;
            border-radius: var(--radius-pill);
            background: var(--accent-blue-dim);
            border: 1px solid rgba(56,189,248,0.22);
            color: #bae6fd;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .source-text {
            color: var(--text-muted);
            font-size: 0.88rem;
            line-height: 1.65;
        }
        .source-inline {
            display: inline-block;
            margin: 0 0.06rem;
            padding: 0.05rem 0.42rem;
            border-radius: var(--radius-pill);
            background: var(--accent-blue-dim);
            border: 1px solid rgba(56,189,248,0.28);
            color: #bae6fd;
            font-weight: 700;
            font-size: 0.85em;
        }
        .sources-line {
            margin-top: 0.8rem;
            padding: 0.7rem 1rem;
            border-radius: var(--radius-sm);
            background: rgba(148,163,184,0.06);
            border: 1px solid var(--border-subtle);
            color: #e2e8f0;
            font-size: 0.88rem;
        }
        .sources-line strong { color: var(--accent-blue); }

        /* ─── CONFIDENCE BADGE ─── */
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.28rem 0.8rem;
            border-radius: var(--radius-pill);
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            font-size: 0.76rem;
            font-weight: 600;
            margin-top: 0.55rem;
        }

        /* ─── SUGGESTION BUTTONS ─── */
        .suggestion-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin-bottom: 1.2rem;
        }
        .suggestion-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-dim);
            margin-bottom: 0.65rem;
        }

        /* ─── TABS ─── */
        [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: transparent !important;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 0.6rem;
        }
        [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid transparent;
            border-radius: var(--radius-pill);
            color: var(--text-muted);
            padding: 0.42rem 1.1rem;
            font-family: "DM Sans", sans-serif;
            font-weight: 500;
            font-size: 0.88rem;
            transition: all 0.15s ease;
        }
        [data-baseweb="tab"]:hover {
            background: var(--glass-bg);
            border-color: var(--border-subtle);
            color: var(--text-main);
        }
        [aria-selected="true"][data-baseweb="tab"] {
            color: #e0f2fe;
            background: rgba(56,189,248,0.13);
            border-color: rgba(56,189,248,0.28);
        }
        [data-baseweb="tab-highlight"] { display: none !important; }

        /* ─── BUTTONS ─── */
        div.stButton > button {
            border-radius: var(--radius-pill);
            border: 1px solid rgba(125,211,252,0.18);
            background: linear-gradient(135deg, rgba(14,165,233,0.18) 0%, rgba(99,102,241,0.14) 100%);
            color: #e0f2fe;
            font-weight: 600;
            font-family: "DM Sans", sans-serif;
            font-size: 0.88rem;
            letter-spacing: 0.01em;
            transition: all 0.18s ease;
            padding: 0.4rem 1rem;
        }
        div.stButton > button:hover {
            border-color: rgba(125,211,252,0.38);
            background: linear-gradient(135deg, rgba(14,165,233,0.28) 0%, rgba(99,102,241,0.22) 100%);
            transform: translateY(-1px);
            box-shadow: 0 5px 18px rgba(14,165,233,0.16);
        }
        div.stButton > button:active { transform: translateY(0); }

        /* ─── FILE UPLOADER ─── */
        div[data-testid="stFileUploader"] section {
            background: var(--glass-bg);
            border: 1.5px dashed rgba(56,189,248,0.22);
            border-radius: var(--radius-lg);
            transition: border-color 0.2s;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: rgba(56,189,248,0.45);
        }

        /* ─── ALERTS ─── */
        .stAlert {
            background: rgba(8,12,24,0.85) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
        }

        /* ─── CHAT INPUT ─── */
        [data-testid="stChatInput"] {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid var(--border-medium) !important;
            border-radius: var(--radius-lg) !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(56,189,248,0.35) !important;
            box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
        }

        /* ─── CHAT SECTION HEADER ─── */
        .chat-section-header {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 1rem;
            padding-bottom: 0.85rem;
            border-bottom: 1px solid var(--border-subtle);
        }
        .chat-section-title {
            font-family: "Syne", sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 0;
        }
        .chat-section-badge {
            padding: 0.18rem 0.65rem;
            border-radius: var(--radius-pill);
            background: var(--accent-blue-dim);
            border: 1px solid rgba(56,189,248,0.22);
            color: #bae6fd;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }

        /* ─── MISC ─── */
        p, label, .stCaption, .stMarkdown { color: var(--text-main); }
        .stCaption { color: var(--text-muted) !important; }
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


def _generate_analysis(context: str) -> str:
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
    chunks = st.session_state.get("analysis_chunks", [])
    chunks = chunks[:20]
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
        st.session_state.active_tab = "overview"

        st.toast("Document indexed successfully")
        st.rerun()

    except Exception as exc:
        st.session_state.processing_error = str(exc)
        st.session_state.processing_success = ""
        logger.error("Pipeline failed: %s", exc)
        st.error(f"Processing failed: {exc}")
    finally:
        st.session_state.is_processing = False


def _format_answer_html(text: str) -> str:
    safe_text = html.escape((text or "No answer available.").strip())
    safe_text = re.sub(
        r"(?im)^Sources:\s*(.+)$",
        lambda m: f"<div class='sources-line'><strong>Sources:</strong> {m.group(1)}</div>",
        safe_text,
    )
    safe_text = re.sub(
        r"\(p\.\s*(\d+)\)",
        lambda m: f"<span class='source-inline'>(p. {m.group(1)})</span>",
        safe_text,
    )
    safe_text = re.sub(
        r"\bPage\s+(\d+)\b",
        lambda m: f"<span class='source-inline'>Page {m.group(1)}</span>",
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
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:var(--text-dim);margin:0.9rem 0 0.5rem'>Sources</div>",
        unsafe_allow_html=True,
    )
    for src in sources:
        page = src.get("page_number", "?")
        text = re.sub(r"\s+", " ", str(src.get("text", "")).strip())
        preview = html.escape(text[:220] + ("…" if len(text) > 220 else ""))
        score = src.get("score")
        score_html = ""
        if isinstance(score, (float, int)):
            score_html = f"<span class='badge-score'>Score {float(score):.3f}</span>"
        st.markdown(
            f"""<div class="source-card">
                    <div class="source-meta">
                        <span class="badge-page">📄 Page {page}</span>
                        {score_html}
                    </div>
                    <div class="source-text">{preview}</div>
                </div>""",
            unsafe_allow_html=True,
        )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """<div class="sb-brand">
                <div class="sb-brand-icon">D</div>
                <span class="sb-brand-name">DocuMind</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("AI document intelligence")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        if has_api_key():
            st.success("LLM configured")
        else:
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
            st.markdown("**Current Document**")
            st.markdown(
                f"<span style='color:var(--text-muted);font-size:0.88rem'>"
                f"{html.escape(st.session_state.document_name or metadata.get('file_name', 'Uploaded PDF'))}"
                f"</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"{metadata.get('page_count', 0)} pages · "
                f"{pipeline.get('total_chunks', 0)} chunks · "
                f"{format_file_size(pipeline['file_size'])} · "
                f"{pipeline.get('processing_time', 0)}s"
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
            """<div class="hero-wrap">
                <div class="eyebrow">Document Intelligence</div>
                <h1 class="hero-title">Unlock insights from any PDF</h1>
                <p class="hero-subtitle">
                    Upload a document to generate grounded analysis, search with citations,
                    and chat with your indexed content — all in a single premium workspace.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    metadata = pipeline["metadata"]
    vector_db = pipeline.get("vector_db")
    vector_ready = vector_db is not None
    vector_count = vector_db.total_chunks if vector_ready else 0
    file_name = html.escape(metadata.get("file_name", st.session_state.document_name))

    st.markdown(
        f"""<div class="hero-wrap">
            <div class="eyebrow">Indexed & Ready</div>
            <h1 class="hero-title">{file_name}</h1>
            <p class="hero-subtitle">
                Retrieval, analysis, and grounded chat are live.
                Every answer stays anchored to your indexed document.
            </p>
            <div class="pill-row">
                <span class="pill blue">{format_file_size(pipeline['file_size'])}</span>
                <span class="pill">{pipeline.get('processing_time', 0)}s processing</span>
                <span class="pill {'green' if vector_ready else ''}">
                    Vector DB {'Ready' if vector_ready else 'Missing'}
                </span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{metadata.get('page_count', 0)}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pipeline.get('total_chunks', 0)}</div>
                <div class="stat-label">Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{vector_count}</div>
                <div class="stat-label">Vectors</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_overview_tab() -> None:
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.markdown(
            """<div class="empty-state">
                <span class="empty-state-icon">📄</span>
                Upload a PDF to generate AI-powered insights, summaries, and analysis.
            </div>""",
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

    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;"
        "color:var(--text-main);margin-bottom:0.7rem'>Executive Summary</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='glass-card'>{html.escape(analysis['summary'] or 'Summary unavailable.')}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if analysis["insights"]:
            bullets_html = "".join(
                f"<div class='bullet-item'><span class='bullet-dot dot-blue'></span>"
                f"<span>{html.escape(item)}</span></div>"
                for item in analysis["insights"]
            )
        else:
            bullets_html = "<div style='color:var(--text-dim);font-size:0.9rem'>No insights available yet.</div>"
        st.markdown(
            f"""<div class="section-card">
                <div class="section-card-title">
                    <span class="icon icon-blue">💡</span>Key Insights
                </div>
                {bullets_html}
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        if analysis["risks"]:
            bullets_html = "".join(
                f"<div class='bullet-item'><span class='bullet-dot dot-amber'></span>"
                f"<span>{html.escape(item)}</span></div>"
                for item in analysis["risks"]
            )
        else:
            bullets_html = "<div style='color:var(--text-dim);font-size:0.9rem'>No risk analysis available yet.</div>"
        st.markdown(
            f"""<div class="section-card">
                <div class="section-card-title">
                    <span class="icon icon-amber">⚠️</span>Risks & Caveats
                </div>
                {bullets_html}
            </div>""",
            unsafe_allow_html=True,
        )


def _render_chat_history() -> None:
    if not st.session_state.chat_history:
        return

    for turn in st.session_state.chat_history:
        st.markdown(
            f"""<div class="chat-outer-user">
                <div class="chat-turn chat-turn-user">
                    <div class="chat-bubble-user">{html.escape(turn["question"])}</div>
                    <div class="chat-avatar avatar-user">You</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        sources_count = len(turn.get("sources", []))
        src_label = f"{sources_count} source{'s' if sources_count != 1 else ''}" if sources_count else ""

        answer_html = _format_answer_html(turn["answer"])

        col_ai, col_spacer = st.columns([0.85, 0.15])
        with col_ai:
            st.markdown(
                f"""<div class="chat-turn">
                    <div class="chat-avatar avatar-ai">DM</div>
                    <div style="flex:1">
                        <div class="chat-bubble-ai">{answer_html}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            if src_label:
                with st.expander(f"View {src_label}", expanded=False):
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
    cleaned = prompt.strip()
    if not cleaned:
        return

    st.markdown(
        f"""<div class="chat-outer-user">
            <div class="chat-turn chat-turn-user">
                <div class="chat-bubble-user">{html.escape(cleaned)}</div>
                <div class="chat-avatar avatar-user">You</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    col_ai, _ = st.columns([0.85, 0.15])
    with col_ai:
        st.markdown(
            """<div class="chat-turn">
                <div class="chat-avatar avatar-ai">DM</div>
                <div style="flex:1">""",
            unsafe_allow_html=True,
        )
        with st.spinner("Generating answer…"):
            result = _answer_question(cleaned, force_refresh=force_refresh)

        answer = result.get("answer") or "No answer available."
        sources = result.get("sources", [])

        st.markdown('<div class="chat-bubble-ai">', unsafe_allow_html=True)
        stream_text(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        if sources:
            valid_scores = [s.get("score") for s in sources if isinstance(s.get("score"), (int, float))]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                conf_pct = min(int(avg_score * 100), 100)
                color = "#34d399" if conf_pct >= 70 else "#fbbf24" if conf_pct >= 45 else "#f87171"
                st.markdown(
                    f"<div class='confidence-badge' style='color:{color};border-color:{color}30'>"
                    f"● Confidence {conf_pct}%</div>",
                    unsafe_allow_html=True,
                )

        src_count = len(sources)
        if src_count:
            with st.expander(f"View {src_count} source{'s' if src_count != 1 else ''}", expanded=False):
                _render_sources(sources)

        regen_key = f"regen_{len(st.session_state.chat_history)}"
        if st.button("↻ Regenerate", key=regen_key):
            if st.session_state.chat_history:
                st.session_state.chat_history.pop()
            st.session_state.pending_prompt = cleaned
            st.session_state["_regen_force"] = True
            st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)

    _append_chat_turn(cleaned, answer, sources)


def _render_suggestions_in_chat(questions: list[str]) -> None:
    if not questions:
        return
    st.markdown(
        "<div class='suggestion-label'>Suggested Questions</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for idx, question in enumerate(questions):
        with cols[idx % 2]:
            if st.button(
                question,
                key=f"sugg_chat::{question}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = question
                st.session_state.active_tab = "qa"
                st.rerun()
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


def _render_chat_tab() -> None:
    pipeline = st.session_state.get("pipeline")

    st.markdown(
        f"""<div class="chat-section-header">
            <span class="chat-section-title">Ask Your Document</span>
            {'<span class="chat-section-badge">LIVE</span>' if pipeline and has_api_key() else ''}
        </div>""",
        unsafe_allow_html=True,
    )

    clear_col, info_col = st.columns([0.22, 0.78])
    with clear_col:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.pending_prompt = ""
            st.rerun()
    with info_col:
        if pipeline is None:
            st.caption("Upload a document to enable chat.")
        elif not has_api_key():
            st.caption("Add `GROQ_API_KEY` to Streamlit secrets to enable AI Q&A.")
        else:
            st.caption("Streaming · Citation-backed · Grounded in your document")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if pipeline is not None and has_api_key():
        _ensure_analysis()
        analysis = st.session_state.analysis_cache or {}
        questions = analysis.get("questions", [])
        if questions and not st.session_state.chat_history:
            _render_suggestions_in_chat(questions)

    if not st.session_state.chat_history:
        if pipeline is None:
            st.markdown(
                """<div class="empty-state">
                    <span class="empty-state-icon">💬</span>
                    Upload a PDF to start asking questions.
                    Answers stream live with page-level citations.
                </div>""",
                unsafe_allow_html=True,
            )
        elif not has_api_key():
            st.markdown(
                """<div class="empty-state">
                    <span class="empty-state-icon">🔑</span>
                    Add your <code>GROQ_API_KEY</code> to enable AI-powered chat.
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        _render_chat_history()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    prompt = st.chat_input(
        "Ask anything about your document…",
        disabled=pipeline is None or not has_api_key(),
    )

    force_refresh = False
    if st.session_state.get("_regen_force"):
        force_refresh = True
        st.session_state["_regen_force"] = False

    incoming_prompt = None
    if st.session_state.pending_prompt:
        incoming_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
    elif prompt:
        incoming_prompt = prompt.strip()

    if incoming_prompt and pipeline is not None and has_api_key():
        _handle_prompt(incoming_prompt, force_refresh=force_refresh)


# ─── Bootstrap ───────────────────────────────────────────────────────────────

_init_state()
_hydrate_pipeline_from_session()
_inject_styles()
_render_sidebar()

st.markdown("<div class='eyebrow'>DocuMind</div>", unsafe_allow_html=True)
_render_hero()

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
    if st.session_state.get("is_processing", False):
        st.info("Processing document… please wait.")
    else:
        st.markdown(
            """<div class="empty-state">
                Upload a PDF in the sidebar to begin analysis.
            </div>""",
            unsafe_allow_html=True,
        )
    st.stop()

if not has_api_key():
    st.info(
        "Add `GROQ_API_KEY` in Streamlit secrets to enable AI analysis and Q&A. "
        "Document indexing is already active."
    )

active = st.session_state.get("active_tab", "overview")
tab_index = 1 if active == "qa" else 0

overview_tab, qa_tab = st.tabs(["Overview", " Ask Questions"])

if tab_index == 1:
    st.session_state.active_tab = "overview"

with overview_tab:
    _render_overview_tab()

with qa_tab:
    _render_chat_tab()