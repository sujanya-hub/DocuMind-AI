from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import streamlit as st
except Exception:
    st = None


@dataclass(frozen=True)
class AppConfig:
    """Typed application settings with safe defaults."""

    groq_api_key: str
    groq_model_name: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    top_k_results: int
    upload_dir: str
    max_tokens: int
    temperature: float
    analysis_context_chars: int
    analysis_max_tokens: int
    max_pages: int
    max_chunks: int
    reasoning_effort: str


def _as_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_str(key: str, default: str) -> str:
    return str(os.environ.get(key, default)).strip()


def _get_secret(key: str, default: str = "") -> str:
    """Read a Streamlit secret when available, otherwise fall back safely."""
    if st is not None:
        try:
            value = st.secrets.get(key)
            if value is not None:
                return str(value).strip()
        except Exception:
            pass
    return _as_str(key, default)


def _debug_env_status(config: AppConfig) -> None:
    """Optionally log which environment-driven settings are present."""
    if _as_str("DOCUMIND_DEBUG_CONFIG", "").lower() not in {"1", "true", "yes", "on"}:
        return

    logger.info(
        "DocuMind config loaded from environment | GROQ_API_KEY=%s | "
        "GROQ_MODEL_NAME=%s | CHUNK_SIZE=%s | CHUNK_OVERLAP=%s | "
        "EMBEDDING_MODEL=%s | EMBEDDING_BATCH_SIZE=%s | TOP_K_RESULTS=%s | UPLOAD_DIR=%s",
        "set" if bool(config.groq_api_key) else "missing",
        "set" if bool(config.groq_model_name) else "missing",
        config.chunk_size,
        config.chunk_overlap,
        config.embedding_model,
        config.embedding_batch_size,
        config.top_k_results,
        config.upload_dir,
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Build and cache the runtime settings from environment variables.

    Caching keeps repeated reruns cheap while centralizing config logic.
    Never raises at import time; missing values fall back to safe defaults.
    """
    chunk_size = max(1, _as_int("CHUNK_SIZE", 800))
    chunk_overlap = _as_int("CHUNK_OVERLAP", 100)

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    config = AppConfig(
        groq_api_key=_get_secret("GROQ_API_KEY", ""),
        groq_model_name=_as_str("GROQ_MODEL_NAME", "openai/gpt-oss-20b"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=_as_str("EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2"),
        embedding_dim=max(1, _as_int("EMBEDDING_DIM", 384)),
        # Was 8. Larger batches make embedding much faster on CPU.
        embedding_batch_size=max(1, _as_int("EMBEDDING_BATCH_SIZE", 32)),
        top_k_results=max(1, _as_int("TOP_K_RESULTS", 6)),
        upload_dir=_as_str("UPLOAD_DIR", "uploads"),
        max_tokens=max(128, _as_int("MAX_TOKENS", 1024)),
        temperature=max(0.0, min(2.0, _as_float("TEMPERATURE", 0.2))),
        analysis_context_chars=max(4000, _as_int("ANALYSIS_CONTEXT_CHARS", 12000)),
        # Reasoning models spend part of this budget on hidden reasoning,
        # so it must be well above the visible answer length.
        analysis_max_tokens=max(512, _as_int("ANALYSIS_MAX_TOKENS", 2500)),
        # Indexing limits. The old hard-coded values (20 pages / 100 chunks)
        # silently dropped the end of longer documents.
        max_pages=max(1, _as_int("MAX_PAGES", 60)),
        max_chunks=max(1, _as_int("MAX_CHUNKS", 400)),
        # "low", "medium" or "high" for gpt-oss models; empty string disables.
        reasoning_effort=_as_str("REASONING_EFFORT", "low").lower(),
    )
    _debug_env_status(config)
    return config


def get_api_key() -> str:
    """Return the configured Groq API key, or an empty string when absent."""
    return get_config().groq_api_key


def has_api_key() -> bool:
    """Return True when a usable Groq API key is configured."""
    return bool(get_api_key())


def validate_config() -> None:
    """
    Raise a clear runtime error when the LLM API key is missing.

    Called lazily by LLM-facing code so document parsing and indexing
    can still succeed in environments that are not fully configured.
    """
    if has_api_key():
        return

    raise EnvironmentError(
        "Missing GROQ_API_KEY. Add it to Streamlit secrets "
        "(.streamlit/secrets.toml locally or the Community Cloud secrets panel)."
    )


_CONFIG = get_config()

GROQ_MODEL_NAME: str = _CONFIG.groq_model_name
CHUNK_SIZE: int = _CONFIG.chunk_size
CHUNK_OVERLAP: int = _CONFIG.chunk_overlap
EMBEDDING_MODEL: str = _CONFIG.embedding_model
EMBEDDING_DIM: int = _CONFIG.embedding_dim
EMBEDDING_BATCH_SIZE: int = _CONFIG.embedding_batch_size
TOP_K_RESULTS: int = _CONFIG.top_k_results
UPLOAD_DIR: str = _CONFIG.upload_dir
MAX_TOKENS: int = _CONFIG.max_tokens
TEMPERATURE: float = _CONFIG.temperature
ANALYSIS_CONTEXT_CHARS: int = _CONFIG.analysis_context_chars
ANALYSIS_MAX_TOKENS: int = _CONFIG.analysis_max_tokens
MAX_PAGES: int = _CONFIG.max_pages
MAX_CHUNKS: int = _CONFIG.max_chunks
REASONING_EFFORT: str = _CONFIG.reasoning_effort