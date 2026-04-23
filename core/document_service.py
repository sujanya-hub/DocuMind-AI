"""
document_service.py - End-to-end document ingestion pipeline orchestrator.

Provides a single public function, run_pipeline, that accepts an uploaded
file object and executes every ingestion step in sequence: save, extract,
chunk, embed, index, and instantiate the AI engines. Returns a structured
dict consumed by session_manager.store_pipeline_result.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import streamlit as st

from core.chunker import chunk_pages
from core.config import UPLOAD_DIR
from core.pdf_reader import extract_pages, get_document_metadata
from core.qa_engine import QAEngine
from core.summarizer import Summarizer
from core.vectordb import VectorDB
from utils.helpers import ensure_directory, file_sha256, save_uploaded_file

logger = logging.getLogger(__name__)

_MIN_PAGE_TEXT_LENGTH = 10
_MIN_CHUNK_COUNT = 1


def _sanitise_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and sanitize extracted page dicts."""
    sanitised: list[dict[str, Any]] = []
    for i, page in enumerate(pages):
        if not isinstance(page, dict):
            logger.warning("Page %d is not a dict (%s); skipping.", i + 1, type(page).__name__)
            continue

        raw_text = page.get("text")
        if not isinstance(raw_text, str):
            logger.warning(
                "Page %d has non-string text (%s); coercing to empty string.",
                i + 1,
                type(raw_text).__name__,
            )
            raw_text = ""

        clean_text = raw_text.strip()
        if len(clean_text) < _MIN_PAGE_TEXT_LENGTH:
            logger.debug("Page %d contains no usable text (len=%d).", i + 1, len(clean_text))

        sanitised.append({**page, "text": clean_text})

    return sanitised


def _has_extractable_text(pages: list[dict[str, Any]]) -> bool:
    """Return True if at least one page contains text above the noise threshold."""
    return any(len(page.get("text", "")) >= _MIN_PAGE_TEXT_LENGTH for page in pages)


def _log_pipeline_summary(
    pages: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    processing_time: float,
) -> None:
    """Emit a structured INFO summary after a successful pipeline run."""
    text_pages = sum(1 for page in pages if len(page.get("text", "")) >= _MIN_PAGE_TEXT_LENGTH)
    empty_pages = len(pages) - text_pages
    logger.info(
        "[DocumentService] Pipeline complete | pages=%d (text=%d, empty=%d) | chunks=%d | elapsed=%.3fs",
        len(pages),
        text_pages,
        empty_pages,
        len(chunks),
        processing_time,
    )


def run_pipeline(
    uploaded_file: Any,
    upload_dir: str = UPLOAD_DIR,
) -> Dict[str, Any]:
    """Execute the full document ingestion pipeline for an uploaded PDF."""
    ensure_directory(upload_dir)
    pipeline_start = time.perf_counter()
    document_hash = file_sha256(uploaded_file)

    with st.status("Processing document...", expanded=True) as status:
        st.write("Saving uploaded file...")
        try:
            pdf_path = save_uploaded_file(uploaded_file, upload_dir)
            file_size = uploaded_file.size
        except Exception as exc:
            raise IOError(f"Failed to save uploaded file: {exc}") from exc

        st.write("Extracting text from pages...")
        try:
            raw_pages = extract_pages(pdf_path)
            metadata = get_document_metadata(pdf_path)
        except Exception as exc:
            raise RuntimeError(f"Page extraction failed: {exc}") from exc

        st.write("Validating and sanitising pages...")
        pages = _sanitise_pages(raw_pages or [])

        if not pages:
            raise ValueError(
                "No pages could be extracted from this document. "
                "The file may be corrupted or password-protected."
            )

        if not _has_extractable_text(pages):
            raise ValueError(
                "No extractable text was found in this document. "
                "The file may be image-only or scanned. OCR is not supported in this version."
            )

        st.write("Chunking text with metadata...")
        try:
            chunks = chunk_pages(pages, source_name=uploaded_file.name)
        except Exception as exc:
            raise RuntimeError(f"Chunking failed: {exc}") from exc

        if not chunks:
            raise ValueError(
                "No chunks were produced from this document. "
                "The extracted text may be too short or malformed."
            )

        total_chunks = len(chunks)

        st.write("Generating embeddings and building vector index...")
        try:
            vector_db = VectorDB()
            vector_db.add_chunks(chunks)
        except RuntimeError as exc:
            if "Embedding unavailable" in str(exc):
                raise RuntimeError(
                    "Embedding unavailable. The lightweight embedding model could not be loaded on this instance."
                ) from exc
            raise RuntimeError(
                f"Embedding or vector index construction failed: {type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Embedding or vector index construction failed: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            index_size = vector_db.index.ntotal if hasattr(vector_db.index, "ntotal") else None
        except Exception:
            index_size = None

        if index_size is not None and index_size < _MIN_CHUNK_COUNT:
            raise ValueError(
                f"Vector index is empty after embedding ({index_size} vectors). "
                "Embedding may have failed silently. Check your embedding model."
            )

        st.write("Initialising AI engines...")
        try:
            qa_engine = QAEngine(vector_db=vector_db)
            summarizer = Summarizer()
        except Exception as exc:
            raise RuntimeError(f"AI engine initialisation failed: {exc}") from exc

        processing_time = round(time.perf_counter() - pipeline_start, 3)
        _log_pipeline_summary(pages, chunks, processing_time)

        status.update(
            label=f"Document indexed successfully. ({total_chunks} chunks | {processing_time}s)",
            state="complete",
            expanded=False,
        )

    return {
        "pdf_path": pdf_path,
        "document_hash": document_hash,
        "metadata": metadata,
        "file_size": file_size,
        "pages": pages,
        "chunks": chunks,
        "vector_db": vector_db,
        "qa_engine": qa_engine,
        "summarizer": summarizer,
        "processing_time": processing_time,
        "total_chunks": total_chunks,
    }
