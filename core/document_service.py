from __future__ import annotations

import logging
import time
from typing import Any, Dict

from core.chunker import chunk_pages
from core.config import UPLOAD_DIR
from core.pdf_reader import extract_pages, get_document_metadata
from core.qa_engine import QAEngine
from core.vectordb import VectorDB
from utils.helpers import ensure_directory, file_sha256, save_uploaded_file

logger = logging.getLogger(__name__)

_MIN_PAGE_TEXT_LENGTH = 10
_MIN_CHUNK_COUNT = 1
MAX_CHUNKS = 100
MAX_PAGES = 20


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
    start_time = time.time()

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    document_hash = file_sha256(uploaded_file)

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    try:
        pdf_path = save_uploaded_file(uploaded_file, upload_dir)
        file_size = uploaded_file.size
    except Exception as exc:
        raise IOError(f"Failed to save uploaded file: {exc}") from exc

    try:
        raw_pages = extract_pages(pdf_path, max_pages=MAX_PAGES)
        metadata = get_document_metadata(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Page extraction failed: {exc}") from exc

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

    try:
        chunks = chunk_pages(
            pages,
            source_name=uploaded_file.name,
            max_chunks=MAX_CHUNKS,
        )
    except Exception as exc:
        raise RuntimeError(f"Chunking failed: {exc}") from exc

    if not chunks:
        raise ValueError("No text extracted from PDF (possibly scanned PDF)")

    total_chunks = min(len(chunks), MAX_CHUNKS)
    chunks = chunks[:total_chunks]

    vector_db = VectorDB()

    try:
        vector_db.add_chunks(chunks)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc

    if vector_db.total_chunks < _MIN_CHUNK_COUNT:
        raise ValueError(
            f"Vector index is empty after embedding ({vector_db.total_chunks} vectors). "
            "Embedding may have failed silently."
        )

    try:
        qa_engine = QAEngine(vector_db=vector_db)
    except Exception as exc:
        raise RuntimeError(f"AI engine initialisation failed: {exc}") from exc

    processing_time = round(time.time() - start_time, 2)
    _log_pipeline_summary(pages, chunks, processing_time)

    return {
        "pdf_path": pdf_path,
        "document_hash": document_hash,
        "metadata": metadata,
        "file_size": file_size,
        "pages": pages,
        "chunks": chunks,
        "vector_db": vector_db,
        "qa_engine": qa_engine,
        "processing_time": processing_time,
        "total_chunks": total_chunks,
    }
