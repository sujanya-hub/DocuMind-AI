"""
document_service.py - End-to-end document ingestion pipeline orchestrator.

FIX SUMMARY (memory crash fixes for 512MB Render free tier):
  1. embeddings are no longer stored in the pipeline result dict.
     Storing them in st.session_state caused Streamlit to serialize the
     entire numpy matrix on every rerun, adding a ~1–2 MB pickle spike
     on top of already-high post-embedding memory usage.
  2. Chunk cap (MAX_CHUNKS) is applied BEFORE chunk_pages processes the
     full page list. Previously chunk_pages() generated all chunks first
     (potentially thousands), then the cap was applied — wasting RAM on
     dicts that were immediately discarded.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

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
MAX_CHUNKS = 40


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
    print("PIPELINE START")
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
        print("FILE SAVED:", pdf_path)
    except Exception as exc:
        raise IOError(f"Failed to save uploaded file: {exc}") from exc

    # ── STEP 1: PDF READ ──────────────────────────────────────────────────────
    print("STEP 1: PDF READ")
    try:
        raw_pages = extract_pages(pdf_path)
        metadata = get_document_metadata(pdf_path)
        print("PDF METADATA:", metadata)
    except Exception as exc:
        raise RuntimeError(f"Page extraction failed: {exc}") from exc

    pages = _sanitise_pages(raw_pages or [])
    print("PAGES:", len(pages))

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

    # ── STEP 2: CHUNKING ──────────────────────────────────────────────────────
    # FIX: Cap pages BEFORE chunking so chunk_pages() never generates more
    # than ~MAX_CHUNKS worth of text in the first place. Previously the cap
    # was applied after chunk_pages() returned the full list — all intermediate
    # chunk dicts were allocated then immediately discarded.
    #
    # With CHUNK_SIZE=500 and a typical page ~2000 chars: ~4 chunks/page.
    # MAX_CHUNKS=40 → cap pages at 10 to stay safely under limit.
    max_pages_to_chunk = max(1, MAX_CHUNKS // 4)
    pages_to_chunk = pages[:max_pages_to_chunk]
    if len(pages) > max_pages_to_chunk:
        print(f"Page cap applied: {len(pages)} pages → chunking first {max_pages_to_chunk}")

    print("STEP 2: CHUNKING")
    try:
        chunks = chunk_pages(pages_to_chunk, source_name=uploaded_file.name)
    except Exception as exc:
        raise RuntimeError(f"Chunking failed: {exc}") from exc

    print("CHUNKS:", len(chunks))

    if not chunks:
        raise ValueError("No text extracted from PDF (possibly scanned PDF)")

    if len(chunks) > MAX_CHUNKS:
        print(f"Limiting chunks: {len(chunks)} → {MAX_CHUNKS}")
        chunks = chunks[:MAX_CHUNKS]

    print("FINAL CHUNKS USED:", len(chunks))

    total_chunks = len(chunks)

    # ── STEP 3: EMBEDDING + VECTOR DB ─────────────────────────────────────────
    print("STEP 3: EMBEDDING START")
    vector_db = VectorDB()

    try:
        # FIX: add_chunks() now returns None — embeddings are discarded after
        # FAISS ingestion and NOT stored anywhere in the pipeline result.
        vector_db.add_chunks(chunks)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc

    print("VECTOR_DB:", vector_db is not None)

    # ── STEP 4: VECTOR DB VALIDATION ──────────────────────────────────────────
    print("STEP 4: VECTOR DB CREATED")
    print("Total vectors:", vector_db.total_chunks)

    if vector_db.total_chunks < _MIN_CHUNK_COUNT:
        raise ValueError(
            f"Vector index is empty after embedding ({vector_db.total_chunks} vectors). "
            "Embedding may have failed silently."
        )

    # ── STEP 5: QA ENGINE ─────────────────────────────────────────────────────
    print("STEP 5: QA ENGINE READY")
    try:
        qa_engine = QAEngine(vector_db=vector_db)
        summarizer = Summarizer()
        print("QA ENGINE READY:", hasattr(qa_engine, "answer"))
    except Exception as exc:
        raise RuntimeError(f"AI engine initialisation failed: {exc}") from exc

    if qa_engine is None:
        raise ValueError("qa_engine not created")

    processing_time = round(time.time() - start_time, 2)
    _log_pipeline_summary(pages, chunks, processing_time)

    # FIX: "embeddings" key is intentionally EXCLUDED from the result dict.
    # Storing the numpy embedding matrix here meant it flowed into
    # st.session_state.pipeline["embeddings"] and got pickled by Streamlit
    # on every rerun — a ~1–2 MB serialization spike at the worst possible
    # moment (right after the memory-intensive embedding pass).
    result = {
        "pdf_path":        pdf_path,
        "document_hash":   document_hash,
        "metadata":        metadata,
        "file_size":       file_size,
        "pages":           pages,
        "chunks":          chunks,
        # "embeddings":   intentionally omitted — see note above
        "vector_db":       vector_db,
        "qa_engine":       qa_engine,
        "summarizer":      summarizer,
        "processing_time": processing_time,
        "total_chunks":    total_chunks,
    }
    print("PIPELINE COMPLETE")
    return result