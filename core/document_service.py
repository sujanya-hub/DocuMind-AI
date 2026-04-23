"""
document_service.py - End-to-end document ingestion pipeline orchestrator.

Provides a single public function, run_pipeline, that accepts an uploaded
file object and executes every ingestion step in sequence: save, extract,
chunk, embed, index, and instantiate the AI engines. Returns a structured
dict consumed by session_manager.store_pipeline_result.

Production enhancements (v2):
  - Full processing-time tracking (total + per-step)
  - Chunk count and page validity logging
  - Defensive page validation with per-page text sanitisation
  - Safe empty-text handling (filters noise before chunking)
  - Guard ensuring VectorDB is never left in an empty state
  - try/except isolation around the embedding step with structured error
  - Additional return keys: processing_time, total_chunks
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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_PAGE_TEXT_LENGTH = 10   # characters; pages below this are treated as empty
_MIN_CHUNK_COUNT      = 1    # VectorDB must contain at least this many chunks


def _sanitise_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate and sanitise extracted page dicts.

    - Ensures every page dict has a 'text' key (str).
    - Strips surrounding whitespace from text.
    - Replaces None / non-string text values with an empty string.
    - Logs a warning for pages that yield no usable text.

    Returns the sanitised list (same order, same length).
    """
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
    return any(len(p.get("text", "")) >= _MIN_PAGE_TEXT_LENGTH for p in pages)


def _log_pipeline_summary(
    pages:           list[dict[str, Any]],
    chunks:          list[dict[str, Any]],
    processing_time: float,
) -> None:
    """Emit a structured INFO summary after a successful pipeline run."""
    text_pages  = sum(1 for p in pages if len(p.get("text", "")) >= _MIN_PAGE_TEXT_LENGTH)
    empty_pages = len(pages) - text_pages
    logger.info(
        "[DocumentService] Pipeline complete | pages=%d (text=%d, empty=%d) "
        "| chunks=%d | elapsed=%.3fs",
        len(pages),
        text_pages,
        empty_pages,
        len(chunks),
        processing_time,
    )


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    uploaded_file: Any,
    upload_dir: str = UPLOAD_DIR,
) -> Dict[str, Any]:
    """
    Execute the full document ingestion pipeline for an uploaded PDF.

    Steps:
        1. Save the uploaded file to disk.
        2. Extract per-page text via PyMuPDF.
        3. Retrieve document-level metadata.
        4. Validate and sanitise extracted pages.
        5. Chunk pages with overlapping sliding windows.
        6. Build a FAISS vector index from chunk embeddings.
        7. Validate that the index is non-empty.
        8. Instantiate QAEngine and Summarizer with the new index.

    Progress is streamed to the Streamlit UI using st.status so the user
    receives live feedback during the embedding step.

    Args:
        uploaded_file: File object from st.file_uploader().
        upload_dir:    Directory where the PDF will be persisted.

    Returns:
        Dict containing:
            pdf_path         (str)         - absolute path to the saved PDF
            metadata         (dict)        - document metadata
            file_size        (int)         - raw byte count
            pages            (list[dict])  - sanitised extracted page dicts
            chunks           (list[dict])  - generated chunk dicts
            vector_db        (VectorDB)    - populated FAISS index
            qa_engine        (QAEngine)    - ready-to-query QA engine
            summarizer       (Summarizer)  - ready-to-use summarizer
            processing_time  (float)       - total wall-clock seconds
            total_chunks     (int)         - number of chunks indexed

    Raises:
        IOError:      If the file cannot be saved to disk.
        ValueError:   If the PDF contains no extractable text, produces no
                      chunks, or the vector index ends up empty.
        RuntimeError: If embedding or index construction fails.
    """
    ensure_directory(upload_dir)
    pipeline_start = time.perf_counter()
    document_hash = file_sha256(uploaded_file)

    with st.status("Processing document...", expanded=True) as status:

        # ── Step 1: Save ──────────────────────────────────────────────
        st.write("Saving uploaded file...")
        _t = time.perf_counter()
        try:
            pdf_path  = save_uploaded_file(uploaded_file, upload_dir)
            file_size = uploaded_file.size
        except Exception as exc:
            raise IOError(f"Failed to save uploaded file: {exc}") from exc
        logger.debug("[DocumentService] Save step: %.3fs", time.perf_counter() - _t)

        # ── Step 2: Extract ───────────────────────────────────────────
        st.write("Extracting text from pages...")
        _t = time.perf_counter()
        try:
            raw_pages = extract_pages(pdf_path)
            metadata  = get_document_metadata(pdf_path)
        except Exception as exc:
            raise RuntimeError(f"Page extraction failed: {exc}") from exc
        logger.debug(
            "[DocumentService] Extraction step: %.3fs | raw pages=%d",
            time.perf_counter() - _t,
            len(raw_pages) if raw_pages else 0,
        )

        # ── Step 3: Validate & sanitise pages ─────────────────────────
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
                "The file may be image-only or scanned. "
                "OCR is not supported in this version."
            )

        text_page_count = sum(
            1 for p in pages if len(p.get("text", "")) >= _MIN_PAGE_TEXT_LENGTH
        )
        logger.info(
            "[DocumentService] Pages validated: total=%d, with-text=%d, empty=%d",
            len(pages),
            text_page_count,
            len(pages) - text_page_count,
        )

        # ── Step 4: Chunk ─────────────────────────────────────────────
        st.write("Chunking text with metadata...")
        _t = time.perf_counter()
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
        logger.info(
            "[DocumentService] Chunking step: %.3fs | chunks=%d",
            time.perf_counter() - _t,
            total_chunks,
        )

        # ── Step 5: Embed & index ─────────────────────────────────────
        st.write("Generating embeddings and building vector index...")
        _t = time.perf_counter()
        try:
            vector_db = VectorDB()
            vector_db.add_chunks(chunks)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding or vector index construction failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        embed_elapsed = time.perf_counter() - _t
        logger.info(
            "[DocumentService] Embedding step: %.3fs | chunks_indexed=%d",
            embed_elapsed,
            total_chunks,
        )

        # ── Step 6: Guard — ensure VectorDB is non-empty ─────────────
        try:
            index_size = (
                vector_db.index.ntotal          # FAISS native attr
                if hasattr(vector_db, "index") and hasattr(vector_db.index, "ntotal")
                else None
            )
        except Exception:
            index_size = None

        if index_size is not None and index_size < _MIN_CHUNK_COUNT:
            raise ValueError(
                f"Vector index is empty after embedding ({index_size} vectors). "
                "Embedding may have failed silently. Check your embedding model."
            )

        if index_size is not None:
            logger.info("[DocumentService] FAISS index size verified: %d vector(s).", index_size)

        # ── Step 7: Initialise AI engines ─────────────────────────────
        st.write("Initialising AI engines...")
        _t = time.perf_counter()
        try:
            # These objects stay lightweight on init and only require an API
            # key when the app actually performs an LLM call.
            qa_engine = QAEngine(vector_db=vector_db)
            summarizer = Summarizer()
        except Exception as exc:
            raise RuntimeError(f"AI engine initialisation failed: {exc}") from exc
        logger.debug("[DocumentService] Engine init step: %.3fs", time.perf_counter() - _t)

        # ── Finalise ──────────────────────────────────────────────────
        processing_time = round(time.perf_counter() - pipeline_start, 3)
        _log_pipeline_summary(pages, chunks, processing_time)

        status.update(
            label=f"Document indexed successfully. ({total_chunks} chunks · {processing_time}s)",
            state="complete",
            expanded=False,
        )

    return {
        "pdf_path":        pdf_path,
        "document_hash":   document_hash,
        "metadata":        metadata,
        "file_size":       file_size,
        "pages":           pages,
        "chunks":          chunks,
        "vector_db":       vector_db,
        "qa_engine":       qa_engine,
        "summarizer":      summarizer,
        "processing_time": processing_time,
        "total_chunks":    total_chunks,
    }
