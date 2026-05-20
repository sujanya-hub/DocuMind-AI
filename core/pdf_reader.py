from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required: pip install pymupdf"
    ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str, max_pages: int | None = None) -> List[Dict[str, Any]]:
    """
    Extract text from every page of a PDF file.

    Returns:
        List of page dicts, each containing:
            page_number (int) - 1-based page index
            text        (str) - cleaned extracted text
            char_count  (int) - character length of text

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If the file cannot be opened as a PDF.
    """
    _assert_file_exists(pdf_path)
   
    content_hash = _file_sha256(pdf_path)
    payload = _read_pdf_payload(pdf_path, content_hash, max_pages=max_pages)
    pages = list(payload["pages"])
    return pages


def get_document_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Return high-level metadata for a PDF document.

    Returns:
        Dict containing:
            file_name  (str)
            page_count (int)
            title      (str)
            author     (str)
            subject    (str)

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If the file cannot be opened as a PDF.
    """
    _assert_file_exists(pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{pdf_path}': {exc}") from exc

    with doc:
        meta = doc.metadata or {}
        return {
            "file_name": os.path.basename(pdf_path),
            "page_count": doc.page_count,
            "title": meta.get("title", "Unknown") or "Unknown",
            "author": meta.get("author", "Unknown") or "Unknown",
            "subject": meta.get("subject", "") or "",
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_file_exists(path: str) -> None:
    """Raise FileNotFoundError if the given path does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")


def _file_sha256(path: str) -> str:
    """
    Return a hex SHA-256 digest of the file at path.

    Used as the lru_cache key instead of mtime so that two different files
    saved to the same path (same-filename re-upload) always get independent
    cache entries regardless of filesystem timestamp resolution.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_text(text: str) -> str:
    """
    Normalise whitespace while preserving paragraph structure.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

_cache: Dict[tuple[str, int | None], Dict[str, Any]] = {}


def _read_pdf_payload(
    pdf_path: str,
    content_hash: str,
    max_pages: int | None = None,
) -> Dict[str, Any]:
    """
    Read the PDF once and cache both pages and metadata keyed on content hash.

    Using a plain dict cache instead of @lru_cache avoids the constraint that
    all arguments must be hashable AND ensures the cache key is truly the file
    content, not the path or mtime.
    """
    cache_key = (content_hash, max_pages)

    if cache_key in _cache:
        return _cache[cache_key]

    pages: List[Dict[str, Any]] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{pdf_path}': {exc}") from exc

    with doc:
        for i, page in enumerate(doc, start=1):
            if max_pages is not None and i > max_pages:
                break
            cleaned = _clean_text(page.get_text("text") or "")
            pages.append(
                {
                    "page_number": i,
                    "text": cleaned,
                    "char_count": len(cleaned),
                }
            )

        meta = doc.metadata or {}
        metadata = {
            "file_name": os.path.basename(pdf_path),
            "page_count": doc.page_count,
            "title": meta.get("title", "Unknown") or "Unknown",
            "author": meta.get("author", "Unknown") or "Unknown",
            "subject": meta.get("subject", "") or "",
        }

    payload = {"pages": pages, "metadata": metadata}


    if len(_cache) >= 16:
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]

    _cache[cache_key] = payload
    return payload
