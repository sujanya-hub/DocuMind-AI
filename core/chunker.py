"""
chunker.py - Sliding-window text chunker with page-level metadata.

Splits extracted PDF pages into overlapping character-level chunks.
Every chunk carries source file name and page number so citation links
survive through retrieval to the UI layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.config import CHUNK_OVERLAP, CHUNK_SIZE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: List[Dict[str, Any]],
    source_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_chunks: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Convert a list of extracted pages into overlapping text chunks.

    Args:
        pages:         Output of ``pdf_reader.extract_pages()``.
        source_name:   Label stored in each chunk, typically the file name.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        List of chunk dicts, each containing:
            chunk_id    (int) - global 0-based index
            text        (str) - chunk content
            page_number (int) - originating page, 1-based
            source      (str) - file name or identifier
            char_start  (int) - start offset within the page text
            char_end    (int) - end offset within the page text

    Raises:
        ValueError: If chunk_size is not strictly greater than chunk_overlap.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be strictly greater than "
            f"chunk_overlap ({chunk_overlap})."
        )

    all_chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for page in pages:
        page_text: str   = page.get("text", "").strip()
        page_number: int = page["page_number"]

        if not page_text:
            continue

        for text, char_start, char_end in _sliding_window(
            page_text, chunk_size, chunk_overlap
        ):
            all_chunks.append(
                {
                    "chunk_id":    chunk_id,
                    "text":        text,
                    "page_number": page_number,
                    "source":      source_name,
                    "char_start":  char_start,
                    "char_end":    char_end,
                }
            )
            chunk_id += 1
            if max_chunks is not None and len(all_chunks) >= max_chunks:
                return all_chunks

    return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a plain string into overlapping chunks without metadata.

    Used by the summarizer for map-reduce passes where chunk provenance
    is not required.

    Args:
        text:          Input string to split.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.

    Returns:
        Ordered list of text strings.
    """
    return [t for t, _, _ in _sliding_window(text, chunk_size, chunk_overlap)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sliding_window(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Tuple[str, int, int]]:
    """
    Produce (chunk_text, char_start, char_end) tuples via a sliding window.

    The window advances by ``chunk_size - chunk_overlap`` characters on
    each step. Empty chunks produced after stripping are silently discarded.

    Args:
        text:          Source text to partition.
        chunk_size:    Window width in characters.
        chunk_overlap: Trailing characters re-included in the next window.

    Returns:
        List of (text, start_offset, end_offset) tuples.
    """
    step = chunk_size - chunk_overlap
    results: List[Tuple[str, int, int]] = []
    start = 0

    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            results.append((chunk, start, end))
        if end == len(text):
            break
        start += step

    return results
