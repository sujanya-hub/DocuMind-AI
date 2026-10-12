from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from core.config import CHUNK_OVERLAP, CHUNK_SIZE

# A chunk must reach at least this fraction of chunk_size before we look for a
# natural break point, so chunks don't become tiny.
_MIN_FILL_RATIO = 0.6

# Trailing fragments shorter than this are dropped when the page already has chunks.
_MIN_TRAILING_CHARS = 40

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s")


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

    Chunks end at a paragraph break, sentence end or word boundary (in that
    order of preference) and never start or end in the middle of a word.

    Returns:
        List of chunk dicts, each containing:
            chunk_id    (int) - global 0-based index
            text        (str) - chunk content (whitespace normalised)
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
        page_text: str = page.get("text", "").strip()
        page_number: int = page["page_number"]

        if not page_text:
            continue

        page_chunk_count = 0
        for text, char_start, char_end in _sliding_window(
            page_text, chunk_size, chunk_overlap
        ):
            if page_chunk_count > 0 and len(text) < _MIN_TRAILING_CHARS:
                continue
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "page_number": page_number,
                    "source": source_name,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            chunk_id += 1
            page_chunk_count += 1
            if max_chunks is not None and len(all_chunks) >= max_chunks:
                return all_chunks

    return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split a plain string into overlapping chunks without metadata."""
    return [t for t, _, _ in _sliding_window(text, chunk_size, chunk_overlap)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(chunk: str) -> str:
    """Rejoin hyphenated line breaks and collapse whitespace."""
    chunk = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", chunk)
    return re.sub(r"\s+", " ", chunk).strip()


def _find_break(text: str, start: int, end: int, chunk_size: int) -> int:
    """
    Choose where a chunk spanning text[start:end] should end.

    Preference: paragraph break, then sentence end, then any whitespace.
    Falls back to the hard limit only if the region has no whitespace at all.
    """
    min_end = min(start + max(1, int(chunk_size * _MIN_FILL_RATIO)), end)
    region = text[min_end:end]

    idx = region.rfind("\n\n")
    if idx != -1:
        return min_end + idx + 2

    best = -1
    for match in _SENTENCE_END.finditer(region):
        best = match.end()
    if best != -1:
        return min_end + best

    idx = max(region.rfind(" "), region.rfind("\n"), region.rfind("\t"))
    if idx != -1:
        return min_end + idx + 1

    return end


def _align_to_word_start(text: str, pos: int, limit: int) -> int:
    """Move pos forward (never past limit) so it does not sit inside a word."""
    n = len(text)
    while pos < min(limit, n) and not text[pos - 1].isspace():
        pos += 1
    return pos


def _sliding_window(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Tuple[str, int, int]]:
    """
    Produce (chunk_text, char_start, char_end) tuples.

    Windows end on natural boundaries, and the overlap with the next window is
    aligned so the next chunk starts at the beginning of a word.
    """
    results: List[Tuple[str, int, int]] = []
    n = len(text)
    start = 0

    while start < n:
        hard_end = min(start + chunk_size, n)
        end = n if hard_end == n else _find_break(text, start, hard_end, chunk_size)

        chunk = _normalise(text[start:end])
        if chunk:
            results.append((chunk, start, end))
        if end >= n:
            break

        next_start = max(end - chunk_overlap, start + 1)
        next_start = _align_to_word_start(text, next_start, end)
        if next_start <= start:
            next_start = end
        start = next_start

    return results