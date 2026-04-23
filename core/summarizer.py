"""
summarizer.py - Document summarization via Groq.

Produces three complementary summary types from document text:
    1. short_summary    - executive-level overview (~100 words)
    2. detailed_summary - comprehensive analytical summary (~300 words)
    3. bullet_summary   - 5–8 grounded, specific bullet points

Short documents are processed in a single LLM call per summary type.
Documents that exceed the character limit are processed using a
map-reduce strategy: windows are summarized independently, then
condensed into each final output type.

All LLM calls are routed through core.ai_engine.generate_response.

Public interface:
    summarizer.full_summary(pages)        → str   (backward-compatible)
    summarizer.summarize(pages)           → dict  {"short", "detailed", "bullets"}
    summarizer.chunk_summary(text)        → str   (backward-compatible)
    summarizer.short_summary(pages)       → str
    summarizer.detailed_summary(pages)    → str
    summarizer.bullet_summary(pages)      → list[str]
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from core.ai_engine import generate_response
from core.chunker import chunk_text
from core.config import MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SUMMARY_CHARS:   int = 12_000   # char threshold for map-reduce
_MAP_WINDOW_SIZE:     int = 10_000   # chars per map-reduce window
_MAP_WINDOW_OVERLAP:  int = 300      # chars of overlap between windows
_MIN_BULLET_COUNT:    int = 5
_MAX_BULLET_COUNT:    int = 8
_SHORT_WORD_TARGET:   int = 100
_DETAILED_WORD_TARGET: int = 300

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_MAP_CHUNK_SYSTEM = """\
You are a precise document excerpt summarizer.
Summarize the following excerpt in 3 to 5 sentences, preserving all key
facts, figures, named entities, and conclusions.
Do NOT speculate or add information not present in the text.
"""

_SHORT_SUMMARY_SYSTEM = """\
You are DocuMind Analyst, an expert document intelligence system.
Write a SHORT executive summary of the provided document text.

Requirements:
- Exactly 90–110 words.
- One cohesive paragraph; no bullet points or headers.
- Capture: main topic, core argument or purpose, and primary conclusion.
- Ground every sentence in the document. Do not speculate.
- Professional, precise tone.

Return only the summary paragraph. No preamble, no labels.
"""

_DETAILED_SUMMARY_SYSTEM = """\
You are DocuMind Analyst, an expert document intelligence system.
Write a DETAILED analytical summary of the provided document text.

Requirements:
- 280–320 words.
- Two to four well-structured paragraphs; no bullet points or headers.
- Cover: main topic, key arguments, supporting evidence, important data
  points or figures, conclusions, and any stated caveats or limitations.
- Distinguish between what the document explicitly states and what it implies.
- Ground every claim in the document. Do not introduce external knowledge.
- Professional, precise, analytical tone.

Return only the summary paragraphs. No preamble, no labels.
"""

_BULLET_SUMMARY_SYSTEM = f"""\
You are DocuMind Analyst, an expert document intelligence system.
Extract the most important facts and findings from the provided document
as a structured bullet list.

Requirements:
- Between {_MIN_BULLET_COUNT} and {_MAX_BULLET_COUNT} bullets.
- Each bullet must be specific and grounded: cite figures, entities,
  dates, or named concepts from the document where available.
- No generic observations (e.g. "the document discusses X").
- Each bullet must be a complete, standalone statement.
- Use this exact format — one bullet per line, starting with a hyphen:
  - <specific finding>

Return only the bullet list. No preamble, no labels, no section headers.
"""

_FULL_SUMMARY_SYSTEM = """\
You are DocuMind Analyst, a professional document summarizer.
Write a clear, well-structured summary of the provided document text.
- Use concise paragraphs; avoid bullet points unless listing discrete facts.
- Capture the main topic, key arguments, and important conclusions.
- Keep the summary under 300 words unless the document is unusually long.
"""

_CHUNK_SUMMARY_SYSTEM = """\
You are a precise text summarizer.
Summarize the following excerpt in 2 to 4 sentences, preserving all key facts.
"""

# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class Summarizer:
    """
    Document summarizer with three output types and map-reduce support
    for long documents.

    Summary types:
        short    (~100 words)  — executive overview, single paragraph
        detailed (~300 words)  — analytical multi-paragraph summary
        bullets  (5–8 items)   — grounded, specific bullet points

    All methods are safe to call on empty or missing text; they return
    empty strings / lists rather than raising in those edge cases.
    """

    def __init__(self) -> None:
        # Keep construction side-effect free so document ingestion can finish
        # even before deployment secrets are configured.
        pass

    # -------------------------------------------------------------------------
    # Primary public API
    # -------------------------------------------------------------------------

    def summarize(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate all three summary types for a document.

        This is the canonical entry point. The three LLM calls are made
        sequentially; each is independently wrapped so a failure in one
        does not prevent the others from completing.

        Args:
            pages: Output of ``pdf_reader.extract_pages()``.

        Returns:
            {
                "short":    str,        # ~100-word executive overview
                "detailed": str,        # ~300-word analytical summary
                "bullets":  list[str],  # 5–8 grounded bullet points
            }

        Raises:
            ValueError: If no extractable text is found in pages.
        """
        full_text = _pages_to_text(pages)
        if not full_text.strip():
            raise ValueError("No extractable text was found in the document.")

        short_text    = self._safe_call(self._generate_short,    full_text, "short")
        detailed_text = self._safe_call(self._generate_detailed, full_text, "detailed")
        bullets_list  = self._safe_bullet_call(full_text)

        return {
            "short":    short_text,
            "detailed": detailed_text,
            "bullets":  bullets_list,
        }

    def short_summary(self, pages: List[Dict[str, Any]]) -> str:
        """
        Generate a ~100-word executive summary paragraph.

        Args:
            pages: Output of ``pdf_reader.extract_pages()``.

        Returns:
            Short summary string.

        Raises:
            ValueError: If no extractable text is found.
        """
        full_text = _pages_to_text(pages)
        if not full_text.strip():
            raise ValueError("No extractable text was found in the document.")
        return self._generate_short(full_text)

    def detailed_summary(self, pages: List[Dict[str, Any]]) -> str:
        """
        Generate a ~300-word detailed analytical summary.

        Args:
            pages: Output of ``pdf_reader.extract_pages()``.

        Returns:
            Detailed summary string.

        Raises:
            ValueError: If no extractable text is found.
        """
        full_text = _pages_to_text(pages)
        if not full_text.strip():
            raise ValueError("No extractable text was found in the document.")
        return self._generate_detailed(full_text)

    def bullet_summary(self, pages: List[Dict[str, Any]]) -> List[str]:
        """
        Generate a 5–8 bullet point summary grounded in the document.

        Args:
            pages: Output of ``pdf_reader.extract_pages()``.

        Returns:
            List of bullet point strings (markers stripped).

        Raises:
            ValueError: If no extractable text is found.
        """
        full_text = _pages_to_text(pages)
        if not full_text.strip():
            raise ValueError("No extractable text was found in the document.")
        return self._generate_bullets(full_text)

    # -------------------------------------------------------------------------
    # Backward-compatible methods
    # -------------------------------------------------------------------------

    def full_summary(self, pages: List[Dict[str, Any]]) -> str:
        """
        Backward-compatible single-string summary (~300 words).

        Delegates to detailed_summary internally.

        Args:
            pages: Output of ``pdf_reader.extract_pages()``.

        Returns:
            Summary string produced by the LLM.

        Raises:
            ValueError:   If no extractable text is found.
            RuntimeError: If any LLM call fails.
        """
        return self.detailed_summary(pages)

    def chunk_summary(self, chunk_text_str: str) -> str:
        """
        Summarize a single text chunk in 2 to 4 sentences.

        Args:
            chunk_text_str: Raw text of one chunk.

        Returns:
            Short summary string.

        Raises:
            ValueError: If the input is blank.
        """
        if not chunk_text_str.strip():
            raise ValueError("Chunk text must not be blank.")

        return generate_response(
            system_prompt=_CHUNK_SUMMARY_SYSTEM,
            user_prompt=chunk_text_str,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

    # -------------------------------------------------------------------------
    # Summary generators (operate on assembled text strings)
    # -------------------------------------------------------------------------

    def _generate_short(self, text: str) -> str:
        """Generate a ~100-word short summary from assembled document text."""
        condensed = self._maybe_map_reduce(text)
        result    = generate_response(
            system_prompt=_SHORT_SUMMARY_SYSTEM,
            user_prompt=f"DOCUMENT TEXT:\n{condensed}",
            max_tokens=300,
            temperature=TEMPERATURE,
        )
        return (result or "").strip()

    def _generate_detailed(self, text: str) -> str:
        """Generate a ~300-word detailed summary from assembled document text."""
        condensed = self._maybe_map_reduce(text)
        result    = generate_response(
            system_prompt=_DETAILED_SUMMARY_SYSTEM,
            user_prompt=f"DOCUMENT TEXT:\n{condensed}",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return (result or "").strip()

    def _generate_bullets(self, text: str) -> List[str]:
        """Generate a 5–8 bullet list from assembled document text."""
        condensed = self._maybe_map_reduce(text)
        raw       = generate_response(
            system_prompt=_BULLET_SUMMARY_SYSTEM,
            user_prompt=f"DOCUMENT TEXT:\n{condensed}",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        bullets = _parse_bullet_list(raw or "")
        bullets = _enforce_bullet_count(bullets, text)
        return bullets

    # -------------------------------------------------------------------------
    # Map-reduce
    # -------------------------------------------------------------------------

    def _maybe_map_reduce(self, text: str) -> str:
        """
        Return text unchanged if short enough; otherwise apply map-reduce.

        Args:
            text: Assembled document text of arbitrary length.

        Returns:
            Either the original text or a condensed map-reduce output that
            fits within _MAX_SUMMARY_CHARS.
        """
        if len(text) <= _MAX_SUMMARY_CHARS:
            return text
        return self._map_reduce(text)

    def _map_reduce(self, text: str) -> str:
        """
        Condense long text through a two-pass map-reduce strategy.

        Map:    Each window is summarised independently in 3–5 sentences.
        Reduce: Partial summaries are joined and re-summarised if still long.

        Args:
            text: Full document text exceeding _MAX_SUMMARY_CHARS.

        Returns:
            Condensed text string suitable for a final LLM summary call.
        """
        windows = chunk_text(
            text,
            chunk_size=_MAP_WINDOW_SIZE,
            chunk_overlap=_MAP_WINDOW_OVERLAP,
        )

        logger.info(
            "[Summarizer] Map-reduce: %d windows (doc_len=%d chars).",
            len(windows),
            len(text),
        )

        partials: List[str] = []
        for i, window in enumerate(windows):
            if not window.strip():
                continue
            try:
                partial = generate_response(
                    system_prompt=_MAP_CHUNK_SYSTEM,
                    user_prompt=window,
                    max_tokens=400,
                    temperature=TEMPERATURE,
                )
                if partial and partial.strip():
                    partials.append(partial.strip())
            except Exception as exc:
                logger.warning(
                    "[Summarizer] Map window %d failed: %s", i, exc
                )

        if not partials:
            # Fallback: hard-truncate and proceed
            logger.warning(
                "[Summarizer] All map windows failed; falling back to truncation."
            )
            return text[:_MAX_SUMMARY_CHARS]

        combined = "\n\n".join(partials)

        # Reduce pass if combined partials still exceed the limit
        if len(combined) > _MAX_SUMMARY_CHARS:
            combined = combined[:_MAX_SUMMARY_CHARS]

        return combined

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _safe_call(
        self,
        fn: Any,
        text: str,
        label: str,
    ) -> str:
        """
        Call a string-returning summary generator with full error isolation.

        Args:
            fn:    Generator method accepting a single text argument.
            text:  Assembled document text.
            label: Human-readable label for log messages.

        Returns:
            Summary string, or an empty string on failure.
        """
        try:
            result = fn(text)
            return result if isinstance(result, str) else ""
        except Exception as exc:
            logger.error(
                "[Summarizer] %s summary generation failed: %s", label, exc
            )
            return ""

    def _safe_bullet_call(self, text: str) -> List[str]:
        """
        Call _generate_bullets with full error isolation.

        Args:
            text: Assembled document text.

        Returns:
            List of bullet strings, or an empty list on failure.
        """
        try:
            return self._generate_bullets(text)
        except Exception as exc:
            logger.error("[Summarizer] Bullet summary generation failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_bullet_list(raw: str) -> List[str]:
    """
    Extract bullet-point items from a raw LLM response.

    Recognises lines beginning with ``-``, ``*``, ``•``, or a digit
    followed by ``.`` or ``)``. Returns stripped, non-empty strings only
    with all bullet markers removed.

    Args:
        raw: Raw string returned by the LLM.

    Returns:
        Ordered list of bullet text strings.
    """
    items: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^[-*•]\s+", "", stripped)
        cleaned = re.sub(r"^\d+[.)]\s+", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            items.append(cleaned)
    return items


def _enforce_bullet_count(
    bullets: List[str],
    source_text: str,
) -> List[str]:
    """
    Ensure the bullet list falls within [_MIN_BULLET_COUNT, _MAX_BULLET_COUNT].

    - If there are more than _MAX_BULLET_COUNT bullets, retain the first
      _MAX_BULLET_COUNT (the LLM typically orders by importance).
    - If there are fewer than _MIN_BULLET_COUNT bullets, log a warning;
      the list is returned as-is rather than padding with fabricated items.

    Args:
        bullets:     Parsed bullet list from the LLM.
        source_text: Original document text (used for length logging only).

    Returns:
        Adjusted bullet list.
    """
    if len(bullets) > _MAX_BULLET_COUNT:
        logger.debug(
            "[Summarizer] Trimming bullets from %d to %d.",
            len(bullets),
            _MAX_BULLET_COUNT,
        )
        return bullets[:_MAX_BULLET_COUNT]

    if len(bullets) < _MIN_BULLET_COUNT:
        logger.warning(
            "[Summarizer] Only %d bullet(s) generated (minimum=%d). "
            "Document may be too short or LLM response was partial.",
            len(bullets),
            _MIN_BULLET_COUNT,
        )

    return bullets


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

def _pages_to_text(pages: List[Dict[str, Any]]) -> str:
    """
    Concatenate page texts with page-break markers.

    Filters out pages with no usable text before joining. Pages with
    non-string or None text values are skipped safely.

    Args:
        pages: List of page dicts from ``pdf_reader.extract_pages()``.

    Returns:
        Single string with ``--- Page N ---`` separators.
    """
    parts: List[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        raw  = page.get("text")
        text = (raw.strip() if isinstance(raw, str) else "")
        if text:
            parts.append(
                f"--- Page {page.get('page_number', '?')} ---\n{text}"
            )
    return "\n\n".join(parts)
