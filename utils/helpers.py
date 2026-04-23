"""
helpers.py - Stateless utility functions for DocuMind Analyst.

Covers file I/O, size formatting, text export formatting, JSON parsing,
execution timing, and error formatting. No Streamlit rendering logic
lives here. This module is independently importable and testable without
a running Streamlit process.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Union


def ensure_directory(path: str) -> str:
    """
    Create path and any required parent directories if they are absent.

    Args:
        path: Target directory path.

    Returns:
        The same path, unchanged, for convenient chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_uploaded_file(uploaded_file: Any, destination_dir: str) -> str:
    """
    Write a Streamlit UploadedFile object to disk.

    Args:
        uploaded_file:   Object returned by st.file_uploader().
        destination_dir: Directory in which the file will be saved.

    Returns:
        Absolute path to the written file.

    Raises:
        IOError: If the file cannot be written.
    """
    ensure_directory(destination_dir)
    file_path = os.path.join(destination_dir, uploaded_file.name)

    try:
        payload = uploaded_file.getbuffer()

        # Avoid rewriting the file when the upload content has not changed.
        if os.path.exists(file_path):
            with open(file_path, "rb") as existing_fh:
                if existing_fh.read() == payload:
                    return file_path

        with open(file_path, "wb") as fh:
            fh.write(payload)
    except OSError as exc:
        raise IOError(
            f"Could not save '{uploaded_file.name}' "
            f"to '{destination_dir}': {exc}"
        ) from exc

    return file_path


def file_sha256(uploaded_file: Any) -> str:
    """Return a stable content hash for a Streamlit UploadedFile."""
    return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()


def format_file_size(num_bytes: int) -> str:
    """
    Convert a raw byte count into a human-readable size string.

    Args:
        num_bytes: File size in bytes.

    Returns:
        Formatted string such as "1.23 MB" or "456.0 KB".
    """
    if num_bytes < 1_024:
        return f"{num_bytes} B"
    if num_bytes < 1_024 ** 2:
        return f"{num_bytes / 1_024:.1f} KB"
    if num_bytes < 1_024 ** 3:
        return f"{num_bytes / 1_024 ** 2:.2f} MB"
    return f"{num_bytes / 1_024 ** 3:.2f} GB"


def format_summary_export(
    summary: str,
    metadata: Dict[str, Any],
) -> str:
    """
    Produce a plain-text export string for a generated document summary.

    Args:
        summary:  Summary text from Summarizer.full_summary().
        metadata: Document metadata dict from pdf_reader.get_document_metadata().

    Returns:
        Formatted multi-line string suitable for download as .txt.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "DOCUMIND ANALYST - DOCUMENT SUMMARY",
        "=" * 60,
        f"Generated   : {timestamp}",
        f"File        : {metadata.get('file_name', 'Unknown')}",
        f"Title       : {metadata.get('title', 'Unknown')}",
        f"Author      : {metadata.get('author', 'Unknown')}",
        f"Pages       : {metadata.get('page_count', 'Unknown')}",
        "=" * 60,
        "",
        summary,
        "",
    ]
    return "\n".join(lines)


def format_chat_export(
    chat_history: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> str:
    """
    Produce a plain-text export string for the full Q&A chat history.

    Args:
        chat_history: List of turn dicts from session_manager.get_chat_history().
        metadata:     Document metadata dict.

    Returns:
        Formatted multi-line string suitable for download as .txt.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "DOCUMIND ANALYST - CHAT HISTORY",
        "=" * 60,
        f"Exported    : {timestamp}",
        f"File        : {metadata.get('file_name', 'Unknown')}",
        f"Total turns : {len(chat_history)}",
        "=" * 60,
        "",
    ]

    for turn in chat_history:
        lines += [
            f"[Turn {turn['turn']}]",
            f"Question : {turn['question']}",
            f"Answer   : {turn['answer']}",
        ]
        sources = turn.get("sources", [])
        if sources:
            pages = sorted({s.get("page_number", "?") for s in sources})
            lines.append(
                f"Sources  : Page(s) {', '.join(str(p) for p in pages)}"
            )
        lines += ["", "-" * 60, ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def safe_json_parse(
    text: str,
    fallback: Optional[Any] = None,
) -> Any:
    """
    Attempt to parse a JSON string, returning a safe fallback on failure
    instead of raising an exception.

    Common LLM response artifacts are stripped before parsing:
    - Leading/trailing whitespace
    - Markdown code-fence wrappers (```json ... ``` or ``` ... ```)

    Args:
        text:     Raw string that may contain JSON, possibly wrapped in
                  markdown code fences or prefixed with natural language.
        fallback: Value returned when parsing fails. Defaults to None.

    Returns:
        Parsed Python object (dict, list, str, int, …) on success, or
        ``fallback`` if the input is not valid JSON after cleaning.

    Examples:
        >>> safe_json_parse('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_parse("```json\\n{\\\"a\\\": 1}\\n```")
        {'a': 1}
        >>> safe_json_parse("not json", fallback={})
        {}
    """
    if not isinstance(text, str) or not text.strip():
        return fallback

    cleaned = text.strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        # Remove opening fence (with optional language tag)
        cleaned = cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------

class Timer:
    """
    A lightweight wall-clock timer with start/stop/lap semantics.

    Usage — manual::

        t = Timer()
        t.start()
        do_work()
        elapsed = t.stop()          # seconds as float
        print(t.format())           # "1.23s"

    Usage — context manager::

        with Timer() as t:
            do_work()
        print(t.elapsed)            # float seconds
        print(t.format())           # "1.23s"

    Usage — one-shot convenience::

        elapsed = Timer.measure(some_callable, arg1, arg2)
    """

    def __init__(self) -> None:
        self._start:  Optional[float] = None
        self._end:    Optional[float] = None
        self._laps:   List[float]     = []

    # -- Core interface ------------------------------------------------------

    def start(self) -> "Timer":
        """Record the start time and return self for chaining."""
        self._start = time.perf_counter()
        self._end   = None
        self._laps  = []
        return self

    def stop(self) -> float:
        """
        Record the stop time.

        Returns:
            Elapsed wall-clock seconds as a float.

        Raises:
            RuntimeError: If start() has not been called.
        """
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start().")
        self._end = time.perf_counter()
        return self.elapsed

    def lap(self) -> float:
        """
        Record a lap split without stopping the timer.

        Returns:
            Seconds since the last lap (or since start if no prior lap).

        Raises:
            RuntimeError: If start() has not been called.
        """
        if self._start is None:
            raise RuntimeError("Timer.lap() called before Timer.start().")
        now      = time.perf_counter()
        previous = self._laps[-1] if self._laps else self._start
        split    = now - previous
        self._laps.append(now)
        return split

    def reset(self) -> None:
        """Clear all recorded times."""
        self._start = None
        self._end   = None
        self._laps  = []

    # -- Properties ----------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """
        Seconds elapsed since start.

        Returns the time to the recorded stop if stop() has been called,
        otherwise the time to *now* (live read while timer is running).

        Returns:
            0.0 if start() has never been called.
        """
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    @property
    def laps(self) -> List[float]:
        """List of absolute perf_counter timestamps for each recorded lap."""
        return list(self._laps)

    @property
    def is_running(self) -> bool:
        """True if the timer has been started but not yet stopped."""
        return self._start is not None and self._end is None

    # -- Formatting ----------------------------------------------------------

    def format(self, decimals: int = 2) -> str:
        """
        Return a human-readable elapsed time string.

        Args:
            decimals: Number of decimal places (default 2).

        Returns:
            Strings like "0.05s", "3.14s", "120.00s".
        """
        return f"{self.elapsed:.{decimals}f}s"

    def format_ms(self, decimals: int = 0) -> str:
        """
        Return elapsed time expressed in milliseconds.

        Args:
            decimals: Number of decimal places (default 0).

        Returns:
            Strings like "50ms", "3140ms".
        """
        return f"{self.elapsed * 1_000:.{decimals}f}ms"

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # -- Class-level convenience ---------------------------------------------

    @classmethod
    def measure(cls, fn: Any, *args: Any, **kwargs: Any) -> float:
        """
        Call ``fn(*args, **kwargs)`` and return the elapsed seconds.

        The return value of ``fn`` is discarded; use a context manager
        if you need both the result and the timing.

        Args:
            fn:      Any callable.
            *args:   Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn.

        Returns:
            Wall-clock seconds as a float.
        """
        t = cls()
        with t:
            fn(*args, **kwargs)
        return t.elapsed


@contextmanager
def timed_block(label: str = "") -> Generator[Timer, None, None]:
    """
    Context manager that yields a running Timer and prints the elapsed
    time on exit — useful for quick ad-hoc profiling in non-UI code.

    Args:
        label: Optional label printed alongside the elapsed time.

    Yields:
        A started Timer instance.

    Example::

        with timed_block("embedding") as t:
            embedder.encode(chunks)
        # prints: "embedding: 0.83s"
    """
    t = Timer()
    t.start()
    try:
        yield t
    finally:
        t.stop()
        tag = f"{label}: " if label else ""
        print(f"{tag}{t.format()}")


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

def format_error(
    exc: Exception,
    context: str = "",
    include_traceback: bool = False,
) -> str:
    """
    Produce a consistent, human-readable error string from an exception.

    Suitable for user-facing error cards in the UI or for structured
    log output — never raises itself.

    Args:
        exc:               The caught exception instance.
        context:           Optional short description of the operation that
                           failed (e.g. "PDF extraction", "FAISS query").
                           Prepended to the message when non-empty.
        include_traceback: When True, append the full formatted traceback.
                           Recommended for debug/log use; avoid in
                           user-facing UI strings.

    Returns:
        A formatted error string. Examples:

        Without context::
            "[ValueError] Input text is empty."

        With context::
            "[PDF extraction] ValueError: Input text is empty."

        With traceback::
            "[PDF extraction] ValueError: Input text is empty.
            Traceback (most recent call last):
              ..."
    """
    exc_type = type(exc).__name__
    exc_msg  = str(exc).strip() or "(no message)"

    if context:
        header = f"[{context}] {exc_type}: {exc_msg}"
    else:
        header = f"[{exc_type}] {exc_msg}"

    if not include_traceback:
        return header

    tb = traceback.format_exc().strip()
    return f"{header}\n{tb}"


def format_error_dict(
    exc: Exception,
    context: str = "",
) -> Dict[str, str]:
    """
    Return a structured dict representation of an exception.

    Useful when the error needs to be stored in session state or
    serialised rather than immediately rendered as a string.

    Args:
        exc:     The caught exception instance.
        context: Optional operation context label.

    Returns:
        Dict with keys ``type``, ``message``, ``context``, and
        ``timestamp`` (ISO-8601 UTC string).

    Example::

        {
            "type":      "ValueError",
            "message":   "Input text is empty.",
            "context":   "PDF extraction",
            "timestamp": "2024-11-01T14:32:07",
        }
    """
    return {
        "type":      type(exc).__name__,
        "message":   str(exc).strip() or "(no message)",
        "context":   context,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def is_retryable_error(exc: Exception) -> bool:
    """
    Heuristic check for whether an exception is likely transient and
    safe to retry (e.g. network timeouts, rate limits).

    Matches exception type names and message substrings against a
    curated list of known transient error patterns. No external
    dependencies are imported; matching is purely string-based so it
    works across HTTP client libraries.

    Args:
        exc: The caught exception instance.

    Returns:
        True if the error looks transient; False otherwise.
    """
    transient_type_fragments = {
        "Timeout", "timeout",
        "ConnectionError", "ConnectionReset",
        "RateLimitError", "TooManyRequests",
        "ServiceUnavailable", "GatewayTimeout",
        "TemporaryFailure", "Retry",
    }
    transient_message_fragments = {
        "rate limit", "rate_limit",
        "too many requests",
        "timeout", "timed out",
        "connection reset", "connection refused",
        "service unavailable", "503", "429",
        "temporary", "retry",
    }

    exc_type_name = type(exc).__name__
    exc_message   = str(exc).lower()

    if any(frag in exc_type_name for frag in transient_type_fragments):
        return True
    if any(frag in exc_message for frag in transient_message_fragments):
        return True
    return False
