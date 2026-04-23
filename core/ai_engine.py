"""Centralized Groq client helpers for DocuMind."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    from groq import Groq
except ImportError as exc:
    raise ImportError("The groq package is required: pip install groq") from exc

from core.config import MAX_TOKENS, TEMPERATURE, get_config, get_api_key, validate_config

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.1


def _resolve_api_key() -> str:
    """Return the active API key or raise a friendly runtime error."""
    validate_config()
    return get_api_key()


def _resolve_model() -> str:
    """Return the active model name from shared runtime config."""
    return get_config().groq_model_name


@lru_cache(maxsize=1)
def _build_client() -> Groq:
    """Construct and cache the authenticated Groq client."""
    api_key = _resolve_api_key()
    try:
        return Groq(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate Groq client: {exc}") from exc


def generate_response(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> str:
    """Send a single chat completion request to Groq and return the text."""
    if not system_prompt.strip():
        raise ValueError("system_prompt must not be blank.")
    if not user_prompt.strip():
        raise ValueError("user_prompt must not be blank.")

    client = _build_client()
    model = _resolve_model()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(
            f"Groq API request failed (model={model}, max_tokens={max_tokens}): {exc}"
        ) from exc


class AIEngine:
    """Small stateful wrapper used by code paths that expect an object API."""

    def __init__(self) -> None:
        # Keep init cheap so document ingestion can complete before the app
        # attempts any LLM call.
        self._model_name = _resolve_model()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate one completion with shared config defaults."""
        return generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=int(MAX_TOKENS if max_tokens is None else max_tokens),
            temperature=float(TEMPERATURE if temperature is None else temperature),
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Alias for ``complete()`` kept for backwards compatibility."""
        return self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def __repr__(self) -> str:
        return f"AIEngine(model={self._model_name!r})"
