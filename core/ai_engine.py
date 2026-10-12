"""
core/ai_engine.py - Centralized Groq client helpers for DocuMind.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from groq import Groq
except ImportError as exc:
    raise ImportError("The groq package is required: pip install groq") from exc

from core.config import get_config, get_api_key, validate_config

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.1
_VALID_EFFORTS = {"low", "medium", "high"}

# Module-level singleton. None means "not yet built". If construction fails
# the variable stays None so the next call retries (an lru_cache would cache
# the failure and keep the app broken until restart).
_client: "Groq | None" = None
_cached_api_key: str = ""


def _resolve_api_key() -> str:
    """Return the active API key or raise a clear EnvironmentError."""
    validate_config()
    return get_api_key()


def _resolve_model() -> str:
    """Return the active model name from shared runtime config."""
    return get_config().groq_model_name


def _get_client() -> "Groq":
    """Return the shared Groq client, rebuilding it if the API key changes."""
    global _client, _cached_api_key

    api_key = _resolve_api_key()

    if _client is None or api_key != _cached_api_key:
        try:
            _client = Groq(api_key=api_key)
            _cached_api_key = api_key
            logger.debug("Groq client (re)built for model=%s", _resolve_model())
        except Exception as exc:
            _client = None
            _cached_api_key = ""
            raise RuntimeError(f"Failed to instantiate Groq client: {exc}") from exc

    return _client


def _reasoning_kwargs(model: str) -> dict:
    """
    Reasoning models (gpt-oss) spend part of max_tokens on hidden reasoning.
    A low effort leaves room for the visible answer and is much faster.
    """
    effort = get_config().reasoning_effort
    if effort in _VALID_EFFORTS and "gpt-oss" in model.lower():
        return {"reasoning_effort": effort}
    return {}


def generate_response(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> str:
    """
    Send a single chat-completion request to Groq and return the response text.

    Raises:
        ValueError:   If either prompt is blank.
        RuntimeError: If the Groq client cannot be built, the API call fails,
                      or the model returns no text.
    """
    if not system_prompt.strip():
        raise ValueError("system_prompt must not be blank.")
    if not user_prompt.strip():
        raise ValueError("user_prompt must not be blank.")

    client = _get_client()
    model = _resolve_model()

    request = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    extra = _reasoning_kwargs(model)

    try:
        try:
            response = client.chat.completions.create(**request, **extra)
        except Exception as exc:
            # If the installed SDK or the model rejects reasoning_effort, retry without it.
            if extra and "reasoning" in str(exc).lower():
                logger.warning("reasoning_effort rejected (%s); retrying without it.", exc)
                response = client.chat.completions.create(**request)
            else:
                raise
    except Exception as exc:
        raise RuntimeError(
            f"Groq API request failed (model={model}, max_tokens={max_tokens}): {exc}"
        ) from exc

    choice = response.choices[0]
    content = (choice.message.content or "").strip()

    if not content:
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError(
                "The model ran out of tokens before writing an answer. "
                "Increase MAX_TOKENS / ANALYSIS_MAX_TOKENS or lower REASONING_EFFORT."
            )
        raise RuntimeError("The model returned an empty response.")

    return content


class AIEngine:
    """
    Thin stateful wrapper used by code paths that expect an object API.
    All actual LLM calls delegate to generate_response().
    """

    def __init__(self) -> None:
        # Keep init cheap: no network call, no heavy object allocation.
        self._model_name = _resolve_model()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate one completion, falling back to shared config defaults."""
        from core.config import MAX_TOKENS, TEMPERATURE

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
        """Alias for complete() — kept for backwards compatibility."""
        return self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def __repr__(self) -> str:
        return f"AIEngine(model={self._model_name!r})"