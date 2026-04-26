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

# FIX: import only what is actually used here; MAX_TOKENS/TEMPERATURE are
# only referenced inside AIEngine.complete() so they are imported there via
# the config module rather than at module level to avoid a misleading
# "unused import" and to guarantee they always reflect the live config value.
from core.config import get_config, get_api_key, validate_config

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.1

# FIX-1 (memory / pipeline): Replace @lru_cache with a plain module-level
# singleton guarded by a key-equality check.
#
# ROOT CAUSE of the previous bug:
#   @lru_cache caches the *first* result, including raised exceptions.
#   If the API key is absent on the very first call, Python caches the
#   EnvironmentError.  Every subsequent call — even after the key is added
#   to the environment — hits the cache and re-raises the stale error,
#   making the app permanently broken until the process is restarted.
#
# FIX: Use a module-level variable.  None means "not yet built".
# If construction fails, the variable stays None, so the next call retries.
# If the API key changes at runtime the client is rebuilt automatically.
_client: "Groq | None" = None
_cached_api_key: str = ""


def _resolve_api_key() -> str:
    """Return the active API key or raise a clear EnvironmentError."""
    validate_config()          # raises EnvironmentError when key is absent
    return get_api_key()


def _resolve_model() -> str:
    """Return the active model name from shared runtime config."""
    return get_config().groq_model_name


def _get_client() -> "Groq":
    """
    Return the shared Groq client, constructing it on first call or whenever
    the API key changes.

    This never caches an exception: if construction fails (key absent,
    network error, etc.) _client stays None and the next call retries.
    """
    global _client, _cached_api_key

    # FIX-1: validate_config() raises before we touch _client when key absent,
    # so _client is never set to a half-constructed or error state.
    api_key = _resolve_api_key()

    if _client is None or api_key != _cached_api_key:
        try:
            _client = Groq(api_key=api_key)
            _cached_api_key = api_key
            logger.debug("Groq client (re)built for model=%s", _resolve_model())
        except Exception as exc:
            # FIX-1: do NOT assign to _client on failure — stays None so
            # the next call retries construction rather than hitting a cache.
            _client = None
            _cached_api_key = ""
            raise RuntimeError(f"Failed to instantiate Groq client: {exc}") from exc

    return _client


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
        RuntimeError: If the Groq client cannot be built or the API call fails.
    """
    if not system_prompt.strip():
        raise ValueError("system_prompt must not be blank.")
    if not user_prompt.strip():
        raise ValueError("user_prompt must not be blank.")

    client = _get_client()
    model  = _resolve_model()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(
            f"Groq API request failed (model={model}, "
            f"max_tokens={max_tokens}): {exc}"
        ) from exc


class AIEngine:
    """
    Thin stateful wrapper used by code paths that expect an object API
    (e.g. AgenticEngine).  All actual LLM calls delegate to generate_response().
    """

    def __init__(self) -> None:
        # Keep init cheap: no network call, no heavy object allocation.
        # AgenticEngine constructs this during document ingestion, so any
        # work here delays the pipeline unnecessarily.
        self._model_name = _resolve_model()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate one completion, falling back to shared config defaults."""
        # FIX-2: import MAX_TOKENS/TEMPERATURE here (not at module level) so
        # they always reflect the live config value and are not "unused" when
        # AIEngine is not instantiated.
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