from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K     = 6
_MIN_CHUNK_LENGTH  = 30      
_MAX_CONTEXT_CHARS = 12_000  

_FALLBACK_ANSWER = (
    "The document does not contain sufficient information to answer this question "
    "with confidence. Please rephrase your query or upload a more relevant document."
)

_VALID_MODES = {"default", "red_team"}


# ---------------------------------------------------------------------------
# Chunk helpers (dedup + noise filter)
# ---------------------------------------------------------------------------

def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    try:
        return str(val).strip() or default
    except Exception:
        return default


def _extract_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return _safe_str(
            chunk.get("text") or chunk.get("content") or chunk.get("page_content")
        )
    return _safe_str(
        getattr(chunk, "text",          None)
        or getattr(chunk, "content",    None)
        or getattr(chunk, "page_content", None)
    )


def _extract_page(chunk: Any) -> Any:
    if isinstance(chunk, dict):
        return chunk.get("page_number") or chunk.get("page") or "?"
    return (
        getattr(chunk, "page_number", None)
        or getattr(chunk, "page",     None)
        or "?"
    )


def _extract_score(chunk: Any) -> float | None:
    raw = None
    if isinstance(chunk, dict):
        raw = (
            chunk.get("similarity")
            or chunk.get("score")
            or chunk.get("distance")
            or chunk.get("relevance_score")
        )
    else:
        raw = (
            getattr(chunk, "similarity",       None)
            or getattr(chunk, "score",         None)
            or getattr(chunk, "distance",      None)
            or getattr(chunk, "relevance_score", None)
        )
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _filter_and_deduplicate(chunks: list[Any]) -> list[Any]:
    """
    1. Remove chunks shorter than _MIN_CHUNK_LENGTH.
    2. Deduplicate by 200-char normalised text fingerprint (keep first).
    """
    seen:   set[str]  = set()
    result: list[Any] = []
    for chunk in chunks:
        text = _extract_text(chunk)
        if len(text) < _MIN_CHUNK_LENGTH:
            continue
        key = " ".join(text.lower().split())[:200]
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


def _build_context_block(chunks: list[Any], max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    lines: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        text  = _extract_text(chunk)
        page  = _extract_page(chunk)
        score = _extract_score(chunk)
        if not text or len(text) < _MIN_CHUNK_LENGTH:
            continue
        score_tag = f" [score={score:.3f}]" if score is not None else ""
        header    = f"[Chunk {i} | Page {page}{score_tag}]"
        entry     = f"{header}\n{text}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                lines.append(entry[:remaining] + "\n...[truncated]")
            break
        lines.append(entry)
        total += len(entry)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(ai_engine: Any, system_prompt: str, user_prompt: str, step_name: str) -> str:
    """
    Invoke ai_engine.complete(system_prompt, user_prompt) with full error isolation.
    Falls back to ai_engine.complete(merged_prompt) for single-arg implementations.
    Returns empty string on failure.
    """
    try:

        try:
            response = ai_engine.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        except TypeError:
     
            merged   = f"{system_prompt}\n\n{user_prompt}"
            response = ai_engine.complete(merged)

        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            return _safe_str(
                response.get("text")
                or response.get("content")
                or response.get("answer")
            )
        return _safe_str(response)
    except Exception as exc:
        logger.error("[AgenticEngine] LLM call failed at step '%s': %s", step_name, exc)
        return ""


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    sources:        list[Any],
    extractor_ok:   bool,
    critic_flagged: bool,
    answer:         str,
) -> float:
    if not answer or answer == _FALLBACK_ANSWER:
        return 0.0

    score = 0.50
    score += min(len(sources), 5) * 0.04

    sim_scores = [s for s in (_extract_score(src) for src in sources) if s is not None]
    if sim_scores:
        avg_sim = sum(sim_scores) / len(sim_scores)
        score  += max(0.0, min(1.0, avg_sim)) * 0.15

    if extractor_ok:
        score += 0.05
    if critic_flagged:
        score -= 0.15
    if len(answer) > 80:
        score += 0.05

    return round(max(0.0, min(0.95, score)), 4)


# ---------------------------------------------------------------------------
# Mode-aware prompts
# ---------------------------------------------------------------------------

_EXTRACTOR_SYSTEM = """\
You are a precise Document Analysis Agent.
Extract only facts, figures, and statements from the document chunks that are
directly relevant to answering the user's question.
Do NOT infer, assume, or add external knowledge.
Format output as a concise bullet list.
If no relevant facts exist, respond with exactly: NO_RELEVANT_FACTS"""

_EXTRACTOR_USER = """\
QUESTION:
{query}

DOCUMENT CHUNKS:
{context}

EXTRACTED FACTS:"""

# ── Default critic ────────────────────────────────────────────────────────────

_CRITIC_DEFAULT_SYSTEM = """\
You are a rigorous Evidence Critic Agent.
Evaluate the extracted facts for quality, consistency, and evidential strength.
Respond in this exact format:
VERDICT: PASS | FLAG
ISSUES: <bullet list of issues, or "None" if PASS>
RECOMMENDATION: <one sentence on how the synthesis agent should proceed>"""

_CRITIC_DEFAULT_USER = """\
QUESTION:
{query}

EXTRACTED FACTS:
{facts}

DOCUMENT CHUNKS (for reference):
{context}

EVALUATE for:
1. Contradictions between facts or with the source chunks
2. Vague or weak evidence
3. Missing critical context
4. Unsupported claims"""

# ── Red-team critic ───────────────────────────────────────────────────────────

_CRITIC_REDTEAM_SYSTEM = """\
You are an adversarial Document Red Team Agent.
Your job is to aggressively challenge the extracted facts and find every weakness,
contradiction, unsupported assumption, missing caveat, and logical flaw.
Be thorough and unsparing. If the evidence is weak, say so explicitly.
Respond in this exact format:
VERDICT: PASS | FLAG
ISSUES: <detailed bullet list of all identified problems>
RECOMMENDATION: <one sentence on how the synthesis agent should handle the weaknesses>"""

_CRITIC_REDTEAM_USER = """\
QUESTION:
{query}

EXTRACTED FACTS:
{facts}

DOCUMENT CHUNKS (for reference):
{context}

RED TEAM ANALYSIS — find all of the following:
1. Factual contradictions within or across chunks
2. Unsupported or unverifiable claims
3. Missing data that would be needed to answer confidently
4. Logical fallacies or overgeneralisation
5. Ambiguous language that could be misinterpreted
6. Confidence-inflating language without supporting evidence"""

# ── Synthesis ─────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are a professional Document Intelligence Synthesis Agent.
Produce a clear, grounded, accurate answer based solely on the verified facts
and source chunks provided.
RULES:
- Ground every claim in the provided facts or chunks.
- Cite page numbers inline, e.g. (p. 4), when referencing specific passages.
- Quote exact figures or names from the document when they answer the question.
- Do NOT introduce external knowledge or assumptions.
- If evidence is insufficient, clearly state what is missing.
- Write in clear, professional paragraphs. Be concise."""

_SYNTHESIS_USER = """\
QUESTION:
{query}

EXTRACTED FACTS:
{facts}

CRITIC ASSESSMENT:
{critique}

DOCUMENT CHUNKS:
{context}

FINAL ANSWER:"""


# ---------------------------------------------------------------------------
# AgenticEngine
# ---------------------------------------------------------------------------

class AgenticEngine:
    """
    Multi-step agentic reasoning engine for document Q&A.

    Pipeline (per query):
        Step 1 — Retrieval:   Fetch top-k chunks (filtered + deduplicated).
        Step 2 — Extractor:   Extract relevant facts via LLM.
        Step 3 — Critic:      Validate facts (normal or red-team mode).
        Step 4 — Synthesis:   Generate the final grounded answer.

    Public interface:
        engine.run(query, top_k, mode)  → dict
        engine.answer(query, top_k)     → dict  (alias)
    """

    def __init__(
        self,
        *,
        qa_engine:  Any = None,
        vector_db:  Any = None,
        ai_engine:  Any,
        top_k:      int = _DEFAULT_TOP_K,
    ) -> None:
        self._retriever = qa_engine if qa_engine is not None else vector_db
        self._ai        = ai_engine
        self._top_k     = max(1, int(top_k))

        if self._retriever is None:
            raise ValueError(
                "AgenticEngine requires either a 'qa_engine' or 'vector_db' argument."
            )
        if self._ai is None:
            raise ValueError("AgenticEngine requires an 'ai_engine' argument.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query:  str,
        top_k:  int | None = None,
        mode:   str = "default",
    ) -> dict[str, Any]:
        """
        Execute the full four-step agentic pipeline.

        Parameters
        ----------
        query : str   — The user's question.
        top_k : int   — Chunks to retrieve. Overrides the instance default.
                        Passed from app.py speed-mode selector.
        mode  : str   — "default" or "red_team".

        Returns
        -------
        {
            "final_answer": str,
            "sources":      list,
            "confidence":   float,
            "steps":        list[dict],
        }
        """
        query = _safe_str(query, default="")
        if not query:
            return self._error_response("Empty query received.")

        # Resolve effective top_k: per-call arg > instance default
        effective_top_k = max(1, int(top_k)) if top_k is not None else self._top_k

        # Normalise mode
        mode = mode if mode in _VALID_MODES else "default"

        steps:   list[dict[str, Any]] = []
        t_start: float = time.perf_counter()

        # ── Step 1: Retrieval ──────────────────────────────────────────
        sources, retrieval_msg = self._step_retrieval(query, effective_top_k)
        steps.append({
            "step":     1,
            "title":    "Retrieval",
            "body":     retrieval_msg,
            "complete": True,
            "active":   False,
        })

        if not sources:
            steps += self._pending_steps(start=2)
            return {
                "final_answer": _FALLBACK_ANSWER,
                "sources":      [],
                "confidence":   0.0,
                "steps":        steps,
            }

        context = _build_context_block(sources)

        # ── Step 2: Extractor ──────────────────────────────────────────
        facts, extractor_ok, analysis_msg = self._step_extractor(query, context)
        steps.append({
            "step":     2,
            "title":    "Analysis",
            "body":     analysis_msg,
            "complete": True,
            "active":   False,
        })

        # ── Step 3: Critic (mode-aware) ────────────────────────────────
        critique, critic_flagged, critique_msg = self._step_critic(
            query, facts, context, mode=mode
        )
        critic_title = "Red Team Critique" if mode == "red_team" else "Critique"
        steps.append({
            "step":     3,
            "title":    critic_title,
            "body":     critique_msg,
            "complete": True,
            "active":   False,
        })

        # ── Step 4: Synthesis ──────────────────────────────────────────
        final_answer, synthesis_msg = self._step_synthesis(
            query, facts, critique, context
        )
        steps.append({
            "step":     4,
            "title":    "Final Answer",
            "body":     synthesis_msg,
            "complete": True,
            "active":   False,
        })

        elapsed    = round(time.perf_counter() - t_start, 3)
        confidence = _compute_confidence(sources, extractor_ok, critic_flagged, final_answer)

        logger.info(
            "[AgenticEngine] Query completed in %.3fs | top_k=%d | mode=%s "
            "| sources=%d | confidence=%.2f",
            elapsed, effective_top_k, mode, len(sources), confidence,
        )

        return {
            "final_answer": final_answer or _FALLBACK_ANSWER,
            "sources":      sources,
            "confidence":   confidence,
            "steps":        steps,
        }

    def answer(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Alias for run() in default mode; maintains backward compatibility."""
        return self.run(query, top_k=top_k, mode="default")

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step_retrieval(
        self, query: str, top_k: int
    ) -> tuple[list[Any], str]:
        """Retrieve, filter, and deduplicate top-k chunks."""
        try:
            raw_chunks = self._retrieve_chunks(query, top_k)
        except Exception as exc:
            logger.error("[AgenticEngine] Retrieval failed: %s", exc)
            return [], f"Retrieval failed: {type(exc).__name__}: {exc}"

        sources = _filter_and_deduplicate(raw_chunks)

        if not sources:
            return [], "No relevant chunks found in the vector index for this query."

        page_refs = list({str(_extract_page(s)) for s in sources})[:5]
        msg = (
            f"Retrieved {len(sources)} chunk(s) from FAISS "
            f"(top_k={top_k}, after dedup). "
            f"Pages referenced: {', '.join(sorted(page_refs))}."
        )
        return sources, msg

    def _step_extractor(
        self, query: str, context: str
    ) -> tuple[str, bool, str]:
        system  = _EXTRACTOR_SYSTEM
        user    = _EXTRACTOR_USER.format(query=query, context=context)
        facts   = _call_llm(self._ai, system, user, "extractor")

        if not facts or facts.strip().upper() == "NO_RELEVANT_FACTS":
            return (
                "No relevant facts could be extracted from the retrieved chunks.",
                False,
                "Extractor Agent: No relevant facts found in the retrieved context.",
            )

        line_count = len([l for l in facts.splitlines() if l.strip()])
        return (
            facts,
            True,
            f"Extractor Agent identified {line_count} relevant fact(s).",
        )

    def _step_critic(
        self, query: str, facts: str, context: str, mode: str = "default"
    ) -> tuple[str, bool, str]:
        if mode == "red_team":
            system = _CRITIC_REDTEAM_SYSTEM
            user   = _CRITIC_REDTEAM_USER.format(query=query, facts=facts, context=context)
        else:
            system = _CRITIC_DEFAULT_SYSTEM
            user   = _CRITIC_DEFAULT_USER.format(query=query, facts=facts, context=context)

        critique = _call_llm(self._ai, system, user, "critic")

        if not critique:
            return (
                "Critic assessment unavailable.",
                False,
                "Critic Agent: Assessment skipped (LLM returned empty response).",
            )

        upper        = critique.upper()
        flagged      = "FLAG" in upper and "PASS" not in upper.split("FLAG")[0]
        verdict_word = "flagged issues" if flagged else "passed validation"
        label        = "Red Team" if mode == "red_team" else "Critic"
        return (
            critique,
            flagged,
            f"{label} Agent {verdict_word} in the extracted evidence.",
        )

    def _step_synthesis(
        self, query: str, facts: str, critique: str, context: str
    ) -> tuple[str, str]:
        system = _SYNTHESIS_SYSTEM
        user   = _SYNTHESIS_USER.format(
            query=query, facts=facts, critique=critique, context=context
        )
        answer = _call_llm(self._ai, system, user, "synthesis")

        if not answer:
            return (
                _FALLBACK_ANSWER,
                "Synthesis Agent: Could not generate a final answer.",
            )

        word_count = len(answer.split())
        return (
            answer,
            f"Synthesis Agent generated a {word_count}-word grounded answer.",
        )

    # ------------------------------------------------------------------
    # Retrieval dispatcher
    # ------------------------------------------------------------------

    def _retrieve_chunks(self, query: str, top_k: int) -> list[Any]:
        """
        Dispatch retrieval to QAEngine or VectorDB, trying multiple signatures.
        top_k is passed explicitly so every call respects the speed-mode setting.
        """
        retriever = self._retriever

        method_candidates = [
            # QAEngine.answer() — preferred; returns dict with "sources"
            ("answer",              (query,),      {"top_k": top_k}),
            ("answer",              (query, top_k), {}),
            # VectorDB.search()
            ("search",              (query,),      {"top_k": top_k}),
            ("search",              (query, top_k), {}),
            ("retrieve",            (query,),      {"top_k": top_k}),
            ("retrieve",            (query, top_k), {}),
            ("query",               (query,),      {"top_k": top_k}),
            ("query",               (query, top_k), {}),
            ("similarity_search",   (query,),      {"k": top_k}),
            ("similarity_search",   (query, top_k), {}),
            ("get_relevant_chunks", (query,),      {"top_k": top_k}),
            ("get_relevant_chunks", (query, top_k), {}),
        ]

        for method_name, args, kwargs in method_candidates:
            method = getattr(retriever, method_name, None)
            if method is None or not callable(method):
                continue
            try:
                result = method(*args, **kwargs)
                chunks = self._normalise_retrieval_result(result)
                if chunks:
                    return chunks
            except TypeError:
                continue
            except Exception as exc:
                logger.warning(
                    "[AgenticEngine] Retrieval method '%s' failed: %s",
                    method_name, exc,
                )
                continue

        logger.error(
            "[AgenticEngine] No compatible retrieval method found on %s.",
            type(retriever).__name__,
        )
        return []

    def _normalise_retrieval_result(self, result: Any) -> list[Any]:
        """Flatten any retriever output format into a list of chunk dicts/objects."""
        if result is None:
            return []

        # Dict response from QAEngine.answer()
        if isinstance(result, dict):
            for key in ("sources", "chunks", "results", "documents", "hits", "matches"):
                val = result.get(key)
                if val and isinstance(val, (list, tuple)):
                    return list(val)
            return []

        # List / tuple
        if isinstance(result, (list, tuple)):
            flat: list[Any] = []
            for item in result:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    chunk, score = item
                    if isinstance(chunk, dict) and _extract_score(chunk) is None and score is not None:
                        chunk = dict(chunk)
                        chunk["score"] = score
                    flat.append(chunk)
                else:
                    flat.append(item)
            return [c for c in flat if c is not None]

        # Object response
        for attr in ("chunks", "results", "documents", "hits"):
            val = getattr(result, attr, None)
            if val and isinstance(val, (list, tuple)):
                return list(val)

        return []

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pending_steps(start: int = 2) -> list[dict[str, Any]]:
        pending_map = {
            2: ("Analysis",     "Extractor Agent skipped — no source chunks available."),
            3: ("Critique",     "Critic Agent skipped — no facts to validate."),
            4: ("Final Answer", "Synthesis Agent skipped — insufficient context."),
        }
        return [
            {
                "step":     idx,
                "title":    title,
                "body":     body,
                "complete": False,
                "active":   False,
            }
            for idx, (title, body) in pending_map.items()
            if idx >= start
        ]

    @staticmethod
    def _error_response(reason: str) -> dict[str, Any]:
        return {
            "final_answer": _FALLBACK_ANSWER,
            "sources":      [],
            "confidence":   0.0,
            "steps": [
                {
                    "step":     1,
                    "title":    "Error",
                    "body":     reason,
                    "complete": False,
                    "active":   False,
                },
                *AgenticEngine._pending_steps(start=2),
            ],
        }
