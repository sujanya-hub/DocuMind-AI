from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.ai_engine import generate_response
from core.config import MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_MIN_INSIGHTS        = 5
_MAX_INSIGHTS        = 8
_MIN_CONFIDENCE      = 0.45  
_MIN_INSIGHT_WORDS   = 8     
_MAX_REGEN_ATTEMPTS  = 2    

_GENERIC_PHRASES: Tuple[str, ...] = (
    "the document provides",
    "the document discusses",
    "the document covers",
    "the document contains",
    "the document highlights",
    "this document",
    "as mentioned",
    "it is important to note",
    "it should be noted",
    "the report states",
    "the report discusses",
    "further analysis is needed",
    "further research is needed",
    "more information is needed",
    "additional information",
    "various aspects",
    "several factors",
    "a number of",
    "in conclusion",
    "in summary",
    "overall, the document",
)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class InsightItem:
    """
    A single grounded insight with an associated confidence score.

    Attributes:
        text:       The insight statement, specific and document-grounded.
        confidence: Float in [0.0, 1.0] reflecting evidential strength.
                    Derived from LLM self-assessment and heuristic signals.
    """
    text:       str
    confidence: float

    def to_dict(self) -> Dict[str, object]:
        """Serialise to the canonical output format."""
        return {"text": self.text, "confidence": round(self.confidence, 4)}


@dataclass
class SuggestedQuestions:
    """
    Stakeholder-segmented questions surfaced from the document content.

    Attributes:
        executive:     High-level strategic questions for leadership.
        analytical:    Data-oriented questions for analysts.
        risk:          Questions focused on uncertainty, gaps, and exposure.
        visualization: Questions suited to charts, tables, or visual analysis.
    """
    executive:     List[str] = field(default_factory=list)
    analytical:    List[str] = field(default_factory=list)
    risk:          List[str] = field(default_factory=list)
    visualization: List[str] = field(default_factory=list)


@dataclass
class ActionableTakeaways:
    """
    Role-oriented recommendations and sequenced next steps.

    Attributes:
        leadership: Strategic actions for senior decision-makers.
        analyst:    Investigative or analytical follow-up actions.
        next_steps: Concrete, sequenced operational actions.
    """
    leadership: List[str] = field(default_factory=list)
    analyst:    List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


@dataclass
class InsightResult:
    """
    Complete structured analytical output for one document.

    Attributes:
        key_insights:         Ranked, scored, grounded InsightItem list.
        suggested_questions:  Questions categorised by stakeholder role.
        actionable_takeaways: Role-oriented actions and sequenced steps.
        enhanced_explanation: Analytical narrative beyond summarisation.
    """
    key_insights:         List[InsightItem]    = field(default_factory=list)
    suggested_questions:  SuggestedQuestions   = field(default_factory=SuggestedQuestions)
    actionable_takeaways: ActionableTakeaways  = field(default_factory=ActionableTakeaways)
    enhanced_explanation: str                  = ""

    def key_insights_as_dicts(self) -> List[Dict[str, object]]:
        """Return key_insights in the canonical serialised format."""
        return [item.to_dict() for item in self.key_insights]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior document intelligence analyst. Your role is to extract
structured, substantive analysis from document text and present it in a
format that directly supports decision-making by executives, analysts,
and operational teams.

Analytical standards:
- Ground every claim in the document. Do not speculate beyond the text.
- Distinguish between what the document states and what it implies.
- Flag caveats, data-quality concerns, weak evidence, or contradictions
  when present in the source material.
- Avoid generic filler, vague language, and unsupported assertions.
- Write for a professional audience that values precision and brevity.
- Each KEY_INSIGHT must reference specific data, entities, dates, figures,
  or named concepts from the document. Generic observations are not permitted.

Return your response using EXACTLY this structure. Do not add prose before
or after the structured output. Do not rename, reorder, or omit any section.

KEY_INSIGHTS:
- <concise, specific finding with material significance — must cite specific
  facts, figures, entities, or named elements from the document>

SUGGESTED_QUESTIONS:
Executive:
- <strategic question for senior leadership>
Analytical:
- <data or evidence-oriented question>
Risk:
- <question about uncertainty, gaps, or exposure>
Visualization:
- <question best answered with a chart or table>

ACTIONABLE_TAKEAWAYS:
Leadership:
- <strategic recommendation for decision-makers>
Analyst:
- <investigative or analytical follow-up action>
Next Steps:
- <concrete, sequenced operational action>

ENHANCED_EXPLANATION:
<A substantive analytical narrative of 3-5 paragraphs. Interpret the
document, explain what the findings mean in context, identify the most
consequential elements, and note caveats or areas requiring further
investigation. Write in clear, professional prose with no bullet points.>
"""

_SCORING_SYSTEM_PROMPT = """\
You are an evidence-quality assessor for a document intelligence system.
You will receive a list of analytical insights extracted from a document,
along with the source document text.

For each insight, assign a confidence score from 0.00 to 1.00 based on:
  - Specificity: Does it cite concrete facts, figures, names, or dates? (high weight)
  - Grounding:   Is it directly supported by the document text? (high weight)
  - Materiality: Is it consequential to understanding the document? (medium weight)
  - Uniqueness:  Does it add information beyond the obvious? (medium weight)
  - Precision:   Is the language precise rather than vague? (low weight)

Scoring bands:
  0.85–1.00  Highly specific, directly evidenced, materially significant
  0.65–0.84  Specific and grounded but moderate materiality or minor vagueness
  0.45–0.64  Partially grounded; some specificity missing or claim is marginal
  0.00–0.44  Generic, vague, or unsupported — should be discarded

Return ONLY a JSON array of objects in this exact format with no prose, no
markdown fences, no explanation:
[{"index": 0, "score": 0.82}, {"index": 1, "score": 0.61}, ...]
"""

_ADDITIONAL_INSIGHTS_PROMPT = """\
The previous analysis produced fewer than {min_needed} strong insights.
Examine the document again and extract {count} additional specific, grounded
insights that were not covered previously.

Previously extracted insights (do not repeat):
{existing}

Return ONLY a bullet list in this format — no section headers, no prose:
- <insight text>
- <insight text>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_insights(document_text: str) -> InsightResult:
    """
    Generate structured analytical insights from document text.

    Pipeline:
        1. Primary LLM call — full structured generation.
        2. Parse all four output sections.
        3. Score each raw insight via a dedicated scoring LLM pass.
        4. Filter insights below _MIN_CONFIDENCE threshold.
        5. If fewer than _MIN_INSIGHTS remain, request additional insights
           (up to _MAX_REGEN_ATTEMPTS extra LLM calls).
        6. Rank survivors by confidence (descending).
        7. Cap at _MAX_INSIGHTS.

    Args:
        document_text: Assembled document text from extracted pages.
                       Callers are responsible for respecting context
                       window limits before calling this function.

    Returns:
        Populated InsightResult with ranked, scored InsightItem list.

    Raises:
        ValueError:   If document_text is blank.
        RuntimeError: If the primary LLM call fails.
    """
    if not document_text.strip():
        raise ValueError("document_text must not be blank.")

    user_prompt = (
        "Analyse the following document and return your structured output "
        "using exactly the format specified in your instructions.\n\n"
        f"DOCUMENT TEXT:\n{document_text}"
    )

    raw_response = generate_response(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    result = _parse_response(raw_response)

    # ── Score, filter, and rank key_insights ────────────────────────────────
    raw_texts   = [item.text for item in result.key_insights]
    scored      = _score_insights(raw_texts, document_text)
    filtered    = _filter_insights(scored)
    regenerated = filtered

    # ── Regenerate if below minimum ──────────────────────────────────────────
    for attempt in range(_MAX_REGEN_ATTEMPTS):
        if len(regenerated) >= _MIN_INSIGHTS:
            break
        needed     = _MIN_INSIGHTS - len(regenerated) + 2   # request a buffer
        additional = _request_additional_insights(
            document_text=document_text,
            existing_texts=[i.text for i in regenerated],
            count=needed,
        )
        if not additional:
            logger.warning(
                "[InsightEngine] Regen attempt %d returned no additional insights.",
                attempt + 1,
            )
            break
        extra_scored   = _score_insights(additional, document_text)
        extra_filtered = _filter_insights(extra_scored)
        regenerated    = regenerated + extra_filtered
        logger.info(
            "[InsightEngine] Regen attempt %d: +%d insights (total=%d).",
            attempt + 1,
            len(extra_filtered),
            len(regenerated),
        )

    # ── Pad with low-confidence survivors if still short ────────────────────
    if len(regenerated) < _MIN_INSIGHTS:
        fallback_needed = _MIN_INSIGHTS - len(regenerated)
        fallback_pool   = _score_insights(raw_texts, document_text)
        # Include items that were filtered out but are least-weak
        already_texts   = {i.text for i in regenerated}
        candidates      = sorted(
            [i for i in fallback_pool if i.text not in already_texts],
            key=lambda x: x.confidence,
            reverse=True,
        )
        regenerated += candidates[:fallback_needed]
        logger.info(
            "[InsightEngine] Padded with %d fallback insight(s) to meet minimum.",
            min(fallback_needed, len(candidates)),
        )

    # ── Rank by confidence, cap at maximum ──────────────────────────────────
    ranked = sorted(regenerated, key=lambda x: x.confidence, reverse=True)
    ranked = ranked[:_MAX_INSIGHTS]

    logger.info(
        "[InsightEngine] Final insight count: %d (min=%d, max=%d).",
        len(ranked), _MIN_INSIGHTS, _MAX_INSIGHTS,
    )

    result.key_insights = ranked
    return result


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _score_insights(
    texts: List[str],
    document_text: str,
) -> List[InsightItem]:
    """
    Assign confidence scores to a list of insight texts via a dedicated
    LLM scoring pass, with heuristic pre-filtering and fallback scoring.

    Args:
        texts:         Raw insight text strings to score.
        document_text: Source document for evidential grounding assessment.

    Returns:
        List of InsightItem with LLM-assigned or heuristic confidence scores.
    """
    if not texts:
        return []

    # ── Heuristic pre-filter: drop obviously generic items before LLM call ───
    pre_filtered = [(i, t) for i, t in enumerate(texts) if not _is_generic(t)]
    if not pre_filtered:
        return []

    indices, clean_texts = zip(*pre_filtered)

    numbered = "\n".join(f'{j}. "{t}"' for j, t in enumerate(clean_texts))
    user_prompt = (
        f"DOCUMENT TEXT (excerpt, first 4000 chars):\n"
        f"{document_text[:4000]}\n\n"
        f"INSIGHTS TO SCORE:\n{numbered}"
    )

    raw = generate_response(
        system_prompt=_SCORING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=800,
        temperature=0.0,
    )

    llm_scores = _parse_score_response(raw, len(clean_texts))

    items: List[InsightItem] = []
    for j, text in enumerate(clean_texts):
        llm_score = llm_scores.get(j)
        if llm_score is not None:
            score = llm_score
        else:
            # Heuristic fallback when LLM score is missing for this index
            score = _heuristic_score(text, document_text)
        items.append(InsightItem(text=text, confidence=round(score, 4)))

    return items


def _parse_score_response(
    raw: str,
    expected_count: int,
) -> Dict[int, float]:
    """
    Parse the JSON array returned by the scoring LLM into an index→score map.
    Robust to markdown fences, leading/trailing prose, and partial responses.

    Args:
        raw:            Raw string from the scoring LLM call.
        expected_count: Number of insights that were scored (for validation).

    Returns:
        Dict mapping 0-based insight index to its float score.
    """
    scores: Dict[int, float] = {}

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()

    # Locate the JSON array
    array_match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if not array_match:
        logger.warning("[InsightEngine] Score response contained no JSON array.")
        return scores

    try:
        import json
        items = json.loads(array_match.group())
        for item in items:
            if not isinstance(item, dict):
                continue
            idx   = item.get("index")
            score = item.get("score")
            if idx is None or score is None:
                continue
            try:
                idx_int   = int(idx)
                score_flt = float(score)
                if 0 <= idx_int < expected_count:
                    scores[idx_int] = max(0.0, min(1.0, score_flt))
            except (TypeError, ValueError):
                continue
    except Exception as exc:
        logger.warning("[InsightEngine] Failed to parse score JSON: %s", exc)

    return scores


def _heuristic_score(text: str, document_text: str) -> float:
    """
    Estimate confidence heuristically when the LLM score is unavailable.

    Signals used:
      - Word count (longer → more specific)
      - Presence of digits / named entities (specificity)
      - Term overlap with document text (grounding)
      - Generic phrase detection (penalty)

    Returns:
        Float in [0.0, 0.95].
    """
    if _is_generic(text):
        return 0.20

    score = 0.45

    words = text.split()
    if len(words) >= 15:
        score += 0.08
    elif len(words) >= 10:
        score += 0.04

    if re.search(r"\d", text):
        score += 0.10

    # Named-entity proxy: capitalised words (excluding first word of sentence)
    caps = re.findall(r"(?<!\.\s)\b[A-Z][a-z]{2,}", text)
    if len(caps) >= 2:
        score += 0.08
    elif len(caps) == 1:
        score += 0.04

    # Term-overlap grounding
    text_lower = text.lower()
    doc_lower  = document_text.lower()
    content_words = [w for w in words if len(w) > 4]
    if content_words:
        overlap = sum(1 for w in content_words if w.lower() in doc_lower)
        overlap_ratio = overlap / len(content_words)
        score += overlap_ratio * 0.15

    return round(min(0.95, score), 4)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _is_generic(text: str) -> bool:
    """
    Return True if the insight text is probably generic or non-grounded.

    Checks:
      - Minimum word count
      - Presence of known generic phrases
    """
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped.split()) < _MIN_INSIGHT_WORDS:
        return True
    lower = stripped.lower()
    return any(phrase in lower for phrase in _GENERIC_PHRASES)


def _filter_insights(items: List[InsightItem]) -> List[InsightItem]:
    """
    Remove insights that are generic or fall below _MIN_CONFIDENCE.

    Args:
        items: Scored InsightItem list.

    Returns:
        Filtered list; original order preserved.
    """
    return [
        item for item in items
        if not _is_generic(item.text) and item.confidence >= _MIN_CONFIDENCE
    ]


# ---------------------------------------------------------------------------
# Regeneration helper
# ---------------------------------------------------------------------------

def _request_additional_insights(
    document_text: str,
    existing_texts: List[str],
    count: int,
) -> List[str]:
    """
    Request additional insights from the LLM to reach the minimum threshold.

    Args:
        document_text:  Source document text.
        existing_texts: Insights already accepted (to avoid duplication).
        count:          How many additional insights to request.

    Returns:
        List of new insight text strings (unscored).
    """
    existing_formatted = "\n".join(f"- {t}" for t in existing_texts)
    user_prompt = (
        _ADDITIONAL_INSIGHTS_PROMPT.format(
            min_needed=_MIN_INSIGHTS,
            count=count,
            existing=existing_formatted or "(none yet)",
        )
        + f"\n\nDOCUMENT TEXT:\n{document_text}"
    )

    try:
        raw = generate_response(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=MAX_TOKENS // 2,
            temperature=TEMPERATURE,
        )
        return _extract_bullet_list(raw)
    except Exception as exc:
        logger.error("[InsightEngine] Additional insight generation failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TOP_LEVEL_PATTERNS: Dict[str, re.Pattern[str]] = {
    "key_insights":         re.compile(r"KEY_INSIGHTS\s*:",         re.IGNORECASE),
    "suggested_questions":  re.compile(r"SUGGESTED_QUESTIONS\s*:",  re.IGNORECASE),
    "actionable_takeaways": re.compile(r"ACTIONABLE_TAKEAWAYS\s*:", re.IGNORECASE),
    "enhanced_explanation": re.compile(r"ENHANCED_EXPLANATION\s*:", re.IGNORECASE),
}

_SUBSECTION_PATTERNS: Dict[str, re.Pattern[str]] = {
    "executive":     re.compile(r"Executive\s*:",     re.IGNORECASE),
    "analytical":    re.compile(r"Analytical\s*:",    re.IGNORECASE),
    "risk":          re.compile(r"Risk\s*:",           re.IGNORECASE),
    "visualization": re.compile(r"Visualization\s*:", re.IGNORECASE),
    "leadership":    re.compile(r"Leadership\s*:",    re.IGNORECASE),
    "analyst":       re.compile(r"Analyst\s*:",       re.IGNORECASE),
    "next_steps":    re.compile(r"Next\s+Steps\s*:",  re.IGNORECASE),
}


def _parse_response(raw: str) -> InsightResult:
    """
    Parse a structured LLM response into an InsightResult.

    Key insights are returned as preliminary InsightItem objects with a
    placeholder confidence of 0.0; they are scored by the pipeline in
    generate_insights() before being returned to the caller.

    Args:
        raw: Raw string returned by generate_response.

    Returns:
        Populated InsightResult. Missing sections default to empty.
    """
    sections = _split_top_level_sections(raw)

    raw_insight_texts = _extract_bullet_list(sections.get("key_insights", ""))
    preliminary_items = [
        InsightItem(text=t, confidence=0.0)
        for t in raw_insight_texts
        if t.strip()
    ]

    return InsightResult(
        key_insights=preliminary_items,
        suggested_questions=_parse_suggested_questions(
            sections.get("suggested_questions", "")
        ),
        actionable_takeaways=_parse_actionable_takeaways(
            sections.get("actionable_takeaways", "")
        ),
        enhanced_explanation=sections.get("enhanced_explanation", "").strip(),
    )


def _split_top_level_sections(text: str) -> Dict[str, str]:
    """
    Locate each top-level section header and extract the body text that
    follows it up to the next recognised section header.

    Args:
        text: Full raw LLM response string.

    Returns:
        Dict mapping section key to its raw body text.
    """
    anchors: List[tuple[int, str]] = []
    for key, pattern in _TOP_LEVEL_PATTERNS.items():
        match = pattern.search(text)
        if match:
            anchors.append((match.end(), key))
    anchors.sort(key=lambda t: t[0])

    result: Dict[str, str] = {}
    for i, (start, key) in enumerate(anchors):
        if i + 1 < len(anchors):
            next_key     = anchors[i + 1][1]
            next_pattern = _TOP_LEVEL_PATTERNS[next_key]
            next_match   = next_pattern.search(text, start)
            end          = next_match.start() if next_match else len(text)
        else:
            end = len(text)
        result[key] = text[start:end].strip()

    return result


def _extract_bullet_list(text: str) -> List[str]:
    """
    Extract bullet-point items from a text block.

    Recognises lines beginning with ``-``, ``*``, or a digit followed
    by ``.`` or ``)``. Returns stripped, non-empty strings only.

    Args:
        text: Raw section body text.

    Returns:
        Ordered list of item strings with bullet markers removed.
    """
    items: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^[-*]\s+", "", stripped)
        cleaned = re.sub(r"^\d+[.)]\s+", "", cleaned)
        if cleaned:
            items.append(cleaned)
    return items


def _extract_subsection(text: str, key: str) -> List[str]:
    """
    Extract bullet items from a named subsection within a section body.

    Args:
        text: Full section body containing multiple subsection headers.
        key:  Subsection key from _SUBSECTION_PATTERNS.

    Returns:
        List of bullet items for that subsection, or an empty list.
    """
    pattern = _SUBSECTION_PATTERNS.get(key)
    if not pattern:
        return []

    match = pattern.search(text)
    if not match:
        return []

    start      = match.end()
    next_start = len(text)

    for other_key, other_pattern in _SUBSECTION_PATTERNS.items():
        if other_key == key:
            continue
        other_match = other_pattern.search(text, start)
        if other_match and other_match.start() < next_start:
            next_start = other_match.start()

    return _extract_bullet_list(text[start:next_start])


def _parse_suggested_questions(text: str) -> SuggestedQuestions:
    return SuggestedQuestions(
        executive=     _extract_subsection(text, "executive"),
        analytical=    _extract_subsection(text, "analytical"),
        risk=          _extract_subsection(text, "risk"),
        visualization= _extract_subsection(text, "visualization"),
    )


def _parse_actionable_takeaways(text: str) -> ActionableTakeaways:
    return ActionableTakeaways(
        leadership= _extract_subsection(text, "leadership"),
        analyst=    _extract_subsection(text, "analyst"),
        next_steps= _extract_subsection(text, "next_steps"),
    )
