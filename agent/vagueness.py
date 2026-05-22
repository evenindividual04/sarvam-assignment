"""Vagueness scoring for clarification gating (F7).

Competitor `goku1610/sarvam` always fires three clarifying chips. Our approach
is restrained: emit a single clarifying question only when the query is
genuinely vague.

The score combines four pure-Python signals computed from the user query text
and the planner output. No external API calls. Thresholded at 0.55 to gate the
``clarification_offered`` event.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent.models import PlannerOutput

# ── Weights (must sum to 1.0; see test_signal_weights_sum_to_one) ────────────
W_ENTITY_COUNT_INV = 0.30
W_QUERY_LENGTH_INV = 0.20
W_AMBIGUITY_FLAG = 0.30
W_WH_BREADTH = 0.20
SIGNAL_WEIGHTS: dict[str, float] = {
    "entity_count_inv": W_ENTITY_COUNT_INV,
    "query_length_inv": W_QUERY_LENGTH_INV,
    "ambiguity_flag": W_AMBIGUITY_FLAG,
    "wh_breadth": W_WH_BREADTH,
}

THRESHOLD: float = 0.55

# ── Regex signals ────────────────────────────────────────────────────────────
# Capitalized multi-word proper nouns (e.g., "World Bank", "Reserve Bank of India").
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+(?:of\s+|the\s+)?[A-Z][a-zA-Z]+)+)\b")
# Single capitalized tokens that aren't sentence-initial filler.
_SINGLE_PROPER_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-zA-Z]{2,})\b")
# Years and pure numbers.
_NUMBER_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b")
# Bare interrogative openings without an immediate constraint clause.
_BARE_WH_RE = re.compile(
    r"^\s*(?:what\s+is|what\s+are|how\s+(?:to|do|does|can)|tell\s+me\s+about|explain|describe)\b",
    re.IGNORECASE,
)
# Constraint markers that, when present, mean the query is NOT just a bare wh-question.
_CONSTRAINT_RE = re.compile(
    r"\b(?:in\s+\d{4}|since\s+\d{4}|per\s+|according\s+to|by\s+\w+|of\s+\w+|"
    r"between|compared|vs\.?|versus|under\s+|for\s+\w+|with\s+respect)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VaguenessScore:
    """Outcome of vagueness scoring.

    Attributes:
        score: weighted vagueness in [0.0, 1.0]; higher = vaguer.
        gate: True iff `score >= THRESHOLD` AND `suggested_question is not None`.
        signals: per-signal raw scores keyed by the four signal names.
        suggested_question: a single clarifying question derived from the
            planner's `success_criteria`; None when the planner gave us nothing
            usable (in which case `gate` is forced False).
        suggested_options: up to 4 short option strings (≤60 chars each).
    """

    score: float
    gate: bool
    signals: dict[str, float]
    suggested_question: Optional[str]
    suggested_options: list[str]


# ── Signal computations ──────────────────────────────────────────────────────


def _count_entities(query: str) -> int:
    """Count capitalized proper nouns, multi-word entities, years, and numbers.

    Multi-word matches consume their tokens first so they aren't double-counted
    by the single-word pass.
    """
    if not query:
        return 0
    multi = _PROPER_NOUN_RE.findall(query)
    remainder = _PROPER_NOUN_RE.sub(" ", query)
    singles = _SINGLE_PROPER_RE.findall(remainder)
    numbers = _NUMBER_RE.findall(query)
    # Strip a leading "What/How/Tell" first token from single-proper hits.
    return len(multi) + len(singles) + len(numbers)


def _entity_count_inv(query: str) -> float:
    n = _count_entities(query)
    return max(0.0, 1.0 - n / 3.0)


def _query_length_inv(query: str) -> float:
    tokens = (query or "").split()
    return max(0.0, 1.0 - len(tokens) / 12.0)


def _ambiguity_flag(planner: Optional[PlannerOutput]) -> float:
    if planner is None:
        return 0.0
    return 1.0 if bool(getattr(planner, "ambiguity_flag", False)) else 0.0


def _wh_breadth(query: str) -> float:
    if not query:
        return 0.0
    if not _BARE_WH_RE.search(query):
        return 0.0
    # Bare wh-opening — vague unless a downstream constraint clause appears.
    if _CONSTRAINT_RE.search(query):
        return 0.0
    return 1.0


# ── Suggested question + options ─────────────────────────────────────────────


def _truncate(s: str, limit: int = 60) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _build_suggestion(
    planner: Optional[PlannerOutput],
) -> tuple[Optional[str], list[str]]:
    """Derive a single clarifying question + ≤4 options from success_criteria.

    Empty criteria → (None, []). The orchestrator MUST skip emitting a
    clarification in that case — a question with no options is noise.
    """
    if planner is None:
        return None, []
    criteria = list(getattr(planner, "success_criteria", []) or [])
    options = [_truncate(c) for c in criteria if c and c.strip()][:4]
    if not options:
        return None, []
    question = "Which of these did you mean to focus on?"
    return question, options


# ── Public API ───────────────────────────────────────────────────────────────


def score(query: str, planner: Optional[PlannerOutput]) -> VaguenessScore:
    """Compute a `VaguenessScore` for a query + planner output pair.

    Pure: no I/O, no API calls. Safe to call inside the orchestrator hot path.
    """
    s_entity = _entity_count_inv(query)
    s_length = _query_length_inv(query)
    s_ambiguity = _ambiguity_flag(planner)
    s_wh = _wh_breadth(query)

    signals = {
        "entity_count_inv": s_entity,
        "query_length_inv": s_length,
        "ambiguity_flag": s_ambiguity,
        "wh_breadth": s_wh,
    }
    total = (
        W_ENTITY_COUNT_INV * s_entity
        + W_QUERY_LENGTH_INV * s_length
        + W_AMBIGUITY_FLAG * s_ambiguity
        + W_WH_BREADTH * s_wh
    )
    total = max(0.0, min(1.0, total))

    suggested_question, suggested_options = _build_suggestion(planner)
    gate = (total >= THRESHOLD) and (suggested_question is not None)

    return VaguenessScore(
        score=total,
        gate=gate,
        signals=signals,
        suggested_question=suggested_question,
        suggested_options=suggested_options,
    )
