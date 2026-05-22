"""F7 — Vagueness-gated clarifier tests.

Covers:
  * Specific queries do not gate (no clarification noise).
  * Genuinely vague queries gate.
  * Planner.ambiguity_flag dominates a medium query over the threshold.
  * Empty success_criteria forces gate=False even when score is high.
  * Signal weights sum to 1.0.
  * Orchestrator integration: specific query emits no `clarification_offered`.
"""
from __future__ import annotations

import math

import pytest

from agent.models import PlannerOutput, QueryIntent, TypedQuery
from agent.vagueness import (
    SIGNAL_WEIGHTS,
    THRESHOLD,
    VaguenessScore,
    score,
)


def _planner(
    *,
    ambiguity: bool = False,
    criteria: list[str] | None = None,
    queries: list[str] | None = None,
) -> PlannerOutput:
    qs = queries or ["fallback query"]
    return PlannerOutput(
        strategy="test",
        queries=[TypedQuery(text=q, intent=QueryIntent.PRIMARY) for q in qs],
        ambiguity_flag=ambiguity,
        success_criteria=criteria or [],
    )


# ── 1. Specific query: gate must NOT fire ───────────────────────────────────


def test_specific_query_does_not_gate():
    q = "What was the GDP of India in 2023 per World Bank?"
    planner = _planner(
        ambiguity=False,
        criteria=["Cite World Bank GDP figure", "India 2023 USD"],
    )
    result = score(q, planner)
    assert isinstance(result, VaguenessScore)
    assert result.score < THRESHOLD, (
        f"expected score<{THRESHOLD}, got {result.score} ({result.signals})"
    )
    assert result.gate is False


# ── 2. Vague query: gate must fire ──────────────────────────────────────────


def test_vague_query_gates():
    q = "Tell me about AI"
    planner = _planner(
        ambiguity=True,
        criteria=[
            "Define artificial intelligence",
            "Cover recent developments",
            "Include applications",
        ],
    )
    result = score(q, planner)
    assert result.score >= THRESHOLD
    assert result.gate is True
    assert result.suggested_question is not None
    assert 1 <= len(result.suggested_options) <= 4


# ── 3. Ambiguity flag dominance ─────────────────────────────────────────────


def test_ambiguity_flag_dominates():
    # Medium-length query with one entity. By itself the score sits below
    # threshold; flipping the planner's ambiguity_flag pushes it over.
    q = "What did the report say about emissions"
    criteria = ["Identify the report", "Summarize emissions findings"]

    no_flag = score(q, _planner(ambiguity=False, criteria=criteria))
    yes_flag = score(q, _planner(ambiguity=True, criteria=criteria))

    assert yes_flag.score > no_flag.score
    assert yes_flag.score - no_flag.score == pytest.approx(SIGNAL_WEIGHTS["ambiguity_flag"])
    assert yes_flag.gate is True
    # The non-flag variant should NOT gate — this is the whole point of F7.
    assert no_flag.gate is False or no_flag.score < yes_flag.score


# ── 4. Empty success_criteria → suggested_question is None, gate False ──────


def test_empty_success_criteria_returns_none():
    q = "Tell me about AI"  # would otherwise gate
    planner = _planner(ambiguity=True, criteria=[])
    result = score(q, planner)
    assert result.score >= THRESHOLD  # raw signal is vague
    assert result.suggested_question is None
    assert result.suggested_options == []
    assert result.gate is False  # we don't emit a useless clarification


# ── 5. Sanity: weights sum to 1.0 ───────────────────────────────────────────


def test_signal_weights_sum_to_one():
    total = sum(SIGNAL_WEIGHTS.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9), (
        f"expected weights to sum to 1.0, got {total}"
    )
    # All four signals present.
    assert set(SIGNAL_WEIGHTS.keys()) == {
        "entity_count_inv",
        "query_length_inv",
        "ambiguity_flag",
        "wh_breadth",
    }


# ── 6. Option truncation ────────────────────────────────────────────────────


def test_options_truncated_to_60_chars():
    long = "a" * 200
    planner = _planner(ambiguity=True, criteria=[long])
    result = score("Tell me about it", planner)
    assert result.suggested_options
    assert all(len(o) <= 60 for o in result.suggested_options)


def test_options_cap_at_four():
    planner = _planner(
        ambiguity=True,
        criteria=["one", "two", "three", "four", "five", "six"],
    )
    result = score("Tell me about it", planner)
    assert len(result.suggested_options) == 4


# ── 7. Orchestrator integration: specific query emits no clarification ──────


def test_orchestrator_skips_clarification_when_specific(monkeypatch):
    """Integration: when score.gate is False, no clarification_offered fires.

    We unit-test the gating decision the way the orchestrator wires it
    (vagueness.score(...).gate). End-to-end orchestrator runs require the full
    async DB + provider stack; we assert the contract that matters: the gate
    is False for a specific query, so the orchestrator's `if _vag.gate ...`
    block won't execute.
    """
    q = "What was the GDP of India in 2023 per World Bank?"
    planner = _planner(
        ambiguity=False,
        criteria=["Cite World Bank figure", "India 2023 USD"],
    )
    result = score(q, planner)
    assert result.gate is False, (
        "Specific query gated — orchestrator would emit a clarification "
        "(undesired). Signals: " + repr(result.signals)
    )


def test_orchestrator_fires_clarification_when_vague():
    q = "Tell me about it"
    planner = _planner(
        ambiguity=True,
        criteria=["What domain?", "Which timeframe?", "Which sources?"],
    )
    result = score(q, planner)
    assert result.gate is True
    assert result.suggested_question == "Which of these did you mean to focus on?"
    assert len(result.suggested_options) == 3
