"""Unit tests for agent.termination_policy.

One test per priority rule + CONTINUE + stop-rag-degraded fallthrough.
The orchestrator-level integration tests live in
``tests/test_adaptive_hop.py`` / ``tests/test_stop_rag.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.termination_policy import (
    HopState,
    TerminationDecision,
    decide_continuation,
)


# A minimal stand-in for agent.stopping.StopDecision (keeps tests
# decoupled from the real dataclass so refactors there don't ripple).
@dataclass(frozen=True)
class _FakeStopDecision:
    another_hop_useful: bool
    confidence: float
    reason: str
    degraded: bool


def _state(
    *,
    hop_index: int = 0,
    max_hops: int = 2,
    cumulative_tokens: int = 0,
    token_budget_total: int = 16_000,
    selected_tokens: int = 5_000,
    web_context_budget: int = 6_400,
    planner_confidence: str = "low",
    difficulty: str = "medium",
) -> HopState:
    """All-low-priority defaults: no rule should fire unless explicitly set."""
    return HopState(
        hop_index=hop_index,
        max_hops=max_hops,
        cumulative_tokens=cumulative_tokens,
        token_budget_total=token_budget_total,
        selected_tokens=selected_tokens,
        web_context_budget=web_context_budget,
        planner_confidence=planner_confidence,
        difficulty=difficulty,
    )


@pytest.mark.asyncio
async def test_stop_rag_fires_marginal_gain_low_when_low_confidence():
    """R1: useful=True but confidence<0.5 → STOP_RAG_GATE / MARGINAL_GAIN_LOW."""
    async def caller():
        return _FakeStopDecision(
            another_hop_useful=True, confidence=0.3,
            reason="meh", degraded=False,
        )
    out = await decide_continuation(_state(hop_index=0, max_hops=2), stop_rag_caller=caller)
    assert out.should_terminate
    assert out.reason == "STOP_RAG_GATE"
    assert out.source == "stop_rag"
    assert out.stop_rag_mapped == "MARGINAL_GAIN_LOW"


@pytest.mark.asyncio
async def test_stop_rag_fires_evidence_sufficient_when_not_useful():
    """R1: useful=False → STOP_RAG_GATE / EVIDENCE_SUFFICIENT."""
    async def caller():
        return _FakeStopDecision(
            another_hop_useful=False, confidence=0.95,
            reason="done", degraded=False,
        )
    out = await decide_continuation(_state(hop_index=0, max_hops=3), stop_rag_caller=caller)
    assert out.should_terminate
    assert out.source == "stop_rag"
    assert out.stop_rag_mapped == "EVIDENCE_SUFFICIENT"


@pytest.mark.asyncio
async def test_max_hops_reached_when_on_last_hop_no_stop_rag():
    """R2: on the last hop, stop-rag is skipped and MAX_HOPS_REACHED fires."""
    async def caller():  # pragma: no cover — must not be invoked
        raise AssertionError("stop_rag must not run on last hop")
    out = await decide_continuation(
        _state(hop_index=1, max_hops=2), stop_rag_caller=caller,
    )
    assert out.should_terminate
    assert out.reason == "MAX_HOPS_REACHED"
    assert out.source == "deterministic"


@pytest.mark.asyncio
async def test_token_budget_exhausted():
    """R3: cumulative > 0.75 * total_budget → TOKEN_BUDGET_EXHAUSTED."""
    out = await decide_continuation(
        _state(
            hop_index=0, max_hops=3,
            cumulative_tokens=13_000, token_budget_total=16_000,
        ),
        stop_rag_caller=None,
    )
    assert out.should_terminate
    assert out.reason == "TOKEN_BUDGET_EXHAUSTED"
    assert out.source == "deterministic"


@pytest.mark.asyncio
async def test_difficulty_easy_skipped():
    """R4: difficulty=easy stops after hop 1."""
    out = await decide_continuation(
        _state(hop_index=0, max_hops=3, difficulty="easy"),
        stop_rag_caller=None,
    )
    assert out.should_terminate
    assert out.reason == "DIFFICULTY_EASY_SKIPPED"


@pytest.mark.asyncio
async def test_confidence_high_enough():
    """R5: planner.confidence != "low" → CONFIDENCE_HIGH_ENOUGH."""
    out = await decide_continuation(
        _state(hop_index=0, max_hops=3, planner_confidence="high"),
        stop_rag_caller=None,
    )
    assert out.should_terminate
    assert out.reason == "CONFIDENCE_HIGH_ENOUGH"


@pytest.mark.asyncio
async def test_evidence_sufficient_when_context_not_thin():
    """R6: selected_tokens >= 50% of web budget → EVIDENCE_SUFFICIENT."""
    out = await decide_continuation(
        _state(
            hop_index=0, max_hops=3,
            selected_tokens=4_000, web_context_budget=6_400,  # 4000 >= 3200
            planner_confidence="low", difficulty="medium",
        ),
        stop_rag_caller=None,
    )
    assert out.should_terminate
    assert out.reason == "EVIDENCE_SUFFICIENT"


@pytest.mark.asyncio
async def test_continue_when_no_rule_fires():
    """No rule fires → should_terminate=False, reason=CONTINUE."""
    out = await decide_continuation(
        _state(
            hop_index=0, max_hops=3,
            cumulative_tokens=100, selected_tokens=1_000,
            planner_confidence="low", difficulty="medium",
        ),
        stop_rag_caller=None,
    )
    assert not out.should_terminate
    assert out.reason == "CONTINUE"


@pytest.mark.asyncio
async def test_stop_rag_degraded_falls_through_to_deterministic():
    """A degraded stop-rag decision must not terminate, but its
    StopDecision should be echoed back for logging."""
    async def caller():
        return _FakeStopDecision(
            another_hop_useful=True, confidence=1.0,
            reason="degraded_timeout", degraded=True,
        )
    out = await decide_continuation(
        _state(
            hop_index=0, max_hops=3,
            cumulative_tokens=100, selected_tokens=1_000,
            planner_confidence="low", difficulty="medium",
        ),
        stop_rag_caller=caller,
    )
    assert not out.should_terminate
    assert out.reason == "CONTINUE"
    assert getattr(out.stop_rag_decision, "degraded", False) is True


@pytest.mark.asyncio
async def test_stop_rag_caller_exception_does_not_raise():
    """Exceptions from the stop-rag caller must degrade gracefully."""
    async def caller():
        raise RuntimeError("boom")
    out = await decide_continuation(
        _state(hop_index=0, max_hops=3, planner_confidence="high"),
        stop_rag_caller=caller,
    )
    # Falls through; deterministic rule fires.
    assert out.should_terminate
    assert out.reason == "CONFIDENCE_HIGH_ENOUGH"


@pytest.mark.asyncio
async def test_priority_stop_rag_beats_deterministic_when_both_would_fire():
    """If both stop-rag AND a deterministic rule would fire, stop-rag wins
    (it's evaluated first, as it was in the legacy orchestrator)."""
    async def caller():
        return _FakeStopDecision(
            another_hop_useful=False, confidence=0.9,
            reason="done", degraded=False,
        )
    out = await decide_continuation(
        _state(
            hop_index=0, max_hops=3,
            # Deterministic rules would also fire here:
            cumulative_tokens=15_000, planner_confidence="high",
            difficulty="easy",
        ),
        stop_rag_caller=caller,
    )
    assert out.reason == "STOP_RAG_GATE"
    assert out.source == "stop_rag"


def test_decision_is_frozen():
    """Immutability check (CLAUDE.md)."""
    d = TerminationDecision(True, "MAX_HOPS_REACHED", "deterministic")
    with pytest.raises((AttributeError, Exception)):
        d.reason = "CONTINUE"  # type: ignore[misc]
