"""Refactor #3 — single source of truth for in-loop hop terminator rules.

Collapses two overlapping decision points that previously lived inline in
``agent/orchestrator.py``:

  1. STOP-RAG LLM gate (``agent/stopping.decide_continue``).
  2. Deterministic adaptive-hop gate (MAX_HOPS, TOKEN_BUDGET, DIFFICULTY,
     CONFIDENCE, EVIDENCE_SUFFICIENT).

Refinement (``agent/refinement_check``) is intentionally OUT OF SCOPE — it
runs post-synthesis, a different stage from the hop loop.

Rule priority is the EXACT order the legacy orchestrator evaluated them:

  R1. STOP_RAG_GATE — only when another hop is even possible (hop+1 < max).
                       Maps to EVIDENCE_SUFFICIENT or MARGINAL_GAIN_LOW.
  R2. MAX_HOPS_REACHED — deterministic hard cap.
  R3. TOKEN_BUDGET_EXHAUSTED — cumulative > 0.75 * total_budget.
  R4. DIFFICULTY_EASY_SKIPPED — planner-tagged easy questions stop early.
  R5. CONFIDENCE_HIGH_ENOUGH — planner.confidence != "low".
  R6. EVIDENCE_SUFFICIENT — context NOT thin (selected >= 50% of web budget).

No behaviour change: see ``tests/test_termination_policy.py`` plus the
existing ``tests/test_adaptive_hop.py`` / ``tests/test_stop_rag.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

logger = logging.getLogger(__name__)


TerminatorReason = Literal[
    "MAX_HOPS_REACHED",
    "TOKEN_BUDGET_EXHAUSTED",
    "DIFFICULTY_EASY_SKIPPED",
    "CONFIDENCE_HIGH_ENOUGH",
    "EVIDENCE_SUFFICIENT",
    "STOP_RAG_GATE",
    "MARGINAL_GAIN_LOW",
    "CONTINUE",
]
TerminatorSource = Literal["deterministic", "stop_rag"]


@dataclass(frozen=True)
class TerminationDecision:
    """Outcome of one policy evaluation."""

    should_terminate: bool
    reason: TerminatorReason
    source: TerminatorSource
    # For STOP_RAG_GATE outcomes only: the legacy ``stop_rag_terminator_mapped``
    # value ("EVIDENCE_SUFFICIENT" or "MARGINAL_GAIN_LOW"). None for
    # deterministic decisions and CONTINUE.
    stop_rag_mapped: str | None = None
    detail: str | None = None
    # Echoed StopDecision (used by orchestrator for run_metadata logging).
    stop_rag_decision: object | None = None


@dataclass(frozen=True)
class HopState:
    """Snapshot of hop-level state evaluated by :func:`decide_continuation`.

    Field semantics mirror the legacy orchestrator block 1:1; do not add
    derived fields here — keep the dataclass a passive data holder.
    """

    hop_index: int  # 0-based: 0 == first hop just completed
    max_hops: int
    cumulative_tokens: int  # prompt+completion + max(all_chunks, selected)
    token_budget_total: int
    selected_tokens: int
    web_context_budget: int
    planner_confidence: str  # "low" | "medium" | "high"
    difficulty: str  # "easy" | "medium" | "hard"


# A "stop-rag caller" is an async no-arg function returning a StopDecision.
# We keep the type loose (``object``) so this module doesn't import
# ``agent.stopping`` and create a cycle.
StopRagCaller = Callable[[], Awaitable[object]]


async def decide_continuation(
    state: HopState,
    stop_rag_caller: StopRagCaller | None = None,
) -> TerminationDecision:
    """Evaluate the prioritized termination rules. Highest priority wins.

    The STOP-RAG rule runs FIRST but only when another hop is even
    possible (``hop_index + 1 < max_hops``); this matches the legacy
    orchestrator gate, which short-circuited stop-rag on the last hop
    so MAX_HOPS_REACHED could fire deterministically.
    """
    # R1: STOP-RAG LLM gate (only if another hop is even possible).
    if stop_rag_caller is not None and state.hop_index + 1 < state.max_hops:
        try:
            decision = await stop_rag_caller()
        except Exception as exc:  # noqa: BLE001 — degrade-to-safer
            logger.warning("termination_policy: stop_rag caller raised: %s", exc)
            decision = None

        if decision is not None and not getattr(decision, "degraded", True):
            useful = bool(getattr(decision, "another_hop_useful", True))
            confidence = float(getattr(decision, "confidence", 1.0))
            stop_now = (not useful) or confidence < 0.5
            if stop_now:
                # Preserve the legacy "mapped" reason: useful=False signals
                # sufficiency; confidence<0.5 signals marginal gain.
                mapped = "EVIDENCE_SUFFICIENT" if not useful else "MARGINAL_GAIN_LOW"
                return TerminationDecision(
                    should_terminate=True,
                    reason="STOP_RAG_GATE",
                    source="stop_rag",
                    stop_rag_mapped=mapped,
                    detail=f"stop_rag confidence {confidence:.2f}",
                    stop_rag_decision=decision,
                )
        # Either degraded, missing, or "continue" verdict — fall through.
        # Preserve the (possibly-degraded) decision so the caller can log it.
        passthrough_decision = decision
    else:
        passthrough_decision = None

    # R2: MAX_HOPS hard cap.
    if state.hop_index + 1 >= state.max_hops:
        return TerminationDecision(
            should_terminate=True,
            reason="MAX_HOPS_REACHED",
            source="deterministic",
            detail=f"hop {state.hop_index + 1}/{state.max_hops}",
            stop_rag_decision=passthrough_decision,
        )

    # R3: token-budget exhaustion (cumulative > 0.75 * total).
    token_threshold = int(0.75 * state.token_budget_total)
    if state.cumulative_tokens > token_threshold:
        return TerminationDecision(
            should_terminate=True,
            reason="TOKEN_BUDGET_EXHAUSTED",
            source="deterministic",
            detail=f"{state.cumulative_tokens}/{state.token_budget_total}",
            stop_rag_decision=passthrough_decision,
        )

    # R4: difficulty=easy short-circuit.
    if state.difficulty == "easy":
        return TerminationDecision(
            should_terminate=True,
            reason="DIFFICULTY_EASY_SKIPPED",
            source="deterministic",
            stop_rag_decision=passthrough_decision,
        )

    # R5: planner confidence is good enough.
    if state.planner_confidence != "low":
        return TerminationDecision(
            should_terminate=True,
            reason="CONFIDENCE_HIGH_ENOUGH",
            source="deterministic",
            stop_rag_decision=passthrough_decision,
        )

    # R6: enough evidence selected (context NOT thin).
    context_thin = state.selected_tokens < int(0.5 * state.web_context_budget)
    if not context_thin:
        return TerminationDecision(
            should_terminate=True,
            reason="EVIDENCE_SUFFICIENT",
            source="deterministic",
            stop_rag_decision=passthrough_decision,
        )

    # No rule fired → caller runs another hop.
    return TerminationDecision(
        should_terminate=False,
        reason="CONTINUE",
        source="deterministic",
        stop_rag_decision=passthrough_decision,
    )
