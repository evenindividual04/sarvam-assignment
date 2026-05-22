"""B3 — Retrieval-grounded reasoning events.

Asserts that the orchestrator emits structured `reasoning` SSE events whose
payloads come from planner JSON (`intent` phase) and selected chunk metadata
(`observation` phase) — NEVER from model-streamed chain-of-thought.

Two events per hop:
  1. After SEARCHING — `phase="intent"` with planner rationales.
  2. After SELECTING — `phase="observation"` with title/domain/score.

Graceful skip: when no TypedQuery carries a rationale, the intent event is
suppressed (anti-CoT discipline: never invent prose).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid

import pytest

_TMP_DB_DIR = tempfile.mkdtemp(prefix="reasoning_events_test_db_")
os.environ["DB_PATH"] = os.path.join(_TMP_DB_DIR, "research.db")

from agent import orchestrator as orch_mod  # noqa: E402
from agent.memory import init_db  # noqa: E402
from agent.models import EVT_REASONING, PlannerOutput, QueryIntent, TypedQuery  # noqa: E402
from utils import provider_router  # noqa: E402

# Reuse the heavy-mock scaffold from the adaptive-hop suite — it already
# stubs every external boundary in the orchestrator.
from tests.test_adaptive_hop import (  # noqa: E402
    _GateProbe,
    _install_common_mocks,
    _patch_plan,
)


@pytest.fixture(autouse=True)
def _init_test_db():
    asyncio.run(init_db())


async def _collect_events(query: str = "test query") -> list:
    orch = orch_mod.ResearchOrchestrator()
    events = []
    async for ev in orch.run(query=query, session_id="s-" + uuid.uuid4().hex):
        events.append(ev)
    await orch.aclose()
    return events


def _reasoning_events(events: list) -> list:
    return [e for e in events if getattr(e, "event_type", None) == EVT_REASONING]


def test_emits_two_reasoning_events_per_hop_with_planner_rationale(monkeypatch):
    """Single-hop run with rationales present → exactly 2 reasoning events."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[4000])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="high",  # blocks hop-2
                queries=[
                    TypedQuery(
                        text="q1",
                        intent=QueryIntent.PRIMARY,
                        rationale="probe canonical sources for baseline definition",
                    ),
                ],
            ),
        ],
    )

    events = asyncio.run(_collect_events())
    reasoning = _reasoning_events(events)

    assert len(reasoning) == 2, f"expected 2 reasoning events for 1 hop, got {len(reasoning)}"

    intent_ev, obs_ev = reasoning
    assert intent_ev.data["phase"] == "intent"
    assert intent_ev.data["hop"] == 1
    assert intent_ev.data["queries"][0]["text"] == "q1"
    assert intent_ev.data["queries"][0]["intent"] == "primary"
    assert "canonical sources" in intent_ev.data["queries"][0]["rationale"]

    assert obs_ev.data["phase"] == "observation"
    assert obs_ev.data["hop"] == 1
    assert len(obs_ev.data["observation"]) >= 1
    row = obs_ev.data["observation"][0]
    assert {"title", "domain", "score"}.issubset(row.keys())
    assert isinstance(row["score"], float)


def test_intent_event_skipped_when_planner_omits_rationale(monkeypatch):
    """Anti-CoT discipline: no rationale → no intent event (never invented)."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[4000])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="high",
                queries=[
                    TypedQuery(text="q1", intent=QueryIntent.PRIMARY, rationale=None),
                ],
            ),
        ],
    )

    events = asyncio.run(_collect_events())
    reasoning = _reasoning_events(events)

    # Only the observation event survives; intent is gracefully suppressed.
    assert len(reasoning) == 1
    assert reasoning[0].data["phase"] == "observation"


def test_two_hops_emit_four_reasoning_events(monkeypatch):
    """Two hops with rationales → 4 reasoning events (2 per hop)."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="low",  # triggers hop 2 (with thin context)
                queries=[
                    TypedQuery(
                        text="q1", intent=QueryIntent.PRIMARY,
                        rationale="primary lookup",
                    ),
                ],
            ),
            PlannerOutput(
                strategy="hop2",
                confidence="medium",
                queries=[
                    TypedQuery(
                        text="q2", intent=QueryIntent.RECENCY_CHECK,
                        rationale="check for newer data",
                    ),
                ],
            ),
        ],
    )

    events = asyncio.run(_collect_events())
    reasoning = _reasoning_events(events)

    assert len(reasoning) == 4
    phases = [(e.data["hop"], e.data["phase"]) for e in reasoning]
    assert phases == [(1, "intent"), (1, "observation"), (2, "intent"), (2, "observation")]


def test_reasoning_payloads_contain_no_freeform_cot_markers(monkeypatch):
    """Defense in depth: payloads must not contain CoT-streaming markers."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[4000])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="high",
                queries=[
                    TypedQuery(
                        text="q1", intent=QueryIntent.PRIMARY,
                        rationale="structured rationale only",
                    ),
                ],
            ),
        ],
    )

    events = asyncio.run(_collect_events())
    reasoning = _reasoning_events(events)
    import json

    forbidden = ("<thinking>", "thought_summary", "reasoning_content", "I think", "Let me")
    for ev in reasoning:
        blob = json.dumps(ev.data, default=str)
        for marker in forbidden:
            assert marker not in blob, f"CoT marker {marker!r} leaked into reasoning payload"
