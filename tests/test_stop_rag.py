"""P1 — Stop-RAG adaptive hop-stopping gate.

Tests cover the gate's three behaviours:

1. Stops the hop loop when the LLM says ``useful=False`` OR ``confidence<0.5``.
2. Degrades-to-safer (continues the loop) on timeout / parse / provider errors.
3. The LLM ``reason`` field is recorded in run_metadata but NEVER appears on
   any SSE event payload (assignment compliance — no hidden CoT streaming).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid

import pytest

# Force the DB into a temp location before memory module imports.
_TMP_DB_DIR = tempfile.mkdtemp(prefix="stop_rag_test_db_")
os.environ["DB_PATH"] = os.path.join(_TMP_DB_DIR, "research.db")

from agent import orchestrator as orch_mod  # noqa: E402
from agent import stopping as stopping_mod  # noqa: E402
from agent.memory import init_db  # noqa: E402
from agent.models import (  # noqa: E402
    EVT_TERMINATOR,
    PlannerOutput,
    QueryIntent,
    TypedQuery,
)
from agent.stopping import decide_continue  # noqa: E402

# Reuse the heavy-mock scaffold from the adaptive-hop test file.
from tests.test_adaptive_hop import (  # noqa: E402
    _GateProbe,
    _install_common_mocks,
    _patch_plan,
)


_MOCK_REASON_SENTINEL = "TOPSECRETSTOPRAGREASON_SHOULD_NOT_LEAK"


@pytest.fixture(autouse=True)
def _init_test_db():
    asyncio.run(init_db())


def _patch_call_groq(monkeypatch, fake):
    """Mock the provider_router.call_groq used by agent.stopping."""
    monkeypatch.setattr("utils.provider_router.call_groq", fake)


def _low_conf_plan(query_text: str = "q1") -> PlannerOutput:
    return PlannerOutput(
        strategy="hop1",
        confidence="low",
        success_criteria=["Confirm X"],
        queries=[
            TypedQuery(
                text=query_text,
                intent=QueryIntent.PRIMARY,
                rationale="probe",
            ),
        ],
    )


async def _collect_events(query: str = "what year did X launch?") -> list:
    orch = orch_mod.ResearchOrchestrator()
    events = []
    async for ev in orch.run(query=query, session_id="s-" + uuid.uuid4().hex):
        events.append(ev)
    await orch.aclose()
    return events


# ── Unit tests: decide_continue directly ─────────────────────────────────


def test_stops_when_llm_returns_not_useful(monkeypatch):
    """useful=False with high confidence → another_hop_useful is False."""
    async def _ok(_prompt, max_tokens=80):
        return json.dumps({
            "another_hop_useful": False,
            "confidence": 0.9,
            "reason": "all criteria grounded",
        })

    _patch_call_groq(monkeypatch, _ok)

    decision = asyncio.run(decide_continue(
        query="q", hop=1, hop_evidence={"grounded": [], "open": []},
        success_criteria=["c1"],
    ))
    assert decision.another_hop_useful is False
    assert decision.confidence == pytest.approx(0.9)
    assert decision.degraded is False
    assert decision.reason == "all criteria grounded"


def test_stops_when_low_confidence(monkeypatch):
    """useful=True but confidence<0.5 → also a stop signal at the orchestrator layer."""
    async def _ok(_prompt, max_tokens=80):
        return json.dumps({
            "another_hop_useful": True,
            "confidence": 0.3,
            "reason": "uncertain",
        })

    _patch_call_groq(monkeypatch, _ok)

    decision = asyncio.run(decide_continue(
        query="q", hop=1, hop_evidence=None, success_criteria=[],
    ))
    assert decision.another_hop_useful is True
    assert decision.confidence == pytest.approx(0.3)
    assert decision.degraded is False


def test_continues_on_timeout(monkeypatch):
    async def _slow(_prompt, max_tokens=80):
        raise asyncio.TimeoutError()

    _patch_call_groq(monkeypatch, _slow)

    decision = asyncio.run(decide_continue(
        query="q", hop=1, hop_evidence=None, success_criteria=[],
    ))
    assert decision.degraded is True
    assert decision.another_hop_useful is True  # degrade-to-safer: continue
    assert decision.confidence == 1.0


def test_continues_on_parse_failure(monkeypatch):
    async def _garbage(_prompt, max_tokens=80):
        return "not json at all — definitely not parseable"

    _patch_call_groq(monkeypatch, _garbage)

    decision = asyncio.run(decide_continue(
        query="q", hop=1, hop_evidence=None, success_criteria=[],
    ))
    assert decision.degraded is True
    assert decision.another_hop_useful is True


def test_continues_on_provider_error(monkeypatch):
    async def _boom(_prompt, max_tokens=80):
        raise RuntimeError("groq is on fire")

    _patch_call_groq(monkeypatch, _boom)

    decision = asyncio.run(decide_continue(
        query="q", hop=1, hop_evidence=None, success_criteria=[],
    ))
    assert decision.degraded is True
    assert decision.another_hop_useful is True


# ── End-to-end: orchestrator wiring ──────────────────────────────────────


def test_decision_recorded_in_run_metadata(monkeypatch):
    """Every gate invocation lands one row in run_metadata.stop_rag_decisions."""
    probe = _GateProbe()
    # Low confidence + thin context → without Stop-RAG, hop 2 would fire.
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(monkeypatch, probe, plan_outputs=[_low_conf_plan(), _low_conf_plan()])

    async def _stop_rag_says_done(_prompt, max_tokens=80):
        return json.dumps({
            "another_hop_useful": False,
            "confidence": 0.95,
            "reason": _MOCK_REASON_SENTINEL,
        })

    _patch_call_groq(monkeypatch, _stop_rag_says_done)

    asyncio.run(_collect_events())

    decisions = probe.persisted_run_metadata.get("stop_rag_decisions") or []
    assert decisions, "expected at least one stop_rag_decisions entry"
    row = decisions[0]
    assert row["hop"] == 1
    assert row["useful"] is False
    assert row["confidence"] == pytest.approx(0.95)
    assert row["reason"] == _MOCK_REASON_SENTINEL
    assert row.get("degraded_reason") is None


def test_gate_short_circuits_hop_loop_on_not_useful(monkeypatch):
    """useful=False from the gate must prevent hop 2 from running."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(monkeypatch, probe, plan_outputs=[_low_conf_plan(), _low_conf_plan()])

    async def _stop_rag_says_done(_prompt, max_tokens=80):
        return json.dumps({
            "another_hop_useful": False,
            "confidence": 0.9,
            "reason": "done",
        })

    _patch_call_groq(monkeypatch, _stop_rag_says_done)

    events = asyncio.run(_collect_events())

    # Only one planner / search call should have fired.
    assert len(probe.plan_calls) == 1
    assert len(probe.search_calls) == 1
    # Terminator must say STOP_RAG_GATE mapped to EVIDENCE_SUFFICIENT.
    assert probe.persisted_run_metadata.get("terminator_fired") == "STOP_RAG_GATE"
    assert (
        probe.persisted_run_metadata.get("stop_rag_terminator_mapped")
        == "EVIDENCE_SUFFICIENT"
    )
    # SSE terminator event renders with the public reason.
    terminators = [e for e in events if getattr(e, "event_type", None) == EVT_TERMINATOR]
    assert len(terminators) == 1
    assert terminators[0].data["reason"] == "EVIDENCE_SUFFICIENT"
    assert "stop_rag confidence" in (terminators[0].data.get("detail") or "")


def test_gate_short_circuits_hop_loop_on_low_confidence(monkeypatch):
    """useful=True but confidence<0.5 maps to MARGINAL_GAIN_LOW."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(monkeypatch, probe, plan_outputs=[_low_conf_plan(), _low_conf_plan()])

    async def _stop_rag_low_conf(_prompt, max_tokens=80):
        return json.dumps({
            "another_hop_useful": True,
            "confidence": 0.2,
            "reason": "low marginal value",
        })

    _patch_call_groq(monkeypatch, _stop_rag_low_conf)

    events = asyncio.run(_collect_events())

    assert len(probe.plan_calls) == 1
    assert len(probe.search_calls) == 1
    assert (
        probe.persisted_run_metadata.get("stop_rag_terminator_mapped")
        == "MARGINAL_GAIN_LOW"
    )
    terminators = [e for e in events if getattr(e, "event_type", None) == EVT_TERMINATOR]
    assert terminators[0].data["reason"] == "MARGINAL_GAIN_LOW"


def test_degraded_decision_does_not_stop_loop(monkeypatch):
    """LLM failure → degrade-to-safer → existing hop-2 logic still runs."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            _low_conf_plan("q1"),
            PlannerOutput(
                strategy="hop2",
                confidence="medium",
                queries=[
                    TypedQuery(
                        text="latest 2026 numbers",
                        intent=QueryIntent.RECENCY_CHECK,
                    ),
                ],
            ),
        ],
    )

    async def _boom(_prompt, max_tokens=80):
        raise RuntimeError("provider down")

    _patch_call_groq(monkeypatch, _boom)

    asyncio.run(_collect_events())

    # Hop 2 ran — gate degraded and existing logic took over.
    assert len(probe.plan_calls) == 2
    decisions = probe.persisted_run_metadata.get("stop_rag_decisions") or []
    assert decisions
    assert decisions[0]["degraded_reason"] == "degraded_provider_error"
    assert decisions[0]["useful"] is True  # degrade-to-safer signal


def test_reason_not_in_sse_payload(monkeypatch):
    """The LLM `reason` must NEVER leak onto a streamed SSE event payload."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(monkeypatch, probe, plan_outputs=[_low_conf_plan(), _low_conf_plan()])

    async def _stop_rag(_prompt, max_tokens=80):
        return json.dumps({
            "another_hop_useful": False,
            "confidence": 0.9,
            "reason": _MOCK_REASON_SENTINEL,
        })

    _patch_call_groq(monkeypatch, _stop_rag)

    events = asyncio.run(_collect_events())

    # Verify the sentinel reason DID get persisted in run_metadata
    # (so we know the mock actually ran).
    decisions = probe.persisted_run_metadata.get("stop_rag_decisions") or []
    assert decisions and decisions[0]["reason"] == _MOCK_REASON_SENTINEL

    # Walk every yielded event's `.data` payload (and `.message`) and assert
    # the sentinel string never appears anywhere in the stream. This is the
    # CoT-streaming compliance gate.
    def _walk(value) -> bool:
        if isinstance(value, str):
            return _MOCK_REASON_SENTINEL in value
        if isinstance(value, dict):
            return any(_walk(v) for v in value.values()) or any(
                _walk(k) for k in value.keys()
            )
        if isinstance(value, (list, tuple, set)):
            return any(_walk(v) for v in value)
        return False

    for ev in events:
        # Inspect both event_type, message, and data — exhaustive sweep.
        assert not _walk(getattr(ev, "data", None)), (
            f"reason leaked onto event_type={getattr(ev, 'event_type', None)} "
            f"data={getattr(ev, 'data', None)!r}"
        )
        assert not _walk(getattr(ev, "message", None))
        assert not _walk(getattr(ev, "step", None))


def test_parse_decision_handles_code_fence():
    """Defensive: the parser strips ```json fences if the model adds them."""
    raw = '```json\n{"another_hop_useful": true, "confidence": 0.8, "reason": "ok"}\n```'
    decision = stopping_mod._parse_decision(raw)
    assert decision is not None
    assert decision.another_hop_useful is True
    assert decision.confidence == pytest.approx(0.8)
