"""Unit + integration tests for the cancellation primitive and orchestrator wiring."""
from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from utils.cancellation import (
    CancellationToken,
    CancellationRegistry,
    OperationCancelledError,
    get_registry,
)


def test_token_starts_unset():
    tok = CancellationToken()
    assert tok.is_set() is False


def test_cancel_sets_event():
    tok = CancellationToken()
    tok.cancel()
    assert tok.is_set() is True


def test_check_raises_on_set():
    tok = CancellationToken()
    tok.cancel()
    with pytest.raises(OperationCancelledError):
        tok.check()


def test_check_no_raise_when_unset():
    tok = CancellationToken()
    tok.check()  # must not raise


@pytest.mark.asyncio
async def test_registry_register_returns_token():
    reg = CancellationRegistry()
    tok = await reg.register("turn-1")
    assert isinstance(tok, CancellationToken)
    assert tok.is_set() is False


@pytest.mark.asyncio
async def test_registry_cancel_unknown_returns_false():
    reg = CancellationRegistry()
    assert await reg.cancel("nope") is False


@pytest.mark.asyncio
async def test_registry_release_removes_token():
    reg = CancellationRegistry()
    await reg.register("turn-x")
    await reg.release("turn-x")
    assert await reg.cancel("turn-x") is False


@pytest.mark.asyncio
async def test_registry_cancel_active_returns_true():
    reg = CancellationRegistry()
    tok = await reg.register("t1")
    assert await reg.cancel("t1") is True
    assert tok.is_set() is True


@pytest.mark.asyncio
async def test_orchestrator_run_respects_cancellation(tmp_path, monkeypatch):
    """Cancel after first planning event; assert state_trace ends with CANCELLED
    and exactly one final error/cancelled event is yielded."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    # Re-import memory after env var is set
    import importlib
    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.orchestrator as orch_mod
    importlib.reload(orch_mod)
    await mem_mod.init_db()

    from utils.cancellation import CancellationToken
    from agent.models import PlannerOutput, QueryIntent, TypedQuery

    async def fake_plan(*a, **kw):
        return PlannerOutput(
            strategy="test", queries=[TypedQuery(text="q", intent=QueryIntent.PRIMARY)]
        )

    from utils import provider_router
    monkeypatch.setattr(provider_router, "plan", fake_plan)

    token = CancellationToken()
    orch = orch_mod.ResearchOrchestrator()
    events = []
    gen = orch.run("test query", "test-session", cancel_token=token, turn_id="t-cancel")
    # Pull first event (planning placeholder), then cancel
    first = await gen.__anext__()
    events.append(first)
    token.cancel()
    async for ev in gen:
        events.append(ev)

    # Last event must be the cancellation marker
    assert events[-1].step == "error"
    assert events[-1].label == "cancelled"
    # The body must have appended CANCELLED to state_trace before persisting
    import aiosqlite
    async with aiosqlite.connect(mem_mod.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT state_trace, response FROM turns WHERE turn_id = ?", ("t-cancel",)
        )
    assert len(rows) == 1
    import json
    state_trace = json.loads(rows[0]["state_trace"])
    assert "CANCELLED" in state_trace
    assert rows[0]["response"] == "[Cancelled by user]"


@pytest.mark.asyncio
async def test_cancellation_during_synth_halts_stream():
    """If cancel_token is set after the 2nd chunk, chunks 3-5 must not be yielded."""
    from agent.synthesizer import stream_synthesis

    async def fake_synth(*a, **kw):
        for i in range(5):
            yield (f"chunk{i}", 10, 5)

    from utils import provider_router
    with patch.object(provider_router, "synthesize", fake_synth):
        tok = CancellationToken()
        collected = []
        async for c, pt, ct in stream_synthesis(
            query="q", context_xml="<x/>", doc_map={}, cancel_token=tok
        ):
            collected.append(c)
            if len(collected) == 2:
                tok.cancel()
        assert collected == ["chunk0", "chunk1"]


# ── Phase 2: extended CancellationToken (approval gate) ──────────────────


@pytest.mark.asyncio
async def test_wait_for_approval_normal_path():
    """Approval event fires before timeout → returns (True, payload)."""
    tok = CancellationToken()

    async def approver():
        await asyncio.sleep(0.05)
        tok.approved_payload = {"sub_queries": ["edited q"]}
        tok.approval_event.set()

    asyncio.create_task(approver())
    approved, payload = await tok.wait_for_approval(timeout=1.0)
    assert approved is True
    assert payload == {"sub_queries": ["edited q"]}
    assert tok.approval_status == "approved"


@pytest.mark.asyncio
async def test_wait_for_approval_timeout():
    """Approval never fires → returns (False, None) and status=timeout."""
    tok = CancellationToken()
    approved, payload = await tok.wait_for_approval(timeout=0.05)
    assert approved is False
    assert payload is None
    assert tok.approval_status == "timeout"
    assert tok.is_set() is False


@pytest.mark.asyncio
async def test_wait_for_approval_cancellation_interrupts():
    """cancel() during wait_for_approval → returns (False, None), token set."""
    tok = CancellationToken()

    async def canceller():
        await asyncio.sleep(0.05)
        tok.cancel()

    asyncio.create_task(canceller())
    approved, payload = await tok.wait_for_approval(timeout=1.0)
    assert approved is False
    assert payload is None
    assert tok.is_set() is True
    assert tok.approval_status == "cancelled"
