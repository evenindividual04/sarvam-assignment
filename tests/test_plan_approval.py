"""Phase 2: plan-approval gate — orchestrator + HTTP endpoint coverage.

Tests live in their own module (rather than appended to test_api.py) so a
single shared event loop + a per-test mocked orchestrator path keep the
assertions isolated and fast. Heavyweight pipeline stages (search/fetch/
synth) are stubbed — we're testing the *gate*, not the agent.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent.models import (
    EVT_PLAN_APPROVAL,
    PlannerOutput,
    QueryIntent,
    TypedQuery,
)


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    """Fresh DB + reloaded main module per test (test_api uses a module-scoped
    fixture — we need per-test isolation here because the approval registry
    lives in-process)."""
    db = tmp_path / "approval.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")

    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.eval_queries as eq_mod
    importlib.reload(eq_mod)
    import main as main_mod
    importlib.reload(main_mod)

    asyncio.run(mem_mod.init_db())
    return TestClient(main_mod.app), main_mod, mem_mod


# ── Stub planner & downstream stages so tests focus on the gate ─────────


async def _fake_plan(query, prior_summary=""):
    return PlannerOutput(
        strategy="stub strategy",
        queries=[
            TypedQuery(text="q one", intent=QueryIntent.PRIMARY),
            TypedQuery(text="q two", intent=QueryIntent.COMPARISON),
        ],
    )


# ── Endpoint tests ──────────────────────────────────────────────────────


def test_approval_endpoint_404_for_unknown_turn(isolated_app):
    client, _, _ = isolated_app
    r = client.post("/research/approve/unknown-turn", json={})
    assert r.status_code == 404


def test_approval_endpoint_409_when_not_waiting(isolated_app):
    """A token exists but its approval gate isn't open yet — 409, not 200."""
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    asyncio.run(registry.register("t-idle"))
    r = client.post("/research/approve/t-idle", json={})
    assert r.status_code == 409
    assert "status=idle" in r.json()["detail"]


def test_approval_endpoint_rejects_too_many_sub_queries(isolated_app):
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-too-many"))
    tok.approval_status = "waiting"
    payload = {"sub_queries": [f"q{i}" for i in range(10)]}
    r = client.post("/research/approve/t-too-many", json=payload)
    assert r.status_code == 400
    assert "too many" in r.json()["detail"].lower()


def test_approval_endpoint_rejects_empty_after_strip(isolated_app):
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-blank"))
    tok.approval_status = "waiting"
    r = client.post("/research/approve/t-blank", json={"sub_queries": ["   "]})
    assert r.status_code == 400


def test_approval_endpoint_strips_html(isolated_app):
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-html"))
    tok.approval_status = "waiting"
    r = client.post(
        "/research/approve/t-html",
        json={"sub_queries": ["<b>bold</b> real query"]},
    )
    assert r.status_code == 200
    body = r.json()
    # HTML tags are stripped; inner text is preserved (it's just opaque text
    # to us anyway — we never render the sub_queries as HTML).
    assert body["sub_queries"] == ["bold real query"]


def test_approval_endpoint_accept_as_is(isolated_app):
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-as-is"))
    tok.approval_status = "waiting"
    r = client.post("/research/approve/t-as-is", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["approved"] is True
    assert body["edited"] is False
    assert tok.approval_event.is_set() is True
    assert tok.approved_payload is None


# ── S1 security fix: session_id binding on /cancel and /approve ─────────


def test_approve_rejects_wrong_session(isolated_app):
    """A turn registered to session A must not be approvable by session B —
    even if B knows the turn_id (which leaks in the first SSE frame)."""
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-owned", session_id="session-a"))
    tok.approval_status = "waiting"
    # Wrong session in body — must look identical to "turn doesn't exist" (404).
    r = client.post(
        "/research/approve/t-owned",
        json={"session_id": "session-b"},
    )
    assert r.status_code == 404
    # The gate must still be open for the legitimate owner.
    assert tok.approval_event.is_set() is False


def test_approve_with_correct_session_succeeds(isolated_app):
    """The legitimate session owner can still approve normally."""
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-owned-ok", session_id="session-a"))
    tok.approval_status = "waiting"
    r = client.post(
        "/research/approve/t-owned-ok",
        json={"session_id": "session-a"},
    )
    assert r.status_code == 200
    assert tok.approval_event.is_set() is True


def test_cancel_rejects_wrong_session(isolated_app):
    """A cancel call with a mismatched session_id must 404 and not set the token."""
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-cancel-owned", session_id="session-a"))
    r = client.post(
        "/research/cancel/t-cancel-owned",
        json={"session_id": "session-b"},
    )
    assert r.status_code == 404
    assert tok.is_set() is False


def test_cancel_with_correct_session_succeeds(isolated_app):
    """The legitimate session owner can cancel."""
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-cancel-ok", session_id="session-a"))
    r = client.post(
        "/research/cancel/t-cancel-ok",
        json={"session_id": "session-a"},
    )
    assert r.status_code == 200
    assert tok.is_set() is True


def test_approval_endpoint_409_for_already_approved(isolated_app):
    client, main_mod, _ = isolated_app
    registry = main_mod.get_registry()
    tok = asyncio.run(registry.register("t-twice"))
    tok.approval_status = "waiting"
    r1 = client.post("/research/approve/t-twice", json={})
    assert r1.status_code == 200
    # Simulate the orchestrator post-await status transition.
    tok.approval_status = "approved"
    r2 = client.post("/research/approve/t-twice", json={})
    assert r2.status_code == 409


# ── Orchestrator integration tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_approval_gate_off_by_default_no_pause(tmp_path, monkeypatch):
    """approval_required=False → no plan_approval event, run completes."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "off.db"))
    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.orchestrator as orch_mod
    importlib.reload(orch_mod)
    await mem_mod.init_db()

    from utils import provider_router
    monkeypatch.setattr(provider_router, "plan", _fake_plan)

    # Stub the search/synthesis chain so we don't hit the network.
    async def fake_search(typed_queries, *a, **kw):
        return []

    monkeypatch.setattr(orch_mod, "search", fake_search, raising=False)

    from utils.cancellation import CancellationToken
    cfg = orch_mod.RuntimeConfig.from_overrides(None, approval_required=False)
    assert cfg.approval_required is False

    tok = CancellationToken()
    orch = orch_mod.ResearchOrchestrator()
    events = []
    async for ev in orch.run(
        "q", "s-off", cancel_token=tok, turn_id="t-off", config=cfg,
    ):
        events.append(ev)
        if ev.step == "done" or ev.step == "error":
            break

    types = [ev.event_type for ev in events if ev.event_type]
    assert EVT_PLAN_APPROVAL not in types


@pytest.mark.asyncio
async def test_approval_gate_blocks_then_resumes_with_edit(tmp_path, monkeypatch):
    """approval_required=True → pause, simulate approve with edited sub_queries,
    orchestrator splices them in and continues."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "blocks.db"))
    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.orchestrator as orch_mod
    importlib.reload(orch_mod)
    await mem_mod.init_db()

    from utils import provider_router
    monkeypatch.setattr(provider_router, "plan", _fake_plan)

    captured_queries: list[list[str]] = []

    async def fake_search(typed_queries, *a, **kw):
        captured_queries.append([tq.text for tq in typed_queries])
        return []

    monkeypatch.setattr(orch_mod, "search", fake_search, raising=False)

    from utils.cancellation import CancellationToken
    cfg = orch_mod.RuntimeConfig.from_overrides(None, approval_required=True)
    assert cfg.approval_required is True

    tok = CancellationToken()
    orch = orch_mod.ResearchOrchestrator()

    async def approver():
        # Wait until the orchestrator is actually paused.
        for _ in range(50):
            await asyncio.sleep(0.02)
            if tok.approval_status == "waiting":
                break
        tok.approved_payload = {"sub_queries": ["edited only query"]}
        tok.approval_event.set()

    asyncio.create_task(approver())

    saw_approval_evt = False
    async for ev in orch.run(
        "q", "s-blocks", cancel_token=tok, turn_id="t-blocks", config=cfg,
    ):
        if ev.event_type == EVT_PLAN_APPROVAL:
            saw_approval_evt = True
        if ev.step == "done" or ev.step == "error":
            break

    assert saw_approval_evt is True
    # Searcher should have been called with the EDITED query, not the original.
    assert captured_queries, "search() was never reached"
    assert captured_queries[0] == ["edited only query"]


@pytest.mark.asyncio
async def test_approval_gate_cancel_path(tmp_path, monkeypatch):
    """Cancelling during approval wait → clean exit (error/cancelled event)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cancel.db"))
    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.orchestrator as orch_mod
    importlib.reload(orch_mod)
    await mem_mod.init_db()

    from utils import provider_router
    monkeypatch.setattr(provider_router, "plan", _fake_plan)

    from utils.cancellation import CancellationToken
    cfg = orch_mod.RuntimeConfig.from_overrides(None, approval_required=True)

    tok = CancellationToken()
    orch = orch_mod.ResearchOrchestrator()

    async def canceller():
        for _ in range(50):
            await asyncio.sleep(0.02)
            if tok.approval_status == "waiting":
                break
        tok.cancel()

    asyncio.create_task(canceller())

    events = []
    async for ev in orch.run(
        "q", "s-cancel", cancel_token=tok, turn_id="t-cancel-approval", config=cfg,
    ):
        events.append(ev)

    assert events[-1].step == "error"
    assert events[-1].label == "cancelled"


@pytest.mark.asyncio
async def test_approval_gate_timeout(tmp_path, monkeypatch):
    """No approve/cancel within timeout → APPROVAL_TIMEOUT persisted."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "timeout.db"))
    # Tiny timeout so the test finishes fast.
    monkeypatch.setenv("APPROVAL_TIMEOUT_S", "0.1")
    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.orchestrator as orch_mod
    importlib.reload(orch_mod)
    await mem_mod.init_db()

    from utils import provider_router
    monkeypatch.setattr(provider_router, "plan", _fake_plan)

    from utils.cancellation import CancellationToken
    cfg = orch_mod.RuntimeConfig.from_overrides(None, approval_required=True)

    tok = CancellationToken()
    orch = orch_mod.ResearchOrchestrator()
    events = []
    async for ev in orch.run(
        "q", "s-timeout", cancel_token=tok, turn_id="t-timeout", config=cfg,
    ):
        events.append(ev)

    # The orchestrator should have persisted a turn with APPROVAL_TIMEOUT in
    # state_trace and run_metadata.terminator_fired.
    import aiosqlite
    async with aiosqlite.connect(mem_mod.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT state_trace, run_metadata_json FROM turns WHERE turn_id = ?",
            ("t-timeout",),
        )
    assert len(rows) == 1
    state_trace = json.loads(rows[0]["state_trace"])
    assert "APPROVAL_TIMEOUT" in state_trace
    meta = json.loads(rows[0]["run_metadata_json"])
    assert meta.get("terminator_fired") == "APPROVAL_TIMEOUT"


# ── Disconnect watcher suspension ───────────────────────────────────────


@pytest.mark.asyncio
async def test_disconnect_watcher_suspends_during_pause():
    """When `token.paused=True`, the watcher must NOT cancel on disconnect."""
    import main as main_mod

    class FakeRequest:
        async def is_disconnected(self):
            return True  # always "disconnected"

    from utils.cancellation import CancellationToken
    tok = CancellationToken()
    tok.paused = True

    watcher_task = asyncio.create_task(
        main_mod._disconnect_watcher(FakeRequest(), tok, "t-watch", interval_s=0.01)
    )
    await asyncio.sleep(0.1)
    # Watcher must not have cancelled the token while paused.
    assert tok.is_set() is False

    # Unpause; the watcher must now fire on the next poll.
    tok.paused = False
    await asyncio.sleep(0.05)
    assert tok.is_set() is True

    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass


# ── Eval-runner safeguard ───────────────────────────────────────────────


def test_eval_runner_assert_approval_required_false(monkeypatch):
    """If env coerces approval_required=True into RuntimeConfig, run_eval
    must AssertionError rather than deadlock."""
    # The RuntimeConfig.from_overrides path doesn't currently read an env var
    # for approval_required (it's request-only), so we simulate the leak by
    # monkeypatching from_overrides itself.
    import eval.eval_runner as runner
    from agent.orchestrator import RuntimeConfig

    real = RuntimeConfig.from_overrides

    def leaky(overrides, *, approval_required=False):
        return real(overrides, approval_required=True)

    monkeypatch.setattr(runner._RC if hasattr(runner, "_RC") else RuntimeConfig,
                        "from_overrides", classmethod(lambda cls, ov, **kw: real(ov, approval_required=True)))

    # Reach the runtime check without doing any DB or LLM work — the guard is
    # the very first body statement after the docstring. Audit L3 converted
    # the original `assert` to a `raise RuntimeError` so `python -O` can't
    # strip it.
    with pytest.raises(RuntimeError, match="approval_required"):
        asyncio.run(runner.run_eval(question_ids=["__never__"]))
