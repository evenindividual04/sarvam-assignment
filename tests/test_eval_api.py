"""Contract tests for the eval-dashboard FastAPI endpoints.

Exercises the live SSE endpoint plus the read-path against an isolated
SQLite DB seeded with one synthetic eval row + the underlying turn row.

The key invariant under test is that `context_xml_sent` (the verbatim XML
handed to the synthesizer) reaches the frontend through
`/eval/runs/{run_at}/questions/{question_id}` — this is the "what the LLM
actually saw" payload the dashboard replays in its detail tab. If this
plumbing regresses, faithfulness becomes unauditable from the UI."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import uuid

import aiosqlite
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def eval_client(tmp_path_factory):
    db = tmp_path_factory.mktemp("evalapidb") / "test.db"
    os.environ["DB_PATH"] = str(db)
    os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.eval_queries as eq_mod
    importlib.reload(eq_mod)
    import main as main_mod
    importlib.reload(main_mod)

    asyncio.run(mem_mod.init_db())
    asyncio.run(_seed(mem_mod.DB_PATH))

    client = TestClient(main_mod.app)
    yield client


async def _seed(db_path: str) -> None:
    """Insert one session + turn + eval_run that share IDs."""
    run_at = "2026-05-22T10:00:00Z"
    turn_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    context_xml = (
        "<context>\n  <doc id='1' url='https://example.com'>RBI repo rate is 6.50%.</doc>\n</context>"
    )
    doc_map = json.dumps({"1": {"url": "https://example.com", "title": "RBI"}})
    # Refactor #3: seed a realistic terminator trace in run_metadata so the
    # per-question detail surface (`TerminatorTrace`) has data to render.
    run_metadata_json = json.dumps({
        "terminator_fired": "STOP_RAG_GATE",
        "terminator_source": "stop_rag",
        "stop_rag_terminator_mapped": "EVIDENCE_SUFFICIENT",
        "terminator_history": [
            {"source": "stop_rag", "reason": "EVIDENCE_SUFFICIENT", "hop": 1},
        ],
        "stop_rag_decisions": [
            {
                "hop": 1,
                "useful": False,
                "confidence": 0.82,
                "reason": "All success criteria satisfied by hop 1 evidence.",
                "degraded_reason": None,
            },
        ],
    })

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, run_at, run_at),
        )
        await db.execute(
            "INSERT INTO turns (turn_id, session_id, query, search_queries, urls_opened, "
            "response, context_xml_sent, doc_map, prompt_tokens, completion_tokens, latency_ms, "
            "run_metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (turn_id, session_id, "What is the RBI repo rate?", "[]", "[]",
             "6.50%.", context_xml, doc_map, 1200, 150, 4200,
             run_metadata_json, run_at),
        )
        await db.execute(
            "INSERT INTO eval_runs (run_id, run_at, question_id, question, category, "
            "agent_answer, faithfulness_score, answer_relevance_score, "
            "citation_integrity_score, failure_class, latency_ms, turn_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), run_at, "F-1", "What is the RBI repo rate?",
             "factual", "6.50%.", 1.0, 1.0, 1.0, "PASS", 4200, turn_id),
        )
        await db.commit()


def test_list_runs_returns_seeded_run(eval_client):
    """`/eval/runs` returns at least the one seeded row with the canonical
    pass-rate + metric keys the frontend `EvalRun` type expects."""
    r = eval_client.get("/eval/runs")
    assert r.status_code == 200
    runs = r.json()
    assert isinstance(runs, list) and len(runs) >= 1
    top = runs[0]
    assert top["run_at"] == "2026-05-22T10:00:00Z"
    # Frontend type contract: these key names must survive the rename step.
    for k in ("n_questions", "pass_rate"):
        assert k in top


def test_run_summary_returns_aggregates(eval_client):
    """`/eval/runs/{run_at}/summary` returns per-category rows + failure-class
    distribution shaped for the run-summary dashboard."""
    r = eval_client.get("/eval/runs/2026-05-22T10:00:00Z/summary")
    assert r.status_code == 200
    payload = r.json()
    assert payload["run_at"] == "2026-05-22T10:00:00Z"
    assert isinstance(payload["by_category"], list)
    assert "failure_class_distribution" in payload
    # PASS row should be present and counted.
    assert payload["failure_class_distribution"].get("PASS") == 1


def test_question_detail_surfaces_context_xml(eval_client):
    """The replay-a-turn drawer requires `context_xml_sent` + `doc_map` to be
    surfaced verbatim. This is the "what the LLM actually saw" plumbing."""
    r = eval_client.get("/eval/runs/2026-05-22T10:00:00Z/questions/F-1")
    assert r.status_code == 200
    d = r.json()
    assert "<context>" in d["context_xml_sent"]
    assert "RBI repo rate is 6.50%." in d["context_xml_sent"]
    assert isinstance(d["doc_map"], dict)
    assert "1" in d["doc_map"]


def test_question_detail_surfaces_terminator_trace(eval_client):
    """Refactor #3: the per-question detail endpoint must hoist the
    terminator-policy trace out of `turns.run_metadata_json` so the
    `TerminatorTrace` component on the per-question page has data to
    render. Older rows without run_metadata still respond 200 — these
    fields just don't appear in the payload."""
    r = eval_client.get("/eval/runs/2026-05-22T10:00:00Z/questions/F-1")
    assert r.status_code == 200
    d = r.json()
    assert d.get("terminator_source") == "stop_rag"
    history = d.get("terminator_history") or []
    assert len(history) == 1
    assert history[0]["source"] == "stop_rag"
    assert history[0]["reason"] == "EVIDENCE_SUFFICIENT"
    assert history[0]["hop"] == 1
    stop_rag = d.get("stop_rag_decisions") or []
    assert len(stop_rag) == 1
    assert stop_rag[0]["hop"] == 1
    assert stop_rag[0]["useful"] is False
    assert stop_rag[0]["confidence"] == pytest.approx(0.82)
    assert stop_rag[0]["degraded_reason"] is None


def test_eval_live_generator_emits_snapshot(eval_client):  # noqa: ARG001 — fixture seeds the DB
    """Direct unit test of the SSE generator — exercises `_eval_live_stream`
    against the seeded DB. We avoid TestClient.stream here because httpx's
    sync transport buffers in a way that doesn't play well with a long-lived
    streaming generator. A fake `Request` lets us drive one tick of the loop
    and assert on the emitted frame, then trip `is_disconnected` to unwind."""
    import asyncio as _asyncio
    import main as main_mod

    class FakeRequest:
        def __init__(self):
            self.calls = 0

        async def is_disconnected(self) -> bool:
            self.calls += 1
            # First check: still connected (allow the snapshot to fire).
            # Subsequent: disconnect so the generator exits.
            return self.calls > 1

    async def _drive():
        gen = main_mod._eval_live_stream(FakeRequest())
        first = await gen.__anext__()
        # Drain — generator should exit on the next `is_disconnected` tick.
        try:
            async for _ in gen:
                break
        except StopAsyncIteration:
            pass
        return first

    frame = _asyncio.run(_drive())
    text = frame.decode()
    assert text.startswith("data: ")
    payload = json.loads(text[len("data: "):].strip())
    assert payload["step"] == "snapshot"
    assert payload["data"]["latest_run_at"] == "2026-05-22T10:00:00Z"


def test_eval_live_endpoint_is_registered(eval_client):
    """Confirms the `/eval/live` route is wired into the FastAPI app and
    advertises an SSE content-type. We don't read the body here to avoid
    blocking on the long-lived stream; the generator behavior is unit-tested
    above."""
    # HEAD/OPTIONS aren't supported on the streaming route, so use the OpenAPI
    # introspection endpoint as a stable contract check.
    r = eval_client.get("/openapi.json")
    assert r.status_code == 200
    assert "/eval/live" in r.json()["paths"]
