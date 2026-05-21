"""FastAPI endpoint contract tests. Uses fastapi.testclient with an isolated
SQLite DB created via the DB_PATH env override."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import time

import aiosqlite
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    db = tmp_path_factory.mktemp("apidb") / "test.db"
    os.environ["DB_PATH"] = str(db)
    os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

    # Reload memory + eval_queries + main so the env override is picked up.
    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.eval_queries as eq_mod
    importlib.reload(eq_mod)
    import main as main_mod
    importlib.reload(main_mod)

    asyncio.run(mem_mod.init_db())

    client = TestClient(main_mod.app)
    yield client, mem_mod, main_mod


def test_health_returns_ok(app_client):
    client, _, _ = app_client
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "v2"}


def test_research_emits_sse_events(app_client, monkeypatch):
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None):
        yield ExecutionEvent("planning", "Planning", data={"strategy": "x"})
        yield ExecutionEvent("searching", "Searching the web")
        yield ExecutionEvent("done", "done", data={"turn_id": turn_id})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)

    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()
    parts = [p for p in body.split("\n\n") if p.startswith("data: ")]
    payloads = [json.loads(p[len("data: "):]) for p in parts]
    steps = [p["step"] for p in payloads]
    assert steps == ["planning", "searching", "done"]
    assert all("label" in p for p in payloads)


def test_research_includes_turn_id_in_first_event(app_client, monkeypatch):
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None):
        yield ExecutionEvent("planning", "Planning", data={"strategy": "x"})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)
    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        body = b"".join(r.iter_bytes()).decode()
    first = json.loads(body.split("\n\n")[0][len("data: "):])
    assert first["step"] == "planning"
    assert "turn_id" in first["data"]


def test_research_cancel_unknown_returns_404(app_client):
    client, _, _ = app_client
    r = client.post("/research/cancel/does-not-exist")
    assert r.status_code == 404


def test_research_cancel_active_turn_returns_200(app_client, monkeypatch):
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent
    from utils.cancellation import OperationCancelledError

    saw_cancel = {"flag": False}

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None):
        yield ExecutionEvent("planning", "Planning", data={"turn_id": turn_id})
        # Spin until cancel_token fires (with a timeout safety net)
        for _ in range(50):
            await asyncio.sleep(0.02)
            if cancel_token is not None and cancel_token.is_set():
                saw_cancel["flag"] = True
                break
        yield ExecutionEvent("error", "cancelled", data="Cancelled by user.")

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)

    # Issue cancel from a background thread once the registry has the token.
    import threading
    from utils.cancellation import get_registry

    cancel_result = {"status": None}

    def do_cancel():
        # Wait until a turn is registered (registry inspected directly).
        reg = get_registry()
        for _ in range(50):
            time.sleep(0.02)
            # Peek private dict — test-only.
            if reg._tokens:
                tid = next(iter(reg._tokens))
                cancel_result["status"] = client.post(f"/research/cancel/{tid}").status_code
                return

    t = threading.Thread(target=do_cancel)
    t.start()
    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        body = b"".join(r.iter_bytes()).decode()
    t.join(timeout=5)
    assert saw_cancel["flag"] is True
    assert cancel_result["status"] == 200
    assert "cancelled" in body


def test_chat_stream_alias_still_works(app_client, monkeypatch):
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None):
        yield ExecutionEvent("done", "done", data={"turn_id": turn_id})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)
    with client.stream("POST", "/chat/stream", json={"query": "q", "session_id": "s"}) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()
    assert "done" in body


def test_cors_headers_present_on_options(app_client):
    client, _, _ = app_client
    r = client.options(
        "/research",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Starlette returns 200 with allow-origin header on preflight
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ── Eval endpoints ─────────────────────────────────────────────────────────


async def _seed_eval(mem_mod, run_at: str, turn_id: str = None):
    async with aiosqlite.connect(mem_mod.DB_PATH) as db:
        await db.execute(
            """INSERT INTO eval_runs (run_id, run_at, question_id, question, category,
               agent_answer, faithfulness_score, answer_relevance_score,
               context_precision_score, citation_integrity_score,
               conflict_adherence_score, session_coherence_score, claim_precision_score,
               judge_reasoning, failure_class, latency_ms, turn_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"run-{run_at}-q1", run_at, "q1", "What is X?", "factual",
                "Answer text", 0.9, 0.85, 0.8, 1.0, 1.0, 0.9, 0.95,
                "judge says ok", "PASS", 1200, turn_id,
            ),
        )
        await db.execute(
            """INSERT INTO eval_runs (run_id, run_at, question_id, question, category,
               agent_answer, faithfulness_score, answer_relevance_score,
               context_precision_score, citation_integrity_score,
               conflict_adherence_score, session_coherence_score, claim_precision_score,
               judge_reasoning, failure_class, latency_ms, turn_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"run-{run_at}-q2", run_at, "q2", "What is Y?", "multihop",
                "Answer text 2", 0.5, 0.6, 0.5, 0.7, 1.0, 0.8, 0.6,
                "judge says weak", "HALLUCINATION", 2200, None,
            ),
        )
        await db.commit()


def test_eval_runs_returns_aggregated_list(app_client):
    client, mem_mod, _ = app_client
    asyncio.run(_seed_eval(mem_mod, "2025-05-21T10:00:00"))
    asyncio.run(_seed_eval(mem_mod, "2025-05-21T11:00:00"))
    r = client.get("/eval/runs")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 2
    row = data[0]
    assert "run_at" in row
    assert "n_questions" in row
    assert "pass_rate" in row
    assert "avg_faithfulness" in row


def test_eval_run_summary_per_category_breakdown(app_client):
    client, mem_mod, _ = app_client
    run_at = "2025-05-21T12:00:00"
    asyncio.run(_seed_eval(mem_mod, run_at))
    r = client.get(f"/eval/runs/{run_at}/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["run_at"] == run_at
    categories = {row["category"] for row in data["by_category"]}
    assert "factual" in categories and "multihop" in categories
    fclasses = set(data["failure_class_distribution"].keys())
    assert "PASS" in fclasses or "HALLUCINATION" in fclasses


def test_eval_question_detail_joins_claim_audit_and_probes(app_client):
    client, mem_mod, _ = app_client
    run_at = "2025-05-21T13:00:00"
    turn_id = "turn-eval-1"

    async def seed():
        await _seed_eval(mem_mod, run_at, turn_id=turn_id)
        async with aiosqlite.connect(mem_mod.DB_PATH) as db:
            await db.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?,?,?)",
                ("sess-eval", "now", "now"),
            )
            await db.execute(
                """INSERT INTO turns (turn_id, session_id, query, search_queries, urls_opened,
                   response, context_xml_sent, doc_map, state_trace, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (turn_id, "sess-eval", "Q?", "[]", "[]", "A", "<context/>",
                 json.dumps({"doc_1": ["t", "u", "d"]}),
                 json.dumps(["PLANNING", "DONE"]), "now"),
            )
            await db.execute(
                """INSERT INTO claim_audit (audit_id, turn_id, claim_idx, claim_text,
                   cited_doc_ids, method, overlap, entity_match, score, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                ("a1", turn_id, 0, "Claim", json.dumps(["doc_1"]), "deterministic",
                 0.5, 0.5, 0.7, "supported", "now"),
            )
            await db.execute(
                """INSERT INTO contradiction_probes (turn_id, has_conflict, conflict_summary,
                   contradictions_json, probe_skipped_reason, probe_ms, prompt_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (turn_id, 0, None, json.dumps([]), None, 100, "conflict_v3", "now"),
            )
            await db.commit()

    asyncio.run(seed())
    r = client.get(f"/eval/runs/{run_at}/questions/q1")
    assert r.status_code == 200
    data = r.json()
    assert data["question_id"] == "q1"
    assert len(data["claim_audit"]) == 1
    assert data["claim_audit"][0]["cited_doc_ids"] == ["doc_1"]
    assert data["contradiction_probes"] is not None
    assert data["doc_map"] == {"doc_1": ["t", "u", "d"]}


def test_eval_question_detail_404_on_unknown(app_client):
    client, _, _ = app_client
    r = client.get("/eval/runs/never/questions/nope")
    assert r.status_code == 404


def test_sessions_turn_detail_joins_claim_audit_and_probes(app_client):
    client, mem_mod, _ = app_client
    turn_id = "turn-sess-1"

    async def seed():
        async with aiosqlite.connect(mem_mod.DB_PATH) as db:
            await db.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?,?,?)",
                ("sess-detail", "now", "now"),
            )
            await db.execute(
                """INSERT INTO turns (turn_id, session_id, query, search_queries, urls_opened,
                   response, context_xml_sent, doc_map, state_trace, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (turn_id, "sess-detail", "Q?", "[]", "[]", "A", "<ctx/>",
                 json.dumps({"doc_1": ["t", "u", "d"]}),
                 json.dumps(["DONE"]), "now"),
            )
            await db.execute(
                """INSERT INTO claim_audit (audit_id, turn_id, claim_idx, claim_text,
                   cited_doc_ids, method, overlap, entity_match, score, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                ("a2", turn_id, 0, "Claim2", json.dumps(["doc_1"]), "llm",
                 None, None, 0.8, "supported", "now"),
            )
            await db.commit()

    asyncio.run(seed())
    r = client.get(f"/sessions/sess-detail/turns/{turn_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["turn_id"] == turn_id
    assert data["claim_audit"][0]["cited_doc_ids"] == ["doc_1"]
