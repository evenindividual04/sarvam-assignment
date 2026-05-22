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

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
        yield ExecutionEvent("planning", "Planning", data={"strategy": "x"})
        yield ExecutionEvent("searching", "Searching the web")
        yield ExecutionEvent("done", "done", data={"turn_id": turn_id})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)

    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()
    # Phase 1.25: SSE frames may include `id:` and `event:` lines before `data:`.
    payloads = []
    for part in body.split("\n\n"):
        data_line = next((l for l in part.split("\n") if l.startswith("data: ")), None)
        if data_line is None:
            continue
        payloads.append(json.loads(data_line[len("data: "):]))
    steps = [p["step"] for p in payloads]
    assert steps == ["planning", "searching", "done"]
    assert all("label" in p for p in payloads)


def test_research_includes_turn_id_in_first_event(app_client, monkeypatch):
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
        yield ExecutionEvent("planning", "Planning", data={"strategy": "x"})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)
    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        body = b"".join(r.iter_bytes()).decode()
    # Find first frame containing a `data:` line.
    first = None
    for part in body.split("\n\n"):
        data_line = next((l for l in part.split("\n") if l.startswith("data: ")), None)
        if data_line:
            first = json.loads(data_line[len("data: "):])
            break
    assert first is not None
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

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
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

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
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


def test_sessions_list_includes_first_query(app_client):
    """GET /sessions surfaces the first turn's `query` so the sidebar can
    render a human-readable title instead of a hash slug."""
    client, mem_mod, _ = app_client

    async def seed():
        async with aiosqlite.connect(mem_mod.DB_PATH) as db:
            await db.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, turn_count) VALUES (?,?,?,?)",
                ("sess-with-turn", "2026-01-01T00:00:00", "2026-01-01T00:00:00", 2),
            )
            # Two turns: the earliest one should be picked.
            await db.execute(
                """INSERT INTO turns (turn_id, session_id, query, search_queries, urls_opened,
                   response, context_xml_sent, doc_map, state_trace, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                ("t-first", "sess-with-turn", "What is India's repo rate?",
                 "[]", "[]", "A", "<ctx/>", "{}", "[]", "2026-01-01T00:00:00"),
            )
            await db.execute(
                """INSERT INTO turns (turn_id, session_id, query, search_queries, urls_opened,
                   response, context_xml_sent, doc_map, state_trace, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                ("t-second", "sess-with-turn", "Follow-up question",
                 "[]", "[]", "A", "<ctx/>", "{}", "[]", "2026-01-01T00:05:00"),
            )
            # An empty session — should report null first_query.
            await db.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, turn_count) VALUES (?,?,?,?)",
                ("sess-empty", "2026-01-01T00:00:00", "2026-01-01T00:00:00", 0),
            )
            await db.commit()

    asyncio.run(seed())
    r = client.get("/sessions")
    assert r.status_code == 200
    rows = {row["session_id"]: row for row in r.json()}

    assert "sess-with-turn" in rows
    assert rows["sess-with-turn"]["first_query"] == "What is India's repo rate?"

    assert "sess-empty" in rows
    assert rows["sess-empty"]["first_query"] is None


# ── Phase 1.25: typed event taxonomy + SSE robustness ──────────────────────


def _parse_sse_frames(body: str) -> list[dict]:
    """Parse SSE frames into {id, event, data, comment} dicts."""
    frames = []
    for part in body.split("\n\n"):
        if not part.strip():
            continue
        frame: dict = {"id": None, "event": None, "data": None, "comment": False}
        for line in part.split("\n"):
            if line.startswith(":"):
                frame["comment"] = True
            elif line.startswith("id: "):
                try:
                    frame["id"] = int(line[len("id: "):].strip())
                except ValueError:
                    pass
            elif line.startswith("event: "):
                frame["event"] = line[len("event: "):].strip()
            elif line.startswith("data: "):
                try:
                    frame["data"] = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    frame["data"] = line[len("data: "):]
            elif line.startswith("retry: "):
                frame["retry"] = int(line[len("retry: "):].strip())
        frames.append(frame)
    return frames


def test_sse_event_has_id_field(app_client, monkeypatch):
    """Phase 1.25: every data frame carries a monotonically increasing `id:`."""
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
        yield ExecutionEvent("planning", "Planning", data={"strategy": "x"})
        yield ExecutionEvent("searching", "Searching the web")
        yield ExecutionEvent("done", "done", data={"turn_id": turn_id})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)
    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        body = b"".join(r.iter_bytes()).decode()
    frames = [f for f in _parse_sse_frames(body) if f["data"] is not None]
    ids = [f["id"] for f in frames]
    assert all(i is not None for i in ids), f"missing id field: {ids}"
    assert ids == sorted(ids), f"ids not monotonic: {ids}"
    assert len(set(ids)) == len(ids), f"duplicate ids: {ids}"


def test_sse_event_carries_type_discriminator(app_client, monkeypatch):
    """Phase 1.25: events with `event_type` are emitted with SSE `event:` line."""
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent, EVT_PHASE_STARTED

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
        yield ExecutionEvent(
            "planning", "Planning",
            data={"name": "planning", "label": "Planning", "idx": 1, "total": 7},
            event_type=EVT_PHASE_STARTED,
        )

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)
    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        body = b"".join(r.iter_bytes()).decode()
    frames = _parse_sse_frames(body)
    typed = [f for f in frames if f["event"] == EVT_PHASE_STARTED]
    assert len(typed) >= 1, f"no phase_started event: {[f['event'] for f in frames]}"


def test_cot_preemit_filter_strips_thinking(app_client, monkeypatch, caplog):
    """The CoT filter scrubs `<thinking>` blocks from text leaves and drops
    frames whose CoT-envelope keys (thought_summary, reasoning_content, …)
    survive scrubbing.

    Behavior split:
    - `data.text="<thinking>…</thinking>"` → text scrubbed to empty, frame
      kept (so a mixed thinking+content stream doesn't lose the content
      portion in a future refactor).
    - `data={"thought_summary": …}` → envelope key survives scrub → frame
      dropped wholesale.

    Invariant: no `<thinking>` / `thought_summary` substring leaks into the
    serialized SSE output — this is the line-103 assignment guarantee.
    """
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
        # First event: normal, should pass through.
        yield ExecutionEvent("planning", "Planning", data={"turn_id": turn_id})
        # Second event: text contains CoT → scrubbed to empty, frame kept.
        yield ExecutionEvent("generating", "Generating answer with citations",
                             data={"text": "<thinking>internal reasoning</thinking>"})
        # Third event: CoT envelope key → frame dropped wholesale.
        yield ExecutionEvent("generating", "Generating answer with citations",
                             data={"thought_summary": "hidden"})
        # Fourth event: clean.
        yield ExecutionEvent("done", "done", data={"turn_id": turn_id})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)
    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        body = b"".join(r.iter_bytes()).decode()
    frames = [f for f in _parse_sse_frames(body) if f["data"] is not None]
    serialized = json.dumps([f["data"] for f in frames])
    # Line-103 assignment invariant: no CoT substring in the wire output.
    assert "<thinking>" not in serialized
    assert "thought_summary" not in serialized
    # Envelope-key frame dropped, scrubbed-text frame kept (planning + scrubbed-generating + done).
    assert len(frames) == 3, f"expected 3 frames, got {len(frames)}: {frames}"
    # The scrubbed-text frame should have an empty (or absent) text field.
    gen_frames = [f for f in frames if f["data"].get("step") == "generating"]
    assert len(gen_frames) == 1
    assert gen_frames[0]["data"]["data"].get("text", "") == ""


def test_last_event_id_replay(app_client, monkeypatch):
    """Phase 1.25: a second stream call with ?turn_id=X&last_event_id=N replays
    frames with id > N from the per-turn ring buffer."""
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
        yield ExecutionEvent("planning", "Planning", data={"strategy": "x"})
        yield ExecutionEvent("searching", "Searching the web", data={"query": "q1"})
        yield ExecutionEvent("done", "done", data={"turn_id": turn_id})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)

    # First run: consume the stream and capture turn_id + ids.
    fixed_turn = "phase-1-25-replay-test"
    with client.stream(
        "POST", f"/research?turn_id={fixed_turn}",
        json={"query": "q", "session_id": "s"},
    ) as r:
        body1 = b"".join(r.iter_bytes()).decode()
    # NB: the per-turn replay buffer is dropped in `finally`. Confirm the test
    # exercises the replay machinery itself (in-memory) by checking the
    # function directly — request-level replay only fires when the consumer
    # *interrupts* mid-stream, which TestClient can't easily simulate.
    from main import _replay_remember, _replay_since
    _replay_remember("replay-direct", 1, "phase_started", {"step": "planning"})
    _replay_remember("replay-direct", 2, "answer_delta", {"step": "generating"})
    _replay_remember("replay-direct", 3, "phase_finished", {"step": "planning"})
    out = _replay_since("replay-direct", last_event_id=1)
    ids_replayed = [eid for (eid, _et, _pl) in out]
    assert ids_replayed == [2, 3]


def test_heartbeat_keeps_stream_alive(app_client, monkeypatch):
    """Phase 1.25: idle SSE streams emit `: ping` comment frames periodically.

    Drives the heartbeat at a very short interval (0.1s) by patching the
    module constant so the test runs in milliseconds rather than 30s.
    """
    client, _, main_mod = app_client
    from agent.models import ExecutionEvent
    import asyncio as _aio

    async def fake_run(self, query, session_id, cancel_token=None, turn_id=None, config=None):
        yield ExecutionEvent("planning", "Planning", data={"strategy": "x"})
        # Stall for long enough to see ≥2 heartbeats at 0.1s interval.
        await _aio.sleep(0.35)
        yield ExecutionEvent("done", "done", data={"turn_id": turn_id})

    monkeypatch.setattr(main_mod.ResearchOrchestrator, "run", fake_run)
    monkeypatch.setattr(main_mod, "_HEARTBEAT_INTERVAL_S", 0.1)

    with client.stream("POST", "/research", json={"query": "q", "session_id": "s"}) as r:
        body = b"".join(r.iter_bytes()).decode()
    pings = body.count(": ping")
    assert pings >= 2, f"expected ≥2 heartbeats, got {pings}; body={body!r}"


def test_stream_labels_endpoint(app_client):
    """Phase 1.25: GET /stream/labels returns the canonical STREAM_LABELS dict."""
    client, _, _ = app_client
    r = client.get("/stream/labels")
    assert r.status_code == 200
    body = r.json()
    assert "labels" in body
    assert "order" in body
    # The 5 required user-facing labels must be present and exact.
    assert body["labels"]["planning"] == "Planning"
    assert body["labels"]["searching"] == "Searching the web"
    assert body["labels"]["fetching"] == "Fetching sources"
    assert body["labels"]["selecting"] == "Selecting relevant context"
    assert body["labels"]["generating"] == "Generating answer with citations"
