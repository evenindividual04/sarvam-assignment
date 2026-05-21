import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiosqlite

from agent import eval_queries
from agent.memory import init_db, DB_PATH
from agent.orchestrator import ResearchOrchestrator
from utils.cancellation import get_registry
from utils.env_check import validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    validate()
except Exception as e:
    logger.error(f"Startup validation failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Cancellation registry is a module-level singleton — nothing else to do.
    yield


app = FastAPI(title="Deep Research Agent", lifespan=lifespan)

_allowed = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: str


def _format_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _research_stream(req: ChatRequest, request: Request):
    """SSE generator: registers a cancellation token, polls disconnect,
    streams ExecutionEvents from the orchestrator."""
    turn_id = str(uuid.uuid4())
    registry = get_registry()
    token = await registry.register(turn_id)
    orchestrator = ResearchOrchestrator()
    first_event = True
    try:
        async for event in orchestrator.run(
            req.query, req.session_id, cancel_token=token, turn_id=turn_id
        ):
            # Disconnect poll between events
            if await request.is_disconnected():
                token.cancel()
            data = event.data
            if first_event and event.step == "planning":
                # Inject turn_id so the client can issue cancel calls before /done.
                if isinstance(data, dict):
                    data = {**data, "turn_id": turn_id}
                else:
                    data = {"turn_id": turn_id}
            payload = {"step": event.step, "label": event.label, "data": data}
            first_event = False
            yield _format_event(payload)
    except Exception as e:
        logger.error("Orchestrator error: %s", e, extra={"turn_id": turn_id})
        yield _format_event({"step": "error", "label": "error", "data": str(e)})
    finally:
        await orchestrator.aclose()
        await registry.release(turn_id)


@app.post("/research")
async def research(req: ChatRequest, request: Request):
    return StreamingResponse(
        _research_stream(req, request),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Deprecated alias for /research — kept until static/app.js is removed."""
    return await research(req, request)


@app.post("/research/cancel/{turn_id}")
async def cancel_research(turn_id: str):
    ok = await get_registry().cancel(turn_id)
    if not ok:
        raise HTTPException(status_code=404, detail="turn_id not active")
    return {"cancelled": True}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "v2"}


@app.get("/sessions")
async def get_sessions():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT session_id, updated_at, turn_count FROM sessions ORDER BY updated_at DESC LIMIT 50"
        )
        return [dict(r) for r in rows]


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT turn_id, session_id, query, response, created_at,
                   state_trace, doc_map, search_queries, urls_opened,
                   context_xml_sent, citation_integrity_score, claim_precision_score,
                   prompt_tokens, completion_tokens, latency_ms,
                   planning_ms, search_ms, fetch_ms, select_ms, synthesize_ms,
                   run_metadata_json
            FROM turns WHERE session_id = ? ORDER BY created_at ASC
            """,
            (session_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            for k in (
                "doc_map",
                "state_trace",
                "search_queries",
                "urls_opened",
                "run_metadata_json",
            ):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except (TypeError, ValueError):
                        pass
            out.append(d)
        return out


@app.get("/sessions/{session_id}/turns/{turn_id}")
async def get_turn_detail(session_id: str, turn_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        t = await db.execute_fetchall(
            "SELECT * FROM turns WHERE session_id = ? AND turn_id = ? LIMIT 1",
            (session_id, turn_id),
        )
        if not t:
            raise HTTPException(status_code=404, detail="turn not found")
        turn = dict(t[0])
        for k in ("doc_map", "state_trace", "search_queries", "urls_opened", "run_metadata_json"):
            if turn.get(k):
                try:
                    turn[k] = json.loads(turn[k])
                except (TypeError, ValueError):
                    pass

        ca = await db.execute_fetchall(
            "SELECT * FROM claim_audit WHERE turn_id = ? ORDER BY claim_idx ASC", (turn_id,)
        )
        claim_audit = [dict(r) for r in ca]
        for c in claim_audit:
            if c.get("cited_doc_ids"):
                try:
                    c["cited_doc_ids"] = json.loads(c["cited_doc_ids"])
                except (TypeError, ValueError):
                    pass

        p = await db.execute_fetchall(
            "SELECT * FROM contradiction_probes WHERE turn_id = ? LIMIT 1", (turn_id,)
        )
        probe = dict(p[0]) if p else None
        if probe and probe.get("contradictions_json"):
            try:
                probe["contradictions_json"] = json.loads(probe["contradictions_json"])
            except (TypeError, ValueError):
                pass

    # Flatten into a single object matching frontend `TurnDetail` shape:
    # turn fields at root + `claim_audit` and `contradiction_probes` (plural) alongside.
    flat = dict(turn)
    flat["claim_audit"] = claim_audit
    flat["contradiction_probes"] = _adapt_probe(probe)
    return flat


# ── Eval endpoints ─────────────────────────────────────────────────────────
#
# The internal `agent.eval_queries` module returns SQL-canonical shapes:
#   - metric columns suffixed `_score` (e.g. `faithfulness_score`)
#   - per-run summary nested as `{by_category, failure_distribution, run_summary, ...}`
#   - question detail nested as `{eval_row, turn, claim_audit, contradiction_probe}`
#
# The Next.js frontend (`frontend/lib/types.ts`) consumes flat shapes with bare
# metric names (`faithfulness`, `avg_relevance`, `failure_class_distribution`).
# These adapter helpers reshape at the HTTP boundary so internal consumers
# (eval_runner.py, ablation_report.py, …) keep using the canonical form.

_EVAL_METRIC_RENAME = {
    "faithfulness_score": "faithfulness",
    "answer_relevance_score": "answer_relevance",
    "context_precision_score": "context_precision",
    "citation_integrity_score": "citation_integrity",
    "claim_precision_score": "claim_precision",
    "conflict_adherence_score": "conflict_adherence",
    "session_coherence_score": "session_coherence",
    "factual_accuracy_score": "factual_accuracy",
}

# Frontend `EvalRun.avg_*` keys (note: `avg_relevance`, not `avg_answer_relevance`).
_EVAL_AVG_RENAME = {
    "avg_faithfulness_score": "avg_faithfulness",
    "avg_answer_relevance_score": "avg_relevance",
    "avg_context_precision_score": "avg_context_precision",
    "avg_citation_integrity_score": "avg_citation_integrity",
    "avg_claim_precision_score": "avg_claim_precision",
    "avg_conflict_adherence_score": "avg_conflict_adherence",
    "avg_session_coherence_score": "avg_session_coherence",
    "avg_factual_accuracy_score": "avg_factual_accuracy",
}


def _rename_metric_keys(d: dict) -> dict:
    """Strip `_score` suffix from metric columns and `_score` from avg_* columns."""
    out = dict(d)
    for old, new in _EVAL_METRIC_RENAME.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
    for old, new in _EVAL_AVG_RENAME.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
    return out


def _adapt_probe(p: dict | None) -> dict | None:
    """Map backend contradiction_probes row → frontend `ContradictionProbeRow`."""
    if p is None:
        return None
    out = dict(p)
    if "contradictions_json" in out:
        out["contradictions"] = out.pop("contradictions_json")
    if "probe_skipped_reason" in out:
        out["skip_reason"] = out.pop("probe_skipped_reason")
    if "has_conflict" in out:
        out["has_conflict"] = bool(out["has_conflict"])
    return out


def _adapt_eval_question(row: dict) -> dict:
    """Map an `eval_runs` row → frontend `EvalQuestion`."""
    out = _rename_metric_keys(row)
    fc = out.get("failure_class")
    out["pass"] = (fc == "PASS") or (fc is None)
    return out


def _rename_category_metric_keys(d: dict) -> dict:
    """For per-category rows: backend uses `avg_X_score`, frontend
    `EvalSummaryCategoryRow` uses bare `X` (no avg_ prefix)."""
    out = dict(d)
    for old, new in _EVAL_AVG_RENAME.items():
        bare = new.removeprefix("avg_")  # avg_faithfulness → faithfulness
        if old in out and bare not in out:
            out[bare] = out.pop(old)
    return out


def _adapt_eval_summary(payload: dict) -> dict:
    """Map `eval_queries.get_run_summary(...)` → frontend `EvalSummary`."""
    rs = payload.get("run_summary") or {}
    failure_dist = payload.get("failure_distribution") or []
    failure_class_distribution = {r["failure_class"]: r["n"] for r in failure_dist}

    by_category = [
        _rename_category_metric_keys(r) for r in (payload.get("by_category") or [])
    ]

    total = sum(failure_class_distribution.values())
    pass_count = failure_class_distribution.get("PASS", 0)
    pass_rate = rs.get("pass_rate")
    if pass_rate is None:
        pass_rate = pass_count / total if total > 0 else 0.0

    return {
        "run_at": payload.get("run_at"),
        "pass_rate": pass_rate,
        "avg_faithfulness": rs.get("mean_faithfulness"),
        "avg_relevance": rs.get("mean_relevance"),
        "avg_context_precision": rs.get("mean_context_precision"),
        "avg_citation_integrity": rs.get("mean_citation_integrity"),
        "avg_claim_precision": rs.get("mean_claim_precision"),
        "avg_conflict_adherence": rs.get("mean_conflict_adherence"),
        "by_category": by_category,
        "failure_class_distribution": failure_class_distribution,
        # Extra sections preserved for any future UI use; frontend types ignore them.
        "cross_language": payload.get("cross_language"),
        "calibration": payload.get("calibration"),
        "run_summary": rs,
    }


def _adapt_question_detail(payload: dict) -> dict:
    """Map `eval_queries.get_question_detail(...)` → frontend `EvalQuestionDetail`.
    Flattens `{eval_row, turn, claim_audit, contradiction_probe}` into one object
    with renamed metric keys and `contradiction_probes` (plural to match TS type).
    """
    eval_row = _rename_metric_keys(payload.get("eval_row") or {})
    turn = payload.get("turn") or {}
    flat = dict(eval_row)
    for k in (
        "context_xml_sent",
        "doc_map",
        "prompt_tokens",
        "completion_tokens",
        "state_trace",
        "claim_verification_json",
    ):
        if k in turn and turn[k] is not None:
            flat[k] = turn[k]
    flat["claim_audit"] = payload.get("claim_audit") or []
    flat["contradiction_probes"] = _adapt_probe(payload.get("contradiction_probe"))
    fc = flat.get("failure_class")
    flat["pass"] = (fc == "PASS") or (fc is None)
    return flat


@app.get("/eval/runs")
async def list_runs():
    rows = await eval_queries.list_eval_runs()
    return [_rename_metric_keys(r) for r in rows]


@app.get("/eval/runs/{run_at}/summary")
async def run_summary(run_at: str):
    payload = await eval_queries.get_run_summary(unquote(run_at))
    return _adapt_eval_summary(payload)


@app.get("/eval/runs/{run_at}/questions")
async def run_questions(run_at: str):
    rows = await eval_queries.get_run_questions(unquote(run_at))
    return [_adapt_eval_question(r) for r in rows]


@app.get("/eval/runs/{run_at}/questions/{question_id}")
async def question_detail(run_at: str, question_id: str):
    detail = await eval_queries.get_question_detail(unquote(run_at), question_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="question not found in run")
    return _adapt_question_detail(detail)


# Mount static files at the root (legacy vanilla-JS frontend)
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
