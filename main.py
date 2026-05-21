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
            SELECT turn_id, query, response, created_at, state_trace, doc_map,
                   claim_precision_score, citation_integrity_score
            FROM turns WHERE session_id = ? ORDER BY created_at ASC
            """,
            (session_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            for k in ("doc_map", "state_trace"):
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

    return {"turn": turn, "claim_audit": claim_audit, "contradiction_probe": probe}


# ── Eval endpoints ─────────────────────────────────────────────────────────


@app.get("/eval/runs")
async def list_runs():
    return await eval_queries.list_eval_runs()


@app.get("/eval/runs/{run_at}/summary")
async def run_summary(run_at: str):
    return await eval_queries.get_run_summary(unquote(run_at))


@app.get("/eval/runs/{run_at}/questions")
async def run_questions(run_at: str):
    return await eval_queries.get_run_questions(unquote(run_at))


@app.get("/eval/runs/{run_at}/questions/{question_id}")
async def question_detail(run_at: str, question_id: str):
    detail = await eval_queries.get_question_detail(unquote(run_at), question_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="question not found in run")
    return detail


# Mount static files at the root (legacy vanilla-JS frontend)
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
