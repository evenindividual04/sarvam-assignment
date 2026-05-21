import asyncio
import json
import logging
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiosqlite

from agent.memory import init_db, DB_PATH
from agent.orchestrator import ResearchOrchestrator
from utils.env_check import validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment variables on startup
try:
    validate()
except Exception as e:
    logger.error(f"Startup validation failed: {e}")
    # Don't exit here, let the app start so we can show an error page, or just log it

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB
    await init_db()
    yield
    # Cleanup if needed

app = FastAPI(title="Deep Research Agent", lifespan=lifespan)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    session_id: str

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        orchestrator = ResearchOrchestrator()
        try:
            async for event in orchestrator.run(req.query, req.session_id):
                # Send SSE formatted data
                payload = {
                    "step": event.step,
                    "message": event.message,
                    "data": event.data
                }
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            yield f"data: {json.dumps({'step': 'error', 'message': str(e), 'data': None})}\n\n"
        finally:
            await orchestrator.aclose()
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/sessions")
async def get_sessions():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("SELECT session_id, updated_at, turn_count FROM sessions ORDER BY updated_at DESC LIMIT 50")
        return [dict(r) for r in rows]

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("""
            SELECT turn_id, query, response, created_at, state_trace, doc_map 
            FROM turns WHERE session_id = ? ORDER BY created_at ASC
        """, (session_id,))
        return [dict(r) for r in rows]

# Mount static files at the root
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
