"""
Thread+queue bridge between async orchestrator and synchronous Streamlit layer.
Do NOT use nest_asyncio — creates subtle re-entrant loop issues.
"""
from __future__ import annotations

import asyncio
import threading
import queue
from typing import Iterator

from agent.models import ExecutionEvent
from agent.orchestrator import ResearchOrchestrator


def run_agent_sync(
    query: str,
    session_id: str,
    agent: ResearchOrchestrator,
) -> Iterator[ExecutionEvent]:
    """
    Runs async agent in a background thread.
    Yields ExecutionEvents via a queue to the synchronous Streamlit layer.
    """
    q: queue.Queue = queue.Queue()

    async def _run() -> None:
        try:
            async for event in agent.run(query, session_id):
                q.put(event)
        except Exception as e:
            q.put(ExecutionEvent("error", "error", data=str(e)))
        finally:
            q.put(None)  # sentinel

    thread = threading.Thread(target=lambda: asyncio.run(_run()), daemon=True)
    thread.start()

    while (event := q.get()) is not None:
        yield event

    thread.join()
