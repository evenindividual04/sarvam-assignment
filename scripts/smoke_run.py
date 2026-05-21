from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_check import validate
from agent.memory import init_db
from agent.orchestrator import ResearchOrchestrator


async def _run_once(query: str) -> None:
    await init_db()
    agent = ResearchOrchestrator()
    session_id = str(uuid.uuid4())

    done_payload = None
    try:
        async for event in agent.run(query, session_id):
            if event.step == "done":
                done_payload = event.data or {}
                break
    finally:
        await agent.aclose()

    if not done_payload:
        raise RuntimeError("Smoke run did not reach done state")

    print("SMOKE_OK")
    print(f"session_id={session_id}")
    print(f"turn_id={done_payload.get('turn_id')}")
    print(f"citation_integrity_score={done_payload.get('citation_integrity_score')}")
    print(f"prompt_tokens={done_payload.get('prompt_tokens')}")
    print(f"completion_tokens={done_payload.get('completion_tokens')}")
    print(f"url_count={len(done_payload.get('urls') or [])}")
    print(f"planning_ms={done_payload.get('planning_ms', 0)}")
    print(f"search_ms={done_payload.get('search_ms', 0)}")
    print(f"fetch_ms={done_payload.get('fetch_ms', 0)}")
    print(f"select_ms={done_payload.get('select_ms', 0)}")
    print(f"synthesize_ms={done_payload.get('synthesize_ms', 0)}")
    meta = done_payload.get('run_metadata', {}) or {}
    print(f"fallback_path_taken={meta.get('fallback_path_taken', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot smoke run for the research agent")
    parser.add_argument(
        "--query",
        default="What is the current repo rate set by the RBI?",
        help="Question used for smoke testing",
    )
    args = parser.parse_args()

    validate()
    asyncio.run(_run_once(args.query))


if __name__ == "__main__":
    main()
