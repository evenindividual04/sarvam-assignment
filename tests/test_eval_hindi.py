"""V3.4 — Hindi eval subset: API-level checks for the `language` column."""
from __future__ import annotations

import asyncio
import importlib
import os

import aiosqlite
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def hindi_app_client(tmp_path_factory):
    db = tmp_path_factory.mktemp("hindi_apidb") / "test.db"
    os.environ["DB_PATH"] = str(db)
    os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.eval_queries as eq_mod
    importlib.reload(eq_mod)
    import main as main_mod
    importlib.reload(main_mod)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(mem_mod.init_db())

    client = TestClient(main_mod.app)
    yield client, mem_mod, main_mod


async def _seed_mixed_language_run(mem_mod, run_at: str) -> None:
    async with aiosqlite.connect(mem_mod.DB_PATH) as db:
        # One English question
        await db.execute(
            """INSERT INTO eval_runs (run_id, run_at, question_id, question, category,
               agent_answer, faithfulness_score, answer_relevance_score,
               context_precision_score, citation_integrity_score,
               claim_precision_score, failure_class, latency_ms, language)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"run-{run_at}-en", run_at, "F-1", "What is X?", "factual",
                "ans", 0.9, 0.85, 0.8, 1.0, 0.95, "PASS", 1200, "en",
            ),
        )
        # One Hindi question
        await db.execute(
            """INSERT INTO eval_runs (run_id, run_at, question_id, question, category,
               agent_answer, faithfulness_score, answer_relevance_score,
               context_precision_score, citation_integrity_score,
               claim_precision_score, failure_class, latency_ms, language)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"run-{run_at}-hi", run_at, "HI-1", "भारत में रेपो दर क्या है?",
                "factual", "उत्तर", 0.8, 0.7, 0.75, 1.0, 0.9, "PASS", 1500, "hi",
            ),
        )
        await db.commit()


def test_eval_runs_includes_per_language_breakdown(hindi_app_client):
    client, mem_mod, _ = hindi_app_client
    run_at = "2026-05-21T10:00:00"
    asyncio.new_event_loop().run_until_complete(_seed_mixed_language_run(mem_mod, run_at))

    r = client.get("/eval/runs")
    assert r.status_code == 200
    rows = r.json()
    target = next((row for row in rows if row["run_at"] == run_at), None)
    assert target is not None
    assert "by_language" in target
    assert set(target["by_language"].keys()) >= {"en", "hi"}
    assert target["by_language"]["hi"]["n_questions"] == 1
    assert target["by_language"]["en"]["n_questions"] == 1


def test_eval_question_detail_includes_language_field(hindi_app_client):
    client, mem_mod, _ = hindi_app_client
    run_at = "2026-05-21T11:00:00"
    asyncio.new_event_loop().run_until_complete(_seed_mixed_language_run(mem_mod, run_at))

    r = client.get(f"/eval/runs/{run_at}/questions/HI-1")
    assert r.status_code == 200
    data = r.json()
    assert data["language"] == "hi"
    assert data["question_id"] == "HI-1"


def test_eval_runs_no_language_breakdown_when_only_english(hindi_app_client):
    """When a run has only English questions, ``by_language`` should be omitted
    to keep the default payload clean."""
    client, mem_mod, _ = hindi_app_client
    run_at = "2026-05-21T12:00:00"

    async def seed():
        async with aiosqlite.connect(mem_mod.DB_PATH) as db:
            await db.execute(
                """INSERT INTO eval_runs (run_id, run_at, question_id, question, category,
                   agent_answer, faithfulness_score, answer_relevance_score,
                   context_precision_score, citation_integrity_score,
                   claim_precision_score, failure_class, latency_ms, language)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"run-{run_at}-en", run_at, "F-1", "What is X?", "factual",
                    "ans", 0.9, 0.85, 0.8, 1.0, 0.95, "PASS", 1200, "en",
                ),
            )
            await db.commit()

    asyncio.new_event_loop().run_until_complete(seed())
    r = client.get("/eval/runs")
    assert r.status_code == 200
    target = next((row for row in r.json() if row["run_at"] == run_at), None)
    assert target is not None
    assert "by_language" not in target
