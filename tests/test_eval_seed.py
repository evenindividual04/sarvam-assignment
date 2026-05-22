"""Smoke tests for utils.eval_seed — JSONL idempotent seed-on-empty behavior."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def isolated_db_and_results(monkeypatch, tmp_path):
    """Point eval_seed at a temp DB + temp results dir."""
    db = tmp_path / "db.sqlite"
    monkeypatch.setattr("agent.memory.DB_PATH", str(db))
    monkeypatch.setattr("utils.eval_seed.DB_PATH", str(db))

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr("utils.eval_seed._RESULTS_DIR", results_dir)

    # Initialize schema
    from agent.memory import init_db
    asyncio.run(init_db())
    return db, results_dir


def test_seed_no_results_dir(monkeypatch, tmp_path):
    """Missing _RESULTS_DIR returns 0 without error."""
    db = tmp_path / "db.sqlite"
    monkeypatch.setattr("agent.memory.DB_PATH", str(db))
    monkeypatch.setattr("utils.eval_seed.DB_PATH", str(db))
    monkeypatch.setattr("utils.eval_seed._RESULTS_DIR", tmp_path / "does-not-exist")
    from agent.memory import init_db
    asyncio.run(init_db())
    from utils.eval_seed import seed_if_empty
    n = asyncio.run(seed_if_empty())
    assert n == 0


def test_seed_no_jsonl_files(isolated_db_and_results):
    """Empty _RESULTS_DIR returns 0."""
    from utils.eval_seed import seed_if_empty
    n = asyncio.run(seed_if_empty())
    assert n == 0


def test_seed_imports_jsonl_rows(isolated_db_and_results):
    """Two rows in one JSONL file should yield 2 inserts."""
    _, results = isolated_db_and_results
    rows = [
        {
            "run_id": "r1", "run_at": "2026-05-20T00:00:00+00:00",
            "question_id": "F-1", "question": "What is X?", "category": "factual",
            "agent_answer": "X is Y.",
            "faithfulness_score": 0.9, "latency_ms": 1500,
        },
        {
            "run_id": "r1", "run_at": "2026-05-20T00:00:01+00:00",
            "question_id": "F-2", "question": "What is Z?", "category": "factual",
            "agent_answer": "Z is W.",
            "faithfulness_score": 0.85, "latency_ms": 1700,
        },
    ]
    (results / "eval_smoke.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    from utils.eval_seed import seed_if_empty
    n = asyncio.run(seed_if_empty())
    assert n == 2


def test_seed_idempotent_when_table_populated(isolated_db_and_results):
    """A second seed call after rows exist must be a no-op."""
    _, results = isolated_db_and_results
    (results / "eval_smoke.jsonl").write_text(
        json.dumps({
            "run_id": "r1", "run_at": "2026-05-20T00:00:00+00:00",
            "question_id": "F-1", "question": "q", "category": "factual",
            "agent_answer": "a", "latency_ms": 1,
        }) + "\n", encoding="utf-8"
    )
    from utils.eval_seed import seed_if_empty
    first = asyncio.run(seed_if_empty())
    assert first == 1
    second = asyncio.run(seed_if_empty())
    assert second == 0  # idempotent


def test_seed_skips_malformed_lines(isolated_db_and_results):
    """A bad JSON line shouldn't abort the whole import."""
    _, results = isolated_db_and_results
    valid = json.dumps({
        "run_id": "r1", "run_at": "2026-05-20T00:00:00+00:00",
        "question_id": "F-1", "question": "q", "category": "factual",
        "agent_answer": "a", "latency_ms": 1,
    })
    (results / "mixed.jsonl").write_text(
        f"not-valid-json\n{valid}\n", encoding="utf-8",
    )
    from utils.eval_seed import seed_if_empty
    n = asyncio.run(seed_if_empty())
    assert n == 1
