"""Smoke tests for eval.ablation_report — delta computation given two legs."""
from __future__ import annotations

import asyncio
import aiosqlite
import pytest

from eval.ablation_report import _safe_mean, _delta, generate_report


def test_safe_mean_empty():
    assert _safe_mean([]) is None
    assert _safe_mean([None, None]) is None


def test_safe_mean_ignores_none():
    assert _safe_mean([1.0, 2.0, None, 3.0]) == 2.0


def test_delta_basic():
    assert _delta(0.5, 0.8) == pytest.approx(0.3)
    assert _delta(None, 0.8) is None
    assert _delta(0.5, None) is None


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    db = tmp_path / "db.sqlite"
    monkeypatch.setattr("agent.memory.DB_PATH", str(db))
    monkeypatch.setattr("eval.ablation_report.DB_PATH", str(db))
    # Force results dir into tmp so generate_report doesn't write to repo
    monkeypatch.setattr("eval.ablation_report._RESULTS_DIR", tmp_path)
    from agent.memory import init_db
    asyncio.run(init_db())
    return db


async def _seed_ablation_rows(db_path: str, ablation_id: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        for mode, faith in [("bm25", 0.6), ("hybrid", 0.85)]:
            await db.execute(
                """
                INSERT INTO eval_runs (
                    run_id, run_at, question_id, question, category,
                    agent_answer, faithfulness_score, context_precision_score,
                    claim_precision_score, retrieval_mode, ablation_id,
                    failure_class, latency_ms
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"r-{mode}", "2026-05-20T00:00:00+00:00", "F-1",
                    "What is X?", "factual", "X is Y.",
                    faith, 0.7, 0.8, mode, ablation_id, "PASS", 1500,
                ),
            )
        await db.commit()


def test_generate_report_computes_overall_delta(isolated_db):
    asyncio.run(_seed_ablation_rows(str(isolated_db), "abl-test-1"))
    report = asyncio.run(generate_report("abl-test-1"))
    assert report["ablation_id"] == "abl-test-1"
    assert report["n_bm25"] == 1
    assert report["n_hybrid"] == 1
    overall = report["overall"]
    assert overall["faithfulness_score"]["bm25"] == pytest.approx(0.6)
    assert overall["faithfulness_score"]["hybrid"] == pytest.approx(0.85)
    assert overall["faithfulness_score"]["delta"] == pytest.approx(0.25)


def test_generate_report_no_rows_returns_error(isolated_db):
    report = asyncio.run(generate_report("does-not-exist"))
    assert "error" in report
