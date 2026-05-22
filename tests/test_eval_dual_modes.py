"""B2 — dual eval modes: --no-judge and --judge-only.

Three assertions:
1. `--no-judge` runs the deterministic-only scorer and never calls any
   LLM-judge function in eval.judge.
2. `--judge-only --run-id X` loads a prior run's cached artifacts (from
   JSONL + the SQLite `turns` table) and re-executes the judge pipeline
   without invoking the live agent.
3. The CLI rejects `--no-judge --judge-only` as mutually exclusive.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from eval import eval_runner


# ── Test 1: --no-judge skips all judge calls ──────────────────────────────


def test_no_judge_uses_deterministic_scorer_only():
    """The deterministic-only path must not touch any LLM-judge function.

    We exercise the scoring helper directly: it's the single entry point
    that the `run_eval(skip_judge=True)` path takes. If a future change
    routes a judge call through here, this test will fail loudly.
    """
    targets = [
        "judge_faithfulness", "judge_relevance", "judge_context_precision",
        "judge_claim_precision", "judge_conflict_adherence",
        "judge_coherence", "judge_with_dual_family",
    ]
    with patch.multiple(
        eval_runner,
        **{name: AsyncMock(side_effect=AssertionError(
            f"{name} called under --no-judge"
        )) for name in targets},
    ):
        scores = eval_runner._deterministic_only_scores(
            internal_answer="The answer is 42 [doc_1].",
            doc_map={"doc_1": ("Example", "https://example.com/a")},
            fetched_urls={"https://example.com/a"},
            q={"id": "T-1", "query": "What is the answer?"},
            answer="The answer is 42 [doc_1].",
        )

    # LLM-judge scores are None (the sentinel that signals "skipped");
    # deterministic citation-integrity score is real.
    assert scores["faithfulness_score"] is None
    assert scores["relevance_score"] is None
    assert scores["context_precision_score"] is None
    assert scores["claim_precision_score"] is None
    assert scores["ci_res"].citation_integrity_score == pytest.approx(1.0)


# ── Test 2: --judge-only loads cache + re-runs judges ─────────────────────


@pytest.mark.asyncio
async def test_judge_only_replays_from_cache(tmp_path, monkeypatch):
    """`--judge-only --run-id <iso>` must:
       (a) discover the prior JSONL by `run_at`,
       (b) pull `context_xml_sent` + `doc_map` from `turns` table,
       (c) call the judge functions with those cached artifacts,
       (d) never touch ResearchOrchestrator / network providers.
    """
    # Redirect results dir + DB to tmp so the test is hermetic.
    monkeypatch.setattr(eval_runner, "_RESULTS_DIR", tmp_path)
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(eval_runner, "DB_PATH", str(db_path))
    # The cache loader also reads agent.memory.DB_PATH indirectly via
    # aiosqlite.connect(DB_PATH) — patch the module-level binding too.
    import agent.memory as _mem
    monkeypatch.setattr(_mem, "DB_PATH", str(db_path))

    # Seed a "prior run" JSONL.
    prior_run_at = "2026-05-22T12:00:00+00:00"
    prior_jsonl = tmp_path / "eval_20260522_120000.jsonl"
    prior_row = {
        "run_at": prior_run_at,
        "question_id": "T-1",
        "question": "What is the capital of France?",
        "category": "factual",
        "language": "en",
        "agent_answer": "The capital is Paris [doc_1].",
        "turn_id": "turn-abc",
        "retrieval_mode": "bm25",
        "latency_ms": 1234,
    }
    prior_jsonl.write_text(json.dumps(prior_row) + "\n", encoding="utf-8")

    # Seed the turns table with the cached artifacts.
    import aiosqlite
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            "CREATE TABLE turns (turn_id TEXT PRIMARY KEY, context_xml_sent TEXT, "
            "doc_map TEXT, urls_opened TEXT, response TEXT)"
        )
        await db.execute(
            "INSERT INTO turns VALUES (?,?,?,?,?)",
            (
                "turn-abc",
                "<context><doc id='doc_1'>Paris is France's capital.</doc></context>",
                json.dumps({"doc_1": "https://en.wikipedia.org/wiki/Paris"}),
                json.dumps(["https://en.wikipedia.org/wiki/Paris"]),
                "The capital is Paris [doc_1].",
            ),
        )
        await db.commit()

    cached = await eval_runner._load_cached_rows_for_replay(prior_run_at)
    assert len(cached) == 1
    row = cached[0]
    assert "Paris is France's capital" in row["_context_xml"]
    assert row["_doc_map"] == {"doc_1": "https://en.wikipedia.org/wiki/Paris"}
    assert row["_fetched_urls"] == {"https://en.wikipedia.org/wiki/Paris"}
    assert row["_full_answer"] == "The capital is Paris [doc_1]."

    # Verify the judge functions are actually invoked with cached context.
    captured: dict[str, object] = {}

    async def _fake_faith(ctx, ans):
        captured["faith_ctx"] = ctx
        captured["faith_ans"] = ans
        return type("R", (), {"faithfulness_score": 0.93})()

    async def _fake_rel(q, ans):
        return type("R", (), {"answer_relevance_score": 0.88})()

    async def _fake_ctxp(q, ctx):
        return type("R", (), {"context_precision_score": 0.81})()

    # Block any accidental orchestrator construction during replay.
    with patch.object(eval_runner, "judge_faithfulness", side_effect=_fake_faith), \
         patch.object(eval_runner, "judge_relevance", side_effect=_fake_rel), \
         patch.object(eval_runner, "judge_context_precision", side_effect=_fake_ctxp), \
         patch.object(eval_runner, "judge_claim_precision", AsyncMock(
             return_value=type("R", (), {
                 "claim_precision_score": 1.0, "reasoning": "ok"
             })()
         )), \
         patch.object(eval_runner, "ResearchOrchestrator",
                      side_effect=AssertionError("agent must not run under --judge-only")):
        await eval_runner.run_judge_only(prior_run_at)

    # The judge saw the cached context (proving cache-replay, not re-search).
    assert "Paris is France's capital" in captured["faith_ctx"]
    assert captured["faith_ans"] == "The capital is Paris [doc_1]."

    # A new JSONL was written with mode="judge_only".
    new_jsonls = list(tmp_path.glob("eval_judgeonly_*.jsonl"))
    assert len(new_jsonls) == 1
    out_rows = [json.loads(line) for line in new_jsonls[0].read_text().splitlines() if line.strip()]
    assert len(out_rows) == 1
    assert out_rows[0]["mode"] == "judge_only"
    assert out_rows[0]["faithfulness_score"] == pytest.approx(0.93)


# ── Test 3: mutual exclusion ──────────────────────────────────────────────


def test_no_judge_and_judge_only_are_mutually_exclusive():
    """argparse must reject both flags simultaneously."""
    result = subprocess.run(
        [sys.executable, "-m", "eval.eval_runner",
         "--no-judge", "--judge-only", "--run-id", "x", "--skip-preflight"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    # argparse's standard mutually-exclusive error message.
    assert "not allowed" in combined or "mutually exclusive" in combined


def test_judge_only_requires_run_id():
    """`--judge-only` without `--run-id` must exit with a clear error."""
    result = subprocess.run(
        [sys.executable, "-m", "eval.eval_runner",
         "--judge-only", "--skip-preflight"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 2
    assert "run-id" in (result.stderr + result.stdout).lower()
