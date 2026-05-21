"""Unit tests for the eval pipeline upgrade — gold-truth factual judge,
cross-language consistency, expanded failure taxonomy, cost model,
calibration math, ablation tagging, and dataset invariants."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Repo root on path for direct imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.judge import (  # noqa: E402
    classify_failure,
    jaccard,
    judge_cross_language_consistency,
    judge_factual_accuracy,
)
from eval.eval_runner import _compute_calibration, _pearson, _percentile  # noqa: E402
from utils.cost_model import cost_for  # noqa: E402


DATASET = json.loads((ROOT / "eval" / "dataset.json").read_text())


# ── Factual accuracy ────────────────────────────────────────────────────────


def test_factual_judge_substring_match():
    score, _ = judge_factual_accuracy("The capital is New Delhi", "New Delhi", None, None)
    assert score == 1.0


def test_factual_judge_alias_match():
    score, reason = judge_factual_accuracy("It's Delhi", "New Delhi", ["Delhi"], None)
    assert score == 1.0
    assert "Delhi" in reason


def test_factual_judge_no_match():
    score, reason = judge_factual_accuracy("The answer is Lyon", "Paris", None, None)
    assert score == 0.0
    assert "No match" in reason


def test_factual_judge_entity_intersection_fallback():
    # Substring doesn't match exact gold, but entity Modi resolves.
    score, _ = judge_factual_accuracy("PM Modi led", "Narendra Modi", None, ["Modi"])
    assert score == 1.0


def test_factual_judge_skips_when_no_gold():
    score, reason = judge_factual_accuracy("anything", None, None, None)
    assert score is None
    assert "skipped" in reason.lower()


# ── Cross-language consistency ──────────────────────────────────────────────


def test_jaccard_perfect_overlap():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial():
    s = jaccard({"a", "b", "c"}, {"b", "c", "d"})
    # |∩|=2 |∪|=4 -> 0.5
    assert s == 0.5


def test_cross_language_jaccard_perfect_overlap_via_db(tmp_path, monkeypatch):
    """Stub a tiny dataset with one pair, write matching answers, run the judge.
    Both answers share the exact same set of extracted entities."""
    asyncio.run(_run_cross_language_check(
        monkeypatch, tmp_path,
        en_answer="india Modi 2024",
        hi_answer="india Modi 2024",
        expect_flagged=False,
    ))


def test_cross_language_flags_low_overlap(tmp_path, monkeypatch):
    asyncio.run(_run_cross_language_check(
        monkeypatch, tmp_path,
        en_answer="Mumbai is large",
        hi_answer="कोलकाता पुराना शहर है",  # totally different entities
        expect_flagged=True,
    ))


async def _run_cross_language_check(monkeypatch, tmp_path, *, en_answer, hi_answer, expect_flagged):
    import aiosqlite

    test_db = tmp_path / "t.db"
    monkeypatch.setenv("DB_PATH", str(test_db))
    # Re-import memory + judge with the new DB_PATH
    import importlib
    import agent.memory as memory
    importlib.reload(memory)
    import eval.judge as judge_mod
    importlib.reload(judge_mod)

    # Stub dataset with a single concept pair.
    fake_dataset = [
        {"id": "EN-1", "language": "en", "concept_id": "test-concept",
         "query": "q", "category": "factual"},
        {"id": "HI-1", "language": "hi", "concept_id": "test-concept",
         "query": "q", "category": "factual"},
    ]
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(json.dumps(fake_dataset))
    monkeypatch.setattr(
        judge_mod, "__file__",
        str(tmp_path / "judge.py"),  # makes Path(__file__).parent / "dataset.json" resolve here
    )

    # Bootstrap DB and seed two answers under one run_at.
    await memory.init_db()
    run_at = "2025-01-01T00:00:00+00:00"
    async with aiosqlite.connect(memory.DB_PATH) as db:
        for qid, ans in (("EN-1", en_answer), ("HI-1", hi_answer)):
            await db.execute(
                """INSERT INTO eval_runs
                (run_id, run_at, question_id, question, category, agent_answer, language)
                VALUES (?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), run_at, qid, "q", "factual", ans, qid[:2].lower()),
            )
        await db.commit()

    rows = await judge_mod.judge_cross_language_consistency(run_at)
    assert len(rows) == 1
    if expect_flagged:
        assert rows[0]["flagged_inconsistent"] is True
        assert rows[0]["jaccard_score"] < 0.6
    else:
        assert rows[0]["flagged_inconsistent"] is False
        assert rows[0]["jaccard_score"] >= 0.6


# ── Failure taxonomy ────────────────────────────────────────────────────────


def test_failure_taxonomy_hallucination_fact_classification():
    cls = classify_failure({
        "faithfulness_score": 0.3,
        "answer_relevance_score": 0.9,
        "context_precision_score": 0.9,
        "citation_integrity_score": 0.9,
        "claim_precision_score": 0.2,
    })
    assert cls == "HALLUCINATION_FACT"


def test_failure_taxonomy_hallucination_attribution_classification():
    # High faith + high citation integrity, but claim precision low -> attribution bucket.
    cls = classify_failure({
        "faithfulness_score": 0.85,
        "answer_relevance_score": 0.9,
        "context_precision_score": 0.9,
        "citation_integrity_score": 0.95,
        "claim_precision_score": 0.3,
    })
    assert cls == "HALLUCINATION_ATTRIBUTION"


def test_failure_taxonomy_preserves_pass():
    cls = classify_failure({
        "faithfulness_score": 0.95,
        "answer_relevance_score": 0.91,
        "context_precision_score": 0.9,
        "citation_integrity_score": 1.0,
        "claim_precision_score": 0.9,
        "conflict_adherence_score": None,
        "session_coherence_score": None,
    })
    assert cls == "PASS"


def test_failure_taxonomy_retrieval_failure():
    cls = classify_failure({
        "faithfulness_score": 0.9,
        "answer_relevance_score": 0.3,  # low relevance triggers retrieval
        "context_precision_score": 0.9,
        "citation_integrity_score": 0.9,
        "claim_precision_score": 0.9,
    })
    assert cls == "RETRIEVAL_FAILURE"


# ── Cost model ──────────────────────────────────────────────────────────────


def test_cost_for_zero_on_free_tier_model():
    assert cost_for("gemini-2.5-flash", 10_000, 5_000) == 0.0
    assert cost_for("groq-llama-3.3-70b", 1_000_000, 1_000_000) == 0.0


def test_cost_for_unknown_model_returns_zero():
    assert cost_for("brand-new-model", 1000, 1000) == 0.0


# ── Calibration math ────────────────────────────────────────────────────────


def test_pearson_perfect_positive():
    assert _pearson([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_pearson_no_signal_zero_variance():
    # If y has no variance, correlation is defined as 0 (not nan) by our impl.
    assert _pearson([1, 2, 3], [5, 5, 5]) == 0.0


def test_calibration_correlation_perfect_signal():
    results = []
    for conf, faith in [("high", 1.0), ("high", 0.95), ("medium", 0.5),
                        ("medium", 0.55), ("low", 0.05), ("low", 0.1)]:
        results.append({
            "faithfulness_score": faith,
            "claim_precision_score": faith,
            "failure_class": "PASS",
            "run_metadata": {"planner_output": {"confidence": conf}},
        })
    out = _compute_calibration(results)
    assert out["correlation"] is not None
    assert out["correlation"] > 0.9


def test_calibration_correlation_no_signal():
    # All buckets same faithfulness -> correlation ~0
    results = [
        {"faithfulness_score": 0.5, "claim_precision_score": 0.5,
         "failure_class": "PASS",
         "run_metadata": {"planner_output": {"confidence": c}}}
        for c in ("low", "low", "medium", "medium", "high", "high")
    ]
    out = _compute_calibration(results)
    assert out["correlation"] == 0.0


def test_percentile_basic():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(xs, 0.5) == 3.0
    assert _percentile(xs, 0.0) == 1.0
    assert _percentile(xs, 1.0) == 5.0


# ── Dataset invariants ──────────────────────────────────────────────────────


def test_dataset_loaded_has_correct_category_counts():
    by_cat = Counter(q["category"] for q in DATASET)
    for cat in ("factual", "multi_hop", "comparison", "insufficient_evidence",
                "conflicting", "multi_turn"):
        assert by_cat[cat] >= 4, f"{cat} has only {by_cat[cat]} questions"


def test_dataset_has_at_least_6_concept_id_pairs():
    pairs: dict[str, dict[str, str]] = defaultdict(dict)
    for q in DATASET:
        cid = q.get("concept_id")
        if cid:
            pairs[cid][q["language"]] = q["id"]
    paired = [c for c, langs in pairs.items() if "en" in langs and "hi" in langs]
    assert len(paired) >= 6, f"only {len(paired)} EN/HI concept pairs found"


def test_dataset_factual_with_gold_have_string_answer():
    for q in DATASET:
        if "gold_answer" in q:
            assert isinstance(q["gold_answer"], str) and q["gold_answer"]
            assert q["category"] == "factual"


def test_dataset_total_count_at_least_40():
    assert len(DATASET) >= 40


# ── Ablation tagging integration ────────────────────────────────────────────


def test_ablation_runner_tags_rows_with_same_id(tmp_path, monkeypatch):
    """Spot-check: when we synthesize two leg-result lists tagged with the
    same ablation_id, they should round-trip through eval_runs preserving
    that grouping."""
    asyncio.run(_ablation_tagging_roundtrip(tmp_path, monkeypatch))


async def _ablation_tagging_roundtrip(tmp_path, monkeypatch):
    import aiosqlite

    test_db = tmp_path / "abl.db"
    monkeypatch.setenv("DB_PATH", str(test_db))
    import importlib
    import agent.memory as memory
    importlib.reload(memory)
    await memory.init_db()

    ablation_id = str(uuid.uuid4())
    rows = []
    for mode in ("bm25", "hybrid"):
        for i in range(3):
            rows.append((
                str(uuid.uuid4()), "2025-01-01T00:00:00Z", f"Q-{i}", "q?", "factual",
                "ans", 0.8, 0.8, 0.8, 1.0, None, None, 0.9, "",
                "PASS", 100, None, "en", mode, None, ablation_id, None,
            ))
    async with aiosqlite.connect(memory.DB_PATH) as db:
        await db.executemany(
            """INSERT INTO eval_runs
            (run_id, run_at, question_id, question, category, agent_answer,
             faithfulness_score, answer_relevance_score, context_precision_score,
             citation_integrity_score, conflict_adherence_score, session_coherence_score,
             claim_precision_score, judge_reasoning, failure_class, latency_ms,
             turn_id, language, retrieval_mode, factual_accuracy_score, ablation_id,
             calibration_correlation)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT retrieval_mode, COUNT(*) FROM eval_runs WHERE ablation_id = ? GROUP BY retrieval_mode",
            (ablation_id,),
        )
        out = await cursor.fetchall()
    assert sorted(out) == [("bm25", 3), ("hybrid", 3)]
