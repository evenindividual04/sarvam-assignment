"""Tier C — cross-family judge rotation tests.

Verifies dual-family score collection, agreement aggregates, and the
quota guard that prevents blowing past GitHub Models' 150/day cap.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from eval import judge as judge_mod
from eval.judge import (
    cohens_kappa_bucketed,
    get_cross_family_call_count,
    judge_with_dual_family,
    pearson_correlation,
    reset_cross_family_counter,
)


@pytest.fixture(autouse=True)
def _reset_counter():
    reset_cross_family_counter()
    yield
    reset_cross_family_counter()


def test_judge_with_dual_family_returns_both_scores(monkeypatch):
    """Both providers respond → both scores + delta computed."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GITHUB_TOKEN", "test-github")

    async def fake_provider_call(prompt: str, provider: str) -> str:
        score = 0.80 if provider == "groq" else 0.65
        return json.dumps({"faithfulness_score": score, "reasoning": "x"})

    monkeypatch.setattr(judge_mod, "_judge_with_provider", fake_provider_call)
    out = asyncio.run(judge_with_dual_family("PROMPT", "faithfulness"))
    assert out["primary_score"] == pytest.approx(0.80)
    assert out["secondary_score"] == pytest.approx(0.65)
    assert out["agreement_delta"] == pytest.approx(0.15, abs=1e-6)


def test_judge_with_dual_family_handles_missing_secondary(monkeypatch):
    """Secondary judge raises (e.g. no GitHub token) → secondary=None, no exception."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    async def fake_provider_call(prompt: str, provider: str) -> str:
        if provider == "github":
            raise RuntimeError("GITHUB_TOKEN not set")
        return json.dumps({"faithfulness_score": 0.9, "reasoning": "ok"})

    monkeypatch.setattr(judge_mod, "_judge_with_provider", fake_provider_call)
    out = asyncio.run(judge_with_dual_family("PROMPT", "faithfulness"))
    assert out["primary_score"] == pytest.approx(0.9)
    assert out["secondary_score"] is None
    assert out["agreement_delta"] is None


def test_cohens_kappa_computation_perfect_agreement():
    """Both raters give identical bucket assignments → κ near 1."""
    primary = [0.1, 0.5, 0.9, 0.2, 0.8, 0.5]
    secondary = [0.15, 0.55, 0.95, 0.25, 0.85, 0.55]
    kappa = cohens_kappa_bucketed(primary, secondary)
    assert kappa is not None
    assert kappa == pytest.approx(1.0, abs=1e-6)


def test_cohens_kappa_computation_random_agreement():
    """Random anti-aligned data → κ near 0 (or negative)."""
    # Construct buckets [0,1,2,0,1,2,0,1,2] vs [2,1,0,2,1,0,2,1,0] — same
    # marginals but anticorrelated → po=1/3, pe=1/3 → κ=0.
    primary = [0.1, 0.5, 0.9] * 3
    secondary = [0.9, 0.5, 0.1] * 3
    kappa = cohens_kappa_bucketed(primary, secondary)
    assert kappa is not None
    # Off-diagonal pairs give κ exactly 0 here.
    assert abs(kappa) < 0.1


def test_cohens_kappa_degenerate_single_bucket():
    """All scores in one bucket → pe=1 → returns None."""
    primary = [0.9] * 5
    secondary = [0.9] * 5
    assert cohens_kappa_bucketed(primary, secondary) is None


def test_pearson_correlation_computation():
    """Known linear data → r=1.0."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    r = pearson_correlation(xs, ys)
    assert r is not None
    assert r == pytest.approx(1.0, abs=1e-6)


def test_pearson_correlation_anticorrelated():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [5.0, 4.0, 3.0, 2.0, 1.0]
    r = pearson_correlation(xs, ys)
    assert r is not None
    assert r == pytest.approx(-1.0, abs=1e-6)


def test_pearson_correlation_degenerate():
    assert pearson_correlation([], []) is None
    assert pearson_correlation([1.0], [2.0]) is None
    assert pearson_correlation([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) is None


def test_cross_family_quota_guard_truncates(monkeypatch):
    """Once GitHub Models calls exceed the budget, secondary judging halts."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GITHUB_TOKEN", "test-github")
    monkeypatch.setenv("CROSS_FAMILY_QUOTA_BUDGET", "3")
    reset_cross_family_counter()

    call_log: list[str] = []

    async def fake_provider_call(prompt: str, provider: str) -> str:
        call_log.append(provider)
        return json.dumps({"faithfulness_score": 0.7, "reasoning": "x"})

    monkeypatch.setattr(judge_mod, "_judge_with_provider", fake_provider_call)

    async def _run():
        results = []
        for _ in range(5):
            results.append(await judge_with_dual_family("P", "faithfulness"))
        return results

    out = asyncio.run(_run())
    # First 3 should have secondary scores; 4th and 5th should be None (truncated).
    truncated = [r for r in out if r["secondary_score"] is None]
    assert len(truncated) >= 2
    # GitHub counter capped at the budget.
    assert get_cross_family_call_count() == 3
