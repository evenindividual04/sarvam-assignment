"""Tests for the C3 calibration metric (model self-confidence vs judge
confidence) defined in `eval/judge.py`.

Definition recap:
  model_self_confidence ∈ [0,1]   from hedge ↔ assertion phrase balance
  judge_confidence      ∈ [0,1]   from faithfulness + context_precision
  per-case error        = |model - judge|
  calibration_score     = 1 - mean(error)        (higher is better)
  brier_score           = 1 - mean(error^2)
"""
from __future__ import annotations

import pytest

from eval.judge import (
    _compute_judge_confidence,
    _compute_model_self_confidence,
    compute_calibration_score,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _row(qid: str, category: str, answer: str, faith: float, ctxp: float) -> dict:
    return {
        "question_id": qid,
        "category": category,
        "agent_answer": answer,
        "faithfulness_score": faith,
        "context_precision_score": ctxp,
    }


_HEDGE_ANSWER = (
    "It is unclear what the exact figure is; we could not find a definitive "
    "source. Evidence is mixed and we are unable to confirm. [UNCERTAINTY] "
    "sources differ on this point and the data is not publicly available."
)

_ASSERT_ANSWER = (
    "The rate is 6.5 percent. According to the central bank, this was "
    "established in the latest review. The figure is confirmed and is "
    "definitively the current policy rate."
)


# ── Model self-confidence ───────────────────────────────────────────────


def test_model_self_confidence_hedge_heavy_is_low():
    score = _compute_model_self_confidence(_HEDGE_ANSWER)
    assert 0.0 <= score < 0.3, f"hedge-heavy must be low, got {score}"


def test_model_self_confidence_assertion_heavy_is_high():
    score = _compute_model_self_confidence(_ASSERT_ANSWER)
    assert 0.7 < score <= 1.0, f"assert-heavy must be high, got {score}"


def test_model_self_confidence_neutral_for_no_signal():
    assert _compute_model_self_confidence("") == pytest.approx(0.5)
    assert _compute_model_self_confidence("xyz qqq") == pytest.approx(0.5)


# ── Judge confidence ────────────────────────────────────────────────────


def test_judge_confidence_zero_to_one_scale():
    assert _compute_judge_confidence(0.2, 0.2) == pytest.approx(0.2)
    assert _compute_judge_confidence(1.0, 1.0) == pytest.approx(1.0)
    assert _compute_judge_confidence(0.0, 0.0) == pytest.approx(0.0)


def test_judge_confidence_handles_missing_components():
    assert _compute_judge_confidence(None, None) is None
    assert _compute_judge_confidence(0.8, None) == pytest.approx(0.8)


# ── compute_calibration_score: required cases from the spec ────────────


def test_empty_rows_returns_perfect_calibration():
    """No cases → no error possible → score = 1.0."""
    out = compute_calibration_score([])
    assert out["calibration_score"] == 1.0
    assert out["brier_score"] == 1.0
    assert out["n_cases"] == 0
    assert out["per_case"] == []


def test_perfectly_calibrated_hedger_with_weak_evidence():
    """Model hedges (low self-conf) AND judge says evidence is weak (low).
    Errors stay small, calibration is high."""
    rows = [
        _row("Q1", "insufficient_evidence", _HEDGE_ANSWER, faith=0.05, ctxp=0.05),
        _row("Q2", "conflicting", _HEDGE_ANSWER, faith=0.0, ctxp=0.1),
    ]
    out = compute_calibration_score(rows)
    assert out["n_cases"] == 2
    # Both ends are near-zero — calibration should be near-perfect.
    assert out["calibration_score"] > 0.9, out
    assert out["mean_abs_error"] < 0.1


def test_overconfident_agent_scores_low():
    """Model asserts (≈0.9) but judge says evidence is weak (≈0.2). Low score."""
    rows = [
        _row("Q1", "insufficient_evidence", _ASSERT_ANSWER, faith=0.2, ctxp=0.2),
        _row("Q2", "insufficient_evidence", _ASSERT_ANSWER, faith=0.2, ctxp=0.2),
    ]
    out = compute_calibration_score(rows)
    assert out["n_cases"] == 2
    assert out["calibration_score"] < 0.4, out
    assert out["mean_abs_error"] > 0.6


def test_overhedger_agent_scores_low_symmetrically():
    """Model hedges (≈0.1) but judge says evidence is strong (≈0.9). Low score."""
    rows = [
        _row("Q1", "conflicting", _HEDGE_ANSWER, faith=0.95, ctxp=0.95),
        _row("Q2", "conflicting", _HEDGE_ANSWER, faith=0.90, ctxp=0.90),
    ]
    out = compute_calibration_score(rows)
    assert out["n_cases"] == 2
    assert out["calibration_score"] < 0.4, out


# ── Scoping + structure ────────────────────────────────────────────────


def test_only_scoped_categories_are_counted():
    """factual / multi_hop rows must be ignored — C3 is only for evidence-
    stressed categories."""
    rows = [
        _row("Q1", "factual", _ASSERT_ANSWER, faith=0.9, ctxp=0.9),
        _row("Q2", "multi_hop", _ASSERT_ANSWER, faith=0.1, ctxp=0.1),
        _row("Q3", "insufficient_evidence", _HEDGE_ANSWER, faith=0.2, ctxp=0.2),
    ]
    out = compute_calibration_score(rows)
    assert out["n_cases"] == 1
    assert out["per_case"][0]["question_id"] == "Q3"


def test_per_category_breakdown_present():
    rows = [
        _row("Q1", "insufficient_evidence", _HEDGE_ANSWER, faith=0.2, ctxp=0.2),
        _row("Q2", "conflicting", _ASSERT_ANSWER, faith=0.2, ctxp=0.2),
    ]
    out = compute_calibration_score(rows)
    assert set(out["per_category"]) == {"insufficient_evidence", "conflicting"}
    for cat_stats in out["per_category"].values():
        assert "calibration_score" in cat_stats
        assert "brier_score" in cat_stats
        assert "n_cases" in cat_stats


def test_brier_lower_for_larger_errors():
    """Brier penalizes large errors more than MAE — verify it reacts."""
    close = compute_calibration_score(
        [_row("Q1", "insufficient_evidence", _HEDGE_ANSWER, 0.2, 0.2)]
    )
    far = compute_calibration_score(
        [_row("Q1", "insufficient_evidence", _ASSERT_ANSWER, 0.2, 0.2)]
    )
    assert close["brier_score"] > far["brier_score"]
    assert close["calibration_score"] > far["calibration_score"]


def test_rows_missing_judge_signal_are_skipped():
    rows = [
        {
            "question_id": "Q1",
            "category": "insufficient_evidence",
            "agent_answer": _HEDGE_ANSWER,
            "faithfulness_score": None,
            "context_precision_score": None,
        },
    ]
    out = compute_calibration_score(rows)
    assert out["n_cases"] == 0
    assert out["calibration_score"] == 1.0
