"""Tests for the three eval-methodology polish items:

1. Bootstrap 95% CI on top-line means (stdlib only).
2. Provenance header in the markdown report.
3. Script-preservation metric for Indic answers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.eval_runner import (
    _provenance_info,
    _render_provenance_block,
    _write_markdown_report,
    bootstrap_ci,
    script_preservation_ratio,
)


# ── Item 1: bootstrap CI ─────────────────────────────────────────────────


def test_bootstrap_ci_symmetric_on_uniform_data():
    """For perfectly uniform data the CI must collapse around the mean."""
    values = [0.5] * 50
    low, high = bootstrap_ci(values, n_resamples=500)
    assert low == pytest.approx(0.5, abs=1e-9)
    assert high == pytest.approx(0.5, abs=1e-9)


def test_bootstrap_ci_brackets_mean():
    """CI must bracket the sample mean and be deterministic with fixed seed."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sample_mean = sum(values) / len(values)
    low_a, high_a = bootstrap_ci(values, n_resamples=2000)
    low_b, high_b = bootstrap_ci(values, n_resamples=2000)
    assert low_a == low_b and high_a == high_b  # deterministic
    assert low_a <= sample_mean <= high_a
    assert high_a - low_a > 0  # non-degenerate


# ── Item 2: provenance header ────────────────────────────────────────────


def test_provenance_header_contains_git_sha(tmp_path: Path):
    prov = _provenance_info("2026-05-22T00:00:00+00:00")
    assert "git_sha" in prov
    assert prov["run_started_at"] == "2026-05-22T00:00:00+00:00"
    # Render and verify the block is present in a full markdown report.
    out = tmp_path / "report.md"
    _write_markdown_report(
        out,
        run_at_iso=prov["run_started_at"],
        retrieval_mode="bm25",
        ablation_id=None,
        results=[
            {
                "question_id": "T-1",
                "question": "Test?",
                "category": "factual",
                "language": "en",
                "agent_answer": "Answer.",
                "faithfulness_score": 0.9,
                "answer_relevance_score": 0.9,
                "context_precision_score": 0.9,
                "citation_integrity_score": 1.0,
                "claim_precision_score": 1.0,
                "failure_class": "PASS",
                "latency_ms": 100,
            }
        ],
        agg={"overall": {"n_questions": 1, "pass_rate": 1.0}, "by_category": {}},
        taxonomy=__import__("collections").Counter({"PASS": 1}),
        calibration={"correlation": None, "buckets": {}, "n_paired": 0},
        cl_rows=[],
    )
    text = out.read_text(encoding="utf-8")
    assert "Provenance" in text
    assert "git_sha:" in text
    assert "dataset_sha256:" in text
    assert "synthesizer_model:" in text
    assert "run_started_at:" in text


def test_provenance_render_block_keys():
    prov = {
        "git_sha": "abc123",
        "git_commit_timestamp": "2026-05-22T00:00:00+00:00",
        "dataset_path": "/x/dataset.json",
        "dataset_sha256": "deadbeef",
        "synthesizer_model": "gemini-2.5-flash",
        "planner_model": "llama-3.3-70b-versatile",
        "judge_model": "gpt-4o-mini",
        "run_started_at": "2026-05-22T00:00:00+00:00",
        "host": "test-host · Python 3.12.0",
    }
    rendered = "\n".join(_render_provenance_block(prov))
    for key in prov:
        assert key in rendered


# ── Item 3: script preservation ──────────────────────────────────────────


def test_script_preservation_flags_transliterated_hindi_answer():
    """A Hinglish/Latin-script answer for a Hindi query must fail the 0.80 gate."""
    transliterated = "RBI ka repo rate 6.5 percent hai abhi."
    ratio = script_preservation_ratio(transliterated, "hi")
    assert ratio is not None
    assert ratio < 0.80, f"transliterated answer should fail, got ratio={ratio}"


def test_script_preservation_passes_devanagari_answer():
    """A dominantly-Devanagari answer (with citations + English numbers stripped)
    should pass the 0.80 threshold."""
    answer = "आरबीआई की रेपो दर अभी ६.५ प्रतिशत है। [doc_1] देखें https://example.com"
    ratio = script_preservation_ratio(answer, "hi")
    assert ratio is not None
    assert ratio >= 0.80


def test_script_preservation_returns_none_for_english():
    assert script_preservation_ratio("Plain English answer.", "en") is None


def test_script_preservation_tamil_and_bengali():
    tamil = "வணக்கம் உலகம்"
    bengali = "নমস্কার পৃথিবী"
    assert script_preservation_ratio(tamil, "ta") == pytest.approx(1.0)
    assert script_preservation_ratio(bengali, "bn") == pytest.approx(1.0)


def test_script_preservation_handles_empty_after_stripping():
    """All-citations-and-URLs answer has no alpha → returns None."""
    assert script_preservation_ratio("[doc_1] https://example.com 123", "hi") is None
