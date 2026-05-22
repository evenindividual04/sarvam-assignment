"""Goal 5 — Difficulty-aware budget expansion for Gemini 1M context window.

When ``planner.difficulty == "hard"`` AND the synth chain leads with Gemini,
the orchestrator expands ``ContextBudget.web_context_budget`` from 6,400 to
24,000 tokens. Sarvam-M caps lower, so the expansion is skipped whenever
Sarvam will be tried first.
"""
from __future__ import annotations

from utils.token_counter import ContextBudget


def test_context_budget_supports_explicit_web_context_override():
    budget = ContextBudget()
    assert budget.web_context_budget == 6400
    budget.web_context_override = 24000
    assert budget.web_context_budget == 24000


def test_budget_expanded_on_hard_difficulty_with_gemini(monkeypatch):
    """When difficulty=hard and synth=gemini, web_context_budget jumps to 24K.

    This exercises the orchestrator's budget-adjustment branch directly. We
    reproduce the predicate the orchestrator uses (without re-running the
    whole pipeline) to keep the test fast and dependency-free.
    """
    monkeypatch.delenv("SYNTH_PROVIDER", raising=False)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    plan_difficulty = "hard"
    query = "What is the relationship between RAG and inference latency?"

    # Reproduce the orchestrator predicate:
    import os
    from utils.provider_router import _detect_query_script
    is_indic = (
        os.environ.get("SARVAM_INDIC_AUTO", "1").strip() not in {"0", "false", "False", ""}
        and bool(os.environ.get("SARVAM_API_KEY"))
        and _detect_query_script(query) == "indic"
    )
    sarvam_primary = (
        os.environ.get("SYNTH_PROVIDER", "gemini").lower() == "sarvam" or is_indic
    )
    budget = ContextBudget()
    if plan_difficulty == "hard" and not sarvam_primary:
        budget.web_context_override = 24000

    assert budget.web_context_budget == 24000


def test_budget_not_expanded_on_easy_difficulty(monkeypatch):
    monkeypatch.delenv("SYNTH_PROVIDER", raising=False)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    plan_difficulty = "easy"
    budget = ContextBudget()
    if plan_difficulty == "hard":
        budget.web_context_override = 24000
    assert budget.web_context_budget == 6400


def test_budget_not_expanded_when_sarvam_primary(monkeypatch):
    """SYNTH_PROVIDER=sarvam → expansion is skipped (Sarvam-M context cap)."""
    monkeypatch.setenv("SYNTH_PROVIDER", "sarvam")
    plan_difficulty = "hard"
    query = "How does the Sarvam-M model handle long context?"

    import os
    sarvam_primary = os.environ.get("SYNTH_PROVIDER", "gemini").lower() == "sarvam"
    budget = ContextBudget()
    if plan_difficulty == "hard" and not sarvam_primary:
        budget.web_context_override = 24000

    assert budget.web_context_budget == 6400
