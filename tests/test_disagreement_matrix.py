"""B4 — structured disagreement matrix on conflict.

When the contradiction probe surfaces a real (non-temporal) conflict, the
synthesizer prompt MUST include explicit Markdown-table formatting
instructions, and the post-synthesis audit MUST flag answers that fail to
emit such a table.
"""
from __future__ import annotations

import re

import pytest

from agent.citation_guard import has_disagreement_matrix
from agent.models import ClaimContradiction, ConflictResult
from utils.provider_router import _build_disagreement_block


def _two_real_contradictions() -> ConflictResult:
    return ConflictResult(
        has_conflict=True,
        conflict_summary="Two sources disagree on the repo rate and on GDP.",
        contradictions=[
            ClaimContradiction(
                claim="RBI repo rate in May 2026",
                doc_ids_a=["doc_1"],
                position_a="6.50%",
                doc_ids_b=["doc_2"],
                position_b="5.50%",
                is_temporal_evolution=False,
                confidence=0.92,
            ),
            ClaimContradiction(
                claim="India GDP growth FY26",
                doc_ids_a=["doc_3"],
                position_a="7.1%",
                doc_ids_b=["doc_4"],
                position_b="6.4%",
                is_temporal_evolution=False,
                confidence=0.88,
            ),
        ],
    )


def test_disagreement_block_contains_markdown_table_instruction():
    """The prompt block must instruct the model to emit a `| Claim | Source A | Source B |`
    table and must include one row per non-temporal contradiction."""
    block = _build_disagreement_block(_two_real_contradictions())

    # Mandatory table-formatting cues.
    assert "**Sources disagree on this:**" in block
    assert "| Claim | Source A | Source B |" in block
    assert "|---|---|---|" in block

    # One row per contradiction with the per-side doc markers.
    assert "[doc_1]" in block and "[doc_2]" in block
    assert "[doc_3]" in block and "[doc_4]" in block

    # Heading text the synthesizer should reproduce verbatim.
    assert re.search(r"\bSources disagree\b", block)


def test_disagreement_block_empty_when_only_temporal_evolution():
    temporal = ConflictResult(
        has_conflict=True,
        contradictions=[
            ClaimContradiction(
                claim="repo rate",
                doc_ids_a=["doc_1"],
                position_a="6.50% in 2024",
                doc_ids_b=["doc_2"],
                position_b="5.50% in 2026",
                is_temporal_evolution=True,
                confidence=0.95,
            )
        ],
    )
    assert _build_disagreement_block(temporal) == ""


def test_disagreement_block_empty_when_no_conflict():
    assert _build_disagreement_block(ConflictResult(has_conflict=False)) == ""


# ── has_disagreement_matrix audit ────────────────────────────────────────────


def test_has_disagreement_matrix_detects_well_formed_table():
    answer = (
        "Some prose.\n\n"
        "**Sources disagree on this:**\n\n"
        "| Claim | Source A | Source B |\n"
        "|---|---|---|\n"
        "| repo rate | [doc_1] | [doc_2] |\n"
    )
    assert has_disagreement_matrix(answer) is True


def test_has_disagreement_matrix_false_when_only_prose():
    answer = (
        "Sources disagree on this point. [doc_1] states 6.50% while "
        "[doc_2] states 5.50%."
    )
    assert has_disagreement_matrix(answer) is False


def test_has_disagreement_matrix_false_on_wrong_columns():
    answer = (
        "| Topic | Pos A | Pos B |\n"
        "|---|---|---|\n"
        "| repo rate | 6.50% | 5.50% |\n"
    )
    assert has_disagreement_matrix(answer) is False


def test_has_disagreement_matrix_handles_empty():
    assert has_disagreement_matrix("") is False
    assert has_disagreement_matrix(None) is False  # type: ignore[arg-type]
