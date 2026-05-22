"""A3 — `**What I'd search next:**` block tests.

Covers the weak-by-output detector and the deterministic append helper used
by the orchestrator when an answer carries weak/missing evidence signals.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import _append_next_steps_block, _is_weak_by_output


DOC_MAP = {
    "doc_1": ("Title One", "https://example.com/a", "example.com"),
    "doc_2": ("Title Two", "https://example.com/b", "example.com"),
    "doc_3": ("Title Three", "https://example.com/c", "example.com"),
}
FETCHED = {"https://example.com/a", "https://example.com/b", "https://example.com/c"}


# ── _is_weak_by_output ──────────────────────────────────────────────────────

def test_weak_when_unverified_count_at_least_two():
    answer = "claim a [doc_1] [UNVERIFIED]. claim b [doc_2] [UNVERIFIED]. claim c [doc_3]."
    weak, uc, gc = _is_weak_by_output(answer, ["doc_1", "doc_2", "doc_3"], DOC_MAP, FETCHED)
    assert weak is True
    assert uc == 2
    assert gc == 3


def test_weak_when_grounded_citations_below_three():
    answer = "claim a [doc_1]. claim b [doc_2]."
    weak, uc, gc = _is_weak_by_output(answer, ["doc_1", "doc_2"], DOC_MAP, FETCHED)
    assert weak is True
    assert gc == 2


def test_not_weak_when_three_grounded_and_no_unverified():
    answer = "a [doc_1]. b [doc_2]. c [doc_3]."
    weak, uc, gc = _is_weak_by_output(answer, ["doc_1", "doc_2", "doc_3"], DOC_MAP, FETCHED)
    assert weak is False
    assert uc == 0
    assert gc == 3


def test_grounded_excludes_unfetched_urls():
    answer = "a [doc_1]. b [doc_2]. c [doc_3]."
    fetched_partial = {"https://example.com/a"}
    weak, _, gc = _is_weak_by_output(
        answer, ["doc_1", "doc_2", "doc_3"], DOC_MAP, fetched_partial,
    )
    assert weak is True
    assert gc == 1


# ── _append_next_steps_block ────────────────────────────────────────────────

def test_append_block_with_three_unverified_markers():
    answer = (
        "claim 1 [doc_1] [UNVERIFIED]. "
        "claim 2 [doc_2] [UNVERIFIED]. "
        "claim 3 [doc_3] [UNVERIFIED]."
    )
    suggestions = ["latest RBI repo rate official", "MoSPI inflation Q3 2025", "RBI MPC minutes 2025"]
    out = _append_next_steps_block(answer, suggestions)
    assert "**What I'd search next:**" in out
    assert "- latest RBI repo rate official" in out
    assert "- MoSPI inflation Q3 2025" in out
    assert "- RBI MPC minutes 2025" in out


def test_append_block_is_idempotent():
    answer = "x.\n\n**What I'd search next:**\n- a\n- b"
    out = _append_next_steps_block(answer, ["c", "d"])
    assert out == answer
    assert out.count("**What I'd search next:**") == 1


def test_append_block_caps_at_three_suggestions():
    out = _append_next_steps_block("body.", ["q1", "q2", "q3", "q4", "q5"])
    assert "- q1" in out and "- q2" in out and "- q3" in out
    assert "- q4" not in out


def test_append_block_skips_empty_suggestions():
    out = _append_next_steps_block("body.", ["", "   ", None])  # type: ignore[list-item]
    assert "**What I'd search next:**" not in out
    assert out == "body."
