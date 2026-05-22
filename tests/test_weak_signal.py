"""Tests for the consolidated :mod:`agent.weak_signal` module.

Covers every value of :class:`WeakSignal.reason` and the boundary
conditions on the ``is_weak`` predicate.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.weak_signal import WeakSignal, compute_weak_signal


DOC_MAP = {
    "doc_1": ("Title One", "https://example.com/a", "example.com"),
    "doc_2": ("Title Two", "https://example.com/b", "example.com"),
    "doc_3": ("Title Three", "https://example.com/c", "example.com"),
}
FETCHED = {
    "https://example.com/a",
    "https://example.com/b",
    "https://example.com/c",
}


def test_strong_reason_when_three_grounded_and_no_unverified():
    answer = "claim a [doc_1]. claim b [doc_2]. claim c [doc_3]."
    sig = compute_weak_signal(
        answer, ["doc_1", "doc_2", "doc_3"], DOC_MAP, FETCHED,
    )
    assert sig.is_weak is False
    assert sig.reason == "strong"
    assert sig.unverified_count == 0
    assert sig.grounded_count == 3


def test_unverified_threshold_reason_dominates():
    # Two UNVERIFIED markers — trips threshold even though all citations
    # are grounded; reason must reflect the dominant driver.
    answer = (
        "a [doc_1] [UNVERIFIED]. b [doc_2] [UNVERIFIED]. c [doc_3]."
    )
    sig = compute_weak_signal(
        answer, ["doc_1", "doc_2", "doc_3"], DOC_MAP, FETCHED,
    )
    assert sig.is_weak is True
    assert sig.reason == "unverified_threshold"
    assert sig.unverified_count == 2
    assert sig.grounded_count == 3


def test_low_grounded_reason_when_under_three_citations_resolve():
    answer = "a [doc_1]. b [doc_2]."
    sig = compute_weak_signal(answer, ["doc_1", "doc_2"], DOC_MAP, FETCHED)
    assert sig.is_weak is True
    assert sig.reason == "low_grounded"
    assert sig.unverified_count == 0
    assert sig.grounded_count == 2


def test_boundary_one_unverified_is_not_weak_alone():
    # Exactly one UNVERIFIED marker — under the >=2 threshold; three
    # grounded citations keep the answer strong.
    answer = "a [doc_1] [UNVERIFIED]. b [doc_2]. c [doc_3]."
    sig = compute_weak_signal(
        answer, ["doc_1", "doc_2", "doc_3"], DOC_MAP, FETCHED,
    )
    assert sig.is_weak is False
    assert sig.reason == "strong"
    assert sig.unverified_count == 1
    assert sig.grounded_count == 3


def test_grounded_excludes_unfetched_urls():
    # Two of three doc_map entries point at URLs not in the fetched pool —
    # grounded drops to 1, so the answer is weak with reason low_grounded.
    answer = "a [doc_1]. b [doc_2]. c [doc_3]."
    fetched_partial = {"https://example.com/a"}
    sig = compute_weak_signal(
        answer, ["doc_1", "doc_2", "doc_3"], DOC_MAP, fetched_partial,
    )
    assert sig.is_weak is True
    assert sig.reason == "low_grounded"
    assert sig.grounded_count == 1


def test_weak_signal_is_frozen():
    sig = WeakSignal(
        is_weak=False,
        unverified_count=0,
        grounded_count=3,
        reason="strong",
    )
    import pytest
    with pytest.raises(Exception):
        sig.is_weak = True  # type: ignore[misc]
