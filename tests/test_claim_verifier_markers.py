"""P6c — verify the per-claim status markers injected by claim_verifier.

Covers the new ``[AMBIGUOUS]`` marker (status ``ambiguous_resolved``) and
re-asserts the existing ``[UNVERIFIED]`` marker (status ``unsupported``).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.claim_verifier import ClaimRecord, _append_unverified_markers


def _record(text: str, status: str) -> ClaimRecord:
    return ClaimRecord(
        claim_text=text,
        doc_ids=("doc_1",),
        overlap=0.5,
        entity_match=0.5,
        method="deterministic",
        score=0.5,
        status=status,
    )


def test_appends_unverified_for_unsupported():
    answer = "Foo is bar. [doc_1]"
    parsed = [("Foo is bar.", ("doc_1",), (0, len(answer)))]
    out = _append_unverified_markers(answer, parsed, [_record("Foo is bar.", "unsupported")])
    assert "[UNVERIFIED]" in out
    assert "[AMBIGUOUS]" not in out


def test_appends_ambiguous_for_ambiguous_resolved():
    answer = "Foo might be bar. [doc_1]"
    parsed = [("Foo might be bar.", ("doc_1",), (0, len(answer)))]
    out = _append_unverified_markers(
        answer, parsed, [_record("Foo might be bar.", "ambiguous_resolved")]
    )
    assert "[AMBIGUOUS]" in out
    assert "[UNVERIFIED]" not in out


def test_no_marker_for_supported():
    answer = "Foo equals bar. [doc_1]"
    parsed = [("Foo equals bar.", ("doc_1",), (0, len(answer)))]
    out = _append_unverified_markers(
        answer, parsed, [_record("Foo equals bar.", "supported")]
    )
    assert "[AMBIGUOUS]" not in out
    assert "[UNVERIFIED]" not in out


def test_ambiguous_is_idempotent():
    answer = "Foo is bar. [doc_1] [AMBIGUOUS]"
    parsed = [("Foo is bar.", ("doc_1",), (0, len("Foo is bar. [doc_1]")))]
    out = _append_unverified_markers(
        answer, parsed, [_record("Foo is bar.", "ambiguous_resolved")]
    )
    assert out.count("[AMBIGUOUS]") == 1


def test_unverified_takes_precedence_over_ambiguous():
    answer = "Foo is bar. [doc_1] [UNVERIFIED]"
    parsed = [("Foo is bar.", ("doc_1",), (0, len("Foo is bar. [doc_1]")))]
    # If a sentence is already tagged as UNVERIFIED, a subsequent ambiguous record
    # must not weaken it by appending [AMBIGUOUS] on top.
    out = _append_unverified_markers(
        answer, parsed, [_record("Foo is bar.", "ambiguous_resolved")]
    )
    assert "[UNVERIFIED]" in out
    assert "[AMBIGUOUS]" not in out
