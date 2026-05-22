"""Unified weak-evidence signal for a single research turn.

Consolidates two previously-duplicated detection paths in
``agent/orchestrator.py``:

1. **Output-derived** weakness — driven by post-synthesis text features:
   ``[UNVERIFIED]`` marker count and the number of citations that resolve
   to a fetched URL. This was the body of the helper formerly named
   ``_is_weak_by_output`` and was invoked from two distinct downstream
   consumers (refinement-loop trigger gate, A3 next-steps block).

2. **Input-derived** weakness — driven by pre-synthesis context state:
   thin selected-context tokens and/or low planner confidence. These
   appeared inline as ``weak_thin`` / ``weak_planner`` locals around the
   Phase 1.5 weak-confidence branch.

The two are *not* the same phenomenon (different pipeline stages, different
mitigations), so we keep them as separate fields on one frozen dataclass
rather than collapsing them. ``is_weak`` is the OR of the output-derived
boolean condition; input-derived flags are read directly by callers that
care.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeakSignal:
    """Immutable summary of weak-evidence detection for one turn.

    Attributes:
        is_weak: True iff output-derived weakness condition trips
            (``unverified_count >= 2`` OR ``grounded_count < 3``).
        unverified_count: Number of ``[UNVERIFIED]`` markers in the answer.
        grounded_count: Number of cited doc-ids whose URL is in the
            fetched pool.
        reason: Categorical label describing the dominant weakness driver.
            One of: ``"unverified_threshold"`` (>=2 UNVERIFIED markers),
            ``"low_grounded"`` (fewer than 3 grounded citations),
            ``"thin_context"`` (input-stage: selected tokens below
            threshold), ``"low_planner_confidence"`` (input-stage:
            planner emitted ``confidence == "low"``), or ``"strong"``
            (no weakness detected).
    """

    is_weak: bool
    unverified_count: int
    grounded_count: int
    reason: str


def compute_weak_signal(
    answer: str,
    cited_ids: list,
    doc_map: dict,
    fetched_urls: set,
) -> WeakSignal:
    """Compute the output-derived weak-evidence signal for an answer.

    Mirrors the historical ``_is_weak_by_output`` semantics exactly:
    a turn is weak iff ``[UNVERIFIED]`` markers >= 2 OR fewer than 3
    cited doc-ids resolve to a fetched URL in ``doc_map``.

    ``cited_ids`` elements and ``doc_map`` keys are typically strings
    (e.g. ``"doc_1"``); the function does not constrain the key type
    so both string- and int-keyed maps work consistently.
    """
    unverified_count = (answer or "").count("[UNVERIFIED]")
    grounded_count = sum(
        1
        for did in (cited_ids or [])
        if did in doc_map and doc_map[did][1] in fetched_urls
    )
    is_weak = unverified_count >= 2 or grounded_count < 3
    if unverified_count >= 2:
        reason = "unverified_threshold"
    elif grounded_count < 3:
        reason = "low_grounded"
    else:
        reason = "strong"
    return WeakSignal(
        is_weak=is_weak,
        unverified_count=unverified_count,
        grounded_count=grounded_count,
        reason=reason,
    )
