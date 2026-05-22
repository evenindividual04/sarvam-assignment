"""Schema tests for `agent.models.RunMetadata`.

Refactor #2: `run_metadata` was an unbounded dict mutated in 60+ sites. This
suite locks the schema contract:

- Empty default validates with sensible defaults (no required field).
- Round-trip dump → validate is stable.
- `extra="allow"` preserves unknown keys (forward-compat).
- A realistic per-turn payload (refinement + cite_quote_map + unreachable_pages)
  validates cleanly.
"""
from __future__ import annotations

import pytest

from agent.models import RunMetadata, RunMetadataHopEvidence, UnreachablePageEntry


@pytest.mark.unit
def test_empty_default_validates() -> None:
    """No fields are required — orchestrator init path must validate."""
    meta = RunMetadata()
    dumped = meta.model_dump(mode="json")

    # All defaults present + no surprises.
    assert dumped["refinement_triggered"] is False
    assert dumped["refinement_count"] == 0
    assert dumped["fallback_path_taken"] == []
    assert dumped["unreachable_pages"] == []
    assert dumped["cite_quote_map"] == {}
    assert dumped["conflict_table_missing"] is False


@pytest.mark.unit
def test_round_trip_preserves_payload() -> None:
    """dump → validate → dump must be a fixed point."""
    payload = {
        "selection_strategy": "heuristic",
        "hop_count": 2,
        "terminator_fired": "STOP_RAG_GATE",
        "terminator_source": "stop_rag",
        "refinement_triggered": True,
        "refinement_count": 1,
        "cite_quote_map": {"doc_1": "the sky is blue"},
        "fallback_path_taken": ["search_timeout_empty"],
        "extraction_fallbacks": {"trafilatura": 3, "readability": 1},
        "unreachable_pages": [
            {"url": "https://x.test/a", "domain": "x.test", "reason": "403"}
        ],
    }
    meta = RunMetadata.model_validate(payload)
    once = meta.model_dump(mode="json")
    twice = RunMetadata.model_validate(once).model_dump(mode="json")
    assert once == twice
    # Spot-check that the nested model also round-trips.
    assert once["unreachable_pages"][0]["url"] == "https://x.test/a"


@pytest.mark.unit
def test_forward_compat_extra_keys_allowed() -> None:
    """extra='allow' — adding a new key in the orchestrator must not raise
    before the schema is updated. Unknown keys survive round-trip."""
    payload = {
        "selection_strategy": "heuristic",
        "totally_new_forensic_key": {"shape": "anything"},
        "another_unknown_list": [1, 2, 3],
    }
    meta = RunMetadata.model_validate(payload)
    dumped = meta.model_dump(mode="json")
    assert dumped["totally_new_forensic_key"] == {"shape": "anything"}
    assert dumped["another_unknown_list"] == [1, 2, 3]


@pytest.mark.unit
def test_realworld_payload_from_orchestrator() -> None:
    """Synthesize a payload matching what `eval_runner` sees after a turn
    with refinement + citations + unreachable pages. Should validate clean."""
    payload = {
        "selection_strategy": "heuristic",
        "effective_config": {"max_hops": 2},
        "retrieval_mode": {"requested": "hybrid", "effective": "hybrid", "reason": None},
        "planner_prompt_id": "planner_v3",
        "synth_prompt_id": "synth_v2",
        "configured_models": {
            "gemini_model": "gemini-2.5-flash",
            "planner_model": "llama-3.3-70b-versatile",
            "judge_model": "gpt-4o-mini",
        },
        "fallback_path_taken": [],
        "timeout_hits": [],
        "retry_counts": {},
        "budget_breach": [],
        "hop_count": 2,
        "terminator_fired": "MAX_HOPS",
        "terminator_source": "deterministic",
        "hop_evidence": [
            {"hop": 1, "grounded": [{"token": "X", "kind": "entity"}], "open": []}
        ],
        "refinement_triggered": True,
        "refinement_count": 1,
        "refinement_reason": "criteria_coverage<0.6",
        "cite_quote_map": {"doc_1": "verbatim quote one", "doc_3": "verbatim quote two"},
        "quote_audit": {"total": 2, "grounded": 2},
        "quote_grounding_ratio": 1.0,
        "numeric_audit": {"total": 3, "grounded": 3},
        "numeric_grounding_ratio": 1.0,
        "criteria_coverage": [True, True, False],
        "conflict_table_missing": False,
        "next_step_suggestions": ["check 2025 update", "compare with source Y"],
        "unreachable_pages": [
            {
                "url": "https://blocked.test/page",
                "domain": "blocked.test",
                "title": "Blocked",
                "snippet": "...",
                "reason": "403",
                "hop": 1,
            }
        ],
        "extraction_fallbacks": {"trafilatura": 2},
        "domain_blocklist_drops": ["spam.test"],
        "reranker_used": "flashrank",
        "budget_distribution": {
            "system": 2400, "history": 4000, "web_context": 6400, "output_reserved": 3200
        },
        "probe_ms": 120,
        "verification_ms": 340,
    }
    meta = RunMetadata.model_validate(payload)
    # Nested types resolved correctly.
    assert isinstance(meta.hop_evidence[0], RunMetadataHopEvidence)
    assert isinstance(meta.unreachable_pages[0], UnreachablePageEntry)
    # JSON-mode dump is suitable for SQLite persistence.
    serialized = meta.model_dump(mode="json")
    assert serialized["refinement_count"] == 1
    assert serialized["unreachable_pages"][0]["reason"] == "403"
    # Re-validating the JSON shape must still work (round-trip persistence).
    again = RunMetadata.model_validate(serialized)
    assert again.refinement_triggered is True
