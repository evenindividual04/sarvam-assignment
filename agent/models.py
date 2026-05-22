"""
All dataclasses and Pydantic models for the research agent.
Plain dataclasses for internal data flow; Pydantic for LLM JSON output parsing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Plain dataclasses ──────────────────────────────────────────────────────

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    domain: str
    retrieved_at: str
    raw_content: Optional[str] = None    # populated by Parallel/Tavily; skips Trafilatura
    intent_origin: Optional[str] = None  # V2.1: intent of the query that produced this result
    relevance: Optional[float] = None    # normalized to [0,1]; see `relevance_source` for provenance
    relevance_source: Optional[str] = None
    # "provider" → field surfaced directly by the provider (e.g. Tavily `score`); an
    #              absolute relevance estimate, can be compared across queries.
    # "rank"     → derived from result array position (Parallel + Serper, which do not
    #              expose a per-result score). Relative within the query only.
    # None       → no signal available (extremely rare; only when provider response is
    #              malformed and rank fallback also fails).


@dataclass
class ContextSnippet:
    doc_id: str
    url: str
    title: str
    domain: str
    text: str
    snippet: str
    token_count: int
    retrieved_at: str
    bm25_score: float = 0.0
    recency_score: float = 0.0
    diversity_score: float = 0.0
    final_score: float = 0.0
    intent_origin: Optional[str] = None  # V2.1: provenance from the originating typed query
    trust_score: float = 0.7  # V2.3: deterministic source trust prior in [0.45, 1.00]
    trust_tier: str = "unknown"
    # Provider-side relevance carried from `SearchResult` so the selector can
    # use it as an additional signal alongside BM25 / FlashRank / recency /
    # trust. Source: `"provider"` (absolute, Tavily score) or `"rank"` (relative
    # to the query, Parallel/Serper).
    provider_relevance: Optional[float] = None
    provider_relevance_source: Optional[str] = None


@dataclass
class Turn:
    turn_id: str
    session_id: str
    query: str
    created_at: str
    plan: Optional[str] = None
    search_queries: list[str] = field(default_factory=list)
    urls_opened: list[str] = field(default_factory=list)
    response: Optional[str] = None
    context_xml_sent: Optional[str] = None
    doc_map: Optional[dict[str, tuple[str, str, str]]] = None  # doc_id -> (title, url, domain)
    citation_integrity_score: float = 1.0
    hallucination_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    planning_ms: int = 0
    search_ms: int = 0
    fetch_ms: int = 0
    select_ms: int = 0
    synthesize_ms: int = 0
    run_metadata_json: Optional[dict[str, Any]] = None
    state_trace: list[str] = field(default_factory=list)
    claim_precision_score: float = 1.0
    claim_verification_json: Optional[str] = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Turn":
        return cls(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            query=row["query"],
            created_at=row["created_at"],
            plan=row.get("plan"),
            search_queries=json.loads(row.get("search_queries") or "[]"),
            urls_opened=json.loads(row.get("urls_opened") or "[]"),
            response=row.get("response"),
            context_xml_sent=row.get("context_xml_sent"),
            doc_map=json.loads(row.get("doc_map") or "{}") or None,
            citation_integrity_score=row.get("citation_integrity_score") if row.get("citation_integrity_score") is not None else 1.0,
            hallucination_count=row.get("hallucination_count") or 0,
            prompt_tokens=row.get("prompt_tokens") or 0,
            completion_tokens=row.get("completion_tokens") or 0,
            latency_ms=row.get("latency_ms") or 0,
            planning_ms=row.get("planning_ms") or 0,
            search_ms=row.get("search_ms") or 0,
            fetch_ms=row.get("fetch_ms") or 0,
            select_ms=row.get("select_ms") or 0,
            synthesize_ms=row.get("synthesize_ms") or 0,
            run_metadata_json=json.loads(row.get("run_metadata_json") or "{}") or None,
            state_trace=json.loads(row.get("state_trace") or "[]"),
            claim_precision_score=row.get("claim_precision_score") if row.get("claim_precision_score") is not None else 1.0,
            claim_verification_json=row.get("claim_verification_json"),
        )


@dataclass
class ExecutionEvent:
    step: str           # planning | searching | fetching | selecting | generating | done | error
    label: str          # human-readable streaming label
    data: Any = None    # step-specific payload
    # Phase 1.25: typed-event discriminator. When set, the SSE layer uses this
    # as the `event:` field. Old emit sites leave it None (back-compat); new
    # sites set one of the constants in EVENT_TYPES below.
    event_type: str | None = None


# Phase 1.25: typed-event discriminators (AG-UI / Vercel AI SDK 5 inspired).
EVT_RUN_STARTED = "run_started"
EVT_PHASE_STARTED = "phase_started"
EVT_PHASE_PROGRESS = "phase_progress"
EVT_PHASE_FINISHED = "phase_finished"
EVT_SEARCH_QUERY = "search_query"
EVT_SOURCE_FOUND = "source_found"
EVT_SOURCE_FETCHED = "source_fetched"
EVT_CONTEXT_SELECTED = "context_selected"
EVT_CONFLICT_DETECTED = "conflict_detected"
EVT_ANSWER_DELTA = "answer_delta"
EVT_CITATION_RESOLVED = "citation_resolved"
EVT_RUN_FINISHED = "run_finished"
EVT_RUN_ERROR = "run_error"
# Phase 1.5: structured uncertainty signal (weak / missing / conflict).
EVT_UNCERTAINTY = "uncertainty"
# Phase 1.875: plan-level clarification (ambiguity_flag) + per-sub-query
# evidence-gap notifications surfaced after SELECTING.
EVT_CLARIFICATION_OFFERED = "clarification_offered"
EVT_EVIDENCE_GAP = "evidence_gap"
# Phase 2: human-in-the-loop plan approval gate. Emitted between PLANNING and
# SEARCHING when the request opts in via `approval_required=True`.
EVT_PLAN_APPROVAL = "plan_approval"


@dataclass
class ContextBundle:
    xml: str
    doc_map: dict[str, tuple[str, str, str]]   # doc_id -> (title, url, domain)
    fetched_urls: set[str]
    conflict_summary: Optional[str] = None


# ── Pydantic models (LLM JSON output) ─────────────────────────────────────

class QueryIntent(str, Enum):
    PRIMARY = "primary"
    COMPARISON = "comparison"
    RECENCY_CHECK = "recency_check"
    CONTRADICTION_PROBE = "contradiction_probe"
    DEFINITION = "definition"


class TypedQuery(BaseModel):
    text: str
    intent: QueryIntent
    rationale: Optional[str] = None


class PlannerOutput(BaseModel):
    strategy: str
    queries: list[TypedQuery]
    confidence: Literal["low", "medium", "high"] = "medium"  # V3.2 adaptive 2-hop gate
    # Phase 1.875: enriched plan-level metadata. All optional with defaults
    # so older serialized planner outputs (without these fields) still parse.
    time_sensitivity: Literal["live", "recent", "static"] = "static"
    expected_source_types: list[Literal["news", "academic", "official", "wiki", "forum"]] = Field(
        default_factory=list
    )
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    ambiguity_flag: bool = False
    success_criteria: list[str] = Field(default_factory=list)  # 1-3 short bullets


class ClaimContradiction(BaseModel):
    claim: str
    doc_ids_a: list[str]
    position_a: str
    doc_ids_b: list[str]
    position_b: str
    is_temporal_evolution: bool
    confidence: float


class ConflictResult(BaseModel):
    has_conflict: bool
    conflict_summary: Optional[str] = None
    contradictions: list[ClaimContradiction] = []
    probe_skipped_reason: Optional[str] = None


class ClaimVerification(BaseModel):
    supported: bool
    reasoning: str = ""


class JudgeScore(BaseModel):
    faithfulness_score: float = 1.0
    answer_relevance_score: float = 1.0
    citation_integrity_score: float = 1.0
    conflict_adherence_score: Optional[float] = None
    session_coherence_score: Optional[float] = None
    reasoning: str = ""
