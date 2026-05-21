"""
All dataclasses and Pydantic models for the research agent.
Plain dataclasses for internal data flow; Pydantic for LLM JSON output parsing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


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
        )


@dataclass
class ExecutionEvent:
    step: str           # planning | searching | fetching | selecting | generating | done | error
    label: str          # human-readable streaming label
    data: Any = None    # step-specific payload


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


class JudgeScore(BaseModel):
    faithfulness_score: float = 1.0
    answer_relevance_score: float = 1.0
    citation_integrity_score: float = 1.0
    conflict_adherence_score: Optional[float] = None
    session_coherence_score: Optional[float] = None
    reasoning: str = ""
