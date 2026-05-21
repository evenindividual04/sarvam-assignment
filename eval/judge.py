"""
5 metric judge functions using GPT-4o-mini via GitHub Models.
Each is a separate call — no anchor bleed between metrics.
All return typed dataclasses. All wrapped in retry.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class FaithfulnessResult:
    faithfulness_score: float
    supported_count: int
    unsupported_count: int
    unsupported_claims: list[str]
    reasoning: str


@dataclass
class RelevanceResult:
    answer_relevance_score: float
    reasoning: str


@dataclass
class ConflictAdherenceResult:
    conflict_adherence_score: float
    identified_conflict: bool
    presented_both_sides: bool
    expressed_uncertainty: bool
    avoided_silent_choice: bool
    reasoning: str


@dataclass
class CoherenceResult:
    session_coherence_score: float
    reasoning: str


@dataclass
class CitationIntegrityResult:
    citation_integrity_score: float
    total_cited: int
    valid_cited: int


@dataclass
class ContextPrecisionResult:
    context_precision_score: float
    reasoning: str


def _parse_json(raw: str) -> dict:
    """Extract first JSON object from LLM response."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(raw[start:end])
    raise ValueError(f"No JSON found in: {raw[:200]}")


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def judge_faithfulness(context_xml: str, answer: str) -> FaithfulnessResult:
    from utils.provider_router import judge as call_judge

    prompt = f"""CONTEXT:
{context_xml[:3000]}

ANSWER:
{answer}

Extract every factual claim from the answer. For each, determine:
SUPPORTED: explicitly in context | UNSUPPORTED: not in context (hallucination)

JSON only: {{"faithfulness_score": float, "supported_count": int, "unsupported_count": int, "unsupported_claims": [str], "reasoning": str}}"""

    raw = await call_judge(prompt)
    data = _parse_json(raw)
    return FaithfulnessResult(
        faithfulness_score=float(data.get("faithfulness_score", 0.5)),
        supported_count=int(data.get("supported_count", 0)),
        unsupported_count=int(data.get("unsupported_count", 0)),
        unsupported_claims=data.get("unsupported_claims", []),
        reasoning=data.get("reasoning", ""),
    )


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def judge_relevance(query: str, answer: str) -> RelevanceResult:
    from utils.provider_router import judge as call_judge

    prompt = f"""QUESTION: {query}

ANSWER:
{answer}

Score 0.0-1.0: Does the answer directly address the question?
0.0=off-topic | 0.5=partial | 1.0=complete

JSON only: {{"answer_relevance_score": float, "reasoning": str}}"""

    raw = await call_judge(prompt)
    data = _parse_json(raw)
    return RelevanceResult(
        answer_relevance_score=float(data.get("answer_relevance_score", 0.5)),
        reasoning=data.get("reasoning", ""),
    )


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def judge_conflict_adherence(query: str, context_xml: str, answer: str) -> ConflictAdherenceResult:
    from utils.provider_router import judge as call_judge

    prompt = f"""QUESTION: {query}

CONTEXT:
{context_xml[:2000]}

ANSWER:
{answer}

This question has known conflicting sources. Score yes/no:
1. Identified the conflict explicitly?
2. Presented both sides with citations?
3. Expressed appropriate uncertainty?
4. Avoided silently choosing one side?
score = yes_count / 4

JSON only: {{"conflict_adherence_score": float, "identified_conflict": bool, "presented_both_sides": bool, "expressed_uncertainty": bool, "avoided_silent_choice": bool, "reasoning": str}}"""

    raw = await call_judge(prompt)
    data = _parse_json(raw)
    return ConflictAdherenceResult(
        conflict_adherence_score=float(data.get("conflict_adherence_score", 0.0)),
        identified_conflict=bool(data.get("identified_conflict", False)),
        presented_both_sides=bool(data.get("presented_both_sides", False)),
        expressed_uncertainty=bool(data.get("expressed_uncertainty", False)),
        avoided_silent_choice=bool(data.get("avoided_silent_choice", False)),
        reasoning=data.get("reasoning", ""),
    )


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def judge_coherence(t1_query: str, t1_answer: str, t2_query: str, t2_answer: str) -> CoherenceResult:
    from utils.provider_router import judge as call_judge

    prompt = f"""TURN_1_Q: {t1_query}
TURN_1_A: {t1_answer[:500]}

TURN_2_Q: {t2_query}
TURN_2_A: {t2_answer[:500]}

Score 0.0-1.0: Does Turn 2 correctly reference and build on Turn 1?
0.0=treats Turn 2 as if Turn 1 never happened | 1.0=correctly incorporates

JSON only: {{"session_coherence_score": float, "reasoning": str}}"""

    raw = await call_judge(prompt)
    data = _parse_json(raw)
    return CoherenceResult(
        session_coherence_score=float(data.get("session_coherence_score", 0.5)),
        reasoning=data.get("reasoning", ""),
    )
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def judge_context_precision(query: str, context_xml: str) -> ContextPrecisionResult:
    """Prompt 5: Did the retrieval layer fetch the necessary information?"""
    from utils.provider_router import judge as call_judge

    prompt = f"""QUESTION: {query}

CONTEXT:
{context_xml[:3000]}

Score 0.0-1.0: Does the context contain sufficient information to fully answer the question?
0.0=none | 0.5=partial | 1.0=complete

JSON only: {{"context_precision_score": float, "reasoning": str}}"""

    raw = await call_judge(prompt)
    data = _parse_json(raw)
    return ContextPrecisionResult(
        context_precision_score=float(data.get("context_precision_score", 0.5)),
        reasoning=data.get("reasoning", ""),
    )



def judge_citation_integrity(answer: str, doc_map: dict, fetched_urls: set) -> CitationIntegrityResult:
    """Deterministic — no LLM. Fraction of [doc_N] citations that map to real fetched URLs."""
    from agent.citation_guard import _extract_doc_ids
    cited_ids = _extract_doc_ids(answer)
    total = len(cited_ids)
    if total == 0:
        return CitationIntegrityResult(citation_integrity_score=1.0, total_cited=0, valid_cited=0)

    valid = sum(
        1 for doc_id in cited_ids
        if doc_id in doc_map and doc_map[doc_id][1] in fetched_urls
    )
    return CitationIntegrityResult(
        citation_integrity_score=valid / total,
        total_cited=total,
        valid_cited=valid,
    )


def classify_failure(r: dict) -> str:
    conflict_score = r.get("conflict_adherence_score")
    coherence_score = r.get("session_coherence_score")
    if r.get("faithfulness_score", 1.0) < 0.7 and r.get("citation_integrity_score", 1.0) < 0.8:
        return "HALLUCINATION"
    elif r.get("faithfulness_score", 1.0) < 0.7:
        return "KNOWLEDGE_BLEED"
    elif r.get("answer_relevance_score", 1.0) < 0.5 or r.get("context_precision_score", 1.0) < 0.5:
        return "RETRIEVAL_FAILURE"
    elif conflict_score is not None and conflict_score < 0.5:
        return "CONFLICT_MISS"
    elif coherence_score is not None and coherence_score < 0.5:
        return "COHERENCE_FAIL"
    return "PASS"
