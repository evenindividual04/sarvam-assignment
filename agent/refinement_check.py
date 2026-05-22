"""
C2 — Answer-refinement classifier.

After the first synthesis pass and citation-guard finalization, this module
runs a lightweight self-check that classifies the answer as one of:

  - CONFIDENT             — answer is well-grounded, no refinement needed.
  - NEEDS_MORE_EVIDENCE   — gaps remain; orchestrator may run ONE additional
                            targeted search hop + re-synthesis. Bounded by
                            ``MAX_REFINEMENTS = 1``.
  - NEEDS_REPHRASE        — evidence is sufficient but the wording is vague;
                            orchestrator may re-synthesize with the SAME
                            context using an explicit "be specific about
                            uncertainty" hint.

Implementation notes:
  * Uses Groq Llama 3.3 70B (cheap, fast, JSON-strict) via ``call_groq``.
  * ``max_tokens=120`` cap — verdict + short reason + optional 1-line query.
  * Never raises: any classifier failure logs and degrades to CONFIDENT
    (i.e. "skip refinement"), so the request always completes.
  * Pure classifier — does NOT execute search/synthesis. Orchestrator owns
    the refinement loop and budget enforcement.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# Hard cap. Refinement runs at most ONCE per turn.
MAX_REFINEMENTS: int = 1

_CLASSIFIER_TIMEOUT_S: float = 6.0
_MAX_TOKENS: int = 120


class RefinementVerdict(BaseModel):
    """JSON-strict classifier output."""

    verdict: Literal["CONFIDENT", "NEEDS_MORE_EVIDENCE", "NEEDS_REPHRASE"]
    reason: str = Field(default="", max_length=400)
    suggested_query: Optional[str] = Field(default=None, max_length=200)


def _build_prompt(
    query: str,
    answer: str,
    citations: list[str],
    unverified_count: int,
    grounded_count: int,
) -> str:
    # Trim answer to keep prompt small — classifier only needs structure.
    snippet = (answer or "")[:2000]
    cites = ", ".join(citations[:12]) or "(none)"
    return (
        "You are a strict QA classifier for a web-research agent's answer.\n"
        "Given the user's question, the answer, and structured signals, decide "
        "whether the answer is (a) CONFIDENT, (b) NEEDS_MORE_EVIDENCE — gaps "
        "that one more targeted web search would close, or (c) NEEDS_REPHRASE "
        "— evidence is fine but wording is vague.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n{snippet}\n\n"
        f"SIGNALS:\n"
        f"- citations_present: [{cites}]\n"
        f"- unverified_count (sentences flagged [UNVERIFIED]): {unverified_count}\n"
        f"- grounded_citation_count (citations resolving to fetched URLs): "
        f"{grounded_count}\n\n"
        "Respond with a SINGLE JSON object, no prose. Schema:\n"
        '{"verdict": "CONFIDENT"|"NEEDS_MORE_EVIDENCE"|"NEEDS_REPHRASE", '
        '"reason": "<one short sentence>", '
        '"suggested_query": "<one targeted web query if NEEDS_MORE_EVIDENCE, '
        'else null>"}'
    )


def _parse_verdict(raw: str) -> Optional[RefinementVerdict]:
    """Extract a JSON object from the LLM response, then validate."""
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        return RefinementVerdict.model_validate_json(raw[start:end])
    except (ValidationError, ValueError, json.JSONDecodeError):
        return None


async def classify_answer(
    query: str,
    answer: str,
    citations: list[str],
    unverified_count: int,
    grounded_count: int,
) -> RefinementVerdict:
    """Classify the synthesized answer. Degrades to CONFIDENT on any failure."""
    prompt = _build_prompt(query, answer, citations, unverified_count, grounded_count)
    try:
        from utils.provider_router import call_groq

        raw = await asyncio.wait_for(
            call_groq(prompt, max_tokens=_MAX_TOKENS),
            timeout=_CLASSIFIER_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Refinement classifier timed out",
            extra={"component": "refinement_check"},
        )
        return RefinementVerdict(
            verdict="CONFIDENT", reason="classifier_timeout", suggested_query=None
        )
    except Exception as e:
        logger.warning(
            "Refinement classifier error: %s",
            e,
            extra={"component": "refinement_check"},
        )
        return RefinementVerdict(
            verdict="CONFIDENT", reason="classifier_error", suggested_query=None
        )

    parsed = _parse_verdict(raw)
    if parsed is None:
        logger.warning(
            "Refinement classifier returned unparseable JSON",
            extra={"component": "refinement_check", "raw_preview": raw[:200]},
        )
        return RefinementVerdict(
            verdict="CONFIDENT", reason="classifier_parse_fail", suggested_query=None
        )
    return parsed
