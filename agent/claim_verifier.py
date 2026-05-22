"""
V2.4 — Claim-level post-generation verification.

Two-tier check per sentence-with-citation:
  Tier 1 (deterministic): token overlap + entity match against cited snippet.
  Tier 2 (LLM fallback): GPT-4o-mini via provider_router.judge() for ambiguous mid-band.

Unsupported claims have `[UNVERIFIED]` appended AFTER the citation block.
The CitationGuard regex matches only `\\[doc_\\d+\\]`, so the marker is invisible
to URL conversion downstream. Never raises — failures degrade to score=1.0.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from agent.citation_guard import parse_claims_with_citations
from agent.models import ClaimVerification

logger = logging.getLogger(__name__)


CLAIM_RE = re.compile(
    r'([^.!?\n]*?[.!?])\s*((?:\[doc_\d+\]\s*)+)',
    re.MULTILINE,
)
DOC_ID_RE = re.compile(r'\[doc_\d+\]')
ENTITY_RE = re.compile(
    r'\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*|\d[\d,.\-/%]*|\$\d[\d,.]*)\b'
)
STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "with", "by", "from", "as", "also", "not", "no", "yes",
})

_HIGH_THRESHOLD = 0.6
_LOW_THRESHOLD = 0.3
_LLM_TIMEOUT_S = 8.0


@dataclass(frozen=True)
class ClaimRecord:
    claim_text: str
    doc_ids: tuple[str, ...]
    overlap: float
    entity_match: float
    # ``method`` discriminates which path resolved the claim. ``llm`` is kept
    # as a back-compat synonym for ``llm_gpt4o`` so existing tests/callers that
    # only check membership-in-{"llm","deterministic","skip"} keep working.
    method: str  # "deterministic" | "llm" | "llm_deepseek" | "llm_gpt4o" | "skip"
    score: float
    status: str  # "supported" | "unsupported" | "ambiguous_resolved"


def _score_pair(claim: str, snippet: str) -> tuple[float, float, float, frozenset]:
    """Return (overlap, entity_match, combined_score, claim_entities)."""
    claim_tokens = set(re.findall(r'\w+', claim.lower())) - STOPWORDS
    snippet_tokens = set(re.findall(r'\w+', snippet.lower())) - STOPWORDS
    overlap = len(claim_tokens & snippet_tokens) / max(1, len(claim_tokens))

    claim_entities = set(ENTITY_RE.findall(claim))
    snippet_entities = set(ENTITY_RE.findall(snippet))
    entity_match = len(claim_entities & snippet_entities) / max(1, len(claim_entities))

    score = 0.6 * overlap + 0.4 * entity_match
    return overlap, entity_match, score, frozenset(claim_entities)


def _parse_supported_json(raw: str) -> Optional[bool]:
    """Extract ``supported`` from a JSON-ish LLM verifier response, or None on parse failure."""
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        parsed = ClaimVerification.model_validate_json(raw[start:end])
        return parsed.supported
    except Exception:
        return None


async def _llm_verify(claim: str, snippet: str) -> tuple[Optional[bool], str]:
    """Tier-2 LLM verification.

    Provider preference (CLAIM_VERIFIER_PROVIDER env, default ``auto``):
      - ``auto``: DeepSeek R1 via OpenRouter when OPENROUTER_API_KEY is set,
        else GPT-4o-mini via GitHub Models judge().
      - ``deepseek``: force DeepSeek R1.
      - ``gpt4o``: force GPT-4o-mini.

    Returns ``(supported_or_None, method)`` where ``method`` is one of
    ``"llm_deepseek"`` | ``"llm_gpt4o"``. ``supported=None`` on parse/timeout
    failure — the caller maps that to "unsupported" for safety.
    """
    pref = os.environ.get("CLAIM_VERIFIER_PROVIDER", "auto").lower().strip()
    have_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))

    use_deepseek = (
        pref == "deepseek"
        or (pref == "auto" and have_openrouter)
    )
    # Back-compat: when no DeepSeek is involved at all (auto + no key, or
    # explicit "gpt4o"/"llm"/"judge" alias), report the legacy ``"llm"`` method
    # label so pre-existing tests/consumers asserting on equality keep working.
    legacy_label = pref == "auto" and not have_openrouter

    if use_deepseek:
        from utils.provider_router import _claim_verify_with_deepseek
        try:
            raw = await asyncio.wait_for(
                _claim_verify_with_deepseek(claim, snippet), timeout=_LLM_TIMEOUT_S
            )
            return _parse_supported_json(raw), "llm_deepseek"
        except asyncio.TimeoutError:
            logger.warning(
                "Claim verifier DeepSeek timeout", extra={"component": "claim_verifier"}
            )
            if pref == "deepseek":
                return None, "llm_deepseek"
            # auto: fall through to gpt4o
        except Exception as e:
            logger.warning(
                "Claim verifier DeepSeek error (%s); falling back to GPT-4o-mini",
                e, extra={"component": "claim_verifier"},
            )
            if pref == "deepseek":
                return None, "llm_deepseek"
            # auto: fall through

    # GPT-4o-mini path (explicit, or auto-fallback from DeepSeek failure, or no OpenRouter key)
    from utils.provider_router import judge

    prompt = (
        f"CLAIM: {claim}\n"
        f"EVIDENCE SNIPPET: {snippet[:1500]}\n"
        "Does the snippet explicitly support this exact claim "
        "(entities, numbers, dates must match)?\n"
        'JSON only: {"supported": bool, "reasoning": str}'
    )
    method = "llm" if legacy_label else "llm_gpt4o"
    try:
        raw = await asyncio.wait_for(judge(prompt), timeout=_LLM_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("Claim verifier LLM timeout", extra={"component": "claim_verifier"})
        return None, method
    except Exception as e:
        logger.warning("Claim verifier LLM error: %s", e, extra={"component": "claim_verifier"})
        return None, method

    return _parse_supported_json(raw), method


async def _classify_claim(
    claim: str,
    doc_ids: tuple[str, ...],
    snippet_lookup: dict[str, str],
) -> ClaimRecord:
    """Score claim against each cited snippet; take max. Escalate mid-band w/ entities to LLM."""
    best_overlap = 0.0
    best_entity = 0.0
    best_score = 0.0
    best_snippet = ""
    claim_entities: frozenset = frozenset()

    for doc_id in doc_ids:
        snippet = snippet_lookup.get(doc_id, "")
        if not snippet:
            continue
        overlap, entity_match, score, entities = _score_pair(claim, snippet)
        if score > best_score:
            best_overlap = overlap
            best_entity = entity_match
            best_score = score
            best_snippet = snippet
            claim_entities = entities

    if best_score >= _HIGH_THRESHOLD:
        return ClaimRecord(
            claim_text=claim,
            doc_ids=doc_ids,
            overlap=best_overlap,
            entity_match=best_entity,
            method="deterministic",
            score=best_score,
            status="supported",
        )

    if best_score < _LOW_THRESHOLD:
        return ClaimRecord(
            claim_text=claim,
            doc_ids=doc_ids,
            overlap=best_overlap,
            entity_match=best_entity,
            method="deterministic",
            score=best_score,
            status="unsupported",
        )

    # Mid-band [0.3, 0.6): gate on entities
    if not claim_entities:
        return ClaimRecord(
            claim_text=claim,
            doc_ids=doc_ids,
            overlap=best_overlap,
            entity_match=best_entity,
            method="skip",
            score=best_score,
            status="supported",
        )

    # Escalate to LLM
    supported, method = await _llm_verify(claim, best_snippet)
    if supported is None:
        # Timeout or parse failure → unsupported for safety
        return ClaimRecord(
            claim_text=claim,
            doc_ids=doc_ids,
            overlap=best_overlap,
            entity_match=best_entity,
            method=method,
            score=best_score,
            status="unsupported",
        )
    # LLM-resolved: distinguish from deterministic via status label
    return ClaimRecord(
        claim_text=claim,
        doc_ids=doc_ids,
        overlap=best_overlap,
        entity_match=best_entity,
        method=method,
        score=best_score,
        status="ambiguous_resolved" if supported else "unsupported",
    )


def _append_unverified_markers(
    answer: str,
    parsed_claims: list[tuple[str, tuple[str, ...], tuple[int, int]]],
    records: list[ClaimRecord],
) -> str:
    """Append `[UNVERIFIED]` after the citation block for unsupported claims. Idempotent."""
    # Pair parsed claims with records by index; mutate from the END so earlier offsets stay valid.
    mutated = answer
    for (_, _, (_, citation_end)), record in reversed(list(zip(parsed_claims, records))):
        if record.status != "unsupported":
            continue
        # Check idempotency: already marked?
        tail = mutated[citation_end:citation_end + 20]
        if tail.lstrip().startswith("[UNVERIFIED]"):
            continue
        mutated = mutated[:citation_end] + " [UNVERIFIED]" + mutated[citation_end:]
    return mutated


async def verify_claims(
    answer: str,
    doc_map: dict[str, tuple[str, str, str]],
    snippet_lookup: dict[str, str],
) -> tuple[float, list[ClaimRecord], str]:
    """
    Verify each sentence-with-citation against its cited snippet.

    Returns (claim_precision_score, audit_records, mutated_answer).
    Never raises — on any error logs and returns (1.0, [], answer).
    """
    try:
        if not doc_map or not snippet_lookup:
            return 1.0, [], answer

        parsed = parse_claims_with_citations(answer)
        if not parsed:
            return 1.0, [], answer

        records: list[ClaimRecord] = []
        for claim_text, doc_ids, _ in parsed:
            record = await _classify_claim(claim_text.strip(), doc_ids, snippet_lookup)
            records.append(record)

        supported_count = sum(
            1 for r in records if r.status in ("supported", "ambiguous_resolved")
        )
        score = supported_count / max(1, len(records))

        mutated = _append_unverified_markers(answer, parsed, records)
        return score, records, mutated
    except Exception as e:
        logger.warning(
            "Claim verification failed: %s", e, extra={"component": "claim_verifier"}
        )
        return 1.0, [], answer
