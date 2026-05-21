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


@dataclass
class ClaimPrecisionResult:
    claim_precision_score: float
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

    prompt = f"""The user query and/or agent answer may be in Hindi. Evaluate based on
semantic correctness regardless of language; treat Devanagari and Latin script claims equivalently.

CONTEXT:
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

    prompt = f"""The user query and/or agent answer may be in Hindi. Evaluate based on
semantic correctness regardless of language.

QUESTION: {query}

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

    prompt = f"""The user query and/or agent answer may be in Hindi. Evaluate based on
semantic correctness regardless of language.

QUESTION: {query}

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

    prompt = f"""The user queries and/or agent answers may be in Hindi. Evaluate based on
semantic correctness regardless of language.

TURN_1_Q: {t1_query}
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

    prompt = f"""The user query and/or context may be in Hindi. Evaluate based on
semantic correctness regardless of language.

QUESTION: {query}

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


async def judge_claim_precision(turn_id: str) -> ClaimPrecisionResult:
    """Deterministic. Aggregates from `claim_audit` populated at synthesis time."""
    import aiosqlite
    from agent.memory import DB_PATH

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT method, status FROM claim_audit WHERE turn_id = ?",
            (turn_id,),
        )
    rows = [dict(r) for r in rows]
    if not rows:
        return ClaimPrecisionResult(
            claim_precision_score=1.0,
            reasoning="No claims to verify.",
        )
    total = len(rows)
    supported = sum(1 for r in rows if r["status"] in ("supported", "ambiguous_resolved"))
    det = sum(1 for r in rows if r["method"] == "deterministic")
    llm = sum(1 for r in rows if r["method"] == "llm")
    skip = sum(1 for r in rows if r["method"] == "skip")
    score = supported / total
    return ClaimPrecisionResult(
        claim_precision_score=score,
        reasoning=f"Verified {supported}/{total} claims ({det} deterministic, {llm} LLM-resolved, {skip} skipped).",
    )


def classify_failure(r: dict) -> str:
    """Failure taxonomy. HALLUCINATION is split into two sub-buckets:
    - HALLUCINATION_FACT: low faithfulness AND low claim precision — the
      answer asserts facts not in any source.
    - HALLUCINATION_ATTRIBUTION: low claim precision but high citation
      integrity — citations resolve to fetched URLs, but the claim doesn't
      actually appear in the cited doc.
    """
    faith = r.get("faithfulness_score", 1.0)
    cite = r.get("citation_integrity_score", 1.0)
    rel = r.get("answer_relevance_score", 1.0)
    ctx = r.get("context_precision_score", 1.0)
    conflict_score = r.get("conflict_adherence_score")
    coherence_score = r.get("session_coherence_score")
    claim_precision = r.get("claim_precision_score")

    if faith < 0.7 and claim_precision is not None and claim_precision < 0.5:
        return "HALLUCINATION_FACT"
    if claim_precision is not None and claim_precision < 0.5 and cite >= 0.8:
        return "HALLUCINATION_ATTRIBUTION"
    if faith < 0.7 and cite < 0.8:
        # legacy combined bucket — kept for backward compatibility when we
        # can't disambiguate (e.g. claim_precision unavailable).
        return "HALLUCINATION_FACT"
    if faith < 0.7:
        return "KNOWLEDGE_BLEED"
    if rel < 0.5 or ctx < 0.5:
        return "RETRIEVAL_FAILURE"
    if conflict_score is not None and conflict_score < 0.5:
        return "CONFLICT_MISS"
    if coherence_score is not None and coherence_score < 0.5:
        return "COHERENCE_FAIL"
    return "PASS"


# ── Factual accuracy (deterministic, gold-truth-based) ──────────────────────


def judge_factual_accuracy(
    answer: str,
    gold_answer: Optional[str],
    gold_aliases: Optional[list[str]] = None,
    gold_entities: Optional[list[str]] = None,
) -> tuple[Optional[float], str]:
    """Deterministic factual-accuracy judge. Returns (score, reasoning).

    Score is None when no gold answer is configured (skip).
    Otherwise: max of
      - substring match against gold_answer or any alias (1.0 / 0.0)
      - entity intersection ratio over gold_entities
    """
    if gold_answer is None:
        return (None, "skipped: no gold answer configured")

    answer_lower = (answer or "").lower()
    candidates: list[str] = [gold_answer]
    if gold_aliases:
        candidates.extend(gold_aliases)

    matched = next(
        (c for c in candidates if c and c.lower() in answer_lower),
        None,
    )
    substring_score = 1.0 if matched else 0.0

    entity_ratio = 0.0
    entity_detail = ""
    if gold_entities:
        from agent.claim_verifier import ENTITY_RE

        answer_entities = {e.strip() for e in ENTITY_RE.findall(answer or "") if e.strip()}
        gold_set = {e.strip() for e in gold_entities if e.strip()}
        if gold_set:
            # Treat a gold entity as matched if any extracted entity contains
            # (or equals) the gold token — e.g. gold="Modi" matches answer
            # entity "PM Modi". This is intentionally loose; entities are an
            # auxiliary signal that backstops the canonical substring check.
            hits = {
                g for g in gold_set
                if any(g.lower() in ae.lower() for ae in answer_entities)
            }
            entity_ratio = len(hits) / len(gold_set)
            entity_detail = (
                f"; entities matched {len(hits)}/{len(gold_set)} ({sorted(hits)})"
                if hits else
                f"; no entity match (expected {sorted(gold_set)})"
            )

    final = max(substring_score, entity_ratio)
    if matched:
        reason = f"Substring match on '{matched}'{entity_detail}"
    elif entity_ratio > 0:
        reason = f"No substring match; entity ratio {entity_ratio:.2f}{entity_detail}"
    else:
        reason = f"No match for any of: {candidates}{entity_detail}"
    return (final, reason)


# ── Cross-language consistency (post-run aggregate) ─────────────────────────

# Devanagari word tokenizer — pure regex, no external libs.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]+")


def _extract_entities_for_consistency(answer: str) -> set[str]:
    """Language-aware entity extraction. Latin-script entities via ENTITY_RE
    (proper nouns / numbers); Devanagari word tokens added when present.
    Lowercased + stripped for set membership."""
    from agent.claim_verifier import ENTITY_RE

    out: set[str] = {e.strip().lower() for e in ENTITY_RE.findall(answer or "") if e.strip()}
    for token in _DEVANAGARI_RE.findall(answer or ""):
        token = token.strip()
        if len(token) >= 2:
            out.add(token.lower())
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


async def judge_cross_language_consistency(run_at: str) -> list[dict]:
    """For every concept_id with both en and hi answers in `run_at`,
    compute Jaccard over their entity sets and flag pairs <0.6.
    Returns one dict per concept_id."""
    import json as _json
    from pathlib import Path

    import aiosqlite
    from agent.memory import DB_PATH

    # Load dataset to map question_id -> concept_id.
    dataset_path = Path(__file__).parent / "dataset.json"
    with open(dataset_path) as fh:
        dataset = _json.load(fh)
    by_qid = {q["id"]: q for q in dataset}
    pairs: dict[str, dict[str, str]] = {}  # concept_id -> {"en": qid, "hi": qid}
    for q in dataset:
        cid = q.get("concept_id")
        if not cid:
            continue
        pairs.setdefault(cid, {})[q.get("language", "en")] = q["id"]

    paired_concepts = [
        (cid, langs["en"], langs["hi"])
        for cid, langs in pairs.items()
        if "en" in langs and "hi" in langs
    ]
    if not paired_concepts:
        return []

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT question_id, agent_answer FROM eval_runs WHERE run_at = ?",
            (run_at,),
        )
    answers = {r["question_id"]: (r["agent_answer"] or "") for r in rows}

    results: list[dict] = []
    for cid, en_qid, hi_qid in paired_concepts:
        if en_qid not in answers or hi_qid not in answers:
            continue
        en_ents = _extract_entities_for_consistency(answers[en_qid])
        hi_ents = _extract_entities_for_consistency(answers[hi_qid])
        score = jaccard(en_ents, hi_ents)
        flagged = score < 0.6
        reason = (
            f"|en|={len(en_ents)} |hi|={len(hi_ents)} |∩|={len(en_ents & hi_ents)} |∪|={len(en_ents | hi_ents)}"
        )
        results.append({
            "concept_id": cid,
            "en_question_id": en_qid,
            "hi_question_id": hi_qid,
            "jaccard_score": score,
            "flagged_inconsistent": flagged,
            "reasoning": reason,
        })
    return results
