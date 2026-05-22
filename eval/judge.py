"""
Metric judge functions. Each is a separate LLM call — no anchor bleed.
All return typed dataclasses. All wrapped in retry.

Default judge provider is Groq Llama-3.3-70B (cross-family vs the Gemini
generator). Override via JUDGE_PROVIDER (groq|github) + JUDGE_MODEL env vars.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ── Tier C: cross-family judge rotation ─────────────────────────────────────

# Per-run counter for GitHub Models calls used during cross-family judging.
# Reset by `reset_cross_family_counter()` at the start of each eval run.
_GITHUB_CROSS_FAMILY_CALLS: int = 0


def reset_cross_family_counter() -> None:
    global _GITHUB_CROSS_FAMILY_CALLS
    _GITHUB_CROSS_FAMILY_CALLS = 0


def get_cross_family_call_count() -> int:
    return _GITHUB_CROSS_FAMILY_CALLS


def _cross_family_quota_budget() -> int:
    """GitHub Models cap is 150/day; default to 140 to leave a buffer."""
    try:
        return int(os.environ.get("CROSS_FAMILY_QUOTA_BUDGET", "140"))
    except ValueError:
        return 140


async def _judge_with_provider(prompt: str, provider: str) -> str:
    """Direct judge call against a specific provider, bypassing JUDGE_PROVIDER env.

    Used by `judge_with_dual_family` to compare primary vs secondary judges.
    Raises on any error so the caller can decide whether to log + skip.
    """
    from openai import AsyncOpenAI
    from utils.provider_router import (
        _GROQ_ROTATOR,
        _JUDGE_PROVIDERS,
        _extract_retry_after_s,
    )

    cfg = _JUDGE_PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(f"unknown judge provider: {provider}")
    key_env = cfg["api_key_env"]
    is_groq = provider == "groq"
    api_key: Optional[str] = None
    if is_groq:
        api_key = _GROQ_ROTATOR.next_key()
        if not api_key:
            raise RuntimeError("No GROQ_API_KEY / GROQ_API_KEYS configured")
    else:
        if not os.environ.get(key_env):
            raise RuntimeError(f"{key_env} not set")
        api_key = os.environ[key_env]
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=cfg["base_url"],
        timeout=60.0,
    )
    try:
        resp = await client.chat.completions.create(
            model=cfg["default_model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.0,
        )
    except Exception as e:
        if is_groq and api_key:
            status = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            if status == 429:
                _GROQ_ROTATOR.mark_throttled(
                    api_key, retry_after_s=_extract_retry_after_s(e)
                )
        raise
    if is_groq and api_key:
        _GROQ_ROTATOR.mark_success(api_key)
    return (resp.choices[0].message.content or "").strip()


async def judge_with_dual_family(prompt: str, metric_name: str) -> dict:
    """Run BOTH Groq Llama (primary) and GitHub GPT-4o-mini (secondary) judges
    and return their scores with the absolute delta.

    The score is extracted from the parsed JSON via the metric's known key:
      faithfulness         → "faithfulness_score"
      answer_relevance     → "answer_relevance_score"
      context_precision    → "context_precision_score"

    Quota guard: counts GitHub calls and short-circuits the secondary leg once
    `CROSS_FAMILY_QUOTA_BUDGET` (default 140) is reached. Returns secondary=None
    in that case + sets the global counter so the runner can flip the
    truncation flag in the summary.

    Returns:
      {"primary_score": float | None,
       "secondary_score": float | None,
       "agreement_delta": float | None}
    """
    global _GITHUB_CROSS_FAMILY_CALLS

    score_key = f"{metric_name}_score"
    primary_provider = os.environ.get("JUDGE_PROVIDER", "groq").lower()
    secondary_provider = "github" if primary_provider == "groq" else "groq"

    primary_score: Optional[float] = None
    secondary_score: Optional[float] = None

    # Primary
    try:
        raw = await _judge_with_provider(prompt, primary_provider)
        data = _parse_json(raw)
        primary_score = float(data.get(score_key, 0.5))
    except Exception as e:
        logger.warning("dual-family primary judge failed: %s", e)

    # Secondary — gated by quota when it's GitHub Models
    if secondary_provider == "github":
        if _GITHUB_CROSS_FAMILY_CALLS >= _cross_family_quota_budget():
            logger.warning(
                "cross-family quota guard tripped at %d calls; skipping secondary",
                _GITHUB_CROSS_FAMILY_CALLS,
            )
            return {
                "primary_score": primary_score,
                "secondary_score": None,
                "agreement_delta": None,
            }
        _GITHUB_CROSS_FAMILY_CALLS += 1

    try:
        raw2 = await _judge_with_provider(prompt, secondary_provider)
        data2 = _parse_json(raw2)
        secondary_score = float(data2.get(score_key, 0.5))
    except Exception as e:
        logger.warning("dual-family secondary judge failed: %s", e)
        secondary_score = None

    delta = (
        abs(primary_score - secondary_score)
        if primary_score is not None and secondary_score is not None
        else None
    )
    return {
        "primary_score": primary_score,
        "secondary_score": secondary_score,
        "agreement_delta": delta,
    }


def _bucket_score(score: float) -> int:
    """Bucket a [0,1] score into 3 ordinal categories for Cohen's κ.

    0 → [0.0, 0.4),  1 → [0.4, 0.7),  2 → [0.7, 1.0]
    """
    if score < 0.4:
        return 0
    if score < 0.7:
        return 1
    return 2


def cohens_kappa_bucketed(
    primary: list[float], secondary: list[float]
) -> Optional[float]:
    """Cohen's κ for two raters on 3-bucket categorization.

    κ = (po - pe) / (1 - pe)
    where po is observed agreement and pe is expected agreement by chance.
    Returns None when pe == 1 (degenerate single-bucket data) or when inputs
    don't line up.
    """
    if not primary or len(primary) != len(secondary):
        return None
    a = [_bucket_score(s) for s in primary]
    b = [_bucket_score(s) for s in secondary]
    n = len(a)
    po = sum(1 for i in range(n) if a[i] == b[i]) / n
    # Marginal probabilities per bucket
    buckets = (0, 1, 2)
    pa = {k: a.count(k) / n for k in buckets}
    pb = {k: b.count(k) / n for k in buckets}
    pe = sum(pa[k] * pb[k] for k in buckets)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def build_faithfulness_prompt(context_xml: str, answer: str) -> str:
    return f"""The user query and/or agent answer may be in Hindi. Evaluate based on
semantic correctness regardless of language; treat Devanagari and Latin script claims equivalently.

CONTEXT:
{context_xml[:3000]}

ANSWER:
{answer}

Extract every factual claim from the answer. For each, determine:
SUPPORTED: explicitly in context | UNSUPPORTED: not in context (hallucination)

JSON only: {{"faithfulness_score": float, "supported_count": int, "unsupported_count": int, "unsupported_claims": [str], "reasoning": str}}"""


def build_relevance_prompt(query: str, answer: str) -> str:
    return f"""The user query and/or agent answer may be in Hindi. Evaluate based on
semantic correctness regardless of language.

QUESTION: {query}

ANSWER:
{answer}

Score 0.0-1.0: Does the answer directly address the question?
0.0=off-topic | 0.5=partial | 1.0=complete

JSON only: {{"answer_relevance_score": float, "reasoning": str}}"""


def build_context_precision_prompt(query: str, context_xml: str) -> str:
    return f"""The user query and/or context may be in Hindi. Evaluate based on
semantic correctness regardless of language.

QUESTION: {query}

CONTEXT:
{context_xml[:3000]}

Score 0.0-1.0: Does the context contain sufficient information to fully answer the question?
0.0=none | 0.5=partial | 1.0=complete

JSON only: {{"context_precision_score": float, "reasoning": str}}"""


def pearson_correlation(xs: list[float], ys: list[float]) -> Optional[float]:
    """Manual Pearson r — no scipy. Returns None on degenerate inputs."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


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


def score_uncertainty_handling(
    category: str,
    answer: str,
    uncertainty_kind: Optional[str],
    follow_up_queries: Optional[list[str]],
) -> Optional[float]:
    """Deterministic uncertainty-handling score (Phase 1.5).

    Only meaningful for `insufficient_evidence` questions. For other
    categories returns None so the aggregator skips them.

    Score is in [0.0, 1.0]:
      * 0.5 for tagging the turn with a non-"none" `uncertainty_kind`
      * +0.3 for an [UNCERTAINTY] block actually rendered in the answer
      * +0.2 for ≥3 non-empty `follow_up_queries`
    Cheap and anchor-free; complements the LLM-based judges.
    """
    if (category or "").lower() != "insufficient_evidence":
        return None
    score = 0.0
    if uncertainty_kind and uncertainty_kind != "none":
        score += 0.5
    if "[UNCERTAINTY]" in (answer or ""):
        score += 0.3
    fu = [q for q in (follow_up_queries or []) if q and q.strip()]
    if len(fu) >= 3:
        score += 0.2
    return min(1.0, score)


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

# Multi-Indic word tokenizer — pure regex, no external libs.
# Covers Devanagari (Hindi/Marathi/Sanskrit), Tamil, and Bengali/Assamese.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]+")
_INDIC_WORD_RE = re.compile(r"[ऀ-ॿ஀-௿ঀ-৿]+")


def _extract_entities_for_consistency(answer: str) -> set[str]:
    """Language-aware entity extraction. Latin-script entities via ENTITY_RE
    (proper nouns / numbers); Indic word tokens (Devanagari, Tamil, Bengali)
    added when present. Lowercased + stripped for set membership."""
    from agent.claim_verifier import ENTITY_RE

    out: set[str] = {e.strip().lower() for e in ENTITY_RE.findall(answer or "") if e.strip()}
    for token in _INDIC_WORD_RE.findall(answer or ""):
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
    """For every concept_id with an English answer plus at least one Indic
    answer (hi / ta / bn / mr) in ``run_at``, compute Jaccard over the entity
    sets and flag pairs <0.6. Returns one dict per (concept_id, indic_language).

    English is anchored as the reference language. Multi-Indic pairing (e.g.
    ta↔bn) is intentionally not emitted — comparing two Indic scripts to each
    other has near-zero token overlap by construction.
    """
    import json as _json
    from pathlib import Path

    import aiosqlite
    from agent.memory import DB_PATH

    INDIC_LANGS = ("hi", "ta", "bn", "mr")

    # Load dataset to map question_id -> concept_id.
    dataset_path = Path(__file__).parent / "dataset.json"
    with open(dataset_path) as fh:
        dataset = _json.load(fh)
    pairs: dict[str, dict[str, str]] = {}  # concept_id -> {lang: qid}
    for q in dataset:
        cid = q.get("concept_id")
        if not cid:
            continue
        pairs.setdefault(cid, {})[q.get("language", "en")] = q["id"]

    paired: list[tuple[str, str, str, str]] = []  # (cid, en_qid, indic_lang, indic_qid)
    for cid, langs in pairs.items():
        if "en" not in langs:
            continue
        for lang in INDIC_LANGS:
            if lang in langs:
                paired.append((cid, langs["en"], lang, langs[lang]))
    if not paired:
        return []

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT question_id, agent_answer FROM eval_runs WHERE run_at = ?",
            (run_at,),
        )
    answers = {r["question_id"]: (r["agent_answer"] or "") for r in rows}

    results: list[dict] = []
    for cid, en_qid, indic_lang, indic_qid in paired:
        if en_qid not in answers or indic_qid not in answers:
            continue
        en_ents = _extract_entities_for_consistency(answers[en_qid])
        indic_ents = _extract_entities_for_consistency(answers[indic_qid])
        score = jaccard(en_ents, indic_ents)
        flagged = score < 0.6
        reason = (
            f"|en|={len(en_ents)} |{indic_lang}|={len(indic_ents)} "
            f"|∩|={len(en_ents & indic_ents)} |∪|={len(en_ents | indic_ents)}"
        )
        results.append({
            "concept_id": cid,
            "en_question_id": en_qid,
            "indic_language": indic_lang,
            # The DB column is named hi_question_id for legacy reasons; we keep
            # populating it with the Indic partner qid regardless of script so
            # the existing schema/saver continues to work.
            "hi_question_id": indic_qid,
            "indic_question_id": indic_qid,
            "jaccard_score": score,
            "flagged_inconsistent": flagged,
            "reasoning": reason,
        })
    return results
