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
        _CEREBRAS_ROTATOR,
        _GROQ_ROTATOR,
        _JUDGE_PROVIDERS,
        _extract_retry_after_s,
    )

    cfg = _JUDGE_PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(f"unknown judge provider: {provider}")
    key_env = cfg["api_key_env"]
    is_groq = provider == "groq"
    is_cerebras = provider == "cerebras"
    api_key: Optional[str] = None
    if is_groq:
        api_key = _GROQ_ROTATOR.next_key()
        if not api_key:
            raise RuntimeError("No GROQ_API_KEY / GROQ_API_KEYS configured")
    elif is_cerebras:
        api_key = _CEREBRAS_ROTATOR.next_key()
        if not api_key:
            raise RuntimeError(
                "No CEREBRAS_API_KEY / CEREBRAS_API_KEYS configured"
            )
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
        if api_key:
            status = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            if status == 429:
                if is_groq:
                    _GROQ_ROTATOR.mark_throttled(
                        api_key, retry_after_s=_extract_retry_after_s(e)
                    )
                elif is_cerebras:
                    _CEREBRAS_ROTATOR.mark_throttled(
                        api_key, retry_after_s=_extract_retry_after_s(e)
                    )
        raise
    if api_key:
        if is_groq:
            _GROQ_ROTATOR.mark_success(api_key)
        elif is_cerebras:
            _CEREBRAS_ROTATOR.mark_success(api_key)
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
    # Cross-family rotation: pair the primary with a judge from an unrelated
    # family. github (gpt-4o-mini, OpenAI) is the default secondary for both
    # groq (Llama) and cerebras (Qwen) primaries. When primary IS github, fall
    # back to groq for the secondary leg.
    secondary_provider = (
        "groq" if primary_provider == "github" else "github"
    )

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
    """Score how well the answer handles conflicting sources.

    Rubric:
      - 0.0 — Answer missed the conflict entirely (silently picked one side
              or failed to acknowledge disagreement).
      - 0.5 — Answer surfaced the conflict but mislabeled its kind
              (self / pair / conditional) or omitted the temporal /
              sub-domain qualifier when one was required.
      - 1.0 — Answer correctly surfaced AND correctly identified the kind
              (with qualifier when kind=conditional).

    The score remains a single float in [0, 1] to preserve the existing
    `JudgeScore.conflict_adherence_score` schema.
    """
    from utils.provider_router import judge as call_judge

    prompt = f"""The user query and/or agent answer may be in Hindi. Evaluate based on
semantic correctness regardless of language.

QUESTION: {query}

CONTEXT:
{context_xml[:2000]}

ANSWER:
{answer}

This question has known conflicting sources. Classify any contradiction
into one of three conflict kinds:
  - \"self\"        : a single source contradicts itself
  - \"pair\"        : two sources disagree on a fact in the same time window
  - \"conditional\" : sources appear to disagree but reconcile under a
                    temporal or sub-domain qualifier (e.g. \"Indian PM\"
                    answered differently by sources from 2014 vs 2024)

Score with this exact rubric (return ONE float in [0, 1]):
  - 0.0 : missed the conflict entirely (silently picked a side or didn't notice)
  - 0.5 : surfaced the conflict but mislabeled the kind, OR omitted the
          qualifier when the conflict was \"conditional\"
  - 1.0 : correctly surfaced the conflict AND correctly identified the kind
          (with qualifier present when kind=\"conditional\")

Also report:
  - identified_conflict: did the answer surface the conflict at all?
  - presented_both_sides: are both positions cited?
  - expressed_uncertainty: is hedging language present?
  - avoided_silent_choice: did the answer avoid picking a winner?

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
    from agent.citation_guard import extract_doc_ids
    cited_ids = extract_doc_ids(answer)
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


# ── C3 calibration: model self-confidence vs judge confidence ───────────────
#
# Measures whether the answer's own hedging language accurately reflects the
# actual evidence quality the judge observed — i.e. does the model know when
# it doesn't know?
#
#   model_self_confidence ∈ [0,1]  — derived from hedge vs assertion phrases
#   judge_confidence      ∈ [0,1]  — derived from faithfulness + context_precision
#   per_case_error        = |model_self_confidence - judge_confidence|
#   calibration_score     = 1 - mean(per_case_error)              # MAE-based
#   brier_score           = 1 - mean((self - judge)^2)            # Brier-style
#
# Scoped to evidence-stressed categories (insufficient_evidence, conflicting)
# where calibration actually matters.

_HEDGE_PHRASES: tuple[str, ...] = (
    "unclear",
    "uncertain",
    "could not",
    "no reliable",
    "limited evidence",
    "cannot determine",
    "[unverified]",
    "[uncertainty]",
    "not publicly available",
    "no definitive",
    "sources differ",
    "evidence is mixed",
    "unable to confirm",
)

# Assertion markers. We deliberately exclude bare copulas ("is", "are", ...)
# — they're too noisy: a hedging sentence "it is unclear" contains "is" too.
# We rely on stronger assertion markers that are rarely present in genuine
# hedging language.
_ASSERTION_PHRASES: tuple[str, ...] = (
    "established",
    "confirmed",
    "definitively",
    "according to",
    "verified",
    "reported that",
    "states that",
    "shows that",
    "demonstrates",
    "the answer is",
    "the result is",
)


def _count_hedge_phrases(answer: str) -> int:
    if not answer:
        return 0
    low = answer.lower()
    return sum(low.count(h) for h in _HEDGE_PHRASES)


def _count_assertion_phrases(answer: str) -> int:
    if not answer:
        return 0
    low = answer.lower()
    return sum(low.count(p) for p in _ASSERTION_PHRASES)


def _compute_model_self_confidence(answer: str) -> float:
    """[0,1] confidence inferred from the answer's hedge ↔ assertion balance.

    Hedge-heavy → low; assertion-heavy → high. Symmetric and well-defined for
    the empty/no-signal case (returns 0.5).
    """
    hedges = _count_hedge_phrases(answer)
    asserts = _count_assertion_phrases(answer)
    denom = hedges + asserts
    if denom == 0:
        return 0.5  # no signal — middle prior
    # (assert - hedge) / (assert + hedge) ∈ [-1, 1]  →  rescaled to [0, 1]
    raw = (asserts - hedges) / denom
    return (raw + 1.0) / 2.0


def _rescale_judge_score(score: Optional[float]) -> Optional[float]:
    """Judges in this codebase emit scores in [0,1] already. We accept either
    the [0,1] convention or the legacy [1,5] convention via clamping. Returns
    None when the input is None."""
    if score is None:
        return None
    if score <= 1.0:
        return max(0.0, float(score))
    # Treat as a 1–5 Likert and rescale to [0,1].
    return max(0.0, min(1.0, (float(score) - 1.0) / 4.0))


def _compute_judge_confidence(
    faithfulness: Optional[float],
    context_precision: Optional[float],
) -> Optional[float]:
    """Mean of rescaled faithfulness + context_precision. None when both
    components are missing (no judge signal at all)."""
    f01 = _rescale_judge_score(faithfulness)
    c01 = _rescale_judge_score(context_precision)
    parts = [v for v in (f01, c01) if v is not None]
    if not parts:
        return None
    return sum(parts) / len(parts)


_DEFAULT_C3_CATEGORIES: tuple[str, ...] = (
    "insufficient_evidence",
    "conflicting",
    "conflicting_sources",  # accept both spellings
)


def compute_calibration_score(
    rows: list[dict],
    categories: tuple[str, ...] = _DEFAULT_C3_CATEGORIES,
) -> dict:
    """C3 calibration aggregate. Pure-Python, no LLM.

    Returns:
      {
        "calibration_score": 1 - mean|self - judge|,   # higher is better
        "brier_score":       1 - mean((self - judge)^2),
        "mean_abs_error":    mean|self - judge|,
        "n_cases":           int,
        "per_category":      {cat: {"calibration_score", "n_cases", ...}},
        "per_case":          [{"question_id", "category",
                               "model_self_confidence", "judge_confidence",
                               "abs_error"}],
      }
    Empty selection → calibration_score = 1.0, brier_score = 1.0, n_cases = 0
    (no error is possible without cases — neutral default).
    """
    cats = {c.lower() for c in categories}
    pairs: list[tuple[str, str, float, float, float]] = []  # (qid, cat, self, judge, err)
    for r in rows:
        cat = (r.get("category") or "").lower()
        if cat not in cats:
            continue
        answer = r.get("agent_answer") or ""
        m_conf = _compute_model_self_confidence(answer)
        j_conf = _compute_judge_confidence(
            r.get("faithfulness_score"),
            r.get("context_precision_score"),
        )
        if j_conf is None:
            continue
        err = abs(m_conf - j_conf)
        pairs.append((r.get("question_id", ""), cat, m_conf, j_conf, err))

    per_case = [
        {
            "question_id": qid,
            "category": cat,
            "model_self_confidence": round(m, 4),
            "judge_confidence": round(j, 4),
            "abs_error": round(err, 4),
        }
        for (qid, cat, m, j, err) in pairs
    ]

    if not pairs:
        return {
            "calibration_score": 1.0,
            "brier_score": 1.0,
            "mean_abs_error": 0.0,
            "n_cases": 0,
            "per_category": {},
            "per_case": [],
        }

    mae = sum(p[4] for p in pairs) / len(pairs)
    brier = sum((p[2] - p[3]) ** 2 for p in pairs) / len(pairs)

    per_category: dict[str, dict] = {}
    for cat in sorted({p[1] for p in pairs}):
        cat_pairs = [p for p in pairs if p[1] == cat]
        cat_mae = sum(p[4] for p in cat_pairs) / len(cat_pairs)
        cat_brier = sum((p[2] - p[3]) ** 2 for p in cat_pairs) / len(cat_pairs)
        per_category[cat] = {
            "n_cases": len(cat_pairs),
            "calibration_score": 1.0 - cat_mae,
            "brier_score": 1.0 - cat_brier,
            "mean_abs_error": cat_mae,
        }

    return {
        "calibration_score": 1.0 - mae,
        "brier_score": 1.0 - brier,
        "mean_abs_error": mae,
        "n_cases": len(pairs),
        "per_category": per_category,
        "per_case": per_case,
    }


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


# ── C4: Cross-script consistency ────────────────────────────────────────────
#
# Same factual question, posed in different scripts (e.g. English vs Hindi),
# must yield answers that cite the *same set of facts*. This is a pair-level
# aggregator over already-completed eval runs — pure Python, no LLM call.
#
# Topics are linked by `concept_id` (existing dataset field). For each topic
# with ≥2 languages, we compute pairwise Jaccard over:
#   1. Cited URLs (domain-normalized) — citation_overlap
#   2. Extracted entities (or gold_entities when present) — entity_overlap
#
# Distinct from `judge_cross_language_consistency` above, which is an
# English-anchored single-language entity check. C4 is multi-pair and adds
# the URL-overlap signal.

from urllib.parse import urlparse

_MD_LINK_URL_RE = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")


def _normalize_url(url: str) -> str:
    """Normalize a URL for set-membership comparison.

    - Lowercase scheme + netloc (strip ``www.`` prefix).
    - Keep path verbatim (case-sensitive — many CMSes are).
    - Drop query string and fragment (often tracking noise).
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
    except (ValueError, AttributeError):
        return url.strip().lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    scheme = (parsed.scheme or "https").lower()
    path = parsed.path or ""
    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def _extract_urls_from_row(row: dict) -> set[str]:
    """Pull a row's cited URLs from explicit fields, falling back to the
    markdown links embedded in ``agent_answer``."""
    explicit = row.get("cited_urls") or row.get("urls_opened")
    if explicit:
        return {_normalize_url(u) for u in explicit if u}
    answer = row.get("agent_answer") or ""
    return {_normalize_url(u) for u in _MD_LINK_URL_RE.findall(answer)}


def _extract_entities_from_row(row: dict) -> set[str]:
    """Prefer explicit ``entities`` / ``gold_entities`` (lowercased) when
    present; otherwise extract from the answer text using the language-aware
    helper used by the single-language consistency judge."""
    explicit = row.get("entities") or row.get("gold_entities")
    if explicit:
        return {str(e).strip().lower() for e in explicit if str(e).strip()}
    return _extract_entities_for_consistency(row.get("agent_answer") or "")


def compute_cross_script_consistency(rows: list[dict]) -> dict:
    """Compute pair-level cross-script citation + entity overlap.

    Groups rows by ``concept_id`` (the dataset's topic identifier). For each
    topic with ≥2 *distinct* languages, emits one record per language pair
    with Jaccard over cited URLs and extracted entities.

    Args:
        rows: result rows from the eval run. Each must carry at minimum
            ``question_id``, ``language``, and one of ``agent_answer`` /
            ``cited_urls`` (URLs are best, but the function degrades to
            extracting markdown links from the answer text).
            The topic key is read from ``concept_id`` (preferred) or
            ``topic_id``.

    Returns:
        ``{"mean_citation_overlap", "mean_entity_overlap", "n_pairs",
           "n_topics", "per_topic", "pairs"}``. When no qualifying topic has
        ≥2 languages, all aggregates are ``None`` and lists are empty.
    """
    # 1. Bucket rows by topic.
    by_topic: dict[str, dict[str, dict]] = {}
    for row in rows:
        topic = row.get("concept_id") or row.get("topic_id")
        lang = row.get("language") or "en"
        if not topic:
            continue
        # First row wins per (topic, lang) — defensive against dupes.
        by_topic.setdefault(topic, {}).setdefault(lang, row)

    pairs_out: list[dict] = []
    per_topic: dict[str, dict] = {}

    for topic, lang_map in by_topic.items():
        if len(lang_map) < 2:
            # Single-language group — metric is N/A per spec.
            continue
        langs = sorted(lang_map.keys())
        topic_citation: list[float] = []
        topic_entity: list[float] = []
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                la, lb = langs[i], langs[j]
                row_a, row_b = lang_map[la], lang_map[lb]
                urls_a = _extract_urls_from_row(row_a)
                urls_b = _extract_urls_from_row(row_b)
                ents_a = _extract_entities_from_row(row_a)
                ents_b = _extract_entities_from_row(row_b)
                citation_overlap = jaccard(urls_a, urls_b)
                entity_overlap = jaccard(ents_a, ents_b)
                pairs_out.append({
                    "concept_id": topic,
                    "lang_a": la,
                    "lang_b": lb,
                    "question_id_a": row_a.get("question_id"),
                    "question_id_b": row_b.get("question_id"),
                    "n_urls_a": len(urls_a),
                    "n_urls_b": len(urls_b),
                    "n_entities_a": len(ents_a),
                    "n_entities_b": len(ents_b),
                    "citation_overlap": citation_overlap,
                    "entity_overlap": entity_overlap,
                })
                topic_citation.append(citation_overlap)
                topic_entity.append(entity_overlap)
        per_topic[topic] = {
            "languages": langs,
            "n_pairs": len(topic_citation),
            "mean_citation_overlap": (
                sum(topic_citation) / len(topic_citation) if topic_citation else None
            ),
            "mean_entity_overlap": (
                sum(topic_entity) / len(topic_entity) if topic_entity else None
            ),
        }

    if not pairs_out:
        return {
            "mean_citation_overlap": None,
            "mean_entity_overlap": None,
            "n_pairs": 0,
            "n_topics": 0,
            "per_topic": {},
            "pairs": [],
        }

    mean_cit = sum(p["citation_overlap"] for p in pairs_out) / len(pairs_out)
    mean_ent = sum(p["entity_overlap"] for p in pairs_out) / len(pairs_out)
    return {
        "mean_citation_overlap": mean_cit,
        "mean_entity_overlap": mean_ent,
        "n_pairs": len(pairs_out),
        "n_topics": len(per_topic),
        "per_topic": per_topic,
        "pairs": pairs_out,
    }
