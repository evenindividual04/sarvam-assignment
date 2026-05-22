"""Phase 1.5 — deterministic follow-up query generation.

Given a failed retrieval outcome (missing/weak/conflict), propose 3
ALTERNATE search queries. Uses Groq Llama 3.3 70B when reachable; falls
back to deterministic heuristics (broaden / narrow / rephrase) otherwise.

Contract: never raises. Always returns exactly 3 strings, each distinct
from every executed query (case-insensitive).
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
from typing import Literal

logger = logging.getLogger(__name__)

Outcome = Literal["missing", "weak", "conflict"]

# ContextVar so concurrent ``propose_follow_ups`` calls each see their own
# provider attribution. Setters run in the caller's task context so values
# propagate back after ``await`` returns.
_LAST_FOLLOW_UP_PROVIDER_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "next_steps.last_follow_up_provider", default="heuristic"
)


def last_follow_up_provider() -> str:
    """Provider used for the most recent ``propose_follow_ups`` invocation."""
    return _LAST_FOLLOW_UP_PROVIDER_VAR.get()


_PROMPT = """The user asked: "{query}"
We tried these queries with insufficient results: {executed}
Outcome: {outcome}  # missing = zero usable sources, weak = low confidence, conflict = sources disagree
Propose 3 ALTERNATIVE search queries (different phrasings or angles) that might yield better evidence.
The new queries MUST be different from the executed ones.
Output JSON only: {{"follow_ups": ["q1", "q2", "q3"]}}"""


def _heuristic_follow_ups(original_query: str, executed_queries: list[str]) -> list[str]:
    """Deterministic fallback when the LLM is unavailable or parse fails.

    Produces three angles: broader (drop most-specific token), narrower
    (year qualifier), and a question rephrasing. Strips duplicates against
    `executed_queries` case-insensitively.
    """
    q = original_query.strip() or "the topic"
    tokens = q.split()
    # Broader: drop the longest token (proxy for "most specific")
    if len(tokens) > 1:
        longest = max(tokens, key=len)
        broader = " ".join(t for t in tokens if t != longest).strip() or q
    else:
        broader = f"overview of {q}"
    # Narrower: pin a year + authority qualifier
    narrower = f"{q} 2026 official source"
    # Question rephrasing
    if q.lower().startswith(("what", "how", "why", "when", "where", "who")):
        rephrased = f"explain {q}"
    else:
        rephrased = f"What is {q}?"

    candidates = [broader, narrower, rephrased]
    executed_lower = {e.strip().lower() for e in executed_queries}

    # Ensure distinct vs executed. If a candidate collides, mutate it.
    out: list[str] = []
    for i, c in enumerate(candidates):
        c = c.strip()
        if c.lower() in executed_lower or c.lower() in {x.lower() for x in out}:
            c = f"{c} (alternative angle {i + 1})"
        out.append(c)
    return out[:3]


def _parse_response(raw: str) -> list[str] | None:
    """Strict JSON extraction. Returns None on any failure."""
    if not raw:
        return None
    # Strip code fences if model wrapped output.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Last-ditch: pluck the first {...} block.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    follow_ups = data.get("follow_ups") if isinstance(data, dict) else None
    if not isinstance(follow_ups, list) or len(follow_ups) < 1:
        return None
    cleaned = [str(x).strip() for x in follow_ups if isinstance(x, (str, int, float)) and str(x).strip()]
    if len(cleaned) < 3:
        return None
    return cleaned[:3]


async def propose_follow_ups(
    original_query: str,
    executed_queries: list[str],
    outcome: Outcome,
) -> list[str]:
    """Return 3 alternate follow-up queries. Never raises.

    Tries Groq first (~500ms, free tier). Falls back to heuristics on any
    failure path. All returned queries are guaranteed distinct from the
    executed list (case-insensitive).
    """
    executed_clean = [q for q in executed_queries if q and q.strip()]
    try:
        from utils.provider_router import call_cerebras, call_groq

        prompt = _PROMPT.format(
            query=original_query,
            executed=json.dumps(executed_clean, ensure_ascii=False),
            outcome=outcome,
        )

        pref = os.environ.get("FOLLOW_UP_PROVIDER", "auto").lower().strip()
        raw: str | None = None
        chosen: str | None = None
        if pref in ("auto", "cerebras") and os.environ.get("CEREBRAS_API_KEY"):
            try:
                raw = await call_cerebras(prompt, max_tokens=300)
                chosen = "cerebras"
            except Exception as e:
                logger.warning(
                    "follow_ups Cerebras failed (%s); falling back to Groq", e,
                    extra={"component": "next_steps"},
                )
                raw = None
        if raw is None and pref != "cerebras":
            # Wrap Groq in its own try so a failure here doesn't mislabel a
            # successful Cerebras attempt above. If Groq raises, the outer
            # ``except`` catches it and falls back to heuristic.
            try:
                raw = await call_groq(prompt, max_tokens=300)
                chosen = "groq"
            except Exception as e:
                logger.warning(
                    "follow_ups Groq failed (%s); falling back to heuristic", e,
                    extra={"component": "next_steps"},
                )
                raw = None
        if raw is None or chosen is None:
            raise ValueError("no provider produced output")
        _LAST_FOLLOW_UP_PROVIDER_VAR.set(chosen)
        parsed = _parse_response(raw)
        if parsed is None:
            raise ValueError("follow_ups parse failed")
        # Enforce distinctness from executed queries.
        executed_lower = {e.strip().lower() for e in executed_clean}
        deduped: list[str] = []
        for q in parsed:
            if q.strip().lower() in executed_lower or q.strip().lower() in {x.lower() for x in deduped}:
                continue
            deduped.append(q.strip())
        if len(deduped) >= 3:
            return deduped[:3]
        # Top up with heuristic candidates if dedupe drained the list.
        for h in _heuristic_follow_ups(original_query, executed_clean):
            if h.lower() in {x.lower() for x in deduped} or h.lower() in executed_lower:
                continue
            deduped.append(h)
            if len(deduped) >= 3:
                break
        return deduped[:3] if len(deduped) >= 3 else _heuristic_follow_ups(original_query, executed_clean)
    except Exception as e:
        logger.warning(
            "propose_follow_ups fallback: %s", e,
            extra={"component": "next_steps"},
        )
        _LAST_FOLLOW_UP_PROVIDER_VAR.set("heuristic")
        return _heuristic_follow_ups(original_query, executed_clean)
