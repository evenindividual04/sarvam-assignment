"""P1 — Adaptive Hop Stopping (Stop-RAG gate).

Inspired by Park et al., "Stop-RAG: Value-Based Retrieval Control for
Iterative RAG" (arXiv:2510.14337, NeurIPS 2025 MTI-LLM workshop). We
approximate the value-based stopping signal with a fast Groq Llama 3.3
70B JSON call that asks: given what we already grounded, is another
search hop useful?

Compliance constraint (sarvam-assignment.md L103: no hidden CoT
streaming): the `reason` field MUST stay inside `run_metadata` only;
callers must never emit it on an SSE event payload. Only the boolean
verdict and the float confidence are safe to surface to the UI.

The function NEVER raises — on any failure it returns a `degraded`
decision that keeps the hop loop running (degrade-to-safer).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


_TIMEOUT_S = 2.5
_MAX_TOKENS = 80


@dataclass(frozen=True)
class StopDecision:
    """Verdict for "should the agent run another retrieval hop?"."""

    another_hop_useful: bool
    confidence: float
    reason: str  # stored in run_metadata only — never streamed via SSE
    degraded: bool  # True when the LLM call failed and we fell back to "continue"


def _format_evidence_summary(hop_evidence: dict | None) -> str:
    """Render the existing hop_evidence ledger as 2-3 lines of facts.

    We deliberately do NOT call an LLM here — the grounded rows come
    from `_compute_hop_evidence()`, which is set-difference + regex.
    """
    if not hop_evidence:
        return "(no grounded evidence yet)"
    grounded = (hop_evidence.get("grounded") or [])[:3]
    if not grounded:
        return "(no grounded evidence yet)"
    lines: list[str] = []
    for row in grounded:
        if not isinstance(row, dict):
            continue
        tok = row.get("token") or ""
        kind = row.get("kind") or ""
        url = row.get("url") or ""
        lines.append(f"- {tok} ({kind}) @ {url}")
    return "\n".join(lines) if lines else "(no grounded evidence yet)"


def _format_open_criteria(hop_evidence: dict | None) -> str:
    if not hop_evidence:
        return "(none)"
    open_rows = (hop_evidence.get("open") or [])[:5]
    if not open_rows:
        return "(none)"
    lines: list[str] = []
    for row in open_rows:
        if not isinstance(row, dict):
            continue
        crit = row.get("criterion") or ""
        reason = row.get("reason") or ""
        lines.append(f"- {crit} [{reason}]")
    return "\n".join(lines) if lines else "(none)"


def _build_prompt(
    query: str,
    hop: int,
    hop_evidence: dict | None,
    success_criteria: list[str] | None,
) -> str:
    criteria_text = (
        "\n".join(f"- {c}" for c in (success_criteria or [])[:5])
        or "(none)"
    )
    evidence_summary = _format_evidence_summary(hop_evidence)
    open_criteria = _format_open_criteria(hop_evidence)
    return (
        "You are deciding whether a deep-research agent should run "
        "ANOTHER web search hop.\n"
        f"The agent has already completed {hop} hop(s). It has gathered "
        "the following grounded evidence:\n"
        f"{evidence_summary}\n\n"
        "The user's original question:\n"
        f"{query}\n\n"
        "The planner's success criteria (what we need to answer):\n"
        f"{criteria_text}\n\n"
        "Outstanding criteria still unmet:\n"
        f"{open_criteria}\n\n"
        "Decide: should we run another hop?\n"
        "Respond with JSON only:\n"
        '{"another_hop_useful": true|false, "confidence": <0.0-1.0>, '
        '"reason": "<one short sentence>"}'
    )


def _parse_decision(raw: str) -> StopDecision | None:
    """Parse the LLM JSON response. Returns None on any parse failure."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    # Strip code fences if the model wrapped its output.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # Best-effort: pull the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    if "another_hop_useful" not in obj or "confidence" not in obj:
        return None
    useful = obj.get("another_hop_useful")
    if not isinstance(useful, bool):
        return None
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return None
    conf = max(0.0, min(1.0, conf))
    reason = obj.get("reason") or ""
    if not isinstance(reason, str):
        reason = str(reason)
    return StopDecision(
        another_hop_useful=useful,
        confidence=conf,
        reason=reason[:200],
        degraded=False,
    )


def _degraded_continue(reason: str) -> StopDecision:
    """Fallback decision — keep the loop running so we don't lose info."""
    return StopDecision(
        another_hop_useful=True,
        confidence=1.0,
        reason=reason,
        degraded=True,
    )


async def decide_continue(
    query: str,
    hop: int,
    hop_evidence: dict | None,
    success_criteria: list[str] | None,
) -> StopDecision:
    """Ask a fast Groq call whether another retrieval hop is worthwhile.

    Never raises. On timeout / parse failure / provider error, returns
    ``StopDecision(another_hop_useful=True, confidence=1.0, degraded=True)``
    so the orchestrator's existing terminator logic still gets a chance
    to fire (degrade-to-safer).
    """
    prompt = _build_prompt(query, hop, hop_evidence, success_criteria)
    try:
        from utils.provider_router import call_groq

        raw = await asyncio.wait_for(
            call_groq(prompt, max_tokens=_MAX_TOKENS),
            timeout=_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("stop_rag: groq call timed out after %.1fs", _TIMEOUT_S)
        return _degraded_continue("degraded_timeout")
    except asyncio.CancelledError:
        # Cancellation is a real signal — re-raise so the run can unwind.
        raise
    except Exception as e:  # noqa: BLE001 — degrade-to-safer
        logger.warning("stop_rag: groq call failed: %s", e)
        return _degraded_continue("degraded_provider_error")

    decision = _parse_decision(raw)
    if decision is None:
        logger.warning(
            "stop_rag: unparseable LLM response (len=%d)", len(raw or "")
        )
        return _degraded_continue("degraded_parse_failure")
    return decision
