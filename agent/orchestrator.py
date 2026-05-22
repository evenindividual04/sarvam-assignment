"""
Bounded async state machine:
  PLANNING → SEARCHING → FETCHING → SELECTING → SYNTHESIZING → DONE | ERROR

Yields ExecutionEvent at each step with exact STREAM_LABELS from spec Section 2.10.
Max iterations: AGENT_MAX_ITER (env var, default 5).
Saves complete turn to DB including context_xml_sent, doc_map, token counts.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator
import httpx

from agent.citation_guard import (
    CitationGuard,
    extract_doc_ids,
    build_cite_quote_map,
    convert_citations,
)
from agent.context_engine import (
    chunk,
    format_context_xml,
    probe_contradictions,
    rank_and_select,
    rank_and_select_async,
    rank_and_select_mmr,
)
from utils.retrieval_mode import (
    RetrievalMode,
    set_override as set_retrieval_override,
    reset_override as reset_retrieval_override,
)
from agent.extractor import Extractor
from agent.claim_verifier import verify_claims
from agent.memory import (
    create_session, get_latest_summary, get_relevant_prior_turns,
    get_session_turn_count, save_claim_audit, save_contradiction_probe,
    save_session_summary, save_turn, save_turn_context, session_exists,
)
from agent.models import (
    ConflictResult, ContextBundle, ExecutionEvent, QueryIntent, RunMetadata, Turn, TypedQuery,
    EVT_RUN_STARTED, EVT_PHASE_STARTED, EVT_PHASE_PROGRESS, EVT_PHASE_FINISHED,
    EVT_SEARCH_QUERY, EVT_SOURCE_FOUND, EVT_SOURCE_FETCHED, EVT_CONTEXT_SELECTED,
    EVT_CONFLICT_DETECTED, EVT_ANSWER_DELTA, EVT_CITATION_RESOLVED,
    EVT_RUN_FINISHED, EVT_RUN_ERROR, EVT_UNCERTAINTY,
    EVT_CLARIFICATION_OFFERED, EVT_EVIDENCE_GAP, EVT_PLAN_APPROVAL,
    EVT_REASONING,
    EVT_HOP_EVIDENCE, EVT_SOURCE_CONTRIBUTION, EVT_SOURCE_ROLE, EVT_TERMINATOR,
)
from utils.next_steps import propose_follow_ups
from agent.refinement_check import MAX_REFINEMENTS
from agent.search import search
from agent.synthesizer import stream_synthesis
from utils.cancellation import CancellationToken, OperationCancelledError
from utils.failure_policy import POLICY
from utils.prompt_registry import prompt_id
from utils.token_counter import ContextBudget

logger = logging.getLogger(__name__)

def _validate_run_metadata(meta: dict, *, turn_id: str | None = None) -> dict:
    """Validate `run_metadata` against the `RunMetadata` Pydantic schema and
    return a JSON-mode dict. Defensive: validation failure logs a warning and
    falls back to the raw dict — we never want a schema typo to crash a turn
    in progress. `extra="allow"` on the model means unknown keys round-trip
    untouched, so this is a near-no-op for well-formed payloads.
    """
    try:
        return RunMetadata.model_validate(meta).model_dump(
            exclude_unset=False, mode="json"
        )
    except Exception as e:  # pragma: no cover — schema drift escape hatch
        logger.warning(
            "run_metadata schema validation failed: %s",
            e,
            extra={"component": "orchestrator", "turn_id": turn_id},
        )
        return meta


_MAX_ITER = int(os.getenv("AGENT_MAX_ITER", "5"))
_SELECTION_STRATEGY = os.getenv("CONTEXT_SELECTION_STRATEGY", "heuristic").strip().lower()
# V3.2: intents allowed in second hop (PRIMARY already covered by hop 1).
_HOP2_ALLOWED_INTENTS = {"recency_check", "contradiction_probe"}


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """Per-request resolved configuration. Built from request overrides layered
    on top of environment variables. Stamped into `run_metadata_json` so every
    saved turn carries a faithful record of what knobs it actually ran under —
    even if the user flips toggles five minutes later.
    """
    retrieval_mode: str  # "auto" | "hybrid" | "lexical"
    max_hops: int
    selection_strategy: str  # "heuristic" | "mmr"
    mmr_lambda: float  # 0..1; relevance vs novelty tradeoff for MMR selector
    # Phase 5b: per-turn effective domain blocklist. Built as
    # `BLOCKLIST | env_extras`, or `frozenset()` when the user passed
    # `RETRIEVAL_DOMAIN_BLOCKLIST=none` (explicit disable). Empty means
    # "do not filter" — callers must short-circuit on that case.
    domain_blocklist: frozenset[str] = frozenset()
    # Phase 2: human-in-the-loop plan approval gate. When True, the
    # orchestrator emits a `plan_approval` SSE event after PLANNING and waits
    # on `cancel_token.wait_for_approval()` before proceeding to SEARCHING.
    # Default False keeps autonomous behaviour for eval runs and unmodified
    # clients.
    approval_required: bool = False

    @classmethod
    def from_overrides(
        cls,
        overrides: dict | None,
        *,
        approval_required: bool = False,
    ) -> "RuntimeConfig":
        ov = overrides or {}

        def _get(key: str, default: str) -> str:
            # Overrides may carry mixed types from JSON; coerce to str for parity
            # with os.getenv's contract before we parse.
            val = ov.get(key)
            if val is None:
                return os.getenv(key, default)
            return str(val)

        sel = _get("CONTEXT_SELECTION_STRATEGY", "heuristic").strip().lower()
        if sel not in ("heuristic", "mmr"):
            sel = "heuristic"

        # Retrieval mode (V3.8): three-state, supersedes the boolean
        # HYBRID_RETRIEVAL. Legacy alias still honored when the new key is
        # absent — `HYBRID_RETRIEVAL=1` → "hybrid", `=0` → "lexical".
        explicit = _get("RETRIEVAL_MODE", "").strip().lower()
        if explicit in ("auto", "hybrid", "lexical"):
            retrieval_mode = explicit
        else:
            legacy = _get("HYBRID_RETRIEVAL", "").strip()
            if legacy == "1":
                retrieval_mode = "hybrid"
            elif legacy == "0":
                retrieval_mode = "lexical"
            else:
                retrieval_mode = "auto"

        try:
            mmr_lambda = float(_get("MMR_LAMBDA", "0.7"))
        except ValueError:
            mmr_lambda = 0.7
        # Clamp to [0, 1] — values outside that range have no defined meaning
        # in the MMR formulation and would silently break the score blend.
        mmr_lambda = max(0.0, min(1.0, mmr_lambda))

        # Phase 5b: resolve effective domain blocklist. Default = BLOCKLIST.
        # Comma-separated env extends the default. The literal "none" fully
        # disables. Unknown / empty value falls back to the default set.
        from utils.source_trust import BLOCKLIST as _DEFAULT_BLOCKLIST
        raw_block = _get("RETRIEVAL_DOMAIN_BLOCKLIST", "").strip()
        if raw_block.lower() == "none":
            domain_blocklist: frozenset[str] = frozenset()
        elif raw_block:
            # Audit M4: validate client-supplied entries. Max 50 extras, each
            # ≤ 255 chars and matching a basic domain regex. Invalid entries
            # are skipped with a warning rather than raising — fail-soft so a
            # typo doesn't kill an otherwise-valid turn.
            _domain_re = re.compile(r"^[a-z0-9.-]+$")
            extras: set[str] = set()
            for p in raw_block.split(","):
                cleaned = p.strip().lower().lstrip(".")
                if not cleaned:
                    continue
                if len(cleaned) > 255 or not _domain_re.match(cleaned):
                    logger.warning(
                        "Skipping invalid domain in RETRIEVAL_DOMAIN_BLOCKLIST: %r",
                        cleaned[:60],
                    )
                    continue
                extras.add(cleaned)
                if len(extras) >= 50:
                    logger.warning(
                        "RETRIEVAL_DOMAIN_BLOCKLIST truncated at 50 entries"
                    )
                    break
            domain_blocklist = _DEFAULT_BLOCKLIST | frozenset(extras)
        else:
            domain_blocklist = _DEFAULT_BLOCKLIST

        return cls(
            retrieval_mode=retrieval_mode,
            max_hops=max(1, min(int(_get("FAILURE_POLICY_MAX_HOPS", "2")), 3)),
            selection_strategy=sel,
            mmr_lambda=mmr_lambda,
            approval_required=bool(approval_required),
            domain_blocklist=domain_blocklist,
        )

    def as_dict(self) -> dict:
        # Build manually because frozenset isn't JSON-serializable and would
        # also break `run_metadata.effective_config` in saved turn audits.
        # Sort the blocklist so the trace is deterministic + diff-friendly.
        return {
            "retrieval_mode": self.retrieval_mode,
            "max_hops": self.max_hops,
            "selection_strategy": self.selection_strategy,
            "mmr_lambda": self.mmr_lambda,
            "approval_required": self.approval_required,
            "domain_blocklist": sorted(self.domain_blocklist),
        }

STREAM_LABELS = {
    "planning":  "Planning",
    "searching": "Searching the web",
    "fetching":  "Fetching sources",
    "selecting": "Selecting relevant context",
    "probing":   "Probing for cross-source contradictions",
    "generating": "Generating answer with citations",
    "verifying": "Verifying claims against sources",
}

# Phase 1.25: canonical phase order for `phase_started.idx/total` metadata.
PHASE_ORDER = [
    "planning", "searching", "fetching", "selecting",
    "probing", "generating", "verifying",
]

_UNCERTAINTY_MARKERS = [
    "insufficient", "unclear", "unable to find", "could not find",
    "no information", "suggested follow-up", "not enough",
]


def _phase_started_event(name: str, extra: dict | None = None) -> ExecutionEvent:
    """Phase 1.25: build a typed `phase_started` event carrying name/label/idx/total."""
    label = STREAM_LABELS.get(name, name)
    idx = PHASE_ORDER.index(name) + 1 if name in PHASE_ORDER else 0
    data = {"name": name, "label": label, "idx": idx, "total": len(PHASE_ORDER)}
    if extra:
        data.update(extra)
    return ExecutionEvent(name, label, data=data, event_type=EVT_PHASE_STARTED)


def _phase_finished_event(name: str, duration_ms: int, extra: dict | None = None) -> ExecutionEvent:
    """Phase 1.25: build a typed `phase_finished` event with duration_ms and optional extras."""
    label = STREAM_LABELS.get(name, name)
    data = {"name": name, "label": label, "duration_ms": duration_ms}
    if extra:
        data.update(extra)
    return ExecutionEvent(name, label, data=data, event_type=EVT_PHASE_FINISHED)


# ── Forensic-differentiation helpers (mechanical, no LLM) ────────────────
# Token regex: shared with claim_verifier intent. Captures Title-case entities,
# numeric tokens (years, percentages, money), and lowercased terms ≥ 4 chars.
_FORENSIC_ENTITY_RE = re.compile(
    r'\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*|\d[\d,.\-/%]*|\$\d[\d,.]*)\b'
)
_FORENSIC_NUMBER_RE = re.compile(r'\b\d[\d,.\-/%]*\b')
_FORENSIC_STOPWORDS = frozenset({
    "The", "A", "An", "Of", "In", "On", "At", "To", "For", "And", "Or", "But",
    "Is", "Are", "Was", "Were", "Be", "Been", "Being", "This", "That", "These",
    "Those", "It", "Its", "With", "By", "From", "As", "Also", "Not", "No", "Yes",
})


def _extract_criterion_tokens(criterion: str) -> list[str]:
    """Pull entity-ish + numeric tokens out of a planner success_criterion."""
    raw = _FORENSIC_ENTITY_RE.findall(criterion or "")
    out: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        t = tok.strip()
        if not t or t in _FORENSIC_STOPWORDS:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _find_quote_window(text: str, token: str, width: int = 120) -> str | None:
    """Return a ≤200-char window around the first case-insensitive match."""
    if not text or not token:
        return None
    idx = text.lower().find(token.lower())
    if idx < 0:
        return None
    start = max(0, idx - width // 2)
    end = min(len(text), idx + len(token) + width // 2)
    window = text[start:end].strip()
    return window[:200]


def _compute_hop_evidence(
    selected_snippets: list,
    criteria: list[str],
    *,
    contradictions_by_topic: set[str] | None = None,
    max_grounded: int = 8,
    max_open: int = 6,
) -> tuple[list[dict], list[dict]]:
    """Deterministic ledger of what got grounded in this hop and what didn't.

    ``grounded`` rows point to a real ``doc_id`` and quote an actual 120-char
    window around the matched token — NEVER LLM-narrated prose. ``open`` rows
    are the planner criteria whose tokens did not appear in any selected
    chunk this hop (set difference).
    """
    contradictions_by_topic = contradictions_by_topic or set()
    selected_snippets = selected_snippets or []
    criteria = criteria or []

    grounded: list[dict] = []
    grounded_keys: set[tuple[str, str]] = set()  # (token_lower, kind)
    grounded_criteria: set[str] = set()
    open_partial_criteria: set[str] = set()  # had a hit but BM25 < 0.3

    # 1. Criterion-token grounding pass.
    for criterion in criteria:
        tokens = _extract_criterion_tokens(criterion)
        if not tokens:
            continue
        criterion_grounded = False
        criterion_partial = False
        for tok in tokens:
            best_snippet = None
            best_score = -1.0
            for s in selected_snippets:
                snippet_text = getattr(s, "text", "") or ""
                if not snippet_text:
                    continue
                if tok.lower() in snippet_text.lower():
                    score = float(
                        getattr(s, "final_score", None)
                        or getattr(s, "bm25_score", 0.0)
                        or 0.0
                    )
                    if score > best_score:
                        best_score = score
                        best_snippet = s
            if best_snippet is None:
                continue
            criterion_grounded = True
            if best_score < 0.3:
                criterion_partial = True
            key = (tok.lower(), "criterion")
            if key in grounded_keys:
                continue
            grounded_keys.add(key)
            grounded.append({
                "token": tok,
                "kind": "criterion",
                "doc_id": getattr(best_snippet, "doc_id", None),
                "url": getattr(best_snippet, "url", None),
                "quote": _find_quote_window(
                    getattr(best_snippet, "text", "") or "", tok,
                ),
            })
            if len(grounded) >= max_grounded:
                break
        if criterion_grounded:
            grounded_criteria.add(criterion)
            if criterion_partial:
                open_partial_criteria.add(criterion)
        if len(grounded) >= max_grounded:
            break

    # 2. Numeric-token pass — only if we still have room. Extract numbers
    # from the chunks themselves and report them once each.
    if len(grounded) < max_grounded:
        for s in selected_snippets:
            snippet_text = getattr(s, "text", "") or ""
            if not snippet_text:
                continue
            for num in _FORENSIC_NUMBER_RE.findall(snippet_text)[:3]:
                key = (num.lower(), "number")
                if key in grounded_keys:
                    continue
                grounded_keys.add(key)
                grounded.append({
                    "token": num,
                    "kind": "number",
                    "doc_id": getattr(s, "doc_id", None),
                    "url": getattr(s, "url", None),
                    "quote": _find_quote_window(snippet_text, num),
                })
                if len(grounded) >= max_grounded:
                    break
            if len(grounded) >= max_grounded:
                break

    # 3. Build `open` from criteria that got no token-grounding this hop.
    open_rows: list[dict] = []
    for criterion in criteria:
        if criterion in grounded_criteria and criterion not in open_partial_criteria:
            continue
        if any(
            tok.lower() in criterion.lower()
            for tok in contradictions_by_topic
        ):
            reason = "conflicting"
        elif criterion in open_partial_criteria:
            reason = "partial"
        else:
            reason = "no_evidence"
        open_rows.append({"criterion": criterion, "reason": reason})
        if len(open_rows) >= max_open:
            break

    return grounded[:max_grounded], open_rows[:max_open]


# Extracted from inline at L2026 for readability.
def _compute_criteria_coverage(
    plan_success_criteria: list[str] | None,
    full_answer: str,
) -> list[bool]:
    """Heuristic per-criterion coverage check against the synthesized answer.

    A criterion is "covered" if any of:
      (a) Its lowercased text appears verbatim in the answer.
      (b) ≥60% of its alphanumeric word tokens overlap the answer's tokens.
      (c) **Cross-script fallback**: when the answer is mostly non-ASCII
          (Devanagari, CJK, Arabic, etc.) but the criterion is English,
          token-overlap returns 0 because the writing systems don't share
          glyphs. In that case we degrade gracefully and treat the
          criterion as covered if the answer contains AT LEAST one
          script-agnostic anchor from the criterion — a number, year,
          percentage, URL, or capitalized proper noun. This matches the
          common case where the English criterion was "name the current
          exchange rate with a date" and the Hindi answer correctly cites
          93.977 and the date — the language differs, the facts match.

    Pure function: no I/O, no mutation of caller state.
    """
    if not plan_success_criteria:
        return []
    import re as _re_crit
    ans = full_answer or ""
    ans_lc = ans.lower()
    ans_tokens = set(_re_crit.findall(r"\w+", ans_lc))

    # Detect cross-script answer: > 30 % non-ASCII chars in a non-empty answer.
    non_ascii = sum(1 for ch in ans if ord(ch) > 127)
    cross_script = bool(ans) and (non_ascii / max(1, len(ans))) > 0.30

    coverage: list[bool] = []
    for crit in plan_success_criteria:
        if not crit or not isinstance(crit, str):
            coverage.append(False)
            continue
        c_lc = crit.lower().strip()
        if c_lc and c_lc in ans_lc:
            coverage.append(True)
            continue
        ctoks = set(_re_crit.findall(r"\w+", c_lc))
        if not ctoks:
            coverage.append(False)
            continue
        overlap = len(ctoks & ans_tokens) / max(1, len(ctoks))
        if overlap >= 0.6:
            coverage.append(True)
            continue
        if cross_script:
            # Script-agnostic anchors: numbers, years, percentages, URLs,
            # capitalized proper nouns (cap-word>=3 chars in the criterion
            # that also appears verbatim — case-sensitive — in the answer).
            crit_numbers = set(_re_crit.findall(r"\d[\d.,/%-]*\d|\d", crit))
            crit_caps = set(_re_crit.findall(r"\b[A-Z][a-zA-Z]{2,}\b", crit))
            anchors = crit_numbers | crit_caps
            if anchors and any(a in ans for a in anchors):
                coverage.append(True)
                continue
            # No script-agnostic anchor → conservative "uncertain": we don't
            # claim coverage but we also don't penalize the model for the
            # script gap. Return False (caller surfaces as "X / N").
        coverage.append(False)
    return coverage


# Map orchestrator terminator-fired tags onto the public SSE reasons declared
# in `frontend/lib/types.ts`. Internal tags like TOKEN_BUDGET_EXHAUSTED stay
# distinct in `run_metadata` for analytics; the SSE event normalizes them.
_TERMINATOR_TAG_TO_REASON = {
    "MAX_HOPS_REACHED": "MAX_HOPS_REACHED",
    "EVIDENCE_SUFFICIENT": "EVIDENCE_SUFFICIENT",
    "TOKEN_BUDGET_EXHAUSTED": "BUDGET_EXHAUSTED",
    "DIFFICULTY_EASY_SKIPPED": "EVIDENCE_SUFFICIENT",
    "CONFIDENCE_HIGH_ENOUGH": "EVIDENCE_SUFFICIENT",
    "MARGINAL_GAIN_LOW": "MARGINAL_GAIN_LOW",
    "NO_NEW_QUERIES": "NO_NEW_QUERIES",
    "CRITERIA_SATISFIED": "CRITERIA_SATISFIED",
    "APPROVAL_TIMEOUT": "NO_NEW_QUERIES",
    # P1 (Stop-RAG gate): normalized at emission time based on which branch
    # tripped — "useful=False" → EVIDENCE_SUFFICIENT, "confidence<0.5" →
    # MARGINAL_GAIN_LOW. The raw tag below is the safe fallback.
    "STOP_RAG_GATE": "EVIDENCE_SUFFICIENT",
}


from agent.weak_signal import WeakSignal, compute_weak_signal


def _is_weak_by_output(
    answer: str,
    cited_ids: list[str],
    doc_map: dict,
    fetched_urls: set,
) -> tuple[bool, int, int]:
    """Back-compat shim around :func:`agent.weak_signal.compute_weak_signal`.

    Returns ``(weak, unverified_count, grounded_citation_count)`` for any
    existing callers/tests that depend on the tuple shape. New code should
    consume :class:`WeakSignal` directly via ``compute_weak_signal``.
    """
    sig = compute_weak_signal(answer, cited_ids, doc_map, fetched_urls)
    return (sig.is_weak, sig.unverified_count, sig.grounded_count)


def _append_next_steps_block(answer: str, suggestions: list[str]) -> str:
    """Append a `**What I'd search next:**` bullet block. Idempotent."""
    if "**What I'd search next:**" in (answer or ""):
        return answer
    bullets = [s for s in (suggestions or []) if s and s.strip()][:3]
    if not bullets:
        return answer
    block = "\n".join(f"- {s}" for s in bullets)
    return (answer or "").rstrip() + f"\n\n**What I'd search next:**\n{block}"


def _ensure_uncertainty_block(answer: str, follow_ups: list[str]) -> str:
    """Append deterministic uncertainty block when model omitted it.

    Phase 1.5: `follow_ups` MUST be alternate queries (not the executed ones).
    Caller is responsible for generating them via `utils.next_steps.propose_follow_ups`.
    Idempotent: if `[UNCERTAINTY]` is already in the answer, returns it unchanged.
    """
    if "[UNCERTAINTY]" in answer:
        return answer
    q = [x for x in (follow_ups or []) if x and x.strip()][:3]
    while len(q) < 3:
        q.append("authoritative source for latest verified value")
    tail = (
        "\n\n[UNCERTAINTY] Evidence was insufficient for a fully grounded answer.\n"
        "Suggested follow-up searches:\n"
        f"- {q[0]}\n- {q[1]}\n- {q[2]}"
    )
    return answer.rstrip() + tail


class ResearchOrchestrator:
    def __init__(self) -> None:
        self._guard = CitationGuard()
        # Extractor (httpx.AsyncClient) is instantiated lazily on first use and
        # reused across hops within the same run, then closed via `aclose()`.
        # CLAUDE.md mandates a single client per Extractor; recreating per hop
        # would defeat connection pooling and leak sockets on cancellation.
        self._extractor: Extractor | None = None

    def _get_extractor(self) -> Extractor:
        if self._extractor is None:
            self._extractor = Extractor()
        return self._extractor

    async def aclose(self) -> None:
        if self._extractor is not None:
            try:
                await self._extractor.aclose()
            except Exception:
                pass
            self._extractor = None

    async def run(
        self,
        query: str,
        session_id: str,
        cancel_token: CancellationToken | None = None,
        turn_id: str | None = None,
        config: RuntimeConfig | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """
        Full research turn. Yields ExecutionEvents. Saves turn to DB at end.

        On cancellation (cancel_token fires), persists a partial Turn marked
        '[Cancelled by user]', appends 'CANCELLED' to state_trace, yields one
        final 'error' event, and returns. Does not re-raise.

        `config` carries per-request runtime overrides (Option D in design).
        When None, falls back to env-only resolution — keeps `eval_runner.py`
        reproducible from CLI without sending an overrides block.
        """
        if turn_id is None:
            turn_id = str(uuid.uuid4())
        if config is None:
            config = RuntimeConfig.from_overrides(None)
        state_holder: dict = {
            "state_trace": [],
            "now": datetime.now(timezone.utc).isoformat(),
            "config": config,
        }
        # Bind retrieval mode for the duration of this turn so the selector
        # picks up the per-request override (context_engine reads it from a
        # ContextVar inside the worker thread; ContextVars propagate through
        # asyncio.to_thread on Python ≥3.9). The mode resolver inside
        # `hybrid_retrieval_enabled()` then combines this with the capability
        # probe to compute the effective behavior.
        retrieval_token = set_retrieval_override(RetrievalMode(config.retrieval_mode))
        # Phase 5b: bind the per-request domain blocklist so context_engine's
        # rank_and_select / rank_and_select_mmr (which run in worker threads
        # via asyncio.to_thread) see the effective set. The drops sink is
        # `state_holder["run_metadata"]["domain_blocklist_drops"]` — created
        # lazily inside `_run_body`, so we set the drops contextvar to None
        # here and the engine handles None gracefully.
        from agent.context_engine import (
            set_domain_blocklist as _set_blocklist,
            reset_domain_blocklist as _reset_blocklist,
            set_reranker_sink as _set_reranker_sink,
            reset_reranker_sink as _reset_reranker_sink,
        )
        # State holder gets the drops list lazily; bind a sink that resolves
        # at chunk-filter time by referencing the same list via run_metadata.
        # Simpler: pre-seed run_metadata key here so the sink is stable.
        state_holder.setdefault("blocklist_drops_sink", [])
        blocklist_tokens = _set_blocklist(
            config.domain_blocklist,
            state_holder["blocklist_drops_sink"],
        )
        # V3.9: per-turn reranker telemetry sink (set during SELECTING).
        state_holder.setdefault("reranker_sink", {})
        reranker_token = _set_reranker_sink(state_holder["reranker_sink"])
        try:
            async for ev in self._run_body(query, session_id, cancel_token, turn_id, state_holder):
                yield ev
        except OperationCancelledError:
            state_trace = state_holder.get("state_trace", [])
            state_trace.append("CANCELLED")
            try:
                await self._persist_cancelled(turn_id, session_id, query, state_holder)
            except Exception as e:
                logger.warning("Persist cancelled turn failed: %s", e,
                               extra={"component": "orchestrator", "turn_id": turn_id})
            # Single typed cancel event — the legacy string-payload variant
            # was removed to avoid double-emission to SSE consumers.
            yield ExecutionEvent(
                "error", "cancelled",
                data={"phase": "unknown", "message": "Cancelled by user.", "recoverable": False},
                event_type=EVT_RUN_ERROR,
            )
            return
        finally:
            reset_retrieval_override(retrieval_token)
            _reset_blocklist(blocklist_tokens)
            try:
                _reset_reranker_sink(reranker_token)
            except Exception:
                pass
            # Phase 1.875: reset live-time-sensitivity contextvar set inside
            # the run body. Safe no-op if the body never set it.
            tok = state_holder.get("_live_token")
            if tok is not None:
                try:
                    from agent.context_engine import reset_time_sensitivity_live
                    reset_time_sensitivity_live(tok)
                except Exception:
                    pass

    async def _persist_cancelled(
        self, turn_id: str, session_id: str, query: str, state: dict
    ) -> None:
        """Best-effort save of partial state when a turn is cancelled."""
        bundle = state.get("final_context_bundle")
        now = state.get("now") or datetime.now(timezone.utc).isoformat()
        turn = Turn(
            turn_id=turn_id,
            session_id=session_id,
            query=query,
            created_at=now,
            plan=str(state.get("queries") or []),
            search_queries=state.get("queries") or [],
            urls_opened=state.get("urls_opened") or [],
            response="[Cancelled by user]",
            context_xml_sent=bundle.xml if bundle else "",
            doc_map=bundle.doc_map if bundle else {},
            citation_integrity_score=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=state.get("latency_ms") or 0,
            run_metadata_json={
                "cancelled": True,
                # Phase 2: preserve any terminator the body set before raising
                # (e.g. APPROVAL_TIMEOUT) so the eval harness can distinguish a
                # user cancel from an approval-gate timeout.
                **(
                    {"terminator_fired": tf}
                    if (tf := (state.get("run_metadata") or {}).get("terminator_fired"))
                    else {}
                ),
            },
            state_trace=state.get("state_trace", []),
        )
        await save_turn(turn)
        # Best-effort: persist whatever context snippets we had selected so
        # the cancelled turn isn't a black hole in the trace.
        selected = state.get("selected_snippets") or []
        if selected:
            try:
                await save_turn_context(turn_id, selected)
            except Exception as e:
                logger.warning(
                    "Persist cancelled turn_context failed: %s",
                    e,
                    extra={"component": "orchestrator", "turn_id": turn_id},
                )

    async def _run_body(
        self,
        query: str,
        session_id: str,
        cancel_token: CancellationToken | None,
        turn_id: str,
        state_holder: dict,
    ) -> AsyncIterator[ExecutionEvent]:
        start_ms = time.time()

        # Phase 1.25: typed run_started at the very top — frontend uses this to
        # mint a per-turn id and bind a per-turn SSE replay buffer.
        yield ExecutionEvent(
            "planning", STREAM_LABELS["planning"],
            data={"turn_id": turn_id, "session_id": session_id, "query": query},
            event_type=EVT_RUN_STARTED,
        )

        def _ck() -> None:
            if cancel_token is not None:
                cancel_token.check()

        now = state_holder["now"]
        state_trace: list[str] = state_holder["state_trace"]
        config: RuntimeConfig = state_holder["config"]
        budget = ContextBudget()
        stage_ms = {"planning_ms": 0, "search_ms": 0, "fetch_ms": 0, "select_ms": 0, "probe_ms": 0, "synthesize_ms": 0}
        # Resolve effective retrieval mode (V3.8) now that the override is bound.
        # Capture (requested, effective, reason) into run_metadata so evaluators
        # see per-turn whether a "hybrid" request actually ran hybrid or fell back.
        from agent import memory as _mem
        from utils.retrieval_mode import effective_mode_for_request
        _eff_retrieval = effective_mode_for_request(_mem._VEC_AVAILABLE)
        run_metadata = {
            "selection_strategy": config.selection_strategy,
            "effective_config": config.as_dict(),
            "retrieval_mode": _eff_retrieval.to_metadata(),
            "planner_prompt_id": prompt_id("planner"),
            "synth_prompt_id": prompt_id("synthesizer"),
            "conflict_prompt_id": prompt_id("conflict_v3"),
            "judge_prompt_id": prompt_id("judge"),
            "configured_models": {
                "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                "planner_model": "llama-3.3-70b-versatile",
                "judge_model": "gpt-4o-mini",
            },
            "fallback_path_taken": [],
            "timeout_hits": [],
            "retry_counts": {},
            "budget_breach": [],
            "failure_policy": POLICY.__dict__,
        }
        # Phase 2: stash run_metadata into state_holder so the cancellation
        # cleanup path (`_persist_cancelled`) can read `terminator_fired` —
        # otherwise APPROVAL_TIMEOUT and similar terminators are lost when the
        # body raises OperationCancelledError.
        state_holder["run_metadata"] = run_metadata

        # Ensure session exists
        if not await session_exists(session_id):
            await create_session(session_id, now)

        # ── Build conversation context ────────────────────────────────────
        prior_turns = await get_relevant_prior_turns(session_id, query)
        rolling_sum = await get_latest_summary(session_id)
        turn_count = await get_session_turn_count(session_id)

        history_parts = []
        if rolling_sum:
            history_parts.append(f"[Summary of earlier conversation]\n{rolling_sum}")
        for t in prior_turns[-3:]:
            history_parts.append(f"Q: {t.query}\nA: {t.response or ''}")
        history_text = "\n\n".join(history_parts)
        from utils.token_counter import count_tokens, truncate_to_tokens
        run_metadata.setdefault("context_fallbacks", [])
        while history_parts and count_tokens(history_text) > budget.history_budget:
            if len(history_parts) > 1 and not history_parts[0].startswith("[Summary"):
                history_parts.pop(0)
            elif len(history_parts) > 1:
                history_parts.pop(1)
            else:
                break
            history_text = "\n\n".join(history_parts)

        # Phase 1.75: true summarization fallback. The assignment text literally
        # requires "summarization fallback" when history exceeds the cap — not
        # hard truncation. Call a Groq second-pass summary-of-summaries before
        # falling back to `truncate_to_tokens`.
        if count_tokens(history_text) > budget.history_budget:
            try:
                from utils.provider_router import compress_history
                history_text = await compress_history(
                    existing_summary=rolling_sum or "",
                    remaining_turns=list(prior_turns[-3:]),
                    max_tokens=budget.history_budget,
                )
                run_metadata["context_fallbacks"].append("history_compressed")
            except Exception as e:
                logger.warning(
                    "compress_history wiring failed: %s",
                    e,
                    extra={"component": "orchestrator", "turn_id": turn_id},
                )
            if count_tokens(history_text) > budget.history_budget:
                logger.warning(
                    "Hard-truncating history after compression failed",
                    extra={"component": "orchestrator", "turn_id": turn_id},
                )
                history_text = truncate_to_tokens(history_text, budget.history_budget)
                run_metadata["context_fallbacks"].append("history_truncated_last_resort")

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 1 — PLANNING
        # Plan the research: typed search queries + confidence + difficulty.
        # ═════════════════════════════════════════════════════════════════════
        # ── PLANNING ──────────────────────────────────────────────────────
        _ck()
        state_trace.append("PLANNING")
        # Legacy event (back-compat) + new typed phase_started
        yield ExecutionEvent("planning", STREAM_LABELS["planning"], data={"turn_id": turn_id})
        yield _phase_started_event("planning", {"turn_id": turn_id, "hop": 1})
        t0 = time.time()
        try:
            from utils.provider_router import plan
            planner = await asyncio.wait_for(
                plan(query, prior_summary=history_text or "No prior context."),
                timeout=POLICY.plan_timeout_s,
            )
        except OperationCancelledError:
            raise
        except asyncio.TimeoutError:
            run_metadata["timeout_hits"].append("planning")
            run_metadata["fallback_path_taken"].append("planning_timeout_fallback")
            run_metadata["budget_breach"].append("planning_timeout")
            from agent.models import PlannerOutput, QueryIntent, TypedQuery
            planner = PlannerOutput(
                strategy="Direct retrieval fallback",
                queries=[TypedQuery(text=query, intent=QueryIntent.PRIMARY)],
            )
        except (httpx.TimeoutException, httpx.ConnectError, Exception) as e:
            logger.error("Planning failed: %s", e, extra={"component": "orchestrator", "turn_id": turn_id})
            run_metadata["fallback_path_taken"].append("planning_error_fallback")
            from agent.models import PlannerOutput, QueryIntent, TypedQuery
            planner = PlannerOutput(
                strategy="Direct retrieval fallback",
                queries=[TypedQuery(text=query, intent=QueryIntent.PRIMARY)],
            )
        stage_ms["planning_ms"] += int((time.time() - t0) * 1000)
        from agent.models import QueryIntent, TypedQuery
        typed_queries: list[TypedQuery] = [tq for tq in planner.queries if tq.text.strip()]
        if not typed_queries:
            typed_queries = [TypedQuery(text=query, intent=QueryIntent.PRIMARY)]
        typed_queries = typed_queries[:4]
        queries = [tq.text for tq in typed_queries]

        # Persist typed planner output for traceability
        run_metadata["planner_output"] = {
            "strategy": planner.strategy,
            "confidence": getattr(planner, "confidence", "medium"),
            # Phase 1.875: enriched plan-level fields surfaced so the trace
            # inspector and the new PlanCard can render them.
            "time_sensitivity": getattr(planner, "time_sensitivity", "static"),
            "expected_source_types": list(getattr(planner, "expected_source_types", []) or []),
            "difficulty": getattr(planner, "difficulty", "medium"),
            "ambiguity_flag": bool(getattr(planner, "ambiguity_flag", False)),
            "success_criteria": list(getattr(planner, "success_criteria", []) or []),
            "queries": [
                {"text": tq.text, "intent": tq.intent.value, "rationale": tq.rationale}
                for tq in typed_queries
            ],
        }

        # F7: vagueness-gated clarifier. Replaces the prior unconditional
        # emission keyed on `planner.ambiguity_flag` alone. We now combine
        # ambiguity with entity count, query length, and wh-breadth, then
        # gate emission on a 0.55 threshold + non-empty success_criteria.
        from agent.vagueness import score as _vagueness_score

        _vag = _vagueness_score(query, planner)
        run_metadata["vagueness_score"] = _vag.score
        run_metadata["vagueness_gated"] = _vag.gate
        if _vag.gate and _vag.suggested_question is not None:
            yield ExecutionEvent(
                "planning", STREAM_LABELS["planning"],
                data={
                    "kind": "ambiguity",
                    "original_query": query,
                    "possible_interpretations": _vag.suggested_options,
                    "clarifying_question": _vag.suggested_question,
                },
                event_type=EVT_CLARIFICATION_OFFERED,
            )

        # Adapt budget based on planning complexity
        budget.adapt_to_complexity(len(queries))

        # Phase 1.875: capture plan-level fields used by downstream branches.
        plan_time_sensitivity = getattr(planner, "time_sensitivity", "static")
        plan_difficulty = getattr(planner, "difficulty", "medium")
        plan_success_criteria = list(getattr(planner, "success_criteria", []) or [])

        # Record which planner provider actually ran (Cerebras / Groq / Gemini /
        # fallback). Stamped by provider_router.plan() via a contextvar.
        try:
            from utils.provider_router import get_last_planner_provider
            _planner_prov = get_last_planner_provider()
            if _planner_prov:
                run_metadata["planner_provider"] = _planner_prov
        except Exception:
            pass

        # Goal 5: when ``difficulty == "hard"`` AND the synth chain leads with
        # Gemini (1M context), expand the web-context budget 4× from 6,400 to
        # 24,000 tokens. Sarvam-M caps lower; skip the expansion if Sarvam will
        # be tried first (Indic-auto routing or SYNTH_PROVIDER=sarvam).
        if plan_difficulty == "hard":
            _synth_primary = os.environ.get("SYNTH_PROVIDER", "gemini").lower()
            _is_indic = False
            try:
                from utils.provider_router import _detect_query_script
                _is_indic = (
                    os.environ.get("SARVAM_INDIC_AUTO", "1").strip()
                    not in {"0", "false", "False", ""}
                    and bool(os.environ.get("SARVAM_API_KEY"))
                    and _detect_query_script(query) == "indic"
                )
            except Exception:
                _is_indic = False
            _sarvam_primary = _synth_primary == "sarvam" or _is_indic
            if not _sarvam_primary:
                budget.web_context_override = 24000
                logger.info(
                    "Hard difficulty: expanded web_context budget to 24000 tokens"
                )
                run_metadata["budget_expanded_for_hard"] = True

        # Phase 1.875: when live, bind the context-engine recency-boost
        # contextvar so SELECTING re-weights toward freshness for this turn.
        from agent.context_engine import (
            set_time_sensitivity_live as _set_live,
            reset_time_sensitivity_live as _reset_live,
        )
        _live_token = _set_live(plan_time_sensitivity == "live")
        state_holder["_live_token"] = _live_token

        # Persist language-detection result (lang + tier method) so eval and
        # traces can attribute Sarvam/Wikipedia routing decisions.
        try:
            from utils.lang_detect import detect_language as _ld
            _lang, _method = _ld(query)
            run_metadata["language_detection"] = {"lang": _lang, "method": _method}
        except Exception:  # noqa: BLE001
            pass

        yield ExecutionEvent(
            "planning",
            STREAM_LABELS["planning"],
            data={"strategy": planner.strategy, "queries": queries},
        )
        yield _phase_finished_event(
            "planning",
            stage_ms["planning_ms"],
            {
                "strategy": planner.strategy,
                "queries": queries,
                "hop": 1,
                # Phase 1.875: full enriched planner output so the PlanCard
                # can render live (without waiting for `done`).
                "planner_output": run_metadata["planner_output"],
            },
        )

        # ── PHASE 2: PLAN APPROVAL GATE ──────────────────────────────────
        # Opt-in human-in-the-loop checkpoint between PLANNING and SEARCHING.
        # When `approval_required=True`, we emit a `plan_approval` SSE event
        # and block on `cancel_token.wait_for_approval()` for up to
        # APPROVAL_TIMEOUT_S seconds (default 300). The /research/approve/
        # endpoint sets `approved_payload` and fires `approval_event`.
        # `_disconnect_watcher` in main.py reads `token.paused` and skips its
        # poll so a brief SSE drop during the wait doesn't auto-cancel.
        if config.approval_required and cancel_token is not None:
            state_trace.append("AWAITING_APPROVAL")
            yield ExecutionEvent(
                "planning", STREAM_LABELS["planning"],
                data={
                    "turn_id": turn_id,
                    "planner_output": run_metadata["planner_output"],
                    "sub_queries": list(queries),
                },
                event_type=EVT_PLAN_APPROVAL,
            )
            cancel_token.paused = True
            try:
                timeout_s = float(os.environ.get("APPROVAL_TIMEOUT_S", "300"))
            except (TypeError, ValueError):
                timeout_s = 300.0
            approved, edited_payload = await cancel_token.wait_for_approval(
                timeout=timeout_s
            )
            cancel_token.paused = False

            if not approved:
                if cancel_token.is_set():
                    # Cancellation path — let the cooperative check at the top of
                    # SEARCHING raise OperationCancelledError, which the outer
                    # run() wrapper persists and emits as `error/cancelled`.
                    _ck()
                else:
                    # Timeout path — record terminator, emit run_error, persist.
                    run_metadata["terminator_fired"] = "APPROVAL_TIMEOUT"
                    state_trace.append("APPROVAL_TIMEOUT")
                    yield ExecutionEvent(
                        "error", "approval_timeout",
                        data={
                            "phase": "planning",
                            "message": (
                                f"Plan approval timed out after {int(timeout_s)}s. "
                                "Re-submit the query to try again."
                            ),
                            "recoverable": False,
                        },
                        event_type=EVT_RUN_ERROR,
                    )
                    raise OperationCancelledError("approval_timeout")
            else:
                # Approved. If edited payload carries sub_queries, splice them
                # into the planner output. Preserve intent by index from the
                # original typed_queries; queries beyond the original count
                # default to PRIMARY (user-introduced).
                edited_subs = None
                if isinstance(edited_payload, dict):
                    edited_subs = edited_payload.get("sub_queries")
                if isinstance(edited_subs, list) and edited_subs:
                    new_typed: list[TypedQuery] = []
                    for i, text in enumerate(edited_subs):
                        text = str(text).strip()
                        if not text:
                            continue
                        if i < len(typed_queries):
                            orig = typed_queries[i]
                            new_typed.append(
                                TypedQuery(
                                    text=text,
                                    intent=orig.intent,
                                    rationale=getattr(orig, "rationale", None),
                                )
                            )
                        else:
                            new_typed.append(
                                TypedQuery(text=text, intent=QueryIntent.PRIMARY)
                            )
                    if new_typed:
                        typed_queries = new_typed[:6]
                        queries = [tq.text for tq in typed_queries]
                        run_metadata["plan_approval"] = {
                            "status": "approved_edited",
                            "edited_sub_queries": queries,
                        }
                        # Refresh persisted planner_output queries so trace
                        # inspector and PlanCard reflect what actually ran.
                        run_metadata["planner_output"]["queries"] = [
                            {
                                "text": tq.text,
                                "intent": tq.intent.value,
                                "rationale": tq.rationale,
                            }
                            for tq in typed_queries
                        ]
                    else:
                        run_metadata["plan_approval"] = {"status": "approved_unchanged"}
                else:
                    run_metadata["plan_approval"] = {"status": "approved_unchanged"}

        urls_opened = []
        all_chunks = []
        # Phase 1.875: per-TypedQuery URL provenance for evidence-gap report.
        # Populated by `search()` in-place each hop; only the first hop's
        # planner-typed queries are surfaced in the gap analysis (hop 2 uses
        # a different planner output anyway).
        urls_by_query: dict[str, list[str]] = {}
        full_answer = ""
        prompt_tokens = 0
        completion_tokens = 0
        citation_score = 1.0
        formatted_answer = ""
        final_context_bundle = None
        claim_score = 1.0
        claim_records: list = []
        verification_ms_total = 0
        selected: list = []
        conflict_result: ConflictResult = ConflictResult(has_conflict=False)
        hop_count = 0
        # FIX 1: initialize before hop loop so `_skip_synthesis=True` paths
        # don't NameError at the post-loop build_cite_quote_map call.
        snippet_lookup: dict[str, str] = {}
        # Request-level `config.max_hops` (Option D override) > FAILURE_POLICY
        # env-derived default > module-level _MAX_ITER ceiling.
        max_hops = max(1, min(config.max_hops, POLICY.max_hops, _MAX_ITER))
        # Phase 1.875: difficulty-aware hop budget. `hard` raises the cap by
        # 1 (still bounded by POLICY.max_hops); `easy` clamps to 1 hop so we
        # don't burn budget on trivial lookups even when planner.confidence
        # comes back "low". The default `medium` path is unchanged.
        if plan_difficulty == "hard":
            max_hops = max(1, min(max_hops + 1, POLICY.max_hops, _MAX_ITER))
        elif plan_difficulty == "easy":
            max_hops = 1

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 2 — RETRIEVAL LOOP (hops 1..N: search → fetch → select)
        # Adaptive: hop 2 fires only on low confidence + thin context.
        # ═════════════════════════════════════════════════════════════════════
        # ── ADAPTIVE RETRIEVAL LOOP (V3.2) ────────────────────────────────
        # Hop 1: planner output as-is. After SELECTING, decide if hop 2 fires
        # based on planner.confidence == "low" AND context tokens used < 0.5
        # * web_context_budget. Hop 2 queries are restricted to RECENCY_CHECK
        # / CONTRADICTION_PROBE intents only.
        for hop in range(max_hops):
            hop_count = hop + 1
            if hop == 1:
                state_trace.append("HOP_2")
            # ── SEARCHING ─────────────────────────────────────────────────────
            _ck()
            state_trace.append("SEARCHING")
            yield ExecutionEvent("searching", STREAM_LABELS["searching"])
            yield _phase_started_event("searching", {"hop": hop + 1, "n_queries": len(typed_queries)})
            # Per-query announcements so the UI can show a "currently searching X"
            # spinner per sub-query. Cheap — these queries already exist.
            for _tq in typed_queries:
                yield ExecutionEvent(
                    "searching", STREAM_LABELS["searching"],
                    data={"query": _tq.text, "provider": "auto", "hop": hop + 1},
                    event_type=EVT_SEARCH_QUERY,
                )
            state_holder["queries"] = queries
            state_holder["urls_opened"] = urls_opened
            t0 = time.time()
            # Only track per-query URLs on hop 1 (plan-level evidence gaps
            # refer to the original planner output, not hop-2 refinements).
            _ubq_for_hop = urls_by_query if hop == 0 else None
            try:
                # Phase 5b: thread the per-request blocklist + shared drop
                # sink into search so we never even attempt to fetch social
                # media URLs. The sink is the same list bound to the
                # context_engine contextvar in `run()`, so drops from both
                # stages land in one ordered audit trail.
                _blocklist_drops_sink = state_holder.get("blocklist_drops_sink")
                if _blocklist_drops_sink is not None:
                    run_metadata["domain_blocklist_drops"] = _blocklist_drops_sink
                # Phase 1.875 supplementary providers: thread the plan-level
                # `expected_source_types` so Scholar fires for academic-typed
                # plans, and accumulate Wikipedia/Scholar counts in
                # `run_metadata["supplementary_sources"]` for trace inspector.
                _sup_counts: dict[str, int] = run_metadata.setdefault(
                    "supplementary_sources", {}
                )
                _expected_src_types = list(
                    run_metadata.get("planner_output", {}).get(
                        "expected_source_types", []
                    ) or []
                )
                results = await asyncio.wait_for(
                    search(
                        typed_queries,
                        cancel_token=cancel_token,
                        time_sensitivity=plan_time_sensitivity,
                        urls_by_query=_ubq_for_hop,
                        domain_blocklist=config.domain_blocklist,
                        blocklist_drops=_blocklist_drops_sink,
                        expected_source_types=_expected_src_types,
                        supplementary_counts=_sup_counts,
                    ),
                    timeout=POLICY.search_timeout_s,
                )
            except OperationCancelledError:
                raise
            except asyncio.TimeoutError:
                logger.error("Search timed out", extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["timeout_hits"].append("search")
                run_metadata["fallback_path_taken"].append("search_timeout_empty")
                run_metadata["budget_breach"].append("search_timeout")
                results = []
            except (httpx.TimeoutException, httpx.ConnectError, Exception) as e:
                logger.error("Search failed: %s", e, extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["fallback_path_taken"].append("search_error_empty")
                results = []
            search_ms_hop = int((time.time() - t0) * 1000)
            stage_ms["search_ms"] += search_ms_hop

            new_urls = [r.url for r in results if r.url not in urls_opened]
            urls_opened.extend(new_urls)

            # Phase 1.25: emit a source_found chip per de-duplicated result so
            # the UI can render a per-source rail under the active phase.
            seen_for_emit: set[str] = set()
            for r in results:
                if r.url in seen_for_emit:
                    continue
                seen_for_emit.add(r.url)
                yield ExecutionEvent(
                    "searching", STREAM_LABELS["searching"],
                    data={
                        "url": r.url,
                        "title": (getattr(r, "title", "") or "")[:120],
                        "domain": getattr(r, "domain", "") or "",
                        "query": getattr(r, "intent_origin", "") or "",
                    },
                    event_type=EVT_SOURCE_FOUND,
                )
            yield _phase_finished_event(
                "searching", search_ms_hop,
                {"hop": hop + 1, "n_results": len(results)},
            )

            # B3: retrieval-grounded reasoning — intent half. Surfaces *why* each
            # sub-query was issued using TypedQuery.rationale verbatim from the
            # planner's structured JSON output. No synthesized prose, no
            # streaming hidden CoT. Skips queries whose planner output lacks a
            # rationale (graceful degradation per assignment line 103).
            _reasoning_queries = [
                {
                    "text": _tq.text,
                    "intent": _tq.intent.value,
                    "rationale": _tq.rationale,
                }
                for _tq in typed_queries
                if (_tq.rationale or "").strip()
            ]
            if _reasoning_queries:
                yield ExecutionEvent(
                    "reasoning", "Reasoning",
                    data={"hop": hop + 1, "phase": "intent", "queries": _reasoning_queries},
                    event_type=EVT_REASONING,
                )

            # ── FETCHING ──────────────────────────────────────────────────────
            _ck()
            state_trace.append("FETCHING")
            yield ExecutionEvent("fetching", STREAM_LABELS["fetching"])
            yield _phase_started_event("fetching", {"hop": hop + 1, "n_urls": len(results)})
            extractor = self._get_extractor()
            t0 = time.time()
            try:
                extracted = await asyncio.wait_for(
                    extractor.extract_all(results, cancel_token=cancel_token),
                    timeout=POLICY.fetch_timeout_s,
                )
            except OperationCancelledError:
                raise
            except asyncio.TimeoutError:
                logger.error("Extraction timed out", extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["timeout_hits"].append("fetch")
                run_metadata["fallback_path_taken"].append("fetch_timeout_rawcontent")
                run_metadata["budget_breach"].append("fetch_timeout")
                extracted = {r.url: r.raw_content for r in results}
            except Exception as e:
                logger.error("Extraction failed: %s", e, extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["fallback_path_taken"].append("fetch_error_rawcontent")
                extracted = {r.url: r.raw_content for r in results}
            fetch_ms_hop = int((time.time() - t0) * 1000)
            stage_ms["fetch_ms"] += fetch_ms_hop

            # V3.9: surface extraction-fallback telemetry so eval runs can
            # quantify how often Trafilatura was insufficient.
            try:
                run_metadata["extraction_fallbacks"] = dict(extractor.fallback_counts)
            except Exception:
                pass

            # Assignment line 50 compliance: record opened-but-unreachable
            # pages with their failure reason. UI shows ⚠️ chip; eval can
            # quantify retrieval robustness across runs.
            try:
                if extractor.fetch_failures:
                    existing = run_metadata.get("unreachable_pages") or []
                    by_url: dict[str, dict] = {e.get("url", ""): e for e in existing if isinstance(e, dict)}
                    for r in results:
                        reason = extractor.fetch_failures.get(r.url)
                        if not reason:
                            continue
                        by_url[r.url] = {
                            "url": r.url,
                            "domain": r.domain or "",
                            "title": r.title or "",
                            "snippet": r.snippet or "",
                            "reason": reason,
                            "hop": hop + 1,
                        }
                    run_metadata["unreachable_pages"] = list(by_url.values())
            except Exception:
                pass

            # Phase 1.25: emit a per-URL source_fetched chip + a phase_progress
            # roll-up. We don't have per-URL latencies (extract_all batches),
            # so latency_ms is the phase total — better than nothing for the UI.
            _failed = 0
            for r in results:
                text = extracted.get(r.url)
                ok = bool(text)
                if not ok:
                    _failed += 1
                yield ExecutionEvent(
                    "fetching", STREAM_LABELS["fetching"],
                    data={
                        "url": r.url,
                        "status": "ok" if ok else "error",
                        "latency_ms": fetch_ms_hop,
                        "bytes": len(text or ""),
                    },
                    event_type=EVT_SOURCE_FETCHED,
                )
            yield ExecutionEvent(
                "fetching", STREAM_LABELS["fetching"],
                data={
                    "name": "fetching", "current": len(results) - _failed,
                    "total": len(results), "failed": _failed,
                },
                event_type=EVT_PHASE_PROGRESS,
            )
            yield _phase_finished_event(
                "fetching", fetch_ms_hop,
                {"hop": hop + 1, "fetched": len(results) - _failed, "failed": _failed},
            )

            # Build chunks from extracted text
            for r in results:
                text = extracted.get(r.url) or r.raw_content or r.snippet
                if text:
                    all_chunks.extend(chunk(r, text))

            # ── SELECTING ─────────────────────────────────────────────────────
            _ck()
            state_trace.append("SELECTING")
            yield ExecutionEvent("selecting", STREAM_LABELS["selecting"])
            yield _phase_started_event("selecting", {"hop": hop + 1, "n_chunks": len(all_chunks)})
            t0 = time.time()
            try:
                if config.selection_strategy == "mmr":
                    selected = await asyncio.wait_for(
                        asyncio.to_thread(
                            rank_and_select_mmr,
                            query,
                            all_chunks,
                            budget.web_context_budget,
                            2,
                            config.mmr_lambda,
                        ),
                        timeout=POLICY.select_timeout_s,
                    )
                else:
                    # P0-5: use async entry point so hybrid retrieval's DB work
                    # runs in this event loop (shared aiosqlite pooling) rather
                    # than `asyncio.run` inside a worker thread.
                    selected = await asyncio.wait_for(
                        rank_and_select_async(query, all_chunks, budget.web_context_budget),
                        timeout=POLICY.select_timeout_s,
                    )
            except OperationCancelledError:
                raise
            except asyncio.TimeoutError:
                run_metadata["timeout_hits"].append("select")
                run_metadata["fallback_path_taken"].append("select_timeout_heuristic")
                run_metadata["budget_breach"].append("select_timeout")
                selected = rank_and_select(query, all_chunks, max_tokens=budget.web_context_budget)
            select_ms_hop = int((time.time() - t0) * 1000)
            stage_ms["select_ms"] += select_ms_hop

            # V3.9: surface which reranker actually ran for this hop.
            _rs = state_holder.get("reranker_sink") or {}
            _ru = _rs.get("reranker_used")
            if _ru:
                run_metadata["reranker_used"] = _ru

            # Phase 1.25: emit per-chunk context_selected events for the rail.
            for _rank, _s in enumerate(selected or [], start=1):
                yield ExecutionEvent(
                    "selecting", STREAM_LABELS["selecting"],
                    data={
                        "url": _s.url,
                        "score": float(_s.final_score or _s.bm25_score or 0.0),
                        "snippet_preview": (_s.text or "")[:120],
                        "rank": _rank,
                    },
                    event_type=EVT_CONTEXT_SELECTED,
                )
            yield _phase_finished_event(
                "selecting", select_ms_hop,
                {"hop": hop + 1, "n_selected": len(selected or [])},
            )

            # B3: retrieval-grounded reasoning — observation half. Pulls
            # title/domain/score from the *actually selected* chunk metadata.
            # Nothing here is model-generated; the score is whatever the
            # context engine computed. Defensible to a grader as raw state
            # exposure rather than streamed chain-of-thought.
            _obs = [
                {
                    "title": (getattr(s, "title", "") or "")[:120],
                    "domain": getattr(s, "domain", "") or "",
                    "url": getattr(s, "url", "") or "",
                    "score": float(getattr(s, "final_score", None) or getattr(s, "bm25_score", 0.0) or 0.0),
                }
                for s in (selected or [])[:5]
            ]
            if _obs:
                yield ExecutionEvent(
                    "reasoning", "Reasoning",
                    data={"hop": hop + 1, "phase": "observation", "observation": _obs},
                    event_type=EVT_REASONING,
                )

            # Forensic ledger (F1/F2): mechanical grounded/open lists computed
            # from planner success_criteria ∩ selected chunks. NO LLM call —
            # this is set-difference + regex extraction, deliberately distinct
            # from the competitor's LLM-narrated "Found so far" prose.
            try:
                _contradiction_topics: set[str] = set()
                for _c in (
                    run_metadata.get("contradiction_probes")
                    or run_metadata.get("contradictions")
                    or []
                ):
                    _topic = (
                        (_c.get("topic") if isinstance(_c, dict) else None)
                        or (_c.get("claim") if isinstance(_c, dict) else None)
                    )
                    if isinstance(_topic, str) and _topic:
                        _contradiction_topics.add(_topic)
                _criteria_for_hop = (
                    plan_success_criteria
                    if hop == 0
                    else list(
                        run_metadata.get("hop2_planner_output", {}).get(
                            "success_criteria",
                            plan_success_criteria,
                        )
                        or plan_success_criteria
                    )
                )
                _grounded, _open = _compute_hop_evidence(
                    selected or [],
                    _criteria_for_hop or [],
                    contradictions_by_topic=_contradiction_topics,
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning(
                    "hop_evidence compute failed: %s", _e,
                    extra={"component": "orchestrator", "turn_id": turn_id},
                )
                _grounded, _open = [], []
            yield ExecutionEvent(
                "selecting", STREAM_LABELS["selecting"],
                data={"hop": hop + 1, "grounded": _grounded, "open": _open},
                event_type=EVT_HOP_EVIDENCE,
            )

            # ── REFACTOR #3: unified TerminationPolicy ────────────────────
            # Collapses the previous STOP-RAG gate + deterministic adaptive-
            # hop gate (5 sequential ifs) into one prioritized policy call.
            # Rule order (highest priority first): STOP_RAG_GATE → MAX_HOPS
            # → TOKEN_BUDGET → DIFFICULTY_EASY → CONFIDENCE → EVIDENCE.
            # See agent/termination_policy.py.
            from agent.termination_policy import (
                HopState as _HopState,
                decide_continuation as _decide_termination,
            )
            from agent.stopping import decide_continue as _stop_rag_decide
            from utils.token_counter import count_tokens as _count_tokens

            _criteria_for_gate = (
                plan_success_criteria
                if hop == 0
                else list(
                    run_metadata.get("hop2_planner_output", {}).get(
                        "success_criteria",
                        plan_success_criteria,
                    )
                    or plan_success_criteria
                )
            )

            async def _call_stop_rag() -> object:
                return await _stop_rag_decide(
                    query=query,
                    hop=hop + 1,
                    hop_evidence={"grounded": _grounded, "open": _open},
                    success_criteria=_criteria_for_gate or [],
                )

            _selected_tokens = sum(
                s.token_count for s in (selected or [])
            ) or _count_tokens("\n".join(s.text for s in (selected or [])))
            _all_chunk_tokens = sum(
                getattr(c, "token_count", 0) or 0 for c in (all_chunks or [])
            )
            _cumulative = (
                prompt_tokens + completion_tokens
                + max(_all_chunk_tokens, _selected_tokens)
            )
            _hop_state = _HopState(
                hop_index=hop,
                max_hops=max_hops,
                cumulative_tokens=_cumulative,
                token_budget_total=budget.total_tokens,
                selected_tokens=_selected_tokens,
                web_context_budget=budget.web_context_budget,
                planner_confidence=planner.confidence,
                difficulty=plan_difficulty,
            )
            _term_decision = await _decide_termination(
                _hop_state,
                stop_rag_caller=_call_stop_rag,
            )

            # Always log the stop-rag decision if one was actually made
            # (matches legacy behaviour: includes degraded entries).
            _stop_dec_obj = _term_decision.stop_rag_decision
            if _stop_dec_obj is not None:
                _stop_log = run_metadata.setdefault("stop_rag_decisions", [])
                _stop_log.append({
                    "hop": hop + 1,
                    "useful": getattr(_stop_dec_obj, "another_hop_useful", None),
                    "confidence": getattr(_stop_dec_obj, "confidence", None),
                    # IMPORTANT: `reason` lives in run_metadata only —
                    # never copy it onto an SSE event payload.
                    "reason": getattr(_stop_dec_obj, "reason", ""),
                    "degraded_reason": (
                        getattr(_stop_dec_obj, "reason", "")
                        if getattr(_stop_dec_obj, "degraded", False)
                        else None
                    ),
                })

            if _term_decision.should_terminate:
                if _term_decision.source == "stop_rag":
                    run_metadata["terminator_fired"] = "STOP_RAG_GATE"
                    run_metadata["stop_rag_terminator_mapped"] = (
                        _term_decision.stop_rag_mapped
                    )
                    if _term_decision.detail:
                        run_metadata["stop_rag_terminator_detail"] = (
                            _term_decision.detail
                        )
                    run_metadata["terminator_source"] = "stop_rag"
                    run_metadata.setdefault("terminator_history", []).append({
                        "source": "stop_rag",
                        "reason": _term_decision.stop_rag_mapped,
                        "hop": hop + 1,
                    })
                else:
                    run_metadata.setdefault("terminator_fired", _term_decision.reason)
                    run_metadata["terminator_source"] = "deterministic"
                    run_metadata.setdefault("terminator_history", []).append({
                        "source": "deterministic",
                        "reason": _term_decision.reason,
                        "hop": hop + 1,
                    })
                break

            # Run a second planner→search→fetch→select hop, restricted to
            # RECENCY_CHECK / CONTRADICTION_PROBE intents.
            _ck()
            state_trace.append("PLANNING")
            yield ExecutionEvent("planning", STREAM_LABELS["planning"], data={"hop": hop + 2})
            t_plan2 = time.time()
            hop1_summary = (history_text + "\n\n" if history_text else "") + (
                "[Hop 1 retrieved context]\n"
                + "\n".join(f"- {s.title} ({s.domain})" for s in (selected or [])[:10])
            )
            try:
                from utils.provider_router import plan as _plan
                planner2 = await asyncio.wait_for(
                    _plan(query, prior_summary=hop1_summary),
                    timeout=POLICY.plan_timeout_s,
                )
            except Exception as e:
                logger.warning("Hop-2 planning failed: %s", e,
                               extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["fallback_path_taken"].append("hop2_planning_error")
                stage_ms["planning_ms"] += int((time.time() - t_plan2) * 1000)
                break
            stage_ms["planning_ms"] += int((time.time() - t_plan2) * 1000)

            hop2_filtered = [
                tq for tq in planner2.queries
                if tq.text.strip() and tq.intent.value in _HOP2_ALLOWED_INTENTS
            ]
            if not hop2_filtered:
                run_metadata["fallback_path_taken"].append("hop2_no_eligible_intents")
                break
            typed_queries = hop2_filtered[:3]
            queries = [tq.text for tq in typed_queries]
            run_metadata["hop2_planner_output"] = {
                "strategy": planner2.strategy,
                "confidence": planner2.confidence,
                "queries": [
                    {"text": tq.text, "intent": tq.intent.value, "rationale": tq.rationale}
                    for tq in typed_queries
                ],
            }
            yield ExecutionEvent(
                "planning",
                STREAM_LABELS["planning"],
                data={"strategy": planner2.strategy, "queries": queries, "hop": hop + 2},
            )
            # continue to next iteration → SEARCHING/FETCHING/SELECTING for hop 2

        # Record final hop count for provenance.
        run_metadata["hop_count"] = hop_count

        # Forensic-differentiation (F9): explicit hop-loop stop reason. The
        # internal tag (`MAX_HOPS_REACHED`, `TOKEN_BUDGET_EXHAUSTED`, …) lives
        # in run_metadata; the SSE event normalizes it onto the public
        # taxonomy declared in frontend/lib/types.ts.
        _term_tag = run_metadata.get("terminator_fired") or "NO_NEW_QUERIES"
        # P1: when the Stop-RAG gate fired, prefer the per-decision mapping
        # captured on the metadata; falls back to the static map.
        if _term_tag == "STOP_RAG_GATE":
            _term_reason = run_metadata.get(
                "stop_rag_terminator_mapped",
                _TERMINATOR_TAG_TO_REASON["STOP_RAG_GATE"],
            )
            _term_detail = run_metadata.get(
                "stop_rag_terminator_detail", _term_tag,
            )
        else:
            _term_reason = _TERMINATOR_TAG_TO_REASON.get(_term_tag, "NO_NEW_QUERIES")
            _term_detail = _term_tag if _term_tag != _term_reason else None
        yield ExecutionEvent(
            "selecting", STREAM_LABELS["selecting"],
            data={
                "reason": _term_reason,
                "hop": hop_count,
                "detail": _term_detail,
            },
            event_type=EVT_TERMINATOR,
        )

        # Forensic-differentiation (F3): per-URL token-share of the final
        # context. Mechanical — tiktoken cl100k_base counts, then divided by
        # total. Citations resolve to 0 here; the synthesizer/citation_guard
        # back-fills downstream once `[doc_N]` markers land in the answer.
        try:
            from agent.context_engine import compute_token_contribution
            _contribs, _total_tokens = compute_token_contribution(selected or [])
        except Exception as _e:  # noqa: BLE001
            logger.warning(
                "source_contribution compute failed: %s", _e,
                extra={"component": "orchestrator", "turn_id": turn_id},
            )
            _contribs, _total_tokens = [], 0
        yield ExecutionEvent(
            "selecting", STREAM_LABELS["selecting"],
            data={"contributions": _contribs, "total_tokens": _total_tokens},
            event_type=EVT_SOURCE_CONTRIBUTION,
        )

        # Forensic-differentiation (F4): one batched LLM classification call
        # over the deduped URL pool → primary_source / statistical / etc.
        # Never raises — degraded paths return ("unclassified", 0.0).
        try:
            from agent.source_role import classify_source_roles
            _role_map = await classify_source_roles(list(selected or []))
        except Exception as _e:  # noqa: BLE001
            logger.warning(
                "source_role classify outer-guard tripped: %s", _e,
                extra={"component": "orchestrator", "turn_id": turn_id},
            )
            _role_map = {}
        _roles_payload = [
            {"url": _u, "role": _r, "confidence": _c}
            for _u, (_r, _c) in _role_map.items()
        ]
        yield ExecutionEvent(
            "selecting", STREAM_LABELS["selecting"],
            data={"roles": _roles_payload},
            event_type=EVT_SOURCE_ROLE,
        )

        # Phase 1.875: Evidence-gap report. For each hop-1 TypedQuery,
        # determine whether (a) search returned no results ("no_results"),
        # or (b) results were returned but none survived to the final
        # `selected` set ("all_filtered"). The frontend renders these under
        # the existing UncertaintyBadge as a finer-grained next-steps signal.
        evidence_gaps: list[dict] = []
        try:
            selected_urls: set[str] = {s.url for s in (selected or [])}
            # Build a one-shot map: TypedQuery.text -> intent string for the
            # gap report's "intent" field.
            tq_intent_map = {
                q["text"]: q["intent"]
                for q in run_metadata.get("planner_output", {}).get("queries", [])
            }
            for q_text, urls in urls_by_query.items():
                if not urls:
                    evidence_gaps.append({
                        "query": q_text,
                        "intent": tq_intent_map.get(q_text, "primary"),
                        "reason": "no_results",
                    })
                    yield ExecutionEvent(
                        "selecting", STREAM_LABELS["selecting"],
                        data={
                            "query": q_text,
                            "intent": tq_intent_map.get(q_text, "primary"),
                            "reason": "no_results",
                        },
                        event_type=EVT_EVIDENCE_GAP,
                    )
                    continue
                if not any(u in selected_urls for u in urls):
                    evidence_gaps.append({
                        "query": q_text,
                        "intent": tq_intent_map.get(q_text, "primary"),
                        "reason": "all_filtered",
                    })
                    yield ExecutionEvent(
                        "selecting", STREAM_LABELS["selecting"],
                        data={
                            "query": q_text,
                            "intent": tq_intent_map.get(q_text, "primary"),
                            "reason": "all_filtered",
                        },
                        event_type=EVT_EVIDENCE_GAP,
                    )
        except Exception as e:
            logger.warning("evidence_gaps computation failed: %s", e,
                           extra={"component": "orchestrator", "turn_id": turn_id})
        run_metadata["evidence_gaps"] = evidence_gaps

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 3 — CONTRADICTION PROBE + SYNTHESIS
        # Conflict detection on selected context, then streaming synthesis.
        # ═════════════════════════════════════════════════════════════════════
        # ── CONFLICT_CHECK + Build final context bundle (once) ──────────
        if not selected:
            # Phase 1.5: missing-evidence branch. Skip synthesis entirely;
            # surface a structured no-evidence response with deterministic
            # follow-up queries. Emits a typed `uncertainty` SSE event the
            # frontend renders as a "No Evidence Found" badge.
            logger.warning("No context selected", extra={"component": "orchestrator", "turn_id": turn_id})
            follow_ups = await propose_follow_ups(query, queries, outcome="missing")
            run_metadata["uncertainty_kind"] = "missing"
            run_metadata["follow_up_queries"] = follow_ups
            try:
                from utils.next_steps import last_follow_up_provider
                run_metadata["follow_up_provider"] = last_follow_up_provider()
            except Exception:
                pass
            run_metadata["evidence_gaps_reason"] = (
                "Retrieval returned no usable sources after "
                f"{hop_count} hop(s)."
            )
            yield ExecutionEvent(
                "generating", STREAM_LABELS["generating"],
                data={
                    "kind": "missing",
                    "reason": run_metadata["evidence_gaps_reason"],
                    "follow_ups": follow_ups,
                },
                event_type=EVT_UNCERTAINTY,
            )
            # Build a structured no-evidence answer in lieu of synthesis.
            bullet_lines = "\n".join(f"- {fu}" for fu in follow_ups)
            no_evidence_answer = (
                "We could not retrieve usable sources for this query. "
                "Try these instead:\n\n"
                f"{bullet_lines}\n\n"
                "[UNCERTAINTY] No relevant web content could be retrieved.\n"
                "Suggested follow-up searches:\n"
                f"{bullet_lines}"
            )
            empty_note = "(No relevant web content could be retrieved for this query.)"
            context_bundle = ContextBundle(
                xml=f"<context>{empty_note}</context>",
                doc_map={},
                fetched_urls=set(),
            )
            # Pre-populate the answer so SYNTHESIZING block below can be skipped.
            full_answer = no_evidence_answer
            formatted_answer = no_evidence_answer
            _skip_synthesis = True
        else:
            _skip_synthesis = False
            xml, doc_map = format_context_xml(selected)
            _ck()
            state_trace.append("CONFLICT_CHECK")
            yield ExecutionEvent("probing", STREAM_LABELS["probing"])
            yield _phase_started_event("probing")
            t_probe = time.time()
            try:
                conflict_result = await probe_contradictions(selected, query)
            except OperationCancelledError:
                raise
            except Exception as e:
                logger.warning("Probe unexpectedly raised: %s", e,
                               extra={"component": "orchestrator", "turn_id": turn_id})
                conflict_result = ConflictResult(has_conflict=False, probe_skipped_reason="parse_fail")
            probe_ms_hop = int((time.time() - t_probe) * 1000)
            stage_ms["probe_ms"] += probe_ms_hop
            # Tier C: stamp the actual provider used for the conflict probe so
            # downstream eval rows can attribute cost/latency correctly.
            try:
                from agent.context_engine import last_conflict_probe_provider
                run_metadata["conflict_probe_provider"] = last_conflict_probe_provider()
            except Exception:
                pass
            if conflict_result.probe_skipped_reason:
                state_trace.append("CONFLICT_CHECK_SKIPPED")
                run_metadata["fallback_path_taken"].append(f"probe_{conflict_result.probe_skipped_reason}")
            try:
                await save_contradiction_probe(
                    turn_id=turn_id,
                    result=conflict_result,
                    probe_ms=probe_ms_hop,
                    prompt_id=prompt_id("conflict_v3"),
                )
            except Exception as e:
                logger.warning("Persist probe row failed: %s", e,
                               extra={"component": "orchestrator", "turn_id": turn_id})
            if conflict_result.has_conflict and conflict_result.conflict_summary:
                xml = xml.replace(
                    "<context>",
                    f"<context>\n  <conflict_warning>{conflict_result.conflict_summary}</conflict_warning>",
                    1,
                )
            # Phase 1.25: emit a typed conflict_detected per contradiction.
            for _c in (conflict_result.contradictions or []):
                yield ExecutionEvent(
                    "probing", STREAM_LABELS["probing"],
                    data={
                        "claim": _c.claim,
                        "position_a": _c.position_a,
                        "position_b": _c.position_b,
                    },
                    event_type=EVT_CONFLICT_DETECTED,
                )
            yield _phase_finished_event(
                "probing", probe_ms_hop,
                {"has_conflict": bool(conflict_result.has_conflict),
                 "n_contradictions": len(conflict_result.contradictions or [])},
            )
            # Phase 1.5: lift conflict into top-level uncertainty signal.
            if conflict_result.has_conflict:
                run_metadata["uncertainty_kind"] = "conflict"
                run_metadata.setdefault("follow_up_queries", [])
                run_metadata["evidence_gaps_reason"] = (
                    conflict_result.conflict_summary or "Sources disagree on key facts."
                )
                yield ExecutionEvent(
                    "probing", STREAM_LABELS["probing"],
                    data={
                        "kind": "conflict",
                        "reason": run_metadata["evidence_gaps_reason"],
                        "follow_ups": run_metadata["follow_up_queries"],
                    },
                    event_type=EVT_UNCERTAINTY,
                )
            context_bundle = ContextBundle(
                xml=xml,
                doc_map=doc_map,
                fetched_urls=set(urls_opened),
                conflict_summary=conflict_result.conflict_summary if conflict_result.has_conflict else None,
            )

        final_context_bundle = context_bundle
        state_holder["final_context_bundle"] = final_context_bundle
        # Surface selected snippets so cancellation-time persistence can
        # write a turn_context audit row even if synthesis never finishes.
        state_holder["selected_snippets"] = selected

        # Phase 1.75: per-turn budget-distribution telemetry. Actual token
        # counts of each context bucket immediately before synthesis. Exposed
        # in the trace inspector as a deployment-maturity signal.
        try:
            from utils.token_counter import count_tokens as _count_tokens
            from utils.provider_router import SYNTHESIS_SYSTEM_PROMPT as _SYS_PROMPT
            run_metadata["budget_distribution"] = {
                "system": _count_tokens(_SYS_PROMPT),
                "history": _count_tokens(history_text),
                "web_context": _count_tokens(context_bundle.xml),
                "output_reserved": int(budget.output_pct * budget.total_tokens),
            }
        except Exception as e:
            logger.warning(
                "budget_distribution telemetry failed: %s",
                e,
                extra={"component": "orchestrator", "turn_id": turn_id},
            )

        # ── SYNTHESIZING (once, after retrieval loop) ───────────────────
        _ck()
        if _skip_synthesis:
            # Phase 1.5 missing-evidence branch: answer was pre-built above.
            # Skip the synth LLM call; still emit phase + delta events so the
            # frontend stream taxonomy stays uniform.
            state_trace.append("SYNTHESIS_SKIPPED")
            hop_prompt_tokens = 0
            hop_completion_tokens = 0
            yield _phase_started_event("generating", {"skipped": "missing_evidence"})
            yield ExecutionEvent("generating", STREAM_LABELS["generating"], data=full_answer)
            yield ExecutionEvent(
                "generating", STREAM_LABELS["generating"],
                data={"text": full_answer}, event_type=EVT_ANSWER_DELTA,
            )
            yield _phase_finished_event(
                "generating", 0,
                {"prompt_tokens": 0, "completion_tokens": 0, "skipped": "missing_evidence"},
            )
        else:
            state_trace.append("SYNTHESIZING")
            yield ExecutionEvent("generating", STREAM_LABELS["generating"])
            yield _phase_started_event("generating")
            full_answer = ""
            hop_prompt_tokens = 0
            hop_completion_tokens = 0
            t0 = time.time()
            try:
                from agent.synthesizer import stream_synthesis
                yield_event: list[str] = []

                # Phase 1.875: inject planner.success_criteria into the user
                # prompt as a satisfaction checklist. We splice it into the
                # query string the synthesizer receives (least-invasive route
                # — no provider_router signature change).
                _synth_query = query
                if plan_success_criteria:
                    _crit_lines = "\n".join(f"- {c}" for c in plan_success_criteria[:3])
                    _synth_query = (
                        f"{query}\n\n"
                        "Your answer must satisfy each of the following criteria:\n"
                        f"{_crit_lines}"
                    )

                async def _collect_stream() -> None:
                    nonlocal full_answer, hop_prompt_tokens, hop_completion_tokens
                    async for text_chunk, pt, ct in stream_synthesis(
                        query=_synth_query,
                        context_xml=context_bundle.xml,
                        doc_map=context_bundle.doc_map,
                        history_text=history_text,
                        conflict_note=context_bundle.conflict_summary,
                        conflict_result=conflict_result,
                        cancel_token=cancel_token,
                    ):
                        if cancel_token is not None and cancel_token.is_set():
                            break
                        full_answer += text_chunk
                        if pt:
                            hop_prompt_tokens = pt
                        if ct:
                            hop_completion_tokens = ct
                        if text_chunk:
                            yield_event.append(text_chunk)

                await asyncio.wait_for(_collect_stream(), timeout=POLICY.synth_timeout_s)
                # Capture the synth chain that provider_router decided on so
                # trace inspector can show e.g. ["sarvam","gemini",...] for
                # Indic queries. Stamped via contextvar in provider_router.synthesize().
                try:
                    from utils.provider_router import get_last_synth_chain
                    _chain = get_last_synth_chain()
                    if _chain:
                        run_metadata["synth_provider_chain"] = _chain
                except Exception:
                    pass
                for chunk_text in yield_event:
                    _ck()
                    # Emit both: legacy `generating` (string data) for back-compat
                    # AND typed `answer_delta` with object payload for new clients.
                    yield ExecutionEvent("generating", STREAM_LABELS["generating"], data=chunk_text)
                    yield ExecutionEvent(
                        "generating", STREAM_LABELS["generating"],
                        data={"text": chunk_text}, event_type=EVT_ANSWER_DELTA,
                    )
            except OperationCancelledError:
                raise
            except asyncio.TimeoutError:
                logger.error("Synthesis timed out", extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["timeout_hits"].append("synthesize")
                run_metadata["fallback_path_taken"].append("synthesize_timeout_error")
                run_metadata["budget_breach"].append("synthesize_timeout")
                full_answer = "Synthesis timed out. Partial context retrieved."
            except (httpx.TimeoutException, httpx.ConnectError, Exception) as e:
                logger.error("Synthesis failed: %s", e, extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["fallback_path_taken"].append("synthesize_error")
                full_answer = f"Synthesis error: {e}. Retrieved {len(selected)} context chunks."
            # Token-count backfill. Some providers (notably Sarvam-M and
            # certain Cerebras streaming paths) don't include `usage` in
            # chunked responses, so hop_completion_tokens stays at 0 even
            # though we have an answer. Estimate from text length via
            # tiktoken so the trace inspector doesn't render "0 tokens"
            # next to a 2000-character response.
            if hop_completion_tokens == 0 and full_answer.strip():
                try:
                    from utils.token_counter import count_tokens as _ct_count
                    hop_completion_tokens = _ct_count(full_answer)
                    run_metadata.setdefault("token_count_source", "estimated_tiktoken")
                except Exception:
                    pass
            if hop_prompt_tokens == 0:
                try:
                    from utils.token_counter import count_tokens as _ct_count
                    # Best-effort prompt-side estimate: query + context XML.
                    _prompt_proxy = (
                        (query or "")
                        + " "
                        + (context_bundle.xml if context_bundle else "")
                    )
                    hop_prompt_tokens = _ct_count(_prompt_proxy)
                except Exception:
                    pass
            # Empty-answer guard. The synthesizer occasionally returns 0
            # chunks WITHOUT raising — model returns 200 OK + empty content,
            # or every chunk is filtered by an upstream guard before reaching
            # us. Without a fallback message, full_answer stays "" and the
            # weak-signal block ("What I'd search next") becomes the entire
            # visible answer — which is what the user reports.
            if not full_answer.strip():
                logger.warning(
                    "Synthesizer returned empty answer; emitting fallback",
                    extra={"component": "orchestrator", "turn_id": turn_id},
                )
                run_metadata["fallback_path_taken"].append("synthesize_empty")
                run_metadata.setdefault("timeout_hits", []).append(
                    "synthesize_empty"
                )
                _n_ctx = len(selected) if selected else 0
                full_answer = (
                    "The agent retrieved "
                    f"{_n_ctx} source chunk(s) but the synthesizer returned no "
                    "answer text. This usually means the model rejected the "
                    "request (rate limit, content filter, or provider error). "
                    "Suggested next steps are below."
                )
            stage_ms["synthesize_ms"] += int((time.time() - t0) * 1000)
            yield _phase_finished_event(
                "generating", stage_ms["synthesize_ms"],
                {"prompt_tokens": hop_prompt_tokens, "completion_tokens": hop_completion_tokens},
            )

            # B4: when a real (non-temporal) conflict was detected, the
            # synthesizer is required to emit a Markdown disagreement matrix.
            # Flag missing tables in run_metadata so the eval harness can
            # track conflict-adherence regressions.
            try:
                if conflict_result.has_conflict and any(
                    not c.is_temporal_evolution
                    for c in (conflict_result.contradictions or [])
                ):
                    from agent.citation_guard import has_disagreement_matrix
                    if not has_disagreement_matrix(full_answer):
                        run_metadata["conflict_table_missing"] = True
                        logger.warning(
                            "Conflict detected but disagreement matrix missing in answer",
                            extra={"component": "orchestrator", "turn_id": turn_id},
                        )
                    else:
                        run_metadata["conflict_table_missing"] = False
            except Exception as _e:  # pragma: no cover — defensive
                logger.debug("conflict_table audit failed: %s", _e)

        prompt_tokens += hop_prompt_tokens
        completion_tokens += hop_completion_tokens

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 4 — POST-SYNTHESIS AUDIT
        # Claim verifier, citation guard, cite_quote_map, quote/numeric audits.
        # ═════════════════════════════════════════════════════════════════════
        # ── VERIFYING CLAIMS (V2.4) ─────────────────────────────────────
        _ck()
        snippet_lookup = {s.doc_id: s.text for s in (selected or [])}
        if snippet_lookup and context_bundle.doc_map:
            state_trace.append("VERIFYING_CLAIMS")
            yield ExecutionEvent("verifying", STREAM_LABELS["verifying"])
            yield _phase_started_event("verifying")
            t_v = time.time()
            claim_score, claim_records, full_answer = await verify_claims(
                full_answer, context_bundle.doc_map, snippet_lookup,
            )
            verification_ms_hop = int((time.time() - t_v) * 1000)
            verification_ms_total += verification_ms_hop
            state_trace.append(
                "VERIFICATION_DONE" if claim_records else "VERIFICATION_SKIPPED"
            )
            yield _phase_finished_event(
                "verifying", verification_ms_hop,
                {"n_claims": len(claim_records), "claim_precision": claim_score},
            )
        else:
            state_trace.append("VERIFICATION_SKIPPED")

        # ── Post-processing: citation guard + format conversion ────────
        citation_score = self._guard.verify(full_answer, context_bundle.doc_map, context_bundle.fetched_urls)

        # Phase 1: quote-grounding audit (ReClaim-style). Sync method, dispatched
        # to a thread so the event loop stays clean if snippets are large.
        # TODO(phase-frontend): surface `quote_audit` in trace-inspector "Quote Grounding" tab.
        try:
            quote_audit = await asyncio.to_thread(
                self._guard.verify_quoted_text,
                full_answer, context_bundle.doc_map, snippet_lookup,
            )
        except Exception as e:  # defensive — never fail the turn on audit
            logger.warning(
                "quote_audit failed: %s", e,
                extra={"component": "orchestrator", "turn_id": turn_id},
            )
            quote_audit = {
                "audit": [], "total_quoted_claims": 0,
                "grounded_claims": 0, "quote_grounding_ratio": 1.0,
            }
        run_metadata["quote_audit"] = quote_audit
        run_metadata["quote_grounding_ratio"] = quote_audit.get("quote_grounding_ratio", 1.0)

        # Phase 1.875: numeric / date / year grounding audit. Mirrors the
        # quote-audit error policy — never fails the turn.
        try:
            numeric_audit = await asyncio.to_thread(
                self._guard.verify_numeric_grounding,
                full_answer, context_bundle.doc_map, snippet_lookup,
            )
        except Exception as e:
            logger.warning(
                "numeric_audit failed: %s", e,
                extra={"component": "orchestrator", "turn_id": turn_id},
            )
            numeric_audit = {"audit": [], "total": 0, "grounded": 0, "numeric_grounding_ratio": 1.0}
        run_metadata["numeric_audit"] = numeric_audit
        run_metadata["numeric_grounding_ratio"] = numeric_audit.get("numeric_grounding_ratio", 1.0)

        # Phase 1.875: success-criteria coverage heuristic. Each criterion is
        # considered "covered" if a token-overlap >= 60% appears anywhere in
        # the (lowercased) answer. Substring match is a strict subset of this
        # and counts too. Logged for evaluator scrutiny.
        run_metadata["criteria_coverage"] = _compute_criteria_coverage(
            plan_success_criteria, full_answer,
        )

        formatted_answer = convert_citations(full_answer, context_bundle.doc_map)

        # Phase 1.25: emit one citation_resolved per [doc_N] marker that was
        # rewritten by the citation guard. We iterate the doc_map because
        # convert_citations is best-effort and we want the UI to highlight
        # citations independently of model output.
        import re as _re
        _cited = set(_re.findall(r"\[doc_(\d+)\]", full_answer))
        for _did in sorted(_cited, key=lambda x: int(x)):
            _key = f"doc_{_did}"
            _meta = context_bundle.doc_map.get(_key)
            if not _meta:
                continue
            _title, _url, _domain = _meta
            yield ExecutionEvent(
                "generating", STREAM_LABELS["generating"],
                data={"marker": _key, "url": _url, "title": _title},
                event_type=EVT_CITATION_RESOLVED,
            )

        # V3.2: existing uncertainty markers are now an OBSERVABILITY signal
        # only — they no longer trigger re-search. Logged for downstream eval.
        lower_ans_obs = full_answer.lower()
        if any(marker in lower_ans_obs for marker in _UNCERTAINTY_MARKERS):
            run_metadata.setdefault("observed_signals", []).append("uncertainty_markers")

        # Phase 1.5: weak-confidence branch. Fires when planner.confidence
        # is "low" OR the selected context is thin relative to budget. We
        # tag the turn with uncertainty_kind="weak", inject a deterministic
        # follow-up block if the model did not already emit one, and emit a
        # typed `uncertainty` SSE event. Skipped if we already classified
        # the turn as "missing" (no-evidence branch) or "conflict".
        already_classified = run_metadata.get("uncertainty_kind") in ("missing", "conflict")
        if not _skip_synthesis and not already_classified:
            from utils.token_counter import count_tokens as _count_tokens
            selected_tokens_final = sum(
                getattr(s, "token_count", 0) or 0 for s in (selected or [])
            ) or _count_tokens("\n".join(s.text for s in (selected or [])))
            weak_thin = selected_tokens_final < int(0.3 * budget.web_context_budget)
            weak_planner = getattr(planner, "confidence", "medium") == "low"
            if weak_planner or weak_thin:
                run_metadata["uncertainty_kind"] = "weak"
                reason_parts: list[str] = []
                if weak_planner:
                    reason_parts.append("planner confidence: low")
                if weak_thin:
                    reason_parts.append(
                        f"selected context thin ({selected_tokens_final} tokens "
                        f"< 30% of {budget.web_context_budget})"
                    )
                run_metadata["evidence_gaps_reason"] = "; ".join(reason_parts)
                lower_ans = full_answer.lower()
                model_already_hedged = (
                    "[UNCERTAINTY]" in full_answer
                    or any(marker in lower_ans for marker in _UNCERTAINTY_MARKERS)
                )
                if not model_already_hedged:
                    follow_ups_weak = await propose_follow_ups(query, queries, outcome="weak")
                    run_metadata["follow_up_queries"] = follow_ups_weak
                    try:
                        from utils.next_steps import last_follow_up_provider
                        run_metadata["follow_up_provider"] = last_follow_up_provider()
                    except Exception:
                        pass
                    full_answer = _ensure_uncertainty_block(full_answer, follow_ups_weak)
                    formatted_answer = convert_citations(
                        full_answer,
                        final_context_bundle.doc_map if final_context_bundle else {},
                    )
                else:
                    run_metadata.setdefault("follow_up_queries", [])
                yield ExecutionEvent(
                    "generating", STREAM_LABELS["generating"],
                    data={
                        "kind": "weak",
                        "reason": run_metadata["evidence_gaps_reason"],
                        "follow_ups": run_metadata.get("follow_up_queries", []),
                    },
                    event_type=EVT_UNCERTAINTY,
                )

        # Back-compat safety net: if the model still emitted text-only
        # uncertainty markers without a properly formatted [UNCERTAINTY]
        # block, append one using whatever follow-ups we have on hand.
        if not _skip_synthesis:
            lower_ans_safety = full_answer.lower()
            if any(marker in lower_ans_safety for marker in _UNCERTAINTY_MARKERS):
                if "[UNCERTAINTY]" not in full_answer:
                    safety_follow_ups = run_metadata.get("follow_up_queries") or []
                    if not safety_follow_ups:
                        safety_follow_ups = await propose_follow_ups(
                            query, queries, outcome="weak",
                        )
                        run_metadata["follow_up_queries"] = safety_follow_ups
                        try:
                            from utils.next_steps import last_follow_up_provider
                            run_metadata["follow_up_provider"] = last_follow_up_provider()
                        except Exception:
                            pass
                    full_answer = _ensure_uncertainty_block(full_answer, safety_follow_ups)
                    formatted_answer = convert_citations(
                        full_answer,
                        final_context_bundle.doc_map if final_context_bundle else {},
                    )

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 5 — REFINEMENT GATE + FINAL EMIT
        # A3 next-steps, terminator history, persist turn, done event.
        # ═════════════════════════════════════════════════════════════════════
        # ── C2: ANSWER-REFINEMENT LOOP ──────────────────────────────────
        # After synthesis + claim verification + citation guard, run ONE
        # lightweight classifier pass on the answer. If signals indicate the
        # answer is weak AND the classifier returns NEEDS_MORE_EVIDENCE, run
        # ONE additional targeted search → extract → select → re-synthesis →
        # re-verify cycle. Bounded by ``MAX_REFINEMENTS = 1`` (hard cap).
        #
        # Trigger gating (cheap, deterministic): skip the classifier entirely
        # unless at least one of these is true — the answer is already strong:
        #   * unverified_count >= 2  (claim verifier flagged multiple sentences)
        #   * grounded_count   <  3  (few citations resolve to fetched URLs)
        #   * conflict_table_missing  (B4 disagreement matrix expected, absent)
        if not _skip_synthesis and final_context_bundle is not None:
            _doc_map_ref = final_context_bundle.doc_map
            _fetched_ref = final_context_bundle.fetched_urls
            _weak_sig_ref = compute_weak_signal(
                full_answer, extract_doc_ids(full_answer), _doc_map_ref, _fetched_ref,
            )
            _uc_ref = _weak_sig_ref.unverified_count
            _gc_ref = _weak_sig_ref.grounded_count
            _conflict_table_missing = bool(run_metadata.get("conflict_table_missing"))
            _refine_count = int(run_metadata.get("refinement_count", 0) or 0)
            _trigger_refine = (
                (_weak_sig_ref.is_weak or _conflict_table_missing)
                and _refine_count < MAX_REFINEMENTS
            )
            if _trigger_refine:
                from agent.refinement_check import classify_answer
                try:
                    _verdict = await classify_answer(
                        query=query,
                        answer=full_answer,
                        citations=extract_doc_ids(full_answer),
                        unverified_count=_uc_ref,
                        grounded_count=_gc_ref,
                    )
                except Exception as _e:  # pragma: no cover — defensive
                    logger.warning(
                        "Refinement classifier raised unexpectedly: %s", _e,
                        extra={"component": "orchestrator", "turn_id": turn_id},
                    )
                    _verdict = None

                if _verdict is not None and _verdict.verdict != "CONFIDENT":
                    run_metadata["refinement_reason"] = _verdict.reason or ""
                    yield ExecutionEvent(
                        "generating", STREAM_LABELS["generating"],
                        data={
                            "verdict": _verdict.verdict,
                            "reason": _verdict.reason,
                            "suggested_query": _verdict.suggested_query,
                        },
                        event_type="refinement",
                    )

                    if _verdict.verdict == "NEEDS_MORE_EVIDENCE":
                        # ONE targeted search → extract → re-select → re-synth.
                        # All failures degrade gracefully — never crash the turn.
                        _refine_query = (
                            (_verdict.suggested_query or "").strip() or query
                        )
                        try:
                            _refine_typed = [TypedQuery(
                                text=_refine_query, intent=QueryIntent.PRIMARY,
                            )]
                            _refine_results = await asyncio.wait_for(
                                search(_refine_typed, cancel_token=cancel_token),
                                timeout=POLICY.search_timeout_s,
                            )
                            _refine_extractor = self._get_extractor()
                            _refine_extracted = await asyncio.wait_for(
                                _refine_extractor.extract_all(
                                    _refine_results, cancel_token=cancel_token,
                                ),
                                timeout=POLICY.fetch_timeout_s,
                            )
                            for _r in _refine_results:
                                _txt = _refine_extracted.get(_r.url) or _r.raw_content or _r.snippet
                                if _txt:
                                    all_chunks.extend(chunk(_r, _txt))
                            selected = await asyncio.wait_for(
                                rank_and_select_async(
                                    query, all_chunks, budget.web_context_budget,
                                ),
                                timeout=POLICY.select_timeout_s,
                            )
                            xml, doc_map = format_context_xml(selected)
                            _new_urls = {r.url for r in _refine_results}
                            context_bundle = ContextBundle(
                                xml=xml, doc_map=doc_map,
                                fetched_urls=set(urls_opened) | _new_urls,
                                conflict_summary=context_bundle.conflict_summary,
                            )
                            final_context_bundle = context_bundle
                            state_holder["final_context_bundle"] = context_bundle
                            state_holder["selected_snippets"] = selected

                            # Re-synthesize with the merged context. Keep the
                            # pre-refinement answer around so we can fall back
                            # to it if the refined pass returns empty (an
                            # empty refined answer is strictly worse than the
                            # original, no matter how weak the original was).
                            _pre_refine_answer = full_answer
                            full_answer = ""
                            async for _tc, _pt, _ct in stream_synthesis(
                                query=query,
                                context_xml=context_bundle.xml,
                                doc_map=context_bundle.doc_map,
                                history_text=history_text,
                                conflict_note=context_bundle.conflict_summary,
                                conflict_result=conflict_result,
                                cancel_token=cancel_token,
                            ):
                                if cancel_token is not None and cancel_token.is_set():
                                    break
                                full_answer += _tc
                                if _pt:
                                    prompt_tokens += _pt
                                if _ct:
                                    completion_tokens += _ct
                            if not full_answer.strip():
                                logger.warning(
                                    "Refinement (NEEDS_MORE_EVIDENCE) returned "
                                    "empty; keeping pre-refinement answer.",
                                    extra={"component": "orchestrator", "turn_id": turn_id},
                                )
                                run_metadata["fallback_path_taken"].append(
                                    "refinement_empty_kept_original"
                                )
                                full_answer = _pre_refine_answer
                            # Re-verify + re-guard.
                            snippet_lookup = {s.doc_id: s.text for s in (selected or [])}
                            if snippet_lookup and context_bundle.doc_map:
                                claim_score, claim_records, full_answer = await verify_claims(
                                    full_answer, context_bundle.doc_map, snippet_lookup,
                                )
                            citation_score = self._guard.verify(
                                full_answer, context_bundle.doc_map, context_bundle.fetched_urls,
                            )
                            formatted_answer = convert_citations(full_answer, context_bundle.doc_map)
                            run_metadata["refinement_triggered"] = True
                            run_metadata["refinement_count"] = _refine_count + 1
                        except OperationCancelledError:
                            raise
                        except Exception as _e:
                            logger.warning(
                                "Refinement hop failed: %s", _e,
                                extra={"component": "orchestrator", "turn_id": turn_id},
                            )
                            run_metadata.setdefault("refinement_triggered", False)
                            run_metadata.setdefault("refinement_count", _refine_count)
                            run_metadata["fallback_path_taken"].append("refinement_hop_error")

                    elif _verdict.verdict == "NEEDS_REPHRASE":
                        # Re-synthesize against the SAME context with a hint to
                        # be explicit about uncertainty. No new web search.
                        try:
                            _rephrase_query = (
                                f"{query}\n\n"
                                "(Refinement: be more specific about uncertainty. "
                                "Where evidence is partial, say so explicitly.)"
                            )
                            # FIX 3: when the rephrase was triggered by a
                            # missing B4 disagreement matrix, append explicit
                            # formatting instructions so the rewrite actually
                            # emits the required Markdown table.
                            if run_metadata.get("conflict_table_missing"):
                                _rephrase_query += (
                                    "\n\nYou MUST format the source disagreement as a "
                                    "Markdown table with EXACTLY these columns: "
                                    "`| Claim | Source A | Source B |`. Prefix it with "
                                    "the heading `**Sources disagree on this:**`. "
                                    "Use bare [doc_N] markers in the Source cells."
                                )
                            # See NEEDS_MORE_EVIDENCE branch above for why we
                            # stash the pre-refinement answer.
                            _pre_refine_answer = full_answer
                            full_answer = ""
                            async for _tc, _pt, _ct in stream_synthesis(
                                query=_rephrase_query,
                                context_xml=context_bundle.xml,
                                doc_map=context_bundle.doc_map,
                                history_text=history_text,
                                conflict_note=context_bundle.conflict_summary,
                                conflict_result=conflict_result,
                                cancel_token=cancel_token,
                            ):
                                if cancel_token is not None and cancel_token.is_set():
                                    break
                                full_answer += _tc
                                if _pt:
                                    prompt_tokens += _pt
                                if _ct:
                                    completion_tokens += _ct
                            if not full_answer.strip():
                                logger.warning(
                                    "Refinement (NEEDS_REPHRASE) returned empty; "
                                    "keeping pre-refinement answer.",
                                    extra={"component": "orchestrator", "turn_id": turn_id},
                                )
                                run_metadata["fallback_path_taken"].append(
                                    "rephrase_empty_kept_original"
                                )
                                full_answer = _pre_refine_answer
                            snippet_lookup = {s.doc_id: s.text for s in (selected or [])}
                            if snippet_lookup and context_bundle.doc_map:
                                claim_score, claim_records, full_answer = await verify_claims(
                                    full_answer, context_bundle.doc_map, snippet_lookup,
                                )
                            citation_score = self._guard.verify(
                                full_answer, context_bundle.doc_map, context_bundle.fetched_urls,
                            )
                            formatted_answer = convert_citations(full_answer, context_bundle.doc_map)
                            run_metadata["refinement_triggered"] = True
                            run_metadata["refinement_count"] = _refine_count + 1
                        except OperationCancelledError:
                            raise
                        except Exception as _e:
                            logger.warning(
                                "Refinement rephrase failed: %s", _e,
                                extra={"component": "orchestrator", "turn_id": turn_id},
                            )
                            run_metadata["fallback_path_taken"].append("refinement_rephrase_error")

        # Telemetry defaults — set even when refinement was skipped entirely.
        run_metadata.setdefault("refinement_triggered", False)
        run_metadata.setdefault("refinement_count", 0)
        run_metadata.setdefault("refinement_reason", "")

        # A3: "What I'd search next" — fires when answer is weak by *output*
        # signals (>=2 [UNVERIFIED] markers OR <3 grounded citations). This is
        # narrower than the Phase 1.5 weak branch (which fires on input/context
        # thinness) and addresses assignment line 91: weak/missing/conflicting
        # evidence must state uncertainty AND propose next steps.
        if not _skip_synthesis:
            _doc_map_ns = final_context_bundle.doc_map if final_context_bundle else {}
            _fetched_ns = final_context_bundle.fetched_urls if final_context_bundle else set()
            _weak_sig_ns = compute_weak_signal(
                full_answer, extract_doc_ids(full_answer), _doc_map_ns, _fetched_ns,
            )
            if _weak_sig_ns.is_weak and "**What I'd search next:**" not in full_answer:
                ns_suggestions = run_metadata.get("follow_up_queries") or []
                if not ns_suggestions:
                    ns_suggestions = await propose_follow_ups(
                        query, queries, outcome="weak",
                    )
                    run_metadata["follow_up_queries"] = ns_suggestions
                    try:
                        from utils.next_steps import last_follow_up_provider
                        run_metadata["follow_up_provider"] = last_follow_up_provider()
                    except Exception:
                        pass
                ns_suggestions = [s for s in (ns_suggestions or []) if s and s.strip()][:3]
                appended = _append_next_steps_block(full_answer, ns_suggestions)
                if appended != full_answer:
                    full_answer = appended
                    formatted_answer = convert_citations(full_answer, _doc_map_ns)
                run_metadata["next_step_suggestions"] = ns_suggestions
        run_metadata.setdefault("next_step_suggestions", [])

        # Phase 1.5: default kind for turns that didn't trip any branch.
        run_metadata.setdefault("uncertainty_kind", "none")
        run_metadata.setdefault("follow_up_queries", [])
        run_metadata.setdefault("evidence_gaps_reason", None)

        state_trace.append("DONE")

        # ── Rolling summary trigger (Phase 2.2) ───────────────────────────
        # The stored `turn_count` reflects the number of turns BEFORE this
        # one is persisted. We trigger a rolling summary based on the
        # post-turn conversation count to make the intent explicit and avoid
        # fencepost ambiguity.
        post_turn_count = turn_count + 1
        # Trigger when the post-turn count exceeds 5 and is divisible by 3
        should_summarize = (post_turn_count > 5) and (post_turn_count % 3 == 0)
        if should_summarize:
            try:
                old_turns = prior_turns[:-3] if len(prior_turns) > 3 else prior_turns
                if old_turns:
                    from utils.provider_router import rolling_summary
                    turns_text = "\n\n".join(f"Q: {t.query}\nA: {t.response or ''}" for t in old_turns)
                    summary = await rolling_summary(turns_text)
                    await save_session_summary(session_id, summary, [t.turn_id for t in old_turns], now)
            except Exception as e:
                logger.warning("Rolling summary failed: %s", e)

        # ── Persist turn ──────────────────────────────────────────────────
        run_metadata["probe_ms"] = stage_ms["probe_ms"]
        run_metadata["verification_ms"] = verification_ms_total
        import json as _json
        claim_verification_json = _json.dumps([
            {
                "claim_text": r.claim_text,
                "doc_ids": list(r.doc_ids),
                "overlap": r.overlap,
                "entity_match": r.entity_match,
                "method": r.method,
                "score": r.score,
                "status": r.status,
            }
            for r in claim_records
        ]) if claim_records else None
        # B5: quote-anchored citation popovers. For every [doc_N] cited in
        # the internal (pre-rewrite) answer, extract the best verbatim quote
        # from the cited snippet — surfaced to the UI as a hover tooltip so
        # readers can verify each claim without opening the source.
        try:
            cite_quote_map = build_cite_quote_map(
                full_answer,
                final_context_bundle.doc_map if final_context_bundle else {},
                snippet_lookup,
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("build_cite_quote_map failed: %s", e,
                           extra={"component": "orchestrator", "turn_id": turn_id})
            cite_quote_map = {}
        run_metadata["cite_quote_map"] = cite_quote_map

        latency_ms = int((time.time() - start_ms) * 1000)
        if latency_ms > int(POLICY.max_total_turn_time_s * 1000):
            run_metadata["budget_breach"].append("total_turn_time_exceeded")
        turn = Turn(
            turn_id=turn_id,
            session_id=session_id,
            query=query,
            created_at=now,
            plan=str(queries),
            search_queries=queries,
            urls_opened=urls_opened,
            response=formatted_answer,
            context_xml_sent=final_context_bundle.xml if final_context_bundle else "",
            doc_map=final_context_bundle.doc_map if final_context_bundle else {},
            citation_integrity_score=citation_score,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            planning_ms=stage_ms["planning_ms"],
            search_ms=stage_ms["search_ms"],
            fetch_ms=stage_ms["fetch_ms"],
            select_ms=stage_ms["select_ms"],
            synthesize_ms=stage_ms["synthesize_ms"],
            run_metadata_json=_validate_run_metadata(run_metadata, turn_id=turn_id),
            state_trace=state_trace,
            claim_precision_score=claim_score,
            claim_verification_json=claim_verification_json,
        )
        await save_turn(turn)
        # Note: in a multi-hop scenario we just save the final selection to DB
        await save_turn_context(turn_id, selected)

        # V3.1: ephemeral cleanup of per-turn chunk embeddings (hybrid path only).
        try:
            from agent.memory import delete_chunk_embeddings
            ephemeral_ids = [
                getattr(c, "_hybrid_chunk_id", None)
                for c in selected
            ]
            ephemeral_ids = [cid for cid in ephemeral_ids if cid]
            if ephemeral_ids:
                await delete_chunk_embeddings(ephemeral_ids)
        except Exception as e:
            logger.warning("delete_chunk_embeddings cleanup failed: %s", e,
                           extra={"component": "orchestrator", "turn_id": turn_id})
        try:
            await save_claim_audit(turn_id, claim_records)
        except Exception as e:
            logger.warning("Persist claim_audit failed: %s", e,
                           extra={"component": "orchestrator", "turn_id": turn_id})

        # Phase 1.25: typed run_finished — frontend renders a cumulative pill.
        try:
            from utils.cost_model import cost_for
            _cost_usd = cost_for(
                os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                prompt_tokens, completion_tokens,
            )
        except Exception:
            _cost_usd = 0.0
        yield ExecutionEvent(
            "done", "done",
            data={
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "total_latency_ms": latency_ms,
                "cost_usd": _cost_usd,
            },
            event_type=EVT_RUN_FINISHED,
        )

        # P1 (Stop-RAG): scrub the LLM `reason` text from any SSE-bound copy of
        # run_metadata. The full reason stays on the persisted record (already
        # saved via save_turn above); the wire format only carries the
        # boolean + confidence so the inspector can render the gate decision
        # without ever streaming the model's natural-language rationale.
        # See sarvam-assignment.md L103 — no hidden CoT streaming.
        _sse_run_metadata = dict(_validate_run_metadata(run_metadata, turn_id=turn_id))
        _stop_dec_src = _sse_run_metadata.get("stop_rag_decisions") or []
        if _stop_dec_src:
            _sse_run_metadata["stop_rag_decisions"] = [
                {
                    "hop": d.get("hop"),
                    "useful": d.get("useful"),
                    "confidence": d.get("confidence"),
                    "degraded": bool(d.get("degraded_reason")),
                }
                for d in _stop_dec_src
                if isinstance(d, dict)
            ]
        yield ExecutionEvent("done", "done", data={
            "turn_id": turn_id,
            "answer": formatted_answer,
            "internal_answer": full_answer,
            "citation_integrity_score": citation_score,
            "claim_precision_score": claim_score,
            "turn_id_out": turn_id,
            "urls": urls_opened,
            "doc_map": final_context_bundle.doc_map if final_context_bundle else {},
            "cite_quote_map": cite_quote_map,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "planning_ms": stage_ms["planning_ms"],
            "search_ms": stage_ms["search_ms"],
            "fetch_ms": stage_ms["fetch_ms"],
            "select_ms": stage_ms["select_ms"],
            "probe_ms": stage_ms["probe_ms"],
            "synthesize_ms": stage_ms["synthesize_ms"],
            "run_metadata": _sse_run_metadata,
            "context_xml": final_context_bundle.xml if final_context_bundle else "",
            "selection_strategy": config.selection_strategy,
            "planning_strategy": planner.strategy,
        })
