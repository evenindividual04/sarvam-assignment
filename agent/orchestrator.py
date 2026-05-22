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

from agent.citation_guard import CitationGuard, convert_citations
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
    ConflictResult, ContextBundle, ExecutionEvent, Turn,
    EVT_RUN_STARTED, EVT_PHASE_STARTED, EVT_PHASE_PROGRESS, EVT_PHASE_FINISHED,
    EVT_SEARCH_QUERY, EVT_SOURCE_FOUND, EVT_SOURCE_FETCHED, EVT_CONTEXT_SELECTED,
    EVT_CONFLICT_DETECTED, EVT_ANSWER_DELTA, EVT_CITATION_RESOLVED,
    EVT_RUN_FINISHED, EVT_RUN_ERROR, EVT_UNCERTAINTY,
    EVT_CLARIFICATION_OFFERED, EVT_EVIDENCE_GAP, EVT_PLAN_APPROVAL,
)
from utils.next_steps import propose_follow_ups
from agent.search import search
from utils.cancellation import CancellationToken, OperationCancelledError
from utils.failure_policy import POLICY
from utils.prompt_registry import prompt_id
from utils.token_counter import ContextBudget

logger = logging.getLogger(__name__)

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

        # Phase 1.875: if the planner flagged the question as ambiguous, emit
        # a `clarification_offered` SSE event between PLANNING and SEARCHING.
        # Phase 2 will add a real approval-gate block; for now this is a pure
        # notification so the frontend can surface a non-blocking prompt.
        if getattr(planner, "ambiguity_flag", False):
            interpretations = list(getattr(planner, "success_criteria", []) or [])
            if not interpretations:
                interpretations = [tq.text for tq in typed_queries[:3]]
            yield ExecutionEvent(
                "planning", STREAM_LABELS["planning"],
                data={
                    "kind": "ambiguity",
                    "original_query": query,
                    "possible_interpretations": interpretations[:3],
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

            # ── V3.2 ADAPTIVE HOP GATE ────────────────────────────────────
            # After SELECTING: decide whether to run a second hop. Phase 1.875
            # layers a token-budget terminator on top, in the spirit of Jina's
            # node-DeepResearch token-aware stopping (cumulative consumption
            # vs total budget).
            if hop + 1 >= max_hops:
                run_metadata.setdefault("terminator_fired", "MAX_HOPS_REACHED")
                break
            from utils.token_counter import count_tokens as _count_tokens
            selected_tokens = sum(s.token_count for s in (selected or [])) or _count_tokens(
                "\n".join(s.text for s in (selected or []))
            )
            context_thin = selected_tokens < int(0.5 * budget.web_context_budget)
            # Cumulative consumption so far: any synthesis tokens spent
            # (zero on hop 1) plus the snippet payload — count whichever pool
            # is larger so we capture both the unfiltered retrieval volume
            # AND any heavy single-snippet selection that already pinned the
            # context budget.
            all_chunk_tokens = sum(
                getattr(c, "token_count", 0) or 0 for c in (all_chunks or [])
            )
            cumulative = (
                prompt_tokens + completion_tokens
                + max(all_chunk_tokens, selected_tokens)
            )
            token_threshold = int(0.75 * budget.total_tokens)
            if cumulative > token_threshold:
                run_metadata.setdefault("terminator_fired", "TOKEN_BUDGET_EXHAUSTED")
                break
            # Difficulty=easy short-circuits the low-confidence trigger.
            if plan_difficulty == "easy":
                run_metadata.setdefault("terminator_fired", "DIFFICULTY_EASY_SKIPPED")
                break
            if planner.confidence != "low":
                run_metadata.setdefault("terminator_fired", "CONFIDENCE_HIGH_ENOUGH")
                break
            if not context_thin:
                run_metadata.setdefault("terminator_fired", "EVIDENCE_SUFFICIENT")
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
            stage_ms["synthesize_ms"] += int((time.time() - t0) * 1000)
            yield _phase_finished_event(
                "generating", stage_ms["synthesize_ms"],
                {"prompt_tokens": hop_prompt_tokens, "completion_tokens": hop_completion_tokens},
            )

        prompt_tokens += hop_prompt_tokens
        completion_tokens += hop_completion_tokens

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
        criteria_coverage: list[bool] = []
        if plan_success_criteria:
            import re as _re_crit
            _ans_lc = (full_answer or "").lower()
            _ans_tokens = set(_re_crit.findall(r"\w+", _ans_lc))
            for crit in plan_success_criteria:
                if not crit or not isinstance(crit, str):
                    criteria_coverage.append(False)
                    continue
                c_lc = crit.lower().strip()
                if c_lc and c_lc in _ans_lc:
                    criteria_coverage.append(True)
                    continue
                ctoks = set(_re_crit.findall(r"\w+", c_lc))
                if not ctoks:
                    criteria_coverage.append(False)
                    continue
                overlap = len(ctoks & _ans_tokens) / max(1, len(ctoks))
                criteria_coverage.append(overlap >= 0.6)
        run_metadata["criteria_coverage"] = criteria_coverage

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
            run_metadata_json=run_metadata,
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

        yield ExecutionEvent("done", "done", data={
            "turn_id": turn_id,
            "answer": formatted_answer,
            "internal_answer": full_answer,
            "citation_integrity_score": citation_score,
            "claim_precision_score": claim_score,
            "turn_id_out": turn_id,
            "urls": urls_opened,
            "doc_map": final_context_bundle.doc_map if final_context_bundle else {},
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "planning_ms": stage_ms["planning_ms"],
            "search_ms": stage_ms["search_ms"],
            "fetch_ms": stage_ms["fetch_ms"],
            "select_ms": stage_ms["select_ms"],
            "probe_ms": stage_ms["probe_ms"],
            "synthesize_ms": stage_ms["synthesize_ms"],
            "run_metadata": run_metadata,
            "context_xml": final_context_bundle.xml if final_context_bundle else "",
            "selection_strategy": config.selection_strategy,
            "planning_strategy": planner.strategy,
        })
