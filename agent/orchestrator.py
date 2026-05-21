"""
Bounded async state machine:
  PLANNING → SEARCHING → FETCHING → SELECTING → SYNTHESIZING → DONE | ERROR

Yields ExecutionEvent at each step with exact STREAM_LABELS from spec Section 2.10.
Max iterations: AGENT_MAX_ITER (env var, default 5).
Saves complete turn to DB including context_xml_sent, doc_map, token counts.
"""
from __future__ import annotations

import asyncio
import logging
import os
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
    rank_and_select_mmr,
)
from agent.extractor import Extractor
from agent.claim_verifier import verify_claims
from agent.memory import (
    create_session, get_latest_summary, get_relevant_prior_turns,
    get_session_turn_count, save_claim_audit, save_contradiction_probe,
    save_session_summary, save_turn, save_turn_context, session_exists,
)
from agent.models import ConflictResult, ContextBundle, ExecutionEvent, Turn
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

STREAM_LABELS = {
    "planning":  "Planning",
    "searching": "Searching the web",
    "fetching":  "Fetching sources",
    "selecting": "Selecting relevant context",
    "probing":   "Probing for cross-source contradictions",
    "generating": "Generating answer with citations",
    "verifying": "Verifying claims against sources",
}

_UNCERTAINTY_MARKERS = [
    "insufficient", "unclear", "unable to find", "could not find",
    "no information", "suggested follow-up", "not enough",
]


def _ensure_uncertainty_block(answer: str, fallback_queries: list[str]) -> str:
    """Append deterministic uncertainty block when model omitted it."""
    if "[UNCERTAINTY]" in answer and "Suggested follow-up searches" in answer:
        return answer
    q = [x for x in fallback_queries if x.strip()][:3]
    while len(q) < 3:
        q.append("authoritative source for latest verified value")
    tail = (
        "\n\n[UNCERTAINTY] Retrieved context does not provide enough verified evidence.\n"
        "Suggested follow-up searches:\n"
        f"- {q[0]}\n- {q[1]}\n- {q[2]}"
    )
    return answer.rstrip() + tail


class ResearchOrchestrator:
    def __init__(self) -> None:
        self._guard = CitationGuard()

    async def aclose(self) -> None:
        pass

    async def run(
        self,
        query: str,
        session_id: str,
        cancel_token: CancellationToken | None = None,
        turn_id: str | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """
        Full research turn. Yields ExecutionEvents. Saves turn to DB at end.

        On cancellation (cancel_token fires), persists a partial Turn marked
        '[Cancelled by user]', appends 'CANCELLED' to state_trace, yields one
        final 'error' event, and returns. Does not re-raise.
        """
        if turn_id is None:
            turn_id = str(uuid.uuid4())
        state_holder: dict = {"state_trace": [], "now": datetime.now(timezone.utc).isoformat()}
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
            yield ExecutionEvent("error", "cancelled", data="Cancelled by user.")
            return

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
            run_metadata_json={"cancelled": True},
            state_trace=state.get("state_trace", []),
        )
        await save_turn(turn)

    async def _run_body(
        self,
        query: str,
        session_id: str,
        cancel_token: CancellationToken | None,
        turn_id: str,
        state_holder: dict,
    ) -> AsyncIterator[ExecutionEvent]:
        start_ms = time.time()

        def _ck() -> None:
            if cancel_token is not None:
                cancel_token.check()

        now = state_holder["now"]
        state_trace: list[str] = state_holder["state_trace"]
        budget = ContextBudget()
        stage_ms = {"planning_ms": 0, "search_ms": 0, "fetch_ms": 0, "select_ms": 0, "probe_ms": 0, "synthesize_ms": 0}
        run_metadata = {
            "selection_strategy": _SELECTION_STRATEGY,
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
        from utils.token_counter import count_tokens
        while history_parts and count_tokens(history_text) > budget.history_budget:
            if len(history_parts) > 1 and not history_parts[0].startswith("[Summary"):
                history_parts.pop(0)
            elif len(history_parts) > 1:
                history_parts.pop(1)
            else:
                history_text = history_parts[0][: budget.history_budget * 3]
                break
            history_text = "\n\n".join(history_parts)

        # ── PLANNING ──────────────────────────────────────────────────────
        _ck()
        state_trace.append("PLANNING")
        yield ExecutionEvent("planning", STREAM_LABELS["planning"], data={"turn_id": turn_id})
        t0 = time.time()
        try:
            from utils.provider_router import plan
            planner = await asyncio.wait_for(
                plan(query, prior_summary=history_text or "No prior context."),
                timeout=POLICY.plan_timeout_s,
            )
        except asyncio.TimeoutError:
            run_metadata["timeout_hits"].append("planning")
            run_metadata["fallback_path_taken"].append("planning_timeout_fallback")
            run_metadata["budget_breach"].append("planning_timeout")
            from agent.models import PlannerOutput, QueryIntent, TypedQuery
            planner = PlannerOutput(
                strategy="Direct retrieval fallback",
                queries=[TypedQuery(text=query, intent=QueryIntent.PRIMARY)],
            )
        except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ConnectError, Exception) as e:
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
            "queries": [
                {"text": tq.text, "intent": tq.intent.value, "rationale": tq.rationale}
                for tq in typed_queries
            ],
        }

        # Adapt budget based on planning complexity
        budget.adapt_to_complexity(len(queries))

        yield ExecutionEvent(
            "planning",
            STREAM_LABELS["planning"],
            data={"strategy": planner.strategy, "queries": queries},
        )

        urls_opened = []
        all_chunks = []
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
        max_hops = max(1, min(POLICY.max_hops, _MAX_ITER))

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
            state_holder["queries"] = queries
            state_holder["urls_opened"] = urls_opened
            t0 = time.time()
            try:
                results = await asyncio.wait_for(
                    search(typed_queries, cancel_token=cancel_token),
                    timeout=POLICY.search_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error("Search timed out", extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["timeout_hits"].append("search")
                run_metadata["fallback_path_taken"].append("search_timeout_empty")
                run_metadata["budget_breach"].append("search_timeout")
                results = []
            except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ConnectError, Exception) as e:
                logger.error("Search failed: %s", e, extra={"component": "orchestrator", "turn_id": turn_id})
                run_metadata["fallback_path_taken"].append("search_error_empty")
                results = []
            stage_ms["search_ms"] += int((time.time() - t0) * 1000)

            new_urls = [r.url for r in results if r.url not in urls_opened]
            urls_opened.extend(new_urls)

            # ── FETCHING ──────────────────────────────────────────────────────
            _ck()
            state_trace.append("FETCHING")
            yield ExecutionEvent("fetching", STREAM_LABELS["fetching"])
            extractor = Extractor()
            t0 = time.time()
            try:
                extracted = await asyncio.wait_for(
                    extractor.extract_all(results, cancel_token=cancel_token),
                    timeout=POLICY.fetch_timeout_s,
                )
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
            finally:
                await extractor.aclose()
            stage_ms["fetch_ms"] += int((time.time() - t0) * 1000)

            # Build chunks from extracted text
            for r in results:
                text = extracted.get(r.url) or r.raw_content or r.snippet
                if text:
                    all_chunks.extend(chunk(r, text))

            # ── SELECTING ─────────────────────────────────────────────────────
            _ck()
            state_trace.append("SELECTING")
            yield ExecutionEvent("selecting", STREAM_LABELS["selecting"])
            t0 = time.time()
            try:
                if _SELECTION_STRATEGY == "mmr":
                    selected = await asyncio.wait_for(
                        asyncio.to_thread(rank_and_select_mmr, query, all_chunks, budget.web_context_budget),
                        timeout=POLICY.select_timeout_s,
                    )
                else:
                    selected = await asyncio.wait_for(
                        asyncio.to_thread(rank_and_select, query, all_chunks, budget.web_context_budget),
                        timeout=POLICY.select_timeout_s,
                    )
            except asyncio.TimeoutError:
                run_metadata["timeout_hits"].append("select")
                run_metadata["fallback_path_taken"].append("select_timeout_heuristic")
                run_metadata["budget_breach"].append("select_timeout")
                selected = rank_and_select(query, all_chunks, max_tokens=budget.web_context_budget)
            stage_ms["select_ms"] += int((time.time() - t0) * 1000)

            # ── V3.2 ADAPTIVE HOP GATE ────────────────────────────────────
            # After SELECTING: decide whether to run a second hop.
            if hop + 1 >= max_hops:
                break
            from utils.token_counter import count_tokens as _count_tokens
            selected_tokens = sum(s.token_count for s in (selected or [])) or _count_tokens(
                "\n".join(s.text for s in (selected or []))
            )
            context_thin = selected_tokens < int(0.5 * budget.web_context_budget)
            if planner.confidence != "low" or not context_thin:
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

        # ── CONFLICT_CHECK + Build final context bundle (once) ──────────
        if not selected:
            logger.warning("No context selected", extra={"component": "orchestrator", "turn_id": turn_id})
            empty_note = "(No relevant web content could be retrieved for this query.)"
            context_bundle = ContextBundle(
                xml=f"<context>{empty_note}</context>",
                doc_map={},
                fetched_urls=set(),
            )
        else:
            xml, doc_map = format_context_xml(selected)
            _ck()
            state_trace.append("CONFLICT_CHECK")
            yield ExecutionEvent("probing", STREAM_LABELS["probing"])
            t_probe = time.time()
            try:
                conflict_result = await probe_contradictions(selected, query)
            except Exception as e:
                logger.warning("Probe unexpectedly raised: %s", e,
                               extra={"component": "orchestrator", "turn_id": turn_id})
                conflict_result = ConflictResult(has_conflict=False, probe_skipped_reason="parse_fail")
            probe_ms_hop = int((time.time() - t_probe) * 1000)
            stage_ms["probe_ms"] += probe_ms_hop
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
            context_bundle = ContextBundle(
                xml=xml,
                doc_map=doc_map,
                fetched_urls=set(urls_opened),
                conflict_summary=conflict_result.conflict_summary if conflict_result.has_conflict else None,
            )

        final_context_bundle = context_bundle
        state_holder["final_context_bundle"] = final_context_bundle

        # ── SYNTHESIZING (once, after retrieval loop) ───────────────────
        _ck()
        state_trace.append("SYNTHESIZING")
        yield ExecutionEvent("generating", STREAM_LABELS["generating"])
        full_answer = ""
        hop_prompt_tokens = 0
        hop_completion_tokens = 0
        t0 = time.time()
        try:
            from agent.synthesizer import stream_synthesis
            yield_event: list[str] = []

            async def _collect_stream() -> None:
                nonlocal full_answer, hop_prompt_tokens, hop_completion_tokens
                async for text_chunk, pt, ct in stream_synthesis(
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
                    full_answer += text_chunk
                    if pt:
                        hop_prompt_tokens = pt
                    if ct:
                        hop_completion_tokens = ct
                    if text_chunk:
                        yield_event.append(text_chunk)

            await asyncio.wait_for(_collect_stream(), timeout=POLICY.synth_timeout_s)
            for chunk_text in yield_event:
                _ck()
                yield ExecutionEvent("generating", STREAM_LABELS["generating"], data=chunk_text)
        except asyncio.TimeoutError:
            logger.error("Synthesis timed out", extra={"component": "orchestrator", "turn_id": turn_id})
            run_metadata["timeout_hits"].append("synthesize")
            run_metadata["fallback_path_taken"].append("synthesize_timeout_error")
            run_metadata["budget_breach"].append("synthesize_timeout")
            full_answer = "Synthesis timed out. Partial context retrieved."
        except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ConnectError, Exception) as e:
            logger.error("Synthesis failed: %s", e, extra={"component": "orchestrator", "turn_id": turn_id})
            run_metadata["fallback_path_taken"].append("synthesize_error")
            full_answer = f"Synthesis error: {e}. Retrieved {len(selected)} context chunks."
        stage_ms["synthesize_ms"] += int((time.time() - t0) * 1000)

        prompt_tokens += hop_prompt_tokens
        completion_tokens += hop_completion_tokens

        # ── VERIFYING CLAIMS (V2.4) ─────────────────────────────────────
        _ck()
        snippet_lookup = {s.doc_id: s.text for s in (selected or [])}
        if snippet_lookup and context_bundle.doc_map:
            state_trace.append("VERIFYING_CLAIMS")
            yield ExecutionEvent("verifying", STREAM_LABELS["verifying"])
            t_v = time.time()
            claim_score, claim_records, full_answer = await verify_claims(
                full_answer, context_bundle.doc_map, snippet_lookup,
            )
            verification_ms_hop = int((time.time() - t_v) * 1000)
            verification_ms_total += verification_ms_hop
            state_trace.append(
                "VERIFICATION_DONE" if claim_records else "VERIFICATION_SKIPPED"
            )
        else:
            state_trace.append("VERIFICATION_SKIPPED")

        # ── Post-processing: citation guard + format conversion ────────
        citation_score = self._guard.verify(full_answer, context_bundle.doc_map, context_bundle.fetched_urls)
        formatted_answer = convert_citations(full_answer, context_bundle.doc_map)

        # V3.2: existing uncertainty markers are now an OBSERVABILITY signal
        # only — they no longer trigger re-search. Logged for downstream eval.
        lower_ans_obs = full_answer.lower()
        if any(marker in lower_ans_obs for marker in _UNCERTAINTY_MARKERS):
            run_metadata.setdefault("observed_signals", []).append("uncertainty_markers")

        # Enforce deterministic uncertainty contract if model missed required block.
        lower_ans = full_answer.lower()
        if any(marker in lower_ans for marker in _UNCERTAINTY_MARKERS):
            full_answer = _ensure_uncertainty_block(full_answer, queries)
            formatted_answer = convert_citations(full_answer, final_context_bundle.doc_map if final_context_bundle else {})

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
        await save_turn_context(turn_id, selected if 'selected' in locals() else [])

        # V3.1: ephemeral cleanup of per-turn chunk embeddings (hybrid path only).
        try:
            from agent.memory import delete_chunk_embeddings
            ephemeral_ids = [
                getattr(c, "_hybrid_chunk_id", None)
                for c in (selected if 'selected' in locals() else [])
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
            "selection_strategy": _SELECTION_STRATEGY,
            "planning_strategy": planner.strategy,
        })
