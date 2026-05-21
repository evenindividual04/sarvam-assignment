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
    detect_conflicts,
    format_context_xml,
    rank_and_select,
    rank_and_select_mmr,
)
from agent.extractor import Extractor
from agent.memory import (
    create_session, get_latest_summary, get_relevant_prior_turns,
    get_session_turn_count, save_session_summary, save_turn, save_turn_context, session_exists,
)
from agent.models import ContextBundle, ExecutionEvent, Turn
from agent.search import search
from utils.failure_policy import POLICY
from utils.prompt_registry import prompt_id
from utils.token_counter import ContextBudget

logger = logging.getLogger(__name__)

_MAX_ITER = int(os.getenv("AGENT_MAX_ITER", "5"))
_SELECTION_STRATEGY = os.getenv("CONTEXT_SELECTION_STRATEGY", "heuristic").strip().lower()

STREAM_LABELS = {
    "planning":  "Planning",
    "searching": "Searching the web",
    "fetching":  "Fetching sources",
    "selecting": "Selecting relevant context",
    "generating": "Generating answer with citations",
}

_UNCERTAINTY_MARKERS = [
    "insufficient", "unclear", "unable to find", "could not find",
    "no information", "suggested follow-up", "not enough",
]


def _extract_followup_queries(text: str) -> list[str]:
    """Parse deterministic follow-up query block from model output."""
    lines = text.splitlines()
    queries: list[str] = []
    capture = False
    for line in lines:
        lower = line.lower().strip()
        if "suggested follow-up searches" in lower:
            capture = True
            continue
        if not capture:
            continue
        stripped = line.strip()
        if stripped.startswith("-") or stripped.startswith("*"):
            q = stripped[1:].strip().strip('"').strip("'")
            if q:
                queries.append(q)
        elif stripped == "":
            continue
        elif queries:
            break
    return queries[:3]


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

    async def run(self, query: str, session_id: str) -> AsyncIterator[ExecutionEvent]:
        """
        Full research turn. Yields ExecutionEvents.
        Saves turn to DB at the end.
        """
        start_ms = time.time()
        turn_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        state_trace: list[str] = []
        budget = ContextBudget()
        stage_ms = {"planning_ms": 0, "search_ms": 0, "fetch_ms": 0, "select_ms": 0, "synthesize_ms": 0}
        run_metadata = {
            "selection_strategy": _SELECTION_STRATEGY,
            "planner_prompt_id": prompt_id("planner"),
            "synth_prompt_id": prompt_id("synthesizer"),
            "conflict_prompt_id": prompt_id("conflict_detection"),
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
        state_trace.append("PLANNING")
        yield ExecutionEvent("planning", STREAM_LABELS["planning"])
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

        for hop in range(_MAX_ITER):
            # ── SEARCHING ─────────────────────────────────────────────────────
            state_trace.append("SEARCHING")
            yield ExecutionEvent("searching", STREAM_LABELS["searching"])
            t0 = time.time()
            try:
                results = await asyncio.wait_for(search(typed_queries), timeout=POLICY.search_timeout_s)
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
            state_trace.append("FETCHING")
            yield ExecutionEvent("fetching", STREAM_LABELS["fetching"])
            extractor = Extractor()
            t0 = time.time()
            try:
                extracted = await asyncio.wait_for(extractor.extract_all(results), timeout=POLICY.fetch_timeout_s)
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

            if not selected:
                # Empty context guard: inform model explicitly
                logger.warning("No context selected", extra={"component": "orchestrator", "turn_id": turn_id})
                empty_note = "(No relevant web content could be retrieved for this query.)"
                context_bundle = ContextBundle(
                    xml=f"<context>{empty_note}</context>",
                    doc_map={},
                    fetched_urls=set(),
                )
            else:
                conflict = await detect_conflicts(selected, query)
                xml, doc_map = format_context_xml(selected)
                if conflict.has_conflict and conflict.conflict_summary:
                    xml = xml.replace(
                        "<context>",
                        f"<context>\n  <conflict_warning>{conflict.conflict_summary}</conflict_warning>",
                        1,
                    )
                context_bundle = ContextBundle(
                    xml=xml,
                    doc_map=doc_map,
                    fetched_urls=set(urls_opened),
                    conflict_summary=conflict.conflict_summary if conflict.has_conflict else None,
                )

            final_context_bundle = context_bundle

            # ── SYNTHESIZING ──────────────────────────────────────────────────
            state_trace.append("SYNTHESIZING")
            yield ExecutionEvent("generating", STREAM_LABELS["generating"])

            full_answer = ""
            hop_prompt_tokens = 0
            hop_completion_tokens = 0

            t0 = time.time()
            try:
                from agent.synthesizer import stream_synthesis
                async def _collect_stream() -> None:
                    nonlocal full_answer, hop_prompt_tokens, hop_completion_tokens
                    async for text_chunk, pt, ct in stream_synthesis(
                        query=query,
                        context_xml=context_bundle.xml,
                        doc_map=context_bundle.doc_map,
                        history_text=history_text,
                        conflict_note=context_bundle.conflict_summary,
                    ):
                        full_answer += text_chunk
                        if pt:
                            hop_prompt_tokens = pt
                        if ct:
                            hop_completion_tokens = ct
                        if text_chunk:
                            yield_event.append(text_chunk)

                yield_event: list[str] = []
                async def _runner():
                    await _collect_stream()
                await asyncio.wait_for(_runner(), timeout=POLICY.synth_timeout_s)
                for chunk_text in yield_event:
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

            # ── Post-processing: citation guard + format conversion ────────────
            citation_score = self._guard.verify(full_answer, context_bundle.doc_map, context_bundle.fetched_urls)
            formatted_answer = convert_citations(full_answer, context_bundle.doc_map)

            # ── Iterative re-search (Phase 2.3) ───────────────────────────────
            lower_ans = full_answer.lower()
            is_uncertain = any(marker in lower_ans for marker in _UNCERTAINTY_MARKERS)
            
            if is_uncertain and hop < _MAX_ITER - 1:
                new_queries = _extract_followup_queries(full_answer)
                if new_queries:
                    queries = new_queries
                    typed_queries = [TypedQuery(text=q, intent=QueryIntent.PRIMARY) for q in new_queries]
                    state_trace.append(f"RE-SEARCH (hop {hop+1})")
                    yield ExecutionEvent(
                        "planning",
                        STREAM_LABELS["planning"],
                        data={"strategy": "Follow-up uncertainty search", "queries": queries},
                    )
                    continue
                else:
                    break
            else:
                break
        
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
        )
        await save_turn(turn)
        # Note: in a multi-hop scenario we just save the final selection to DB
        await save_turn_context(turn_id, selected if 'selected' in locals() else [])

        yield ExecutionEvent("done", "done", data={
            "turn_id": turn_id,
            "answer": formatted_answer,
            "internal_answer": full_answer,
            "citation_integrity_score": citation_score,
            "urls": urls_opened,
            "doc_map": final_context_bundle.doc_map if final_context_bundle else {},
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "planning_ms": stage_ms["planning_ms"],
            "search_ms": stage_ms["search_ms"],
            "fetch_ms": stage_ms["fetch_ms"],
            "select_ms": stage_ms["select_ms"],
            "synthesize_ms": stage_ms["synthesize_ms"],
            "run_metadata": run_metadata,
            "context_xml": final_context_bundle.xml if final_context_bundle else "",
            "selection_strategy": _SELECTION_STRATEGY,
            "planning_strategy": planner.strategy,
        })
