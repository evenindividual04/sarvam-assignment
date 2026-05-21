"""
Evaluation runner — sequential execution to respect rate limits.
Writes JSONL incrementally. Prints summary table at end.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from utils.logging_config import setup_logging
setup_logging()

from agent.memory import init_db
from agent.orchestrator import ResearchOrchestrator
from eval.judge import (
    classify_failure, judge_citation_integrity, judge_claim_precision,
    judge_conflict_adherence, judge_coherence, judge_faithfulness,
    judge_relevance, judge_context_precision,
)

_DATASET = Path(__file__).parent / "dataset.json"
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)
_PER_QUESTION_TIMEOUT_S = int(os.getenv("EVAL_PER_QUESTION_TIMEOUT_S", "240"))


async def _persist_eval_run(result: dict) -> None:
    """Mirror a JSONL row into the eval_runs SQLite table for dashboard queries."""
    import aiosqlite

    from agent.memory import DB_PATH

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT OR REPLACE INTO eval_runs (
                    run_id, run_at, question_id, question, category, agent_answer,
                    faithfulness_score, answer_relevance_score, context_precision_score,
                    citation_integrity_score, conflict_adherence_score,
                    session_coherence_score, claim_precision_score, judge_reasoning,
                    failure_class, latency_ms, turn_id, language
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result["run_id"], result["run_at"], result["question_id"],
                    result["question"], result["category"], result.get("agent_answer", ""),
                    result.get("faithfulness_score"),
                    result.get("answer_relevance_score"),
                    result.get("context_precision_score"),
                    result.get("citation_integrity_score"),
                    result.get("conflict_adherence_score"),
                    result.get("session_coherence_score"),
                    result.get("claim_precision_score"),
                    result.get("claim_precision_reasoning", ""),
                    result.get("failure_class"),
                    result.get("latency_ms", 0),
                    result.get("turn_id") or None,
                    result.get("language", "en"),
                ),
            )
            await db.commit()
    except Exception as exc:  # pragma: no cover — DB errors should never break the run
        import logging
        logging.getLogger(__name__).warning("eval_runs persist failed: %s", exc)


async def run_eval() -> None:
    await init_db()

    with open(_DATASET) as f:
        questions = json.load(f)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = _RESULTS_DIR / f"eval_{run_ts}.jsonl"
    agent = ResearchOrchestrator()

    # Track multi-turn scenario context
    scenario_sessions: dict[str, tuple[str, str, str]] = {}  # scenario -> (session_id, t1_q, t1_a)
    results_summary = []

    print(f"\n{'='*72}")
    print(f"Deep Research Agent — Evaluation Run {run_ts}")
    print(f"{'='*72}")
    print(f"{'ID':<10} {'Category':<20} {'Faith':>6} {'Rel':>6} {'CitI':>6} {'CtxP':>6} {'Fail':<20}")
    print("-" * 79)

    with open(out_path, "w") as outf:
        for q in questions:
            qid = q["id"]
            category = q["category"]
            query = q["query"]
            language = q.get("language", "en")
            is_multiturn = q.get("is_multiturn", False)
            scenario = q.get("scenario")
            turn = q.get("turn", 1)

            # Determine session for multi-turn scenarios
            if is_multiturn and scenario:
                if scenario not in scenario_sessions:
                    session_id = str(uuid.uuid4())
                    scenario_sessions[scenario] = (session_id, "", "")
                session_id = scenario_sessions[scenario][0]
            else:
                session_id = str(uuid.uuid4())

            print(f"Running {qid}: {query[:50]}…")

            # Run agent
            answer = ""
            internal_answer = ""
            context_xml = ""
            doc_map = {}
            fetched_urls: set = set()
            turn_id_out: str = ""
            latency_ms = 0
            planning_ms = 0
            search_ms = 0
            fetch_ms = 0
            select_ms = 0
            synthesize_ms = 0
            run_metadata = {}

            try:
                async with asyncio.timeout(_PER_QUESTION_TIMEOUT_S):
                    async for event in agent.run(query, session_id):
                        if event.step == "generating" and isinstance(event.data, str):
                            answer += event.data
                        elif event.step == "done" and event.data:
                            d = event.data
                            answer = d.get("answer", answer)
                            internal_answer = d.get("internal_answer", answer)
                            context_xml = d.get("context_xml", "")
                            doc_map = d.get("doc_map", {}) or {}
                            latency_ms = d.get("latency_ms", 0) or 0
                            planning_ms = d.get("planning_ms", 0) or 0
                            search_ms = d.get("search_ms", 0) or 0
                            fetch_ms = d.get("fetch_ms", 0) or 0
                            select_ms = d.get("select_ms", 0) or 0
                            synthesize_ms = d.get("synthesize_ms", 0) or 0
                            run_metadata = d.get("run_metadata", {}) or {}
                            fetched_urls = set(d.get("urls", []))
                            turn_id_out = d.get("turn_id_out", "") or ""
            except TimeoutError:
                print(f"  TIMEOUT running agent after {_PER_QUESTION_TIMEOUT_S}s")
                answer = f"[Agent timeout after {_PER_QUESTION_TIMEOUT_S}s]"
                internal_answer = answer
            except Exception as e:
                print(f"  ERROR running agent: {e}")
                answer = f"[Agent error: {e}]"
                internal_answer = answer

            # ── Judge calls ──────────────────────────────────────────────
            faithfulness_score = 0.5
            relevance_score = 0.5
            context_precision_score = 0.5
            conflict_adherence_score = None
            coherence_score = None

            try:
                fres = await judge_faithfulness(context_xml or "(no context)", answer)
                faithfulness_score = fres.faithfulness_score
            except Exception as e:
                print(f"  Faithfulness judge error: {e}")

            try:
                rres = await judge_relevance(query, answer)
                relevance_score = rres.answer_relevance_score
            except Exception as e:
                print(f"  Relevance judge error: {e}")

            try:
                cp_res = await judge_context_precision(query, context_xml or "(no context)")
                context_precision_score = cp_res.context_precision_score
            except Exception as e:
                print(f"  Context Precision judge error: {e}")

            ci_res = judge_citation_integrity(internal_answer, doc_map, fetched_urls)

            claim_precision_score = 1.0
            claim_precision_reasoning = ""
            if turn_id_out:
                try:
                    cp = await judge_claim_precision(turn_id_out)
                    claim_precision_score = cp.claim_precision_score
                    claim_precision_reasoning = cp.reasoning
                except Exception as e:
                    print(f"  Claim Precision judge error: {e}")

            if category == "conflicting":
                try:
                    cres = await judge_conflict_adherence(query, context_xml or "", answer)
                    conflict_adherence_score = cres.conflict_adherence_score
                except Exception as e:
                    print(f"  Conflict judge error: {e}")

            if is_multiturn and scenario and turn == 2:
                t1_info = scenario_sessions.get(scenario)
                if t1_info and t1_info[1]:
                    try:
                        coh = await judge_coherence(t1_info[1], t1_info[2], query, answer)
                        coherence_score = coh.session_coherence_score
                    except Exception as e:
                        print(f"  Coherence judge error: {e}")

            # Store turn 1 data for multi-turn scenarios
            if is_multiturn and scenario and turn == 1:
                session_id_stored = scenario_sessions[scenario][0]
                scenario_sessions[scenario] = (session_id_stored, query, answer)

            result = {
                "run_id": str(uuid.uuid4()),
                "run_at": datetime.now(timezone.utc).isoformat(),
                "question_id": qid,
                "question": query,
                "category": category,
                "language": language,
                "agent_answer": answer[:1000],
                "faithfulness_score": faithfulness_score,
                "answer_relevance_score": relevance_score,
                "context_precision_score": context_precision_score,
                "citation_integrity_score": ci_res.citation_integrity_score,
                "claim_precision_score": claim_precision_score,
                "claim_precision_reasoning": claim_precision_reasoning,
                "turn_id": turn_id_out,
                "conflict_adherence_score": conflict_adherence_score,
                "session_coherence_score": coherence_score,
                "latency_ms": latency_ms,
                "planning_ms": planning_ms,
                "search_ms": search_ms,
                "fetch_ms": fetch_ms,
                "select_ms": select_ms,
                "synthesize_ms": synthesize_ms,
                "run_metadata": run_metadata,
                "failure_class": classify_failure({
                    "faithfulness_score": faithfulness_score,
                    "answer_relevance_score": relevance_score,
                    "context_precision_score": context_precision_score,
                    "citation_integrity_score": ci_res.citation_integrity_score,
                    "claim_precision_score": claim_precision_score,
                    "conflict_adherence_score": conflict_adherence_score,
                    "session_coherence_score": coherence_score,
                }),
            }
            outf.write(json.dumps(result) + "\n")
            outf.flush()
            results_summary.append(result)
            await _persist_eval_run(result)

            print(
                f"{qid:<10} {category:<20} "
                f"{faithfulness_score:>6.2f} {relevance_score:>6.2f} "
                f"{ci_res.citation_integrity_score:>6.2f} {context_precision_score:>6.2f} "
                f"{result['failure_class']:<20}"
            )

    await agent.aclose()

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    if results_summary:
        avg_faith = sum(r["faithfulness_score"] for r in results_summary) / len(results_summary)
        avg_rel = sum(r["answer_relevance_score"] for r in results_summary) / len(results_summary)
        avg_cp = sum(r["context_precision_score"] for r in results_summary) / len(results_summary)
        avg_ci = sum(r["citation_integrity_score"] for r in results_summary) / len(results_summary)
        passes = sum(1 for r in results_summary if r["failure_class"] == "PASS")
        print(f"Total questions: {len(results_summary)}")
        print(f"PASS: {passes} / {len(results_summary)}")
        print(f"Avg Faithfulness:         {avg_faith:.3f}")
        print(f"Avg Answer Relevance:     {avg_rel:.3f}")
        print(f"Avg Context Precision:    {avg_cp:.3f}")
        print(f"Avg Citation Integrity:   {avg_ci:.3f}")

    from collections import Counter
    taxonomy = Counter(r["failure_class"] for r in results_summary)
    print("\nFailure taxonomy:")
    for k, v in taxonomy.most_common():
        print(f"  {k}: {v}")

    # ── Per-language breakdown (V3.4) ──────────────────────────────────────
    languages = sorted({r.get("language", "en") for r in results_summary})
    if len(languages) > 1:
        print("\nPer-language breakdown:")
        print(
            f"  {'lang':<6}{'n':>4}{'pass%':>8}{'faith':>8}{'rel':>8}{'ctxP':>8}{'citI':>8}"
        )
        for lang in languages:
            rows = [r for r in results_summary if r.get("language", "en") == lang]
            n = len(rows)
            if n == 0:
                continue
            pass_pct = 100.0 * sum(1 for r in rows if r["failure_class"] == "PASS") / n
            avg_f = sum(r["faithfulness_score"] for r in rows) / n
            avg_r = sum(r["answer_relevance_score"] for r in rows) / n
            avg_cp = sum(r["context_precision_score"] for r in rows) / n
            avg_ci = sum(r["citation_integrity_score"] for r in rows) / n
            print(
                f"  {lang:<6}{n:>4}{pass_pct:>7.1f}%{avg_f:>8.2f}{avg_r:>8.2f}{avg_cp:>8.2f}{avg_ci:>8.2f}"
            )

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_eval())
