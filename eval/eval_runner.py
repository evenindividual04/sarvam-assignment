"""
Evaluation runner — sequential execution to respect rate limits.
Writes JSONL incrementally. Prints summary table at end.

Flags:
  --ablate    Run twice (BM25 then Hybrid) tagged with a shared ablation_id,
              then emit eval/results/ablation_<id>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from utils.logging_config import setup_logging
setup_logging()

from agent.memory import (
    DB_PATH,
    init_db,
    save_cross_language_consistency,
    save_eval_run_summary,
)
from agent.orchestrator import ResearchOrchestrator
from eval.judge import (
    classify_failure,
    judge_citation_integrity,
    judge_claim_precision,
    judge_conflict_adherence,
    judge_coherence,
    judge_cross_language_consistency,
    judge_factual_accuracy,
    judge_faithfulness,
    judge_relevance,
    judge_context_precision,
)
from utils.cost_model import DEFAULT_MODEL, cost_for

logger = logging.getLogger(__name__)

_DATASET = Path(__file__).parent / "dataset.json"
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)
_PER_QUESTION_TIMEOUT_S = int(os.getenv("EVAL_PER_QUESTION_TIMEOUT_S", "240"))


async def _persist_eval_run(result: dict) -> None:
    """Mirror a JSONL row into the eval_runs SQLite table for dashboard queries."""
    import aiosqlite

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT OR REPLACE INTO eval_runs (
                    run_id, run_at, question_id, question, category, agent_answer,
                    faithfulness_score, answer_relevance_score, context_precision_score,
                    citation_integrity_score, conflict_adherence_score,
                    session_coherence_score, claim_precision_score, judge_reasoning,
                    failure_class, latency_ms, turn_id, language, retrieval_mode,
                    factual_accuracy_score, ablation_id, calibration_correlation
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    result.get("retrieval_mode", "bm25"),
                    result.get("factual_accuracy_score"),
                    result.get("ablation_id"),
                    result.get("calibration_correlation"),
                ),
            )
            await db.commit()
    except Exception as exc:  # pragma: no cover — DB errors should never break the run
        logger.warning("eval_runs persist failed: %s", exc)


async def _update_calibration_correlation(run_at: str, correlation: float) -> None:
    """After post-run calibration is computed, fan the single correlation value
    out to every row in this run for downstream queries."""
    import aiosqlite
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE eval_runs SET calibration_correlation = ? WHERE run_at = ?",
            (correlation, run_at),
        )
        await db.commit()


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    """Pearson correlation coefficient, manual implementation (no scipy)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


_CONF_ORDINAL = {"low": 1, "medium": 2, "high": 3}


def _compute_calibration(results: list[dict]) -> dict:
    """Bucket per planner.confidence; compute means and Pearson correlation."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    paired_for_corr: list[tuple[int, float]] = []
    for r in results:
        meta = r.get("run_metadata") or {}
        planner = meta.get("planner_output") or {}
        conf = planner.get("confidence")
        if conf not in _CONF_ORDINAL:
            continue
        buckets[conf].append(r)
        if r.get("faithfulness_score") is not None:
            paired_for_corr.append((_CONF_ORDINAL[conf], r["faithfulness_score"]))

    bucket_stats: dict[str, dict] = {}
    for conf in ("low", "medium", "high"):
        rs = buckets.get(conf, [])
        if not rs:
            bucket_stats[conf] = {"n": 0}
            continue
        bucket_stats[conf] = {
            "n": len(rs),
            "mean_faithfulness": mean(r["faithfulness_score"] for r in rs),
            "mean_claim_precision": mean(
                r["claim_precision_score"] for r in rs
                if r.get("claim_precision_score") is not None
            ) if any(r.get("claim_precision_score") is not None for r in rs) else None,
            "pass_rate": sum(1 for r in rs if r["failure_class"] == "PASS") / len(rs),
        }

    correlation = None
    if len(paired_for_corr) >= 2:
        xs = [float(p[0]) for p in paired_for_corr]
        ys = [p[1] for p in paired_for_corr]
        correlation = _pearson(xs, ys)

    return {"buckets": bucket_stats, "correlation": correlation, "n_paired": len(paired_for_corr)}


def _percentile(values: list[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile, no numpy dependency."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * pct
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _compute_aggregates(results: list[dict]) -> dict:
    """Means, P50/P95 latency, per-category breakdown."""
    if not results:
        return {}

    def _safe(rs: list[dict], key: str) -> Optional[float]:
        vals = [r[key] for r in rs if r.get(key) is not None]
        return mean(vals) if vals else None

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    metric_keys = [
        "faithfulness_score", "answer_relevance_score", "context_precision_score",
        "citation_integrity_score", "claim_precision_score", "factual_accuracy_score",
    ]

    overall = {
        "n_questions": len(results),
        "pass_rate": sum(1 for r in results if r["failure_class"] == "PASS") / len(results),
        "p50_latency_ms": int(_percentile(latencies, 0.50) or 0),
        "p95_latency_ms": int(_percentile(latencies, 0.95) or 0),
        "total_cost_usd": sum(r.get("cost_usd", 0.0) or 0.0 for r in results),
    }
    for k in metric_keys:
        overall[f"mean_{k.replace('_score', '')}"] = _safe(results, k)
        vals = [r[k] for r in results if r.get(k) is not None]
        if vals:
            overall[f"p50_{k.replace('_score', '')}"] = _percentile(vals, 0.50)
            overall[f"p95_{k.replace('_score', '')}"] = _percentile(vals, 0.95)

    categories: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in results}):
        rs = [r for r in results if r["category"] == cat]
        categories[cat] = {
            "n": len(rs),
            "pass_rate": sum(1 for r in rs if r["failure_class"] == "PASS") / len(rs),
            **{f"mean_{k.replace('_score', '')}": _safe(rs, k) for k in metric_keys},
        }

    return {"overall": overall, "by_category": categories}


def _print_aggregates(agg: dict, taxonomy: Counter) -> None:
    print(f"\n{'='*72}")
    print("AGGREGATES")
    print(f"{'='*72}")
    o = agg.get("overall", {})
    print(f"n={o.get('n_questions', 0)}  pass_rate={o.get('pass_rate', 0):.2%}  "
          f"p50_latency={o.get('p50_latency_ms', 0)}ms  p95_latency={o.get('p95_latency_ms', 0)}ms  "
          f"cost=${o.get('total_cost_usd', 0):.4f}")
    print(f"\n{'metric':<28}{'mean':>10}{'p50':>10}{'p95':>10}")
    print("-" * 58)
    for k in ("faithfulness", "relevance", "context_precision", "citation_integrity",
              "claim_precision", "factual_accuracy"):
        m = o.get(f"mean_{k}"); p50 = o.get(f"p50_{k}"); p95 = o.get(f"p95_{k}")
        m_s = f"{m:.3f}" if m is not None else "—"
        p50_s = f"{p50:.3f}" if p50 is not None else "—"
        p95_s = f"{p95:.3f}" if p95 is not None else "—"
        print(f"{k:<28}{m_s:>10}{p50_s:>10}{p95_s:>10}")

    print("\nFailure taxonomy:")
    for k, v in taxonomy.most_common():
        print(f"  {k}: {v}")


async def run_eval(ablation_id: Optional[str] = None) -> str:
    """Run the full dataset once. Returns the run_at timestamp."""
    await init_db()

    with open(_DATASET) as f:
        questions = json.load(f)

    retrieval_mode = "hybrid" if os.environ.get("HYBRID_RETRIEVAL") == "1" else "bm25"
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_at_iso = datetime.now(timezone.utc).isoformat()
    out_path = _RESULTS_DIR / f"eval_{run_ts}.jsonl"
    agent = ResearchOrchestrator()

    scenario_sessions: dict[str, tuple[str, str, str]] = {}
    results_summary: list[dict] = []

    print(f"\n{'='*72}")
    print(f"Deep Research Agent — Evaluation Run {run_ts}  (mode={retrieval_mode}"
          + (f", ablation_id={ablation_id}" if ablation_id else "") + ")")
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

            if is_multiturn and scenario:
                if scenario not in scenario_sessions:
                    session_id = str(uuid.uuid4())
                    scenario_sessions[scenario] = (session_id, "", "")
                session_id = scenario_sessions[scenario][0]
            else:
                session_id = str(uuid.uuid4())

            print(f"Running {qid}: {query[:50]}…")

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
            prompt_tokens = 0
            completion_tokens = 0

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
                            prompt_tokens = d.get("prompt_tokens", 0) or 0
                            completion_tokens = d.get("completion_tokens", 0) or 0
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
            factual_accuracy_score: Optional[float] = None
            factual_accuracy_reasoning = ""

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

            if "gold_answer" in q:
                try:
                    factual_accuracy_score, factual_accuracy_reasoning = judge_factual_accuracy(
                        answer,
                        q.get("gold_answer"),
                        q.get("gold_aliases"),
                        q.get("gold_entities"),
                    )
                except Exception as e:
                    print(f"  Factual judge error: {e}")

            if is_multiturn and scenario and turn == 1:
                session_id_stored = scenario_sessions[scenario][0]
                scenario_sessions[scenario] = (session_id_stored, query, answer)

            cost_usd = cost_for(DEFAULT_MODEL, prompt_tokens, completion_tokens)

            result = {
                "run_id": str(uuid.uuid4()),
                "run_at": run_at_iso,
                "question_id": qid,
                "question": query,
                "category": category,
                "language": language,
                "retrieval_mode": retrieval_mode,
                "ablation_id": ablation_id,
                "agent_answer": answer[:1000],
                "faithfulness_score": faithfulness_score,
                "answer_relevance_score": relevance_score,
                "context_precision_score": context_precision_score,
                "citation_integrity_score": ci_res.citation_integrity_score,
                "claim_precision_score": claim_precision_score,
                "claim_precision_reasoning": claim_precision_reasoning,
                "factual_accuracy_score": factual_accuracy_score,
                "factual_accuracy_reasoning": factual_accuracy_reasoning,
                "turn_id": turn_id_out,
                "conflict_adherence_score": conflict_adherence_score,
                "session_coherence_score": coherence_score,
                "latency_ms": latency_ms,
                "planning_ms": planning_ms,
                "search_ms": search_ms,
                "fetch_ms": fetch_ms,
                "select_ms": select_ms,
                "synthesize_ms": synthesize_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
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

    # ── Post-run aggregates ───────────────────────────────────────────────
    taxonomy = Counter(r["failure_class"] for r in results_summary)
    agg = _compute_aggregates(results_summary)
    _print_aggregates(agg, taxonomy)

    # ── Cross-language consistency ────────────────────────────────────────
    try:
        cl_rows = await judge_cross_language_consistency(run_at_iso)
        if cl_rows:
            await save_cross_language_consistency(run_at_iso, cl_rows)
            print("\nCross-language consistency:")
            for r in cl_rows:
                flag = "FLAG" if r["flagged_inconsistent"] else "ok  "
                print(
                    f"  [{flag}] {r['concept_id']:<32} {r['en_question_id']:>6} ↔ {r['hi_question_id']:<6} "
                    f"jaccard={r['jaccard_score']:.3f}"
                )
    except Exception as exc:
        logger.warning("cross-language consistency failed: %s", exc)
        cl_rows = []

    # ── Confidence calibration ────────────────────────────────────────────
    calibration = _compute_calibration(results_summary)
    if calibration["correlation"] is not None:
        await _update_calibration_correlation(run_at_iso, calibration["correlation"])
    calib_path = _RESULTS_DIR / f"calibration_{run_ts}.json"
    with open(calib_path, "w") as fh:
        json.dump(calibration, fh, indent=2)
    print(f"\nCalibration correlation (confidence ↔ faithfulness): {calibration['correlation']}")
    print(f"Calibration written to: {calib_path}")

    # ── Per-language breakdown (V3.4 — preserved) ──────────────────────────
    languages = sorted({r.get("language", "en") for r in results_summary})
    if len(languages) > 1:
        print("\nPer-language breakdown:")
        print(f"  {'lang':<6}{'n':>4}{'pass%':>8}{'faith':>8}{'rel':>8}{'ctxP':>8}{'citI':>8}")
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

    # ── Persist run-level summary row + JSON snapshot ──────────────────────
    overall = agg.get("overall", {})
    summary_row = {
        "run_at": run_at_iso,
        "ablation_id": ablation_id,
        "n_questions": overall.get("n_questions", len(results_summary)),
        "pass_rate": overall.get("pass_rate", 0.0),
        "mean_faithfulness": overall.get("mean_faithfulness"),
        "mean_relevance": overall.get("mean_relevance"),
        "mean_context_precision": overall.get("mean_context_precision"),
        "mean_citation_integrity": overall.get("mean_citation_integrity"),
        "mean_claim_precision": overall.get("mean_claim_precision"),
        "mean_factual_accuracy": overall.get("mean_factual_accuracy"),
        "p50_latency_ms": overall.get("p50_latency_ms"),
        "p95_latency_ms": overall.get("p95_latency_ms"),
        "total_cost_usd": overall.get("total_cost_usd", 0.0),
        "retrieval_mode": retrieval_mode,
        "calibration_correlation": calibration.get("correlation"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await save_eval_run_summary(summary_row)

    summary_path = _RESULTS_DIR / f"summary_{run_ts}.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {
                **summary_row,
                "aggregates": agg,
                "failure_taxonomy": dict(taxonomy),
                "calibration": calibration,
                "cross_language": cl_rows,
            },
            fh, indent=2,
        )
    print(f"\nResults written to: {out_path}")
    print(f"Summary written to: {summary_path}")
    return run_at_iso


async def run_ablation() -> None:
    """Run BM25 leg then Hybrid leg, sharing a single ablation_id."""
    ablation_id = str(uuid.uuid4())
    print(f"\n{'#'*72}")
    print(f"ABLATION RUN  ·  id={ablation_id}")
    print(f"{'#'*72}\n")

    # Leg 1 — BM25 baseline.
    os.environ["HYBRID_RETRIEVAL"] = "0"
    await run_eval(ablation_id=ablation_id)

    # Leg 2 — Hybrid. Probe the *real* capability (not just package install):
    # some Python builds (notably Apple's /usr/bin/python3) import
    # `sqlite_vec` fine but lack `enable_load_extension`, so the extension
    # never loads and the hybrid leg silently degrades to BM25.
    from agent.memory import sqlite_vec_capability
    if not await sqlite_vec_capability():
        print(
            "\nWARNING: sqlite-vec cannot load on this Python build. The "
            "hybrid leg will silently degrade to BM25 and the ablation delta "
            "will be ~0. Use a Python compiled with "
            "--enable-loadable-sqlite-extensions: conda-forge/miniforge, "
            "modern Homebrew python@3.x, pyenv with that configure flag, or "
            "the project's python:3.12-slim Docker image. macOS works fine "
            "on those — only Apple's /usr/bin/python3 is the gotcha.\n"
        )
    os.environ["HYBRID_RETRIEVAL"] = "1"
    await run_eval(ablation_id=ablation_id)

    from eval.ablation_report import generate_report
    await generate_report(ablation_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Research Agent eval runner")
    parser.add_argument("--ablate", action="store_true",
                        help="Run BM25+Hybrid ablation and emit delta report")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ablate:
        asyncio.run(run_ablation())
    else:
        asyncio.run(run_eval())
