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
    build_context_precision_prompt,
    build_faithfulness_prompt,
    build_relevance_prompt,
    classify_failure,
    cohens_kappa_bucketed,
    get_cross_family_call_count,
    judge_citation_integrity,
    judge_claim_precision,
    judge_conflict_adherence,
    judge_coherence,
    judge_cross_language_consistency,
    judge_factual_accuracy,
    judge_faithfulness,
    judge_relevance,
    judge_context_precision,
    judge_with_dual_family,
    pearson_correlation,
    reset_cross_family_counter,
    score_uncertainty_handling,
)
from utils.cost_model import DEFAULT_MODEL, cost_for

logger = logging.getLogger(__name__)

_DATASET = Path(__file__).parent / "dataset.json"
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)
_PER_QUESTION_TIMEOUT_S = int(os.getenv("EVAL_PER_QUESTION_TIMEOUT_S", "240"))


async def _persist_eval_run(result: dict) -> None:
    """Mirror a JSONL row into the eval_runs SQLite table for dashboard queries.

    Tier A (Phase 1+) promotes per-turn quality metrics and routing fields
    from `run_metadata` into first-class columns so the dashboard can
    surface them without re-parsing JSON for every row.
    """
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
                    factual_accuracy_score, ablation_id, calibration_correlation,
                    quote_grounding_ratio, numeric_grounding_ratio,
                    criteria_coverage_ratio, terminator_fired, planner_provider,
                    reranker_used, language_method
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    result.get("quote_grounding_ratio"),
                    result.get("numeric_grounding_ratio"),
                    result.get("criteria_coverage_ratio"),
                    result.get("terminator_fired"),
                    result.get("planner_provider"),
                    result.get("reranker_used"),
                    (result.get("language_detection") or {}).get("method")
                    if isinstance(result.get("language_detection"), dict) else None,
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


def _criteria_coverage_ratio(coverage: list[bool] | None) -> Optional[float]:
    """Mean of a list of bools (success-criteria coverage). None if empty/missing.

    Tier A (Phase 1.875): planner emits `success_criteria` and the orchestrator
    records a parallel list[bool] of whether each criterion was met. We surface
    the ratio so the dashboard can plot it as a per-turn quality signal.
    """
    if not coverage:
        return None
    truthy = [bool(c) for c in coverage]
    if not truthy:
        return None
    return sum(1 for c in truthy if c) / len(truthy)


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
        "quote_grounding_ratio",
        # Phase 1.5: deterministic uncertainty-handling score (insufficient_evidence subset).
        "uncertainty_handling_score",
        # Tier A Phase 1+: per-turn quality metrics promoted from run_metadata.
        "numeric_grounding_ratio",
        "criteria_coverage_ratio",
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

    # Tier A Phase 1+: categorical distributions surfaced for the routing
    # section of the markdown report and the dashboard.
    def _dist(key: str) -> dict[str, int]:
        counter: Counter = Counter()
        for r in results:
            v = r.get(key)
            if v:
                counter[str(v)] += 1
        return dict(counter)

    def _chain_primary_dist() -> dict[str, int]:
        counter: Counter = Counter()
        for r in results:
            chain = r.get("synth_provider_chain")
            if isinstance(chain, list) and chain:
                counter[str(chain[0])] += 1
            elif isinstance(chain, str) and chain:
                counter[chain] += 1
        return dict(counter)

    def _lang_method_dist() -> dict[str, int]:
        counter: Counter = Counter()
        for r in results:
            ld = r.get("language_detection") or {}
            method = ld.get("method") if isinstance(ld, dict) else None
            if method:
                counter[str(method)] += 1
        return dict(counter)

    overall["terminator_fired_distribution"] = _dist("terminator_fired")
    overall["planner_provider_distribution"] = _dist("planner_provider")
    overall["reranker_used_distribution"] = _dist("reranker_used")
    overall["synth_chain_primary_distribution"] = _chain_primary_dist()
    overall["language_detection_method_distribution"] = _lang_method_dist()

    # Evidence-quality counters.
    total_evidence_gaps = sum(int(r.get("evidence_gap_count") or 0) for r in results)
    turns_with_extraction_fallbacks = sum(
        1 for r in results
        if isinstance(r.get("extraction_fallbacks"), dict)
        and any((r["extraction_fallbacks"] or {}).values())
    )
    turns_with_context_fallbacks = sum(
        1 for r in results
        if isinstance(r.get("context_fallbacks"), dict)
        and any((r["context_fallbacks"] or {}).values())
    )
    overall["total_evidence_gaps"] = total_evidence_gaps
    overall["turns_with_extraction_fallbacks"] = turns_with_extraction_fallbacks
    overall["turns_with_context_fallbacks"] = turns_with_context_fallbacks

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
              "claim_precision", "factual_accuracy", "quote_grounding_ratio",
              "numeric_grounding_ratio", "criteria_coverage_ratio",
              "uncertainty_handling"):
        m = o.get(f"mean_{k}"); p50 = o.get(f"p50_{k}"); p95 = o.get(f"p95_{k}")
        m_s = f"{m:.3f}" if m is not None else "—"
        p50_s = f"{p50:.3f}" if p50 is not None else "—"
        p95_s = f"{p95:.3f}" if p95 is not None else "—"
        print(f"{k:<28}{m_s:>10}{p50_s:>10}{p95_s:>10}")

    print("\nFailure taxonomy:")
    for k, v in taxonomy.most_common():
        print(f"  {k}: {v}")


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def _worst_rows(results: list[dict], n: int = 3) -> list[dict]:
    """Three lowest-faithfulness rows, used to surface concrete failure cases."""
    scored = [r for r in results if r.get("faithfulness_score") is not None]
    return sorted(scored, key=lambda r: r["faithfulness_score"])[:n]


def _write_markdown_report(
    path: Path,
    *,
    run_at_iso: str,
    retrieval_mode: str,
    ablation_id: Optional[str],
    results: list[dict],
    agg: dict,
    taxonomy: Counter,
    calibration: dict,
    cl_rows: list[dict],
    cross_family: Optional[dict] = None,
) -> None:
    """Emit a human-readable markdown report alongside the JSON summary."""
    overall = agg.get("overall", {})
    by_category = agg.get("by_category", {})
    languages = sorted({r.get("language") or "en" for r in results})

    lines: list[str] = []
    lines.append(f"# Eval Report — `{run_at_iso}`")
    lines.append("")
    lines.append(
        f"- **Retrieval mode:** `{retrieval_mode}`"
        + (f"  ·  **Ablation id:** `{ablation_id}`" if ablation_id else "")
    )
    lines.append(f"- **Questions:** {overall.get('n_questions', len(results))}")
    lines.append(f"- **Pass rate:** {overall.get('pass_rate', 0):.1%}")
    lines.append(
        f"- **Latency:** p50 {overall.get('p50_latency_ms', 0)} ms  ·  "
        f"p95 {overall.get('p95_latency_ms', 0)} ms"
    )
    lines.append(f"- **Cost:** ${overall.get('total_cost_usd', 0):.4f}")
    lines.append(
        f"- **Quote grounding ratio (Phase 1):** {_fmt(overall.get('mean_quote_grounding_ratio'))}"
    )
    lines.append(
        f"- **Uncertainty handling (Phase 1.5, insufficient_evidence subset):** "
        f"{_fmt(overall.get('mean_uncertainty_handling'))}"
    )

    lines.append("")
    lines.append("## Overall metrics")
    lines.append("")
    lines.append("| Metric | Mean | P50 | P95 |")
    lines.append("|---|---:|---:|---:|")
    for k in (
        "faithfulness", "relevance", "context_precision",
        "citation_integrity", "claim_precision", "factual_accuracy",
        "quote_grounding_ratio", "uncertainty_handling",
    ):
        lines.append(
            f"| {k.replace('_', ' ')} "
            f"| {_fmt(overall.get(f'mean_{k}'))} "
            f"| {_fmt(overall.get(f'p50_{k}'))} "
            f"| {_fmt(overall.get(f'p95_{k}'))} |"
        )

    if by_category:
        lines.append("")
        lines.append("## By category")
        lines.append("")
        lines.append("| Category | n | Pass% | Faithfulness | Relevance | Citation |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cat, row in by_category.items():
            lines.append(
                f"| {cat} | {row['n']} | {row['pass_rate']:.0%} "
                f"| {_fmt(row.get('mean_faithfulness'))} "
                f"| {_fmt(row.get('mean_relevance'))} "
                f"| {_fmt(row.get('mean_citation_integrity'))} |"
            )

    if len(languages) > 1:
        lines.append("")
        lines.append("## By language")
        lines.append("")
        lines.append("| Lang | n | Pass% | Faithfulness | Relevance | Ctx Precision |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for lang in languages:
            rows = [r for r in results if (r.get("language") or "en") == lang]
            if not rows:
                continue
            n = len(rows)
            pass_pct = sum(1 for r in rows if r["failure_class"] == "PASS") / n
            mean_f = mean(r["faithfulness_score"] for r in rows
                          if r.get("faithfulness_score") is not None) if any(
                r.get("faithfulness_score") is not None for r in rows) else None
            mean_r = mean(r["answer_relevance_score"] for r in rows
                          if r.get("answer_relevance_score") is not None) if any(
                r.get("answer_relevance_score") is not None for r in rows) else None
            mean_cp = mean(r["context_precision_score"] for r in rows
                           if r.get("context_precision_score") is not None) if any(
                r.get("context_precision_score") is not None for r in rows) else None
            lines.append(
                f"| {lang} | {n} | {pass_pct:.0%} "
                f"| {_fmt(mean_f)} | {_fmt(mean_r)} | {_fmt(mean_cp)} |"
            )

    lines.append("")
    lines.append("## Failure taxonomy")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("|---|---:|")
    for k, v in taxonomy.most_common():
        lines.append(f"| {k} | {v} |")

    # ── Tier A: per-turn quality metrics (Phase 1+) ──────────────────────
    lines.append("")
    lines.append("## Per-turn quality metrics (Phase 1+)")
    lines.append("")
    lines.append("| Metric | Mean | P50 | P95 |")
    lines.append("|---|---:|---:|---:|")
    for k in ("quote_grounding_ratio", "numeric_grounding_ratio", "criteria_coverage_ratio"):
        lines.append(
            f"| {k.replace('_', ' ')} "
            f"| {_fmt(overall.get(f'mean_{k}'))} "
            f"| {_fmt(overall.get(f'p50_{k}'))} "
            f"| {_fmt(overall.get(f'p95_{k}'))} |"
        )

    # ── Tier A: routing distribution ─────────────────────────────────────
    def _fmt_dist(d: dict[str, int]) -> str:
        if not d:
            return "—"
        return ", ".join(f"{k}: {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))

    lines.append("")
    lines.append("## Routing distribution")
    lines.append("")
    lines.append("| Component | Counts |")
    lines.append("|---|---|")
    lines.append(f"| Planner provider | {_fmt_dist(overall.get('planner_provider_distribution', {}))} |")
    lines.append(f"| Synth chain primary | {_fmt_dist(overall.get('synth_chain_primary_distribution', {}))} |")
    lines.append(f"| Reranker used | {_fmt_dist(overall.get('reranker_used_distribution', {}))} |")
    lines.append(f"| Terminator fired | {_fmt_dist(overall.get('terminator_fired_distribution', {}))} |")
    lines.append(f"| Language detection | {_fmt_dist(overall.get('language_detection_method_distribution', {}))} |")

    # ── Tier A: evidence quality ─────────────────────────────────────────
    lines.append("")
    lines.append("## Evidence quality")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total evidence_gaps | {overall.get('total_evidence_gaps', 0)} |")
    lines.append(
        f"| Turns with extraction fallbacks fired | {overall.get('turns_with_extraction_fallbacks', 0)} |"
    )
    lines.append(
        f"| Turns with context_fallbacks (compress_history) | {overall.get('turns_with_context_fallbacks', 0)} |"
    )

    if calibration.get("correlation") is not None:
        lines.append("")
        lines.append("## Confidence calibration")
        lines.append("")
        lines.append(
            f"Pearson correlation between agent self-reported confidence and "
            f"judge faithfulness score: **{calibration['correlation']:.3f}**"
        )

    if cross_family:
        sample_n = cross_family.get("cross_family_sample_size", 0)
        n_total = len(results)
        metrics = ", ".join(cross_family.get("metrics_double_judged") or [])
        pearson = cross_family.get("inter_rater_agreement_pearson")
        mad = cross_family.get("mean_abs_delta")
        kappa = cross_family.get("cohens_kappa_bucketed")
        lines.append("")
        lines.append("## Inter-Rater Agreement (Cross-Family Judge)")
        lines.append("")
        lines.append(f"- Sample size: {sample_n} / {n_total} questions")
        lines.append(f"- Metrics double-judged: {metrics}")
        lines.append(f"- Pearson correlation: {_fmt(pearson)}")
        lines.append(f"- Mean absolute delta: {_fmt(mad)}")
        lines.append(f"- Cohen's κ (bucketed): {_fmt(kappa)}")
        if cross_family.get("cross_family_judging_truncated"):
            lines.append("- **Note:** judging truncated by GitHub Models quota guard.")

    if cl_rows:
        flagged = sum(1 for r in cl_rows if r["flagged_inconsistent"])
        lines.append("")
        lines.append("## Cross-language consistency")
        lines.append("")
        lines.append(
            f"{len(cl_rows)} EN↔HI concept pair(s) evaluated  ·  "
            f"{flagged} flagged inconsistent"
        )

    worst = _worst_rows(results, 3)
    if worst:
        lines.append("")
        lines.append("## Worst three rows (lowest faithfulness)")
        lines.append("")
        for r in worst:
            lines.append(
                f"### `{r['question_id']}` — {r['category']} / "
                f"{r.get('language') or 'en'} → **{r['failure_class']}**"
            )
            lines.append("")
            lines.append(f"> {r['question']}")
            lines.append("")
            lines.append(
                f"- faithfulness {_fmt(r.get('faithfulness_score'))}  ·  "
                f"relevance {_fmt(r.get('answer_relevance_score'))}  ·  "
                f"citation {_fmt(r.get('citation_integrity_score'))}  ·  "
                f"ctx_precision {_fmt(r.get('context_precision_score'))}"
            )
            answer = (r.get("agent_answer") or "").strip().replace("\n", " ")
            if len(answer) > 280:
                answer = answer[:280] + "…"
            lines.append(f"- agent answer: _{answer}_" if answer else "- agent answer: _(empty)_")
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_CROSS_FAMILY_METRICS = ("faithfulness", "answer_relevance", "context_precision")


def _select_cross_family_sample(
    questions: list[dict],
    sample_size: int = 20,
    seed: int = 42,
) -> set[str]:
    """Deterministic sample of question IDs for cross-family double-judging.

    Uses a seeded random.shuffle so the same dataset always produces the same
    sampled IDs. Returns at most `sample_size` IDs.
    """
    import random
    n = min(sample_size, len(questions))
    if n <= 0:
        return set()
    rng = random.Random(seed)
    ids = [q["id"] for q in questions]
    rng.shuffle(ids)
    return set(ids[:n])


async def run_eval(
    ablation_id: Optional[str] = None,
    question_ids: Optional[list[str]] = None,
    cross_family_judge: bool = False,
    cross_family_sample_size: int = 20,
) -> str:
    """Run the eval dataset once. Returns the run_at timestamp.

    If `question_ids` is provided, only those questions run (used by the
    web smoke-eval endpoint to bound cost and latency). When None, the full
    dataset is used.
    """
    # Phase 2 safeguard: the plan-approval gate must remain off during eval —
    # it's a benchmark, not an interactive session, and a stray True would
    # deadlock the harness forever. We assert at runtime (not just by
    # convention) so a future contributor flipping a fixture toggle fails
    # loudly instead of silently hanging the run.
    #
    # `run_eval` doesn't construct ChatRequest (it drives ResearchOrchestrator
    # directly), so the only way `approval_required=True` could leak in is via
    # the env-derived RuntimeConfig path. Guard both surfaces here.
    from agent.orchestrator import RuntimeConfig as _RC
    _resolved = _RC.from_overrides(None)
    # Audit L3: must be `raise`, not `assert` — `python -O` strips assertions
    # and would silently let an approval-required eval deadlock the harness.
    if _resolved.approval_required is True:
        raise RuntimeError(
            "Eval harness must run with approval_required=False; got True from "
            "RuntimeConfig — unset any approval_required override before running eval."
        )

    await init_db()

    with open(_DATASET) as f:
        questions = json.load(f)
    if question_ids:
        wanted = set(question_ids)
        questions = [q for q in questions if q.get("id") in wanted]
        if not questions:
            raise ValueError(f"No dataset questions matched ids={question_ids}")

    # P1.4: persist the actually-effective retrieval mode rather than reading
    # the legacy env flag directly. Falls back to "lexical" when the capability
    # probe can't load vec (e.g. sqlite without loadable extensions).
    from utils.retrieval_mode import effective_mode_for_request
    from agent import memory as _memory_mod
    try:
        _eff = effective_mode_for_request(_memory_mod._VEC_AVAILABLE)
        retrieval_mode = "hybrid" if _eff.is_hybrid else "bm25"
    except RuntimeError:
        # `hybrid` requested but vec unavailable — raise here would abort eval.
        # Mark as bm25 since that's what the orchestrator will fall back to.
        retrieval_mode = "bm25"
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_at_iso = datetime.now(timezone.utc).isoformat()
    out_path = _RESULTS_DIR / f"eval_{run_ts}.jsonl"
    agent = ResearchOrchestrator()

    scenario_sessions: dict[str, tuple[str, str, str]] = {}
    results_summary: list[dict] = []

    # Tier C: sample question IDs that will receive cross-family double-judging.
    cross_family_sample_ids: set[str] = set()
    if cross_family_judge:
        cross_family_sample_ids = _select_cross_family_sample(
            questions, sample_size=cross_family_sample_size
        )
        reset_cross_family_counter()
        print(
            f"Cross-family judging enabled — sampling {len(cross_family_sample_ids)} "
            f"of {len(questions)} questions (deterministic seed=42)"
        )

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

            # ── Tier C: cross-family double-judging on sampled subset ───────
            cross_family_scores: dict[str, dict] = {}
            if cross_family_judge and qid in cross_family_sample_ids:
                prompts = {
                    "faithfulness": build_faithfulness_prompt(
                        context_xml or "(no context)", answer
                    ),
                    "answer_relevance": build_relevance_prompt(query, answer),
                    "context_precision": build_context_precision_prompt(
                        query, context_xml or "(no context)"
                    ),
                }
                for metric_name, p in prompts.items():
                    try:
                        cross_family_scores[metric_name] = await judge_with_dual_family(
                            p, metric_name
                        )
                    except Exception as e:
                        print(f"  Cross-family judge error ({metric_name}): {e}")
                        cross_family_scores[metric_name] = {
                            "primary_score": None,
                            "secondary_score": None,
                            "agreement_delta": None,
                        }

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
                "quote_grounding_ratio": (run_metadata or {}).get("quote_grounding_ratio"),
                # Phase 1.5: deterministic uncertainty-handling score. Only
                # populated for insufficient_evidence category; None otherwise.
                "uncertainty_handling_score": score_uncertainty_handling(
                    category,
                    answer,
                    (run_metadata or {}).get("uncertainty_kind"),
                    (run_metadata or {}).get("follow_up_queries"),
                ),
                "uncertainty_kind": (run_metadata or {}).get("uncertainty_kind"),
                "follow_up_queries": (run_metadata or {}).get("follow_up_queries"),
                # ── Tier A (Phase 1+) per-row metrics & routing ────────────
                "numeric_grounding_ratio": (run_metadata or {}).get("numeric_grounding_ratio"),
                "criteria_coverage_ratio": _criteria_coverage_ratio(
                    (run_metadata or {}).get("criteria_coverage")
                ),
                "terminator_fired": (run_metadata or {}).get("terminator_fired"),
                "evidence_gap_count": len((run_metadata or {}).get("evidence_gaps") or []),
                "planner_provider": (run_metadata or {}).get("planner_provider"),
                "synth_provider_chain": (run_metadata or {}).get("synth_provider_chain"),
                "reranker_used": (run_metadata or {}).get("reranker_used"),
                "supplementary_sources": (run_metadata or {}).get("supplementary_sources", {}),
                "extraction_fallbacks": (run_metadata or {}).get("extraction_fallbacks", {}),
                "language_detection": (run_metadata or {}).get("language_detection", {}),
                "context_fallbacks": (run_metadata or {}).get("context_fallbacks", {}),
                "budget_distribution": (run_metadata or {}).get("budget_distribution", {}),
                "run_metadata": run_metadata,
                # Tier C: cross-family dual-judge scores (None if not sampled)
                "faithfulness_score_primary": cross_family_scores.get(
                    "faithfulness", {}
                ).get("primary_score"),
                "faithfulness_score_secondary": cross_family_scores.get(
                    "faithfulness", {}
                ).get("secondary_score"),
                "faithfulness_agreement_delta": cross_family_scores.get(
                    "faithfulness", {}
                ).get("agreement_delta"),
                "answer_relevance_score_primary": cross_family_scores.get(
                    "answer_relevance", {}
                ).get("primary_score"),
                "answer_relevance_score_secondary": cross_family_scores.get(
                    "answer_relevance", {}
                ).get("secondary_score"),
                "answer_relevance_agreement_delta": cross_family_scores.get(
                    "answer_relevance", {}
                ).get("agreement_delta"),
                "context_precision_score_primary": cross_family_scores.get(
                    "context_precision", {}
                ).get("primary_score"),
                "context_precision_score_secondary": cross_family_scores.get(
                    "context_precision", {}
                ).get("secondary_score"),
                "context_precision_agreement_delta": cross_family_scores.get(
                    "context_precision", {}
                ).get("agreement_delta"),
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

    # ── Tier C: cross-family inter-rater agreement aggregates ─────────────
    cross_family_summary: dict = {}
    if cross_family_judge:
        primary_all: list[float] = []
        secondary_all: list[float] = []
        deltas: list[float] = []
        sampled_count = 0
        for r in results_summary:
            if r["question_id"] not in cross_family_sample_ids:
                continue
            sampled_count += 1
            for metric in _CROSS_FAMILY_METRICS:
                p = r.get(f"{metric}_score_primary")
                s = r.get(f"{metric}_score_secondary")
                if p is not None and s is not None:
                    primary_all.append(p)
                    secondary_all.append(s)
                    deltas.append(abs(p - s))
        pearson = pearson_correlation(primary_all, secondary_all)
        mad = (sum(deltas) / len(deltas)) if deltas else None
        kappa = cohens_kappa_bucketed(primary_all, secondary_all)
        truncated = get_cross_family_call_count() >= int(
            os.environ.get("CROSS_FAMILY_QUOTA_BUDGET", "140")
        )
        cross_family_summary = {
            "cross_family_sample_size": sampled_count,
            "double_judged_score_pairs": len(deltas),
            "inter_rater_agreement_pearson": pearson,
            "mean_abs_delta": mad,
            "cohens_kappa_bucketed": kappa,
            "cross_family_judging_truncated": truncated,
            "github_models_calls_used": get_cross_family_call_count(),
            "metrics_double_judged": list(_CROSS_FAMILY_METRICS),
        }
        print("\nCross-family inter-rater agreement:")
        print(f"  sample_size={sampled_count}  pairs={len(deltas)}")
        print(f"  pearson={pearson}  mean|Δ|={mad}  κ={kappa}")
        if truncated:
            print("  WARNING: cross-family judging truncated by quota guard")

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

    # Groq key-rotator telemetry — surfaces how many keys we ran with and
    # whether any were still throttled at run-end (signals near-quota).
    from utils.provider_router import _GROQ_ROTATOR
    groq_rotator_snapshot = _GROQ_ROTATOR.snapshot()

    summary_path = _RESULTS_DIR / f"summary_{run_ts}.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {
                **summary_row,
                "aggregates": agg,
                "failure_taxonomy": dict(taxonomy),
                "calibration": calibration,
                "cross_language": cl_rows,
                "cross_family_judge": cross_family_summary,
                "groq_key_count": groq_rotator_snapshot["total_keys"],
                "groq_key_rotator": groq_rotator_snapshot,
            },
            fh, indent=2,
        )
    report_path = _RESULTS_DIR / f"report_{run_ts}.md"
    _write_markdown_report(
        report_path,
        run_at_iso=run_at_iso,
        retrieval_mode=retrieval_mode,
        ablation_id=ablation_id,
        results=results_summary,
        agg=agg,
        taxonomy=taxonomy,
        calibration=calibration,
        cl_rows=cl_rows,
        cross_family=cross_family_summary,
    )

    print(f"\nResults written to: {out_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Report  written to: {report_path}")
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
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the provider health pre-flight check")
    parser.add_argument(
        "--cross-family-judge", action="store_true",
        help="Double-judge a deterministic sample (default 20) with Groq + GitHub "
             "GPT-4o-mini to compute inter-rater agreement (Pearson, κ, mean |Δ|)."
    )
    parser.add_argument(
        "--cross-family-sample-size", type=int, default=20,
        help="Sample size for cross-family judging (default: 20)."
    )
    return parser.parse_args()


async def _run_preflight() -> None:
    """Probe providers; exit 2 if any required capability group is missing."""
    from utils.provider_health import eval_preflight

    print("Pre-flight: probing providers…")
    result = await eval_preflight()
    snap = result.snapshot
    if snap:
        ok_n = sum(1 for p in snap.providers if p.status == "ok")
        print(f"  {ok_n}/{len(snap.providers)} providers OK  ·  overall={snap.overall}")
        for p in snap.providers:
            marker = {"ok": "✓", "degraded": "~", "down": "✗",
                      "missing_key": "·", "not_configured": "·"}.get(p.status, "?")
            latency = f"{p.latency_ms} ms" if p.latency_ms is not None else "—"
            print(f"  {marker} {p.name:<14} {p.role:<16} {p.status:<12} {latency:>8}"
                  + (f"  {p.detail}" if p.detail else ""))
    for w in result.warnings:
        print(f"  warning: {w}")
    if not result.ok:
        print("\nPre-flight FAILED — eval cannot proceed:")
        for b in result.blockers:
            print(f"  · {b}")
        print("\nFix the blockers above, or re-run with --skip-preflight to bypass.")
        sys.exit(2)
    print("Pre-flight passed.\n")


async def _main(args: argparse.Namespace) -> None:
    if not args.skip_preflight:
        await _run_preflight()
    if args.ablate:
        await run_ablation()
    else:
        await run_eval(
            cross_family_judge=args.cross_family_judge,
            cross_family_sample_size=args.cross_family_sample_size,
        )


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
