"""
Evaluation runner — sequential execution to respect rate limits.
Writes JSONL incrementally. Prints summary table at end.

Flags:
  --ablate          Run twice (BM25 then Hybrid) tagged with a shared
                    ablation_id, then emit eval/results/ablation_<id>.json.
  --no-judge        Skip all LLM-judge calls; emit only deterministic metrics
                    (citation integrity, factual accuracy alias match,
                    quote/numeric grounding, script preservation, latencies).
                    Fast and offline-safe. Tags rows with mode="deterministic".
  --judge-only      Cache replay — re-run only the LLM judge passes against
                    answers/context/doc_map persisted by a prior run.
                    Requires --run-id <iso-timestamp>. Skips search/extract/
                    synthesize entirely so no network is needed for those
                    stages. Tags rows with mode="judge_only".

  --no-judge and --judge-only are mutually exclusive.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import socket
import subprocess
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
    compute_cross_script_consistency,
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


async def _compute_cross_script_for_run(rows: list[dict]) -> dict:
    """Enrich result rows with cited URLs from the ``turns`` table and run
    the C4 pair-level cross-script consistency aggregator.

    Result rows don't carry ``urls_opened`` (only ``turn_id``); we hydrate
    that here before delegating to ``compute_cross_script_consistency``.
    Rows without a ``turn_id`` fall back to the markdown links embedded in
    ``agent_answer`` (already handled by the aggregator).
    """
    import aiosqlite

    from agent.memory import DB_PATH

    turn_ids = [r.get("turn_id") for r in rows if r.get("turn_id")]
    url_by_turn: dict[str, list[str]] = {}
    # SQLite's compile-time SQLITE_MAX_VARIABLE_NUMBER is 999 on older builds;
    # chunk the IN-clause to stay well below that and keep query plans cheap.
    _BATCH_SIZE = 500
    if turn_ids:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            for start in range(0, len(turn_ids), _BATCH_SIZE):
                batch = turn_ids[start:start + _BATCH_SIZE]
                placeholders = ",".join("?" * len(batch))
                cursor = await db.execute(
                    f"SELECT turn_id, urls_opened FROM turns WHERE turn_id IN ({placeholders})",
                    batch,
                )
                for row in await cursor.fetchall():
                    try:
                        url_by_turn[row["turn_id"]] = json.loads(row["urls_opened"] or "[]")
                    except (json.JSONDecodeError, TypeError):
                        url_by_turn[row["turn_id"]] = []

    enriched: list[dict] = []
    for r in rows:
        tid = r.get("turn_id")
        enriched.append({**r, "cited_urls": url_by_turn.get(tid, [])} if tid else r)

    return compute_cross_script_consistency(enriched)


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


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI via stdlib random.choices resampling.

    Returns (low, high). Uses a fixed seed for reproducibility. For an empty
    or single-element input, returns the degenerate (val, val) or (0.0, 0.0).
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = rng.choices(values, k=n)
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    low = means[int(math.floor(alpha * n_resamples))]
    high_idx = int(math.ceil((1.0 - alpha) * n_resamples)) - 1
    high = means[max(0, min(high_idx, n_resamples - 1))]
    return (low, high)


# ── Indic script-preservation metric (deterministic, no LLM) ─────────────────
_DEVANAGARI_RANGE = (0x0900, 0x097F)
_TAMIL_RANGE = (0x0B80, 0x0BFF)
_BENGALI_RANGE = (0x0980, 0x09FF)

_SCRIPT_RANGES_BY_LANG: dict[str, tuple[int, int]] = {
    "hi": _DEVANAGARI_RANGE,
    "mr": _DEVANAGARI_RANGE,
    "ta": _TAMIL_RANGE,
    "bn": _BENGALI_RANGE,
}

# Strip citation markers ([doc_N], markdown links), URLs, digits before counting.
_DOC_MARKER_RE = re.compile(r"\[doc_\d+\]")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")


def _strip_for_script_count(text: str) -> str:
    text = _DOC_MARKER_RE.sub("", text)
    # Keep the link text, drop the URL part.
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    # Drop digits — they're script-neutral.
    return re.sub(r"\d+", "", text)


def script_preservation_ratio(answer: str, language: str) -> Optional[float]:
    """Fraction of alphabetic chars in `answer` that are in the expected script.

    Returns None when the language has no Indic-script expectation, or when
    the answer has no alphabetic characters after stripping.
    """
    script_range = _SCRIPT_RANGES_BY_LANG.get((language or "").lower())
    if script_range is None:
        return None
    cleaned = _strip_for_script_count(answer or "")
    in_script = 0
    alpha_total = 0
    lo, hi = script_range
    for ch in cleaned:
        if ch.isalpha():
            alpha_total += 1
            cp = ord(ch)
            if lo <= cp <= hi:
                in_script += 1
    if alpha_total == 0:
        return None
    return in_script / alpha_total


_SCRIPT_PRESERVATION_THRESHOLD = 0.80


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
        base = k.replace("_score", "")
        overall[f"mean_{base}"] = _safe(results, k)
        vals = [r[k] for r in results if r.get(k) is not None]
        if vals:
            overall[f"p50_{base}"] = _percentile(vals, 0.50)
            overall[f"p95_{base}"] = _percentile(vals, 0.95)
            # 95% bootstrap CI on the top-line mean.
            low, high = bootstrap_ci(vals)
            overall[f"mean_{base}_ci_low"] = low
            overall[f"mean_{base}_ci_high"] = high

    # ── Script preservation (Indic subset only, deterministic) ────────────
    indic_ratios: list[float] = []
    for r in results:
        ratio = script_preservation_ratio(
            r.get("agent_answer", "") or "",
            r.get("language", "en") or "en",
        )
        if ratio is not None:
            indic_ratios.append(ratio)
    if indic_ratios:
        overall["script_preservation_mean"] = mean(indic_ratios)
        overall["script_preservation_pass_rate"] = sum(
            1 for v in indic_ratios if v >= _SCRIPT_PRESERVATION_THRESHOLD
        ) / len(indic_ratios)
        overall["script_preservation_n"] = len(indic_ratios)
    else:
        overall["script_preservation_mean"] = None
        overall["script_preservation_pass_rate"] = None
        overall["script_preservation_n"] = 0

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


def _print_aggregates(agg: dict, taxonomy: Counter, modes: Optional[set[str]] = None) -> None:
    """Print the summary table.

    When `modes` contains both "deterministic" and a judge-bearing mode
    (`full` or `judge_only`), the header splits into two columns. With a
    single mode, only the relevant column is labelled — saves the reader
    from interpreting `0.000` rows as real LLM scores.
    """
    print(f"\n{'='*72}")
    if modes and len(modes) > 1:
        print(f"AGGREGATES  (modes: {', '.join(sorted(modes))})")
    elif modes:
        label = next(iter(modes))
        col = "Deterministic" if label == "deterministic" else "Judge"
        print(f"AGGREGATES  ({col} mode)")
    else:
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


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _dataset_sha256() -> str:
    try:
        return hashlib.sha256(_DATASET.read_bytes()).hexdigest()
    except OSError:
        return "unknown"


def _provenance_info(run_at_iso: str) -> dict[str, str]:
    """Capture git, dataset, model, host metadata for the report header."""
    return {
        "git_sha": _git_output(["rev-parse", "HEAD"]),
        "git_commit_timestamp": _git_output(["log", "-1", "--format=%cI", "HEAD"]),
        "dataset_path": str(_DATASET),
        "dataset_sha256": _dataset_sha256(),
        "synthesizer_model": os.getenv("SYNTH_MODEL")
            or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "planner_model": os.getenv("PLANNER_MODEL", "llama-3.3-70b-versatile"),
        "judge_model": os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        "run_started_at": run_at_iso,
        "host": f"{socket.gethostname()} · Python {platform.python_version()}",
    }


def _render_provenance_block(prov: dict[str, str]) -> list[str]:
    lines = ["```", "Provenance"]
    for key in (
        "git_sha", "git_commit_timestamp", "dataset_path", "dataset_sha256",
        "synthesizer_model", "planner_model", "judge_model",
        "run_started_at", "host",
    ):
        lines.append(f"  {key}: {prov.get(key, 'unknown')}")
    lines.append("```")
    return lines


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
    # Provenance block — git/dataset/model/host metadata for reproducibility.
    lines.extend(_render_provenance_block(_provenance_info(run_at_iso)))
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
    lines.append("| Metric | Mean [95% CI] | P50 | P95 |")
    lines.append("|---|---:|---:|---:|")
    for k in (
        "faithfulness", "relevance", "context_precision",
        "citation_integrity", "claim_precision", "factual_accuracy",
        "quote_grounding_ratio", "uncertainty_handling",
    ):
        m = overall.get(f"mean_{k}")
        ci_low = overall.get(f"mean_{k}_ci_low")
        ci_high = overall.get(f"mean_{k}_ci_high")
        if m is not None and ci_low is not None and ci_high is not None:
            mean_cell = f"{m:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
        else:
            mean_cell = _fmt(m)
        lines.append(
            f"| {k.replace('_', ' ')} "
            f"| {mean_cell} "
            f"| {_fmt(overall.get(f'p50_{k}'))} "
            f"| {_fmt(overall.get(f'p95_{k}'))} |"
        )

    # Script preservation (Indic subset only). Emit explicit null when subset is empty.
    sp_mean = overall.get("script_preservation_mean")
    sp_pass = overall.get("script_preservation_pass_rate")
    sp_n = overall.get("script_preservation_n", 0)
    lines.append("")
    lines.append("## Script preservation (Indic subset)")
    lines.append("")
    if sp_n and sp_mean is not None:
        lines.append(
            f"- n = {sp_n}  ·  mean ratio = {sp_mean:.3f}  ·  "
            f"pass rate (≥0.80) = {sp_pass:.1%}"
        )
    else:
        lines.append("- n = 0  ·  mean ratio = null  ·  pass rate = null")

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

    # C3 calibration — pure-Python, hedge-vs-evidence calibration on
    # insufficient_evidence + conflicting cases.
    c3 = calibration.get("c3") if isinstance(calibration, dict) else None
    if c3 and c3.get("n_cases", 0) > 0:
        lines.append("")
        lines.append("## C3 calibration (hedge vs evidence)")
        lines.append("")
        lines.append(
            f"Calibration score: **{c3['calibration_score']:.3f}**  ·  "
            f"Brier score: **{c3['brier_score']:.3f}**  ·  "
            f"n={c3['n_cases']}"
        )
        if c3.get("per_category"):
            lines.append("")
            lines.append("| Category | n | Calibration | MAE |")
            lines.append("|---|---:|---:|---:|")
            for cat, stats in c3["per_category"].items():
                lines.append(
                    f"| {cat} | {stats['n_cases']} | "
                    f"{stats['calibration_score']:.3f} | "
                    f"{stats['mean_abs_error']:.3f} |"
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


async def _run_all_judges(
    *,
    query: str,
    answer: str,
    internal_answer: str,
    context_xml: str,
    doc_map: dict,
    fetched_urls: set,
    category: str,
    is_multiturn: bool,
    scenario: Optional[str],
    turn: int,
    scenario_sessions: dict,
    turn_id_out: str,
    q: dict,
    cross_family_judge: bool,
    qid: str,
    cross_family_sample_ids: set,
) -> dict:
    """Run every LLM-judge (and deterministic) scorer for a single row.

    Extracted so both the live run and the cache-replay (`--judge-only`)
    path call exactly the same scoring logic. Returns a flat dict of
    score fields ready to splice into the result row.
    """
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

    return {
        "faithfulness_score": faithfulness_score,
        "relevance_score": relevance_score,
        "context_precision_score": context_precision_score,
        "ci_res": ci_res,
        "claim_precision_score": claim_precision_score,
        "claim_precision_reasoning": claim_precision_reasoning,
        "conflict_adherence_score": conflict_adherence_score,
        "coherence_score": coherence_score,
        "factual_accuracy_score": factual_accuracy_score,
        "factual_accuracy_reasoning": factual_accuracy_reasoning,
        "cross_family_scores": cross_family_scores,
    }


def _deterministic_only_scores(
    *,
    internal_answer: str,
    doc_map: dict,
    fetched_urls: set,
    q: dict,
    answer: str,
) -> dict:
    """Compute only the deterministic (no-LLM) scorers used by --no-judge.

    Citation integrity is rule-based; factual accuracy uses gold-alias
    matching when gold data is present. All LLM-judge scores are set to
    None so the dashboard renders them as "—" rather than the
    misleading 0.5 sentinel that the live path uses.
    """
    ci_res = judge_citation_integrity(internal_answer, doc_map, fetched_urls)
    factual_accuracy_score: Optional[float] = None
    factual_accuracy_reasoning = ""
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
    return {
        "faithfulness_score": None,
        "relevance_score": None,
        "context_precision_score": None,
        "ci_res": ci_res,
        "claim_precision_score": None,
        "claim_precision_reasoning": "",
        "conflict_adherence_score": None,
        "coherence_score": None,
        "factual_accuracy_score": factual_accuracy_score,
        "factual_accuracy_reasoning": factual_accuracy_reasoning,
        "cross_family_scores": {},
    }


async def _load_cached_rows_for_replay(run_id: str) -> list[dict]:
    """Load prior eval rows for --judge-only replay.

    Strategy:
      1. Find the JSONL whose first row's `run_at` equals `run_id`
         (run_id is the ISO-timestamp stamped on every result row).
         If not found, fall back to interpreting `run_id` as a file path
         or filename stem so users can pass `eval_20260520_142539` directly.
      2. For each row, load the matching `turns` table record via
         `turn_id` to recover `context_xml_sent`, `doc_map`, and
         `urls_opened` — these are NOT stored in the JSONL.

    Rows without a `turn_id` (e.g. agent-timeout rows from the original
    run) are still returned, but with empty context/doc_map so the
    judge calls still execute and score the answer in isolation.
    """
    import aiosqlite

    candidates: list[Path] = []
    # 1. Direct file/stem reference.
    direct = Path(run_id)
    if direct.exists():
        candidates.append(direct)
    stem_match = _RESULTS_DIR / f"{run_id}.jsonl" if not run_id.endswith(".jsonl") \
        else _RESULTS_DIR / run_id
    if stem_match.exists() and stem_match not in candidates:
        candidates.append(stem_match)

    matched_path: Optional[Path] = candidates[0] if candidates else None
    raw_rows: list[dict] = []
    # 2. Otherwise scan all eval_*.jsonl files for run_at == run_id.
    if matched_path is None:
        for jsonl in sorted(_RESULTS_DIR.glob("eval_*.jsonl")):
            try:
                with open(jsonl) as fh:
                    first = fh.readline()
                if not first.strip():
                    continue
                row = json.loads(first)
                if row.get("run_at") == run_id:
                    matched_path = jsonl
                    break
            except (OSError, json.JSONDecodeError):
                continue

    if matched_path is None:
        raise FileNotFoundError(
            f"No prior eval run matches --run-id={run_id!r}. "
            f"Pass an ISO timestamp from a prior run's `run_at` field, a "
            f"path to its JSONL, or the filename stem (e.g. "
            f"`eval_20260520_142539`)."
        )

    with open(matched_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))

    if not raw_rows:
        return []

    # 3. Pull context_xml_sent + doc_map + urls_opened for each turn_id.
    turn_ids = [r.get("turn_id") for r in raw_rows if r.get("turn_id")]
    turn_artifacts: dict[str, dict] = {}
    if turn_ids:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            placeholders = ",".join("?" * len(turn_ids))
            cursor = await db.execute(
                f"SELECT turn_id, context_xml_sent, doc_map, urls_opened, response "
                f"FROM turns WHERE turn_id IN ({placeholders})",
                turn_ids,
            )
            async for row in cursor:
                turn_artifacts[row["turn_id"]] = {
                    "context_xml_sent": row["context_xml_sent"] or "",
                    "doc_map": json.loads(row["doc_map"]) if row["doc_map"] else {},
                    "urls_opened": json.loads(row["urls_opened"] or "[]"),
                    "response": row["response"] or "",
                }

    enriched: list[dict] = []
    for r in raw_rows:
        tid = r.get("turn_id")
        artifact = turn_artifacts.get(tid, {}) if tid else {}
        enriched.append({
            **r,
            "_context_xml": artifact.get("context_xml_sent", ""),
            "_doc_map": artifact.get("doc_map", {}),
            "_fetched_urls": set(artifact.get("urls_opened", [])),
            # Prefer the full DB-persisted response over the JSONL's
            # 1000-char truncated `agent_answer`.
            "_full_answer": artifact.get("response") or r.get("agent_answer", ""),
        })
    return enriched


async def run_judge_only(
    run_id: str,
    cross_family_judge: bool = False,
    cross_family_sample_size: int = 20,
) -> str:
    """Cache-replay: re-run only LLM-judge scoring against a prior run.

    No search, no fetch, no synthesis — everything comes from the
    persisted JSONL + `turns` table. Useful for re-judging with a
    different judge model, or re-judging old runs after fixing a
    scoring bug.
    """
    await init_db()

    cached = await _load_cached_rows_for_replay(run_id)
    if not cached:
        raise ValueError(f"--judge-only: no cached rows found for run_id={run_id!r}")

    with open(_DATASET) as f:
        questions_by_id = {q["id"]: q for q in json.load(f)}

    cross_family_sample_ids: set[str] = set()
    if cross_family_judge:
        cross_family_sample_ids = _select_cross_family_sample(
            list(questions_by_id.values()), sample_size=cross_family_sample_size
        )
        reset_cross_family_counter()

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_at_iso = datetime.now(timezone.utc).isoformat()
    out_path = _RESULTS_DIR / f"eval_judgeonly_{run_ts}.jsonl"
    scenario_sessions: dict[str, tuple[str, str, str]] = {}
    results_summary: list[dict] = []

    print(f"\n{'='*72}")
    print(f"Deep Research Agent — JUDGE-ONLY Replay {run_ts}  "
          f"(source_run_id={run_id})")
    print(f"{'='*72}")
    print(f"{'ID':<10} {'Category':<20} {'Faith':>6} {'Rel':>6} {'CitI':>6} {'CtxP':>6} {'Fail':<20}")
    print("-" * 79)

    with open(out_path, "w") as outf:
        for cached_row in cached:
            qid = cached_row["question_id"]
            q = questions_by_id.get(qid, {
                "id": qid,
                "query": cached_row.get("question", ""),
                "category": cached_row.get("category", "unknown"),
            })
            category = cached_row.get("category", q.get("category", "unknown"))
            query = cached_row.get("question", q.get("query", ""))
            language = cached_row.get("language", "en")
            is_multiturn = q.get("is_multiturn", False)
            scenario = q.get("scenario")
            turn = q.get("turn", 1)

            answer = cached_row["_full_answer"]
            context_xml = cached_row["_context_xml"]
            doc_map = cached_row["_doc_map"]
            fetched_urls = cached_row["_fetched_urls"]
            turn_id_out = cached_row.get("turn_id") or ""

            scores = await _run_all_judges(
                query=query, answer=answer, internal_answer=answer,
                context_xml=context_xml, doc_map=doc_map,
                fetched_urls=fetched_urls, category=category,
                is_multiturn=is_multiturn, scenario=scenario, turn=turn,
                scenario_sessions=scenario_sessions, turn_id_out=turn_id_out,
                q=q, cross_family_judge=cross_family_judge, qid=qid,
                cross_family_sample_ids=cross_family_sample_ids,
            )
            if is_multiturn and scenario and turn == 1:
                scenario_sessions[scenario] = (
                    scenario_sessions.get(scenario, (str(uuid.uuid4()),))[0],
                    query, answer,
                )

            result = _build_result_row(
                qid=qid, query=query, category=category, language=language,
                retrieval_mode=cached_row.get("retrieval_mode", "bm25"),
                ablation_id=None, answer=answer, scores=scores,
                turn_id_out=turn_id_out, run_at_iso=run_at_iso,
                latency_ms=cached_row.get("latency_ms", 0),
                planning_ms=0, search_ms=0, fetch_ms=0,
                select_ms=0, synthesize_ms=0,
                run_metadata=cached_row.get("run_metadata", {}) or {},
                prompt_tokens=cached_row.get("prompt_tokens", 0),
                completion_tokens=cached_row.get("completion_tokens", 0),
                mode="judge_only",
            )
            outf.write(json.dumps(result) + "\n")
            outf.flush()
            results_summary.append(result)
            await _persist_eval_run(result)
            print(
                f"{qid:<10} {category:<20} "
                f"{(scores['faithfulness_score'] or 0):>6.2f} "
                f"{(scores['relevance_score'] or 0):>6.2f} "
                f"{scores['ci_res'].citation_integrity_score:>6.2f} "
                f"{(scores['context_precision_score'] or 0):>6.2f} "
                f"{result['failure_class']:<20}"
            )

    taxonomy = Counter(r["failure_class"] for r in results_summary)
    agg = _compute_aggregates(results_summary)
    _print_aggregates(agg, taxonomy, modes={r.get("mode", "judge_only") for r in results_summary})
    print(f"\nJudge-only results written to: {out_path}")
    return run_at_iso


def _build_result_row(
    *,
    qid: str, query: str, category: str, language: str,
    retrieval_mode: str, ablation_id: Optional[str],
    answer: str, scores: dict, turn_id_out: str, run_at_iso: str,
    latency_ms: int, planning_ms: int, search_ms: int, fetch_ms: int,
    select_ms: int, synthesize_ms: int, run_metadata: dict,
    prompt_tokens: int, completion_tokens: int,
    mode: str = "full",
) -> dict:
    """Assemble a result row. Same shape regardless of mode; mode field
    distinguishes downstream consumers."""
    ci_res = scores["ci_res"]
    cross_family_scores = scores.get("cross_family_scores", {})
    cost_usd = cost_for(DEFAULT_MODEL, prompt_tokens, completion_tokens)
    return {
        "run_id": str(uuid.uuid4()),
        "run_at": run_at_iso,
        "mode": mode,
        "question_id": qid,
        "question": query,
        "category": category,
        "language": language,
        "retrieval_mode": retrieval_mode,
        "ablation_id": ablation_id,
        "agent_answer": (answer or "")[:1000],
        "faithfulness_score": scores["faithfulness_score"],
        "answer_relevance_score": scores["relevance_score"],
        "context_precision_score": scores["context_precision_score"],
        "citation_integrity_score": ci_res.citation_integrity_score,
        "claim_precision_score": scores["claim_precision_score"],
        "claim_precision_reasoning": scores["claim_precision_reasoning"],
        "factual_accuracy_score": scores["factual_accuracy_score"],
        "factual_accuracy_reasoning": scores["factual_accuracy_reasoning"],
        "turn_id": turn_id_out,
        "conflict_adherence_score": scores["conflict_adherence_score"],
        "session_coherence_score": scores["coherence_score"],
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
        "uncertainty_handling_score": score_uncertainty_handling(
            category, answer,
            (run_metadata or {}).get("uncertainty_kind"),
            (run_metadata or {}).get("follow_up_queries"),
        ),
        "uncertainty_kind": (run_metadata or {}).get("uncertainty_kind"),
        "follow_up_queries": (run_metadata or {}).get("follow_up_queries"),
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
        "faithfulness_score_primary": cross_family_scores.get("faithfulness", {}).get("primary_score"),
        "faithfulness_score_secondary": cross_family_scores.get("faithfulness", {}).get("secondary_score"),
        "faithfulness_agreement_delta": cross_family_scores.get("faithfulness", {}).get("agreement_delta"),
        "answer_relevance_score_primary": cross_family_scores.get("answer_relevance", {}).get("primary_score"),
        "answer_relevance_score_secondary": cross_family_scores.get("answer_relevance", {}).get("secondary_score"),
        "answer_relevance_agreement_delta": cross_family_scores.get("answer_relevance", {}).get("agreement_delta"),
        "context_precision_score_primary": cross_family_scores.get("context_precision", {}).get("primary_score"),
        "context_precision_score_secondary": cross_family_scores.get("context_precision", {}).get("secondary_score"),
        "context_precision_agreement_delta": cross_family_scores.get("context_precision", {}).get("agreement_delta"),
        "failure_class": classify_failure({
            "faithfulness_score": scores["faithfulness_score"] if scores["faithfulness_score"] is not None else 1.0,
            "answer_relevance_score": scores["relevance_score"] if scores["relevance_score"] is not None else 1.0,
            "context_precision_score": scores["context_precision_score"] if scores["context_precision_score"] is not None else 1.0,
            "citation_integrity_score": ci_res.citation_integrity_score,
            "claim_precision_score": scores["claim_precision_score"] if scores["claim_precision_score"] is not None else 1.0,
            "conflict_adherence_score": scores["conflict_adherence_score"],
            "session_coherence_score": scores["coherence_score"],
        }),
    }


async def run_eval(
    ablation_id: Optional[str] = None,
    question_ids: Optional[list[str]] = None,
    cross_family_judge: bool = False,
    cross_family_sample_size: int = 20,
    skip_judge: bool = False,
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

            # ── Score calls (judge or deterministic-only) ───────────────
            if skip_judge:
                scores = _deterministic_only_scores(
                    internal_answer=internal_answer, doc_map=doc_map,
                    fetched_urls=fetched_urls, q=q, answer=answer,
                )
            else:
                scores = await _run_all_judges(
                    query=query, answer=answer, internal_answer=internal_answer,
                    context_xml=context_xml, doc_map=doc_map,
                    fetched_urls=fetched_urls, category=category,
                    is_multiturn=is_multiturn, scenario=scenario, turn=turn,
                    scenario_sessions=scenario_sessions,
                    turn_id_out=turn_id_out, q=q,
                    cross_family_judge=cross_family_judge, qid=qid,
                    cross_family_sample_ids=cross_family_sample_ids,
                )

            if is_multiturn and scenario and turn == 1:
                session_id_stored = scenario_sessions[scenario][0]
                scenario_sessions[scenario] = (session_id_stored, query, answer)

            result = _build_result_row(
                qid=qid, query=query, category=category, language=language,
                retrieval_mode=retrieval_mode, ablation_id=ablation_id,
                answer=answer, scores=scores, turn_id_out=turn_id_out,
                run_at_iso=run_at_iso, latency_ms=latency_ms,
                planning_ms=planning_ms, search_ms=search_ms,
                fetch_ms=fetch_ms, select_ms=select_ms,
                synthesize_ms=synthesize_ms, run_metadata=run_metadata,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                mode="deterministic" if skip_judge else "full",
            )
            outf.write(json.dumps(result) + "\n")
            outf.flush()
            results_summary.append(result)
            await _persist_eval_run(result)

            ci_score = scores["ci_res"].citation_integrity_score
            faith_disp = f"{scores['faithfulness_score']:>6.2f}" \
                if scores["faithfulness_score"] is not None else f"{'—':>6}"
            rel_disp = f"{scores['relevance_score']:>6.2f}" \
                if scores["relevance_score"] is not None else f"{'—':>6}"
            ctxp_disp = f"{scores['context_precision_score']:>6.2f}" \
                if scores["context_precision_score"] is not None else f"{'—':>6}"
            print(
                f"{qid:<10} {category:<20} "
                f"{faith_disp} {rel_disp} {ci_score:>6.2f} {ctxp_disp} "
                f"{result['failure_class']:<20}"
            )

    await agent.aclose()

    # ── Post-run aggregates ───────────────────────────────────────────────
    taxonomy = Counter(r["failure_class"] for r in results_summary)
    agg = _compute_aggregates(results_summary)
    _print_aggregates(
        agg, taxonomy,
        modes={r.get("mode", "full") for r in results_summary},
    )

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

    # ── C4: Cross-script consistency (pair-level URL + entity overlap) ────
    # Enrich rows with cited URLs from the `turns` table, then aggregate
    # pairwise across languages that share a `concept_id`. Pure-Python.
    cross_script: dict = {}
    try:
        cross_script = await _compute_cross_script_for_run(results_summary)
        if cross_script.get("n_pairs"):
            print("\nCross-script consistency (C4):")
            print(
                f"  pairs={cross_script['n_pairs']}  topics={cross_script['n_topics']}  "
                f"mean_citation_overlap={cross_script['mean_citation_overlap']:.3f}  "
                f"mean_entity_overlap={cross_script['mean_entity_overlap']:.3f}"
            )
            for p in cross_script["pairs"]:
                print(
                    f"    {p['concept_id']:<32} {p['lang_a']}↔{p['lang_b']} "
                    f"cite={p['citation_overlap']:.2f}  ent={p['entity_overlap']:.2f}"
                )
    except Exception as exc:
        logger.warning("cross-script consistency failed: %s", exc)
        cross_script = {}

    # ── C3 calibration: model self-confidence ↔ judge confidence ─────────
    # Novel metric, scoped to insufficient_evidence + conflicting categories.
    # See eval/judge.py::compute_calibration_score for definition.
    from eval.judge import compute_calibration_score as _compute_c3
    c3_calibration = _compute_c3(results_summary)
    print(
        "\nC3 calibration (model self-confidence vs judge confidence): "
        f"score={c3_calibration['calibration_score']:.3f}  "
        f"brier={c3_calibration['brier_score']:.3f}  "
        f"n={c3_calibration['n_cases']}"
    )
    for cat, stats in c3_calibration.get("per_category", {}).items():
        print(
            f"  {cat:<24} n={stats['n_cases']:>2}  "
            f"score={stats['calibration_score']:.3f}  "
            f"mae={stats['mean_abs_error']:.3f}"
        )

    # ── Confidence calibration ────────────────────────────────────────────
    calibration = _compute_calibration(results_summary)
    # Attach C3 so the markdown report can render it without a signature change.
    calibration["c3"] = c3_calibration
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
        "c3_calibration_score": c3_calibration.get("calibration_score"),
        "c3_brier_score": c3_calibration.get("brier_score"),
        "c3_n_cases": c3_calibration.get("n_cases", 0),
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
                "c3_calibration": c3_calibration,
                "cross_language": cl_rows,
                "cross_script": cross_script,
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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--no-judge", action="store_true",
        help="Skip all LLM-judge calls — emit only deterministic metrics "
             "(citation integrity, factual accuracy, grounding ratios). Fast, "
             "offline-safe, suitable for CI sweeps."
    )
    mode_group.add_argument(
        "--judge-only", action="store_true",
        help="Cache replay: re-run judge passes against a prior run's cached "
             "answer + context + doc_map. Requires --run-id. Skips search/"
             "extract/synthesize entirely."
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Prior run identifier for --judge-only (ISO timestamp matching "
             "`run_at` on a previous JSONL row, a path to a JSONL file, or "
             "an eval_<ts> filename stem)."
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
    if args.judge_only and not args.run_id:
        print("ERROR: --judge-only requires --run-id <prior-run-id>", file=sys.stderr)
        sys.exit(2)

    # --judge-only is a pure cache replay: never run preflight (network may
    # be offline) and never start the live agent.
    if args.judge_only:
        await run_judge_only(
            run_id=args.run_id,
            cross_family_judge=args.cross_family_judge,
            cross_family_sample_size=args.cross_family_sample_size,
        )
        return

    if not args.skip_preflight:
        await _run_preflight()
    if args.ablate:
        await run_ablation()
    else:
        await run_eval(
            cross_family_judge=args.cross_family_judge,
            cross_family_sample_size=args.cross_family_sample_size,
            skip_judge=args.no_judge,
        )


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
