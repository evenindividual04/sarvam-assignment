"""Read-only query helpers for the eval dashboard endpoints.
Kept separate from agent/memory.py to isolate ad-hoc reporting SQL from core CRUD."""
from __future__ import annotations

import json
import logging
from typing import Optional

import aiosqlite

from agent.memory import DB_PATH

logger = logging.getLogger(__name__)


_METRIC_COLS = [
    "faithfulness_score",
    "answer_relevance_score",
    "context_precision_score",
    "citation_integrity_score",
    "conflict_adherence_score",
    "session_coherence_score",
    "claim_precision_score",
]


async def list_eval_runs() -> list[dict]:
    """Aggregate per run_at timestamp. JOINs the eval_run_summary table when
    rows exist there (post-V3.5) for canonical aggregates; otherwise falls
    back to inline AVGs. Includes per-language breakdown (V3.4)."""
    sql = f"""
    SELECT
        e.run_at,
        COUNT(*) AS n_questions,
        SUM(CASE WHEN e.failure_class = 'PASS' OR e.failure_class IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS pass_rate,
        COALESCE(MAX(e.retrieval_mode), 'bm25') AS retrieval_mode,
        MAX(e.ablation_id) AS ablation_id,
        MAX(e.calibration_correlation) AS calibration_correlation,
        AVG(e.factual_accuracy_score) AS avg_factual_accuracy_score,
        {", ".join(f"AVG(e.{c}) AS avg_{c}" for c in _METRIC_COLS)},
        MAX(s.total_cost_usd) AS total_cost_usd,
        MAX(s.p50_latency_ms) AS p50_latency_ms,
        MAX(s.p95_latency_ms) AS p95_latency_ms
    FROM eval_runs e
    LEFT JOIN eval_run_summary s ON s.run_at = e.run_at
    GROUP BY e.run_at
    ORDER BY e.run_at DESC
    """
    lang_sql = f"""
    SELECT
        run_at,
        COALESCE(language, 'en') AS language,
        COUNT(*) AS n_questions,
        SUM(CASE WHEN failure_class = 'PASS' OR failure_class IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS pass_rate,
        {", ".join(f"AVG({c}) AS avg_{c}" for c in _METRIC_COLS)}
    FROM eval_runs
    GROUP BY run_at, COALESCE(language, 'en')
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = [dict(r) for r in await db.execute_fetchall(sql)]
        lang_rows = [dict(r) for r in await db.execute_fetchall(lang_sql)]

    by_run: dict[str, dict[str, dict]] = {}
    for lr in lang_rows:
        ra = lr.pop("run_at")
        lang = lr.pop("language")
        by_run.setdefault(ra, {})[lang] = lr

    for row in rows:
        breakdown = by_run.get(row["run_at"], {})
        # Only include the breakdown when at least one non-default language is present.
        if any(k != "en" for k in breakdown):
            row["by_language"] = breakdown
    return rows


async def get_run_summary(run_at: str) -> dict:
    """Per-category averages + failure_class distribution + cross-language +
    calibration sub-sections."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cat_sql = f"""
        SELECT category,
               COUNT(*) AS n,
               AVG(factual_accuracy_score) AS avg_factual_accuracy_score,
               {", ".join(f"AVG({c}) AS avg_{c}" for c in _METRIC_COLS)}
        FROM eval_runs
        WHERE run_at = ?
        GROUP BY category
        """
        by_category = [dict(r) for r in await db.execute_fetchall(cat_sql, (run_at,))]

        fail_sql = """
        SELECT COALESCE(failure_class, 'PASS') AS failure_class, COUNT(*) AS n
        FROM eval_runs WHERE run_at = ? GROUP BY failure_class
        """
        failure_dist = [dict(r) for r in await db.execute_fetchall(fail_sql, (run_at,))]

        cl_rows = [
            dict(r) for r in await db.execute_fetchall(
                "SELECT * FROM cross_language_consistency WHERE run_at = ? ORDER BY concept_id",
                (run_at,),
            )
        ]

        # SQLite forbids referencing SELECT aliases in WHERE on some builds —
        # inline the json_extract expression in both clauses to stay portable.
        conf_sql = """
        SELECT
          json_extract(t.run_metadata_json, '$.planner_output.confidence') AS confidence,
          AVG(e.faithfulness_score) AS mean_faithfulness,
          AVG(e.claim_precision_score) AS mean_claim_precision,
          COUNT(*) AS n
        FROM eval_runs e LEFT JOIN turns t ON t.turn_id = e.turn_id
        WHERE e.run_at = ?
          AND json_extract(t.run_metadata_json, '$.planner_output.confidence') IS NOT NULL
        GROUP BY json_extract(t.run_metadata_json, '$.planner_output.confidence')
        """
        try:
            calib_buckets = [dict(r) for r in await db.execute_fetchall(conf_sql, (run_at,))]
        except Exception:
            calib_buckets = []

        summary_row = await db.execute_fetchall(
            "SELECT * FROM eval_run_summary WHERE run_at = ? LIMIT 1", (run_at,),
        )
        summary_row = dict(summary_row[0]) if summary_row else {}

        # Tier A (Phase 1+): per-turn quality means promoted from `eval_runs`
        # columns. Wrapped in try/except so older DBs without these columns
        # still render the dashboard (they'll just see "—").
        per_turn_quality: dict = {}
        try:
            ptq = await db.execute_fetchall(
                "SELECT AVG(quote_grounding_ratio)  AS mean_quote_grounding_ratio, "
                "       AVG(numeric_grounding_ratio) AS mean_numeric_grounding_ratio, "
                "       AVG(criteria_coverage_ratio) AS mean_criteria_coverage_ratio "
                "FROM eval_runs WHERE run_at = ?",
                (run_at,),
            )
            per_turn_quality = dict(ptq[0]) if ptq else {}
        except Exception:
            per_turn_quality = {}

    flagged = [r for r in cl_rows if r.get("flagged_inconsistent")]
    mean_jaccard = (
        sum(r["jaccard_score"] for r in cl_rows) / len(cl_rows)
        if cl_rows else None
    )

    return {
        "run_at": run_at,
        "by_category": by_category,
        "failure_distribution": failure_dist,
        "cross_language": {
            "rows": cl_rows,
            "flagged": flagged,
            "mean_jaccard": mean_jaccard,
        },
        "calibration": {
            "buckets": calib_buckets,
            "correlation": summary_row.get("calibration_correlation"),
        },
        "run_summary": summary_row,
        "per_turn_quality": per_turn_quality,
    }


async def get_run_questions(run_at: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT e.*, t.prompt_tokens, t.completion_tokens "
            "FROM eval_runs e LEFT JOIN turns t ON t.turn_id = e.turn_id "
            "WHERE e.run_at = ? ORDER BY e.question_id ASC",
            (run_at,),
        )
        return [dict(r) for r in rows]


async def get_question_detail(run_at: str, question_id: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchall(
            "SELECT * FROM eval_runs WHERE run_at = ? AND question_id = ? LIMIT 1",
            (run_at, question_id),
        )
        if not row:
            return None
        rec = dict(row[0])
        turn_id = rec.get("turn_id")

        turn_row: dict = {}
        claim_audit: list[dict] = []
        probe: Optional[dict] = None
        if turn_id:
            t = await db.execute_fetchall(
                "SELECT context_xml_sent, doc_map, claim_verification_json, state_trace, "
                "prompt_tokens, completion_tokens, run_metadata_json "
                "FROM turns WHERE turn_id = ? LIMIT 1",
                (turn_id,),
            )
            if t:
                turn_row = dict(t[0])
                if turn_row.get("doc_map"):
                    try:
                        turn_row["doc_map"] = json.loads(turn_row["doc_map"])
                    except (TypeError, ValueError):
                        pass
                if turn_row.get("state_trace"):
                    try:
                        turn_row["state_trace"] = json.loads(turn_row["state_trace"])
                    except (TypeError, ValueError):
                        pass
                # Surface run_metadata as a parsed dict so callers can pull
                # the terminator trace (refactor #3) and other routing
                # signals without re-reading the column. Renamed to
                # `run_metadata` (no `_json` suffix) since it's now a dict.
                if turn_row.get("run_metadata_json"):
                    try:
                        turn_row["run_metadata"] = json.loads(
                            turn_row["run_metadata_json"]
                        )
                    except (TypeError, ValueError):
                        turn_row["run_metadata"] = None
                turn_row.pop("run_metadata_json", None)

            ca = await db.execute_fetchall(
                "SELECT * FROM claim_audit WHERE turn_id = ? ORDER BY claim_idx ASC", (turn_id,)
            )
            claim_audit = [dict(r) for r in ca]
            for c in claim_audit:
                if c.get("cited_doc_ids"):
                    try:
                        c["cited_doc_ids"] = json.loads(c["cited_doc_ids"])
                    except (TypeError, ValueError):
                        pass

            p = await db.execute_fetchall(
                "SELECT * FROM contradiction_probes WHERE turn_id = ? LIMIT 1", (turn_id,)
            )
            if p:
                probe = dict(p[0])
                if probe.get("contradictions_json"):
                    try:
                        probe["contradictions_json"] = json.loads(probe["contradictions_json"])
                    except (TypeError, ValueError):
                        pass
                # P3: surface dominant_kind, defaulting NULL/missing rows
                # (pre-migration data) to "none" rather than raising.
                if not probe.get("dominant_kind"):
                    probe["dominant_kind"] = "none"

        return {
            "eval_row": rec,
            "turn": turn_row,
            "claim_audit": claim_audit,
            "contradiction_probe": probe,
        }
