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
    """Aggregate per run_at timestamp. Includes a per-language breakdown
    (V3.4) when any non-default language is present in the run."""
    sql = f"""
    SELECT
        run_at,
        COUNT(*) AS n_questions,
        SUM(CASE WHEN failure_class = 'PASS' OR failure_class IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS pass_rate,
        COALESCE(MAX(retrieval_mode), 'bm25') AS retrieval_mode,
        {", ".join(f"AVG({c}) AS avg_{c}" for c in _METRIC_COLS)}
    FROM eval_runs
    GROUP BY run_at
    ORDER BY run_at DESC
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
    """Per-category averages + failure_class distribution."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cat_sql = f"""
        SELECT category,
               COUNT(*) AS n,
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

    return {"run_at": run_at, "by_category": by_category, "failure_distribution": failure_dist}


async def get_run_questions(run_at: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT * FROM eval_runs WHERE run_at = ? ORDER BY question_id ASC", (run_at,)
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
                "SELECT context_xml_sent, doc_map, claim_verification_json, state_trace "
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

        return {
            "eval_row": rec,
            "turn": turn_row,
            "claim_audit": claim_audit,
            "contradiction_probe": probe,
        }
