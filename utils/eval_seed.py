"""
Seed `eval_runs` from JSONL files on first boot.

Why: the deployed HF Space starts with an empty SQLite. Without seed data,
the `/eval` page shows "No eval runs recorded. Run python eval/eval_runner.py"
— a useless instruction for someone visiting a deployed website. This module
imports any `eval/results/*.jsonl` files into the `eval_runs` table on
lifespan startup, but ONLY when the table is empty (no risk of duplicates
on subsequent boots).

The JSONL files are committed to the repo and bundled into the Docker image,
so a fresh Space deploy comes pre-populated with the real 2026-05-20 eval
run for reviewers to inspect.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import aiosqlite

from agent.memory import DB_PATH

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parent.parent / "eval" / "results"


async def _eval_runs_empty(db: aiosqlite.Connection) -> bool:
    db.row_factory = aiosqlite.Row
    row = await db.execute_fetchall("SELECT COUNT(*) AS n FROM eval_runs")
    return (row[0]["n"] if row else 0) == 0


async def seed_if_empty() -> int:
    """Bulk-import JSONL eval files into `eval_runs` if the table is empty.
    Returns the count of rows inserted. Idempotent: no-op on subsequent boots
    once any row exists in the table.

    Schema-tolerant: rows missing optional columns get NULL — the schema
    has defaults for everything except primary keys. Lines that fail to
    parse are skipped with a warning rather than aborting the seed.
    """
    if not _RESULTS_DIR.exists():
        logger.info("eval_seed: %s does not exist; skipping seed", _RESULTS_DIR)
        return 0

    jsonl_files = sorted(_RESULTS_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.info("eval_seed: no JSONL files in %s; skipping seed", _RESULTS_DIR)
        return 0

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            if not await _eval_runs_empty(db):
                logger.info(
                    "eval_seed: eval_runs already populated; skipping seed (idempotent)"
                )
                return 0

            inserted = 0
            for path in jsonl_files:
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.warning("eval_seed: cannot read %s: %s", path, exc)
                    continue
                if not text.strip():
                    continue
                # Parse all rows first so we can canonicalize the run_at across
                # the file. The historical JSONL files had a *unique* run_at
                # per row (each question used its own timestamp), which made
                # the dashboard show each question as its own "run". For seed
                # purposes we want one logical run per JSONL file, so we pin
                # every row in this file to the earliest run_at present.
                parsed_rows: list[dict] = []
                for line_no, raw in enumerate(text.splitlines(), start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        parsed_rows.append(json.loads(raw))
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "eval_seed: skipping %s line %d: %s", path.name, line_no, exc,
                        )
                if not parsed_rows:
                    continue
                canonical_run_at = min(
                    (r.get("run_at") for r in parsed_rows if r.get("run_at")),
                    default=None,
                )
                for row in parsed_rows:
                    if canonical_run_at:
                        row["run_at"] = canonical_run_at
                    try:
                        await db.execute(
                            """
                            INSERT OR REPLACE INTO eval_runs (
                                run_id, run_at, question_id, question, category, agent_answer,
                                faithfulness_score, answer_relevance_score, context_precision_score,
                                citation_integrity_score, conflict_adherence_score,
                                session_coherence_score, claim_precision_score, judge_reasoning,
                                failure_class, latency_ms, turn_id, language, retrieval_mode,
                                factual_accuracy_score, ablation_id, calibration_correlation
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                row.get("run_id"),
                                row.get("run_at"),
                                row.get("question_id"),
                                row.get("question"),
                                row.get("category"),
                                row.get("agent_answer", ""),
                                row.get("faithfulness_score"),
                                row.get("answer_relevance_score"),
                                row.get("context_precision_score"),
                                row.get("citation_integrity_score"),
                                row.get("conflict_adherence_score"),
                                row.get("session_coherence_score"),
                                row.get("claim_precision_score"),
                                row.get("claim_precision_reasoning") or row.get("judge_reasoning", ""),
                                row.get("failure_class"),
                                row.get("latency_ms", 0),
                                row.get("turn_id"),
                                row.get("language", "en"),
                                row.get("retrieval_mode", "bm25"),
                                row.get("factual_accuracy_score"),
                                row.get("ablation_id"),
                                row.get("calibration_correlation"),
                            ),
                        )
                        inserted += 1
                    except Exception as exc:
                        logger.warning(
                            "eval_seed: insert failed for %s: %s",
                            path.name, exc,
                        )
            await db.commit()
            logger.info(
                "eval_seed: imported %d rows from %d JSONL file(s)",
                inserted, len(jsonl_files),
            )
            return inserted
    except Exception as exc:
        logger.warning("eval_seed: failed: %s", exc)
        return 0
