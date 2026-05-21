"""
aiosqlite CRUD + FTS5 keyword search over session history.
All tables created with IF NOT EXISTS — safe to call init_db() repeatedly.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import aiosqlite

from datetime import datetime, timezone

from agent.models import ConflictResult, ContextSnippet, Turn

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "research.db")


CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    turn_count  INTEGER DEFAULT 0
);
"""

CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id                  TEXT PRIMARY KEY,
    session_id               TEXT NOT NULL REFERENCES sessions(session_id),
    query                    TEXT NOT NULL,
    plan                     TEXT,
    search_queries           TEXT NOT NULL,
    urls_opened              TEXT NOT NULL,
    response                 TEXT,
    context_xml_sent         TEXT,
    doc_map                  TEXT,
    citation_integrity_score REAL,
    hallucination_count      INTEGER DEFAULT 0,
    prompt_tokens            INTEGER DEFAULT 0,
    completion_tokens        INTEGER DEFAULT 0,
    latency_ms               INTEGER,
    planning_ms              INTEGER DEFAULT 0,
    search_ms                INTEGER DEFAULT 0,
    fetch_ms                 INTEGER DEFAULT 0,
    select_ms                INTEGER DEFAULT 0,
    synthesize_ms            INTEGER DEFAULT 0,
    run_metadata_json        TEXT,
    state_trace              TEXT,
    created_at               TEXT NOT NULL
);
"""

CREATE_TURN_CONTEXT = """
CREATE TABLE IF NOT EXISTS turn_context (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id       TEXT NOT NULL REFERENCES turns(turn_id),
    doc_id        TEXT NOT NULL,
    url           TEXT NOT NULL,
    title         TEXT,
    domain        TEXT,
    snippet       TEXT NOT NULL,
    bm25_score    REAL,
    recency_score REAL,
    final_score   REAL,
    retrieved_at  TEXT
);
"""

CREATE_SESSION_SUMMARIES = """
CREATE TABLE IF NOT EXISTS session_summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    summary       TEXT NOT NULL,
    turns_covered TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""

CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_content USING fts5(
    turn_id UNINDEXED,
    content,
    tokenize="porter ascii"
);
"""

CREATE_CONTRADICTION_PROBES = """
CREATE TABLE IF NOT EXISTS contradiction_probes (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id               TEXT NOT NULL REFERENCES turns(turn_id),
    has_conflict          INTEGER NOT NULL,
    conflict_summary      TEXT,
    contradictions_json   TEXT,
    probe_skipped_reason  TEXT,
    probe_ms              INTEGER,
    prompt_id             TEXT,
    created_at            TEXT NOT NULL
);
"""

CREATE_CONTRADICTION_PROBES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_probes_turn ON contradiction_probes(turn_id);
"""

CREATE_EVAL_RUNS = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id                   TEXT PRIMARY KEY,
    run_at                   TEXT NOT NULL,
    question_id              TEXT NOT NULL,
    question                 TEXT NOT NULL,
    category                 TEXT NOT NULL,
    turn_count               INTEGER DEFAULT 1,
    agent_answer             TEXT,
    faithfulness_score       REAL,
    answer_relevance_score   REAL,
    context_precision_score  REAL,
    citation_integrity_score REAL,
    conflict_adherence_score REAL,
    session_coherence_score  REAL,
    judge_reasoning          TEXT,
    failure_class            TEXT,
    latency_ms               INTEGER
);
"""


async def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_SESSIONS)
        await db.execute(CREATE_TURNS)
        await db.execute(CREATE_TURN_CONTEXT)
        await db.execute(CREATE_SESSION_SUMMARIES)
        await db.execute(CREATE_FTS)
        await db.execute(CREATE_CONTRADICTION_PROBES)
        await db.execute(CREATE_CONTRADICTION_PROBES_INDEX)
        await db.execute(CREATE_EVAL_RUNS)
        try:
            await db.execute("ALTER TABLE eval_runs ADD COLUMN context_precision_score REAL")
        except aiosqlite.OperationalError:
            pass
        for col, ddl in [
            ("planning_ms", "ALTER TABLE turns ADD COLUMN planning_ms INTEGER DEFAULT 0"),
            ("search_ms", "ALTER TABLE turns ADD COLUMN search_ms INTEGER DEFAULT 0"),
            ("fetch_ms", "ALTER TABLE turns ADD COLUMN fetch_ms INTEGER DEFAULT 0"),
            ("select_ms", "ALTER TABLE turns ADD COLUMN select_ms INTEGER DEFAULT 0"),
            ("synthesize_ms", "ALTER TABLE turns ADD COLUMN synthesize_ms INTEGER DEFAULT 0"),
            ("run_metadata_json", "ALTER TABLE turns ADD COLUMN run_metadata_json TEXT"),
        ]:
            try:
                await db.execute(ddl)
            except aiosqlite.OperationalError:
                pass
        await db.commit()
    logger.info("DB initialised", extra={"component": "memory"})


async def create_session(session_id: str, now: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        await db.commit()


async def save_turn(turn: Turn) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO turns
            (turn_id, session_id, query, plan, search_queries, urls_opened, response,
             context_xml_sent, doc_map, citation_integrity_score, hallucination_count,
             prompt_tokens, completion_tokens, latency_ms, planning_ms, search_ms,
             fetch_ms, select_ms, synthesize_ms, run_metadata_json, state_trace, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                turn.turn_id, turn.session_id, turn.query, turn.plan,
                json.dumps(turn.search_queries), json.dumps(turn.urls_opened),
                turn.response, turn.context_xml_sent,
                json.dumps(turn.doc_map) if turn.doc_map else None,
                turn.citation_integrity_score, turn.hallucination_count,
                turn.prompt_tokens, turn.completion_tokens,
                turn.latency_ms, turn.planning_ms, turn.search_ms, turn.fetch_ms,
                turn.select_ms, turn.synthesize_ms,
                json.dumps(turn.run_metadata_json) if turn.run_metadata_json else None,
                json.dumps(turn.state_trace),
                turn.created_at,
            ),
        )
        # FTS5 index — concatenate query + response for keyword retrieval
        fts_text = f"{turn.query} {turn.response or ''}".strip()
        await db.execute(
            "INSERT INTO fts_content (turn_id, content) VALUES (?, ?)",
            (turn.turn_id, fts_text),
        )
        await db.execute(
            "UPDATE sessions SET updated_at = ?, turn_count = turn_count + 1 WHERE session_id = ?",
            (turn.created_at, turn.session_id),
        )
        await db.commit()
    logger.info("Turn saved", extra={"component": "memory", "turn_id": turn.turn_id})


async def save_turn_context(turn_id: str, snippets: list[ContextSnippet]) -> None:
    """Persist selected context snippets for retrieval traceability."""
    if not snippets:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            """
            INSERT INTO turn_context
            (turn_id, doc_id, url, title, domain, snippet, bm25_score, recency_score, final_score, retrieved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    turn_id,
                    s.doc_id,
                    s.url,
                    s.title,
                    s.domain,
                    s.text,
                    s.bm25_score,
                    s.recency_score,
                    s.final_score,
                    s.retrieved_at,
                )
                for s in snippets
            ],
        )
        await db.commit()


async def save_contradiction_probe(
    turn_id: str,
    result: ConflictResult,
    probe_ms: int,
    prompt_id: str,
) -> None:
    """Persist one row per turn capturing the probe outcome (including skips)."""
    contradictions_json = json.dumps([c.model_dump() for c in result.contradictions])
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO contradiction_probes
            (turn_id, has_conflict, conflict_summary, contradictions_json,
             probe_skipped_reason, probe_ms, prompt_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                1 if result.has_conflict else 0,
                result.conflict_summary,
                contradictions_json,
                result.probe_skipped_reason,
                probe_ms,
                prompt_id,
                now,
            ),
        )
        await db.commit()


async def get_session_turn_count(session_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchall(
            "SELECT turn_count FROM sessions WHERE session_id = ?", (session_id,)
        )
    return row[0]["turn_count"] if row else 0


async def get_relevant_prior_turns(session_id: str, current_query: str, limit: int = 3) -> list[Turn]:
    """FTS5 keyword retrieval of the most relevant prior turns (not just last-N)."""
    env_limit = int(os.getenv("HISTORY_RETRIEVAL_LIMIT", str(limit)))
    # FTS5-safe normalization: retain alnum and hyphen/underscore, strip operators.
    cleaned_tokens: list[str] = []
    for raw in current_query.split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
        if token:
            cleaned_tokens.append(token)
    fts_q = " ".join(cleaned_tokens)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = []
        if fts_q:
            try:
                rows = await db.execute_fetchall(
                    """
                    SELECT t.* FROM turns t
                    JOIN fts_content f ON f.turn_id = t.turn_id
                    WHERE f.content MATCH ? AND t.session_id = ?
                    ORDER BY rank LIMIT ?
                    """,
                    (fts_q, session_id, env_limit),
                )
            except Exception:
                rows = []
        if not rows:
            rows = await db.execute_fetchall(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, env_limit),
            )
    return [Turn.from_row(dict(r)) for r in rows]


async def get_session_turns(session_id: str) -> list[Turn]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        )
    return [Turn.from_row(dict(r)) for r in rows]


async def save_session_summary(session_id: str, summary: str, turn_ids: list[str], now: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO session_summaries (session_id, summary, turns_covered, created_at) VALUES (?,?,?,?)",
            (session_id, summary, json.dumps(turn_ids), now),
        )
        await db.commit()


async def get_latest_summary(session_id: str) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT summary FROM session_summaries WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
    return rows[0]["summary"] if rows else None


async def session_exists(session_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        rows = await db.execute_fetchall(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        )
    return bool(rows)
