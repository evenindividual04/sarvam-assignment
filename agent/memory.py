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

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "research.db"),
)
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)


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

CREATE_CLAIM_AUDIT = """
CREATE TABLE IF NOT EXISTS claim_audit (
    audit_id        TEXT PRIMARY KEY,
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    claim_idx       INTEGER NOT NULL,
    claim_text      TEXT NOT NULL,
    cited_doc_ids   TEXT NOT NULL,
    method          TEXT NOT NULL,
    overlap         REAL,
    entity_match    REAL,
    score           REAL NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""

CREATE_CLAIM_AUDIT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_claim_audit_turn ON claim_audit(turn_id);
"""

CREATE_CIRCUIT_EVENTS = """
CREATE TABLE IF NOT EXISTS circuit_events (
    event_id     TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    from_state   TEXT NOT NULL,
    to_state     TEXT NOT NULL,
    reason       TEXT,
    at           TEXT NOT NULL
);
"""

CREATE_CIRCUIT_EVENTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_circuit_events_provider ON circuit_events(provider);
"""

# V3.1 — sqlite-vec virtual table for ephemeral per-turn chunk embeddings.
# Created idempotently in init_db() AFTER the extension is loaded; if loading
# fails the hybrid path silently degrades to BM25-only.
CREATE_CHUNK_EMBEDDINGS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[384]
);
"""

# Module-level flag set by init_db(); helpers read it to short-circuit when
# sqlite-vec isn't available on this build.
_VEC_AVAILABLE: bool = False


CREATE_CROSS_LANGUAGE_CONSISTENCY = """
CREATE TABLE IF NOT EXISTS cross_language_consistency (
    run_at               TEXT NOT NULL,
    concept_id           TEXT NOT NULL,
    en_question_id       TEXT NOT NULL,
    hi_question_id       TEXT NOT NULL,
    jaccard_score        REAL NOT NULL,
    flagged_inconsistent INTEGER NOT NULL,
    reasoning            TEXT,
    PRIMARY KEY (run_at, concept_id)
);
"""

CREATE_EVAL_RUN_SUMMARY = """
CREATE TABLE IF NOT EXISTS eval_run_summary (
    run_at                       TEXT PRIMARY KEY,
    ablation_id                  TEXT,
    n_questions                  INTEGER NOT NULL,
    pass_rate                    REAL NOT NULL,
    mean_faithfulness            REAL,
    mean_relevance               REAL,
    mean_context_precision       REAL,
    mean_citation_integrity      REAL,
    mean_claim_precision         REAL,
    mean_factual_accuracy        REAL,
    p50_latency_ms               INTEGER,
    p95_latency_ms               INTEGER,
    total_cost_usd               REAL DEFAULT 0.0,
    retrieval_mode               TEXT,
    calibration_correlation      REAL,
    created_at                   TEXT NOT NULL
);
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


async def _try_load_sqlite_vec(db: aiosqlite.Connection) -> bool:
    """Best-effort load of the sqlite-vec extension. Returns True on success.

    Two failure modes worth distinguishing:
      (a) `sqlite_vec` Python package not installed → ImportError.
      (b) Python's stdlib `sqlite3` was compiled without
          `--enable-loadable-sqlite-extensions` → `enable_load_extension`
          raises AttributeError. The package imports fine, but the load fails.

    This is *not* OS-specific. Builds that work include conda-forge/miniforge,
    modern Homebrew `python@3.x`, pyenv (when built with
    `PYTHON_CONFIGURE_OPTS=--enable-loadable-sqlite-extensions`), recent
    python.org installers, and the `python:3.12-slim` Docker image. The one
    common build that fails is Apple's bundled `/usr/bin/python3` on macOS.
    """
    try:
        import sqlite_vec  # lazy
    except ImportError as exc:
        logger.warning(
            "sqlite-vec package not installed; hybrid retrieval disabled: %s",
            exc, extra={"component": "memory"},
        )
        return False
    try:
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        return True
    except AttributeError as exc:
        logger.warning(
            "This Python's sqlite3 was compiled without "
            "--enable-loadable-sqlite-extensions; hybrid retrieval disabled. "
            "Switch to a Python build that has it (conda-forge/miniforge, "
            "modern Homebrew python@3.x, pyenv with the right configure flag, "
            "or the project's python:3.12-slim Docker image). Apple's "
            "/usr/bin/python3 on macOS is the typical offender. (%s)",
            exc, extra={"component": "memory"},
        )
        return False
    except Exception as exc:
        logger.warning(
            "sqlite-vec extension load failed; hybrid retrieval disabled: %s",
            exc, extra={"component": "memory"},
        )
        return False


async def sqlite_vec_capability() -> bool:
    """Probe (without persisting) whether sqlite-vec can actually load on this
    Python build. Returns True only if both the package is installed AND
    `enable_load_extension` is available AND the extension loads cleanly.
    """
    db = await aiosqlite.connect(":memory:")
    try:
        return await _try_load_sqlite_vec(db)
    finally:
        await db.close()


async def init_db() -> None:
    global _VEC_AVAILABLE
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        # Attempt to load sqlite-vec BEFORE any CREATE so virtual table works.
        _VEC_AVAILABLE = await _try_load_sqlite_vec(db)
        if _VEC_AVAILABLE:
            try:
                await db.execute(CREATE_CHUNK_EMBEDDINGS)
            except aiosqlite.Error as exc:
                logger.warning("chunk_embeddings create failed: %s", exc,
                               extra={"component": "memory"})
                _VEC_AVAILABLE = False
        await db.execute(CREATE_SESSIONS)
        await db.execute(CREATE_TURNS)
        await db.execute(CREATE_TURN_CONTEXT)
        await db.execute(CREATE_SESSION_SUMMARIES)
        await db.execute(CREATE_FTS)
        await db.execute(CREATE_CONTRADICTION_PROBES)
        await db.execute(CREATE_CONTRADICTION_PROBES_INDEX)
        await db.execute(CREATE_EVAL_RUNS)
        await db.execute(CREATE_CLAIM_AUDIT)
        await db.execute(CREATE_CLAIM_AUDIT_INDEX)
        await db.execute(CREATE_CIRCUIT_EVENTS)
        await db.execute(CREATE_CIRCUIT_EVENTS_INDEX)
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
            ("trust_score", "ALTER TABLE turn_context ADD COLUMN trust_score REAL DEFAULT 0.7"),
            ("claim_precision_score_turns", "ALTER TABLE turns ADD COLUMN claim_precision_score REAL DEFAULT 1.0"),
            ("claim_verification_json", "ALTER TABLE turns ADD COLUMN claim_verification_json TEXT"),
            ("claim_precision_score_eval", "ALTER TABLE eval_runs ADD COLUMN claim_precision_score REAL"),
            ("eval_turn_id", "ALTER TABLE eval_runs ADD COLUMN turn_id TEXT"),
            ("eval_language", "ALTER TABLE eval_runs ADD COLUMN language TEXT DEFAULT 'en'"),
            ("eval_retrieval_mode", "ALTER TABLE eval_runs ADD COLUMN retrieval_mode TEXT DEFAULT 'bm25'"),
            ("eval_factual_accuracy", "ALTER TABLE eval_runs ADD COLUMN factual_accuracy_score REAL"),
            ("eval_ablation_id", "ALTER TABLE eval_runs ADD COLUMN ablation_id TEXT"),
            ("eval_calibration_correlation", "ALTER TABLE eval_runs ADD COLUMN calibration_correlation REAL"),
            ("turn_context_provider_relevance", "ALTER TABLE turn_context ADD COLUMN provider_relevance REAL"),
            ("turn_context_provider_relevance_source", "ALTER TABLE turn_context ADD COLUMN provider_relevance_source TEXT"),
            # Tier A Phase 1+: promote per-turn quality + routing fields from
            # run_metadata into first-class eval_runs columns for dashboard queries.
            ("eval_quote_grounding_ratio", "ALTER TABLE eval_runs ADD COLUMN quote_grounding_ratio REAL"),
            ("eval_numeric_grounding_ratio", "ALTER TABLE eval_runs ADD COLUMN numeric_grounding_ratio REAL"),
            ("eval_criteria_coverage_ratio", "ALTER TABLE eval_runs ADD COLUMN criteria_coverage_ratio REAL"),
            ("eval_terminator_fired", "ALTER TABLE eval_runs ADD COLUMN terminator_fired TEXT"),
            ("eval_planner_provider", "ALTER TABLE eval_runs ADD COLUMN planner_provider TEXT"),
            ("eval_reranker_used", "ALTER TABLE eval_runs ADD COLUMN reranker_used TEXT"),
            ("eval_language_method", "ALTER TABLE eval_runs ADD COLUMN language_method TEXT"),
        ]:
            try:
                await db.execute(ddl)
            except aiosqlite.OperationalError:
                pass
        await db.execute(CREATE_CROSS_LANGUAGE_CONSISTENCY)
        await db.execute(CREATE_EVAL_RUN_SUMMARY)
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
             fetch_ms, select_ms, synthesize_ms, run_metadata_json, state_trace,
             claim_precision_score, claim_verification_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                turn.claim_precision_score,
                turn.claim_verification_json,
                turn.created_at,
            ),
        )
        # FTS5 index — concatenate query + response for keyword retrieval.
        # FTS5 has no unique constraint, so on replay (cancel-then-rerun under the
        # same turn_id) we must delete the prior FTS row before inserting.
        await db.execute("DELETE FROM fts_content WHERE turn_id = ?", (turn.turn_id,))
        fts_text = f"{turn.query} {turn.response or ''}".strip()
        await db.execute(
            "INSERT INTO fts_content (turn_id, content) VALUES (?, ?)",
            (turn.turn_id, fts_text),
        )
        # Derive turn_count from the source of truth (turns table) rather than
        # incrementing — that's both idempotent on `INSERT OR REPLACE` retries
        # and race-safe under concurrent /research calls on the same session.
        await db.execute(
            """
            UPDATE sessions
               SET updated_at = ?,
                   turn_count = (SELECT COUNT(*) FROM turns WHERE session_id = ?)
             WHERE session_id = ?
            """,
            (turn.created_at, turn.session_id, turn.session_id),
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
            (turn_id, doc_id, url, title, domain, snippet, bm25_score, recency_score,
             final_score, retrieved_at, trust_score,
             provider_relevance, provider_relevance_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    s.trust_score,
                    s.provider_relevance,
                    s.provider_relevance_source,
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


async def save_claim_audit(turn_id: str, records: list) -> None:
    """Persist one row per ClaimRecord. Caller passes list[ClaimRecord]."""
    if not records:
        return
    now = datetime.now(timezone.utc).isoformat()
    import uuid as _uuid
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            """
            INSERT INTO claim_audit
            (audit_id, turn_id, claim_idx, claim_text, cited_doc_ids,
             method, overlap, entity_match, score, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(_uuid.uuid4()),
                    turn_id,
                    idx,
                    r.claim_text,
                    json.dumps(list(r.doc_ids)),
                    r.method,
                    r.overlap,
                    r.entity_match,
                    r.score,
                    r.status,
                    now,
                )
                for idx, r in enumerate(records)
            ],
        )
        await db.commit()


async def save_circuit_event(event: dict) -> None:
    """Persist a circuit-breaker state transition. Non-fatal on failure."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO circuit_events
            (event_id, provider, from_state, to_state, reason, at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["provider"],
                event["from_state"],
                event["to_state"],
                event.get("reason"),
                event.get("at", now),
            ),
        )
        await db.commit()


# ── V3.1 chunk embedding helpers ───────────────────────────────────────────

async def _connect_with_vec() -> aiosqlite.Connection:
    """Open a connection and load sqlite-vec. Caller closes."""
    db = await aiosqlite.connect(DB_PATH)
    try:
        import sqlite_vec
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
    except Exception:
        await db.close()
        raise
    return db


async def save_chunk_embeddings(
    chunk_ids: list[str], embeddings: list[list[float]]
) -> None:
    """Persist ephemeral embeddings for the current turn's chunks."""
    if not _VEC_AVAILABLE or not chunk_ids:
        return
    import struct
    db = await _connect_with_vec()
    try:
        rows = [
            (cid, struct.pack(f"{len(emb)}f", *emb))
            for cid, emb in zip(chunk_ids, embeddings)
        ]
        await db.executemany(
            "INSERT OR REPLACE INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)",
            rows,
        )
        await db.commit()
    finally:
        await db.close()


async def vector_search(
    query_embedding: list[float], chunk_ids: list[str], top_k: int = 30,
) -> list[tuple[str, float]]:
    """KNN over the supplied chunk_id scope. Returns list of (chunk_id, distance)."""
    if not _VEC_AVAILABLE or not chunk_ids:
        return []
    import struct
    qbytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    placeholders = ",".join("?" * len(chunk_ids))
    sql = (
        f"SELECT chunk_id, distance FROM chunk_embeddings "
        f"WHERE embedding MATCH ? AND chunk_id IN ({placeholders}) "
        f"AND k = ? ORDER BY distance"
    )
    db = await _connect_with_vec()
    try:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(sql, (qbytes, *chunk_ids, top_k))
    finally:
        await db.close()
    return [(r["chunk_id"], float(r["distance"])) for r in rows]


async def delete_chunk_embeddings(chunk_ids: list[str]) -> None:
    """Per-turn cleanup so the corpus stays ephemeral."""
    if not _VEC_AVAILABLE or not chunk_ids:
        return
    placeholders = ",".join("?" * len(chunk_ids))
    db = await _connect_with_vec()
    try:
        await db.execute(
            f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        await db.commit()
    except Exception as exc:
        logger.warning("delete_chunk_embeddings failed: %s", exc,
                       extra={"component": "memory"})
    finally:
        await db.close()


async def save_cross_language_consistency(run_at: str, rows: list[dict]) -> None:
    """Persist per-concept cross-language consistency rows for a given run."""
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            """
            INSERT OR REPLACE INTO cross_language_consistency
            (run_at, concept_id, en_question_id, hi_question_id,
             jaccard_score, flagged_inconsistent, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_at,
                    r["concept_id"],
                    r["en_question_id"],
                    r["hi_question_id"],
                    float(r["jaccard_score"]),
                    1 if r["flagged_inconsistent"] else 0,
                    r.get("reasoning", ""),
                )
                for r in rows
            ],
        )
        await db.commit()


async def save_eval_run_summary(summary: dict) -> None:
    """Persist a single per-run aggregate row."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO eval_run_summary
            (run_at, ablation_id, n_questions, pass_rate,
             mean_faithfulness, mean_relevance, mean_context_precision,
             mean_citation_integrity, mean_claim_precision, mean_factual_accuracy,
             p50_latency_ms, p95_latency_ms, total_cost_usd,
             retrieval_mode, calibration_correlation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary["run_at"],
                summary.get("ablation_id"),
                summary["n_questions"],
                summary["pass_rate"],
                summary.get("mean_faithfulness"),
                summary.get("mean_relevance"),
                summary.get("mean_context_precision"),
                summary.get("mean_citation_integrity"),
                summary.get("mean_claim_precision"),
                summary.get("mean_factual_accuracy"),
                summary.get("p50_latency_ms"),
                summary.get("p95_latency_ms"),
                summary.get("total_cost_usd", 0.0),
                summary.get("retrieval_mode"),
                summary.get("calibration_correlation"),
                summary["created_at"],
            ),
        )
        await db.commit()


async def session_exists(session_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        rows = await db.execute_fetchall(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        )
    return bool(rows)
