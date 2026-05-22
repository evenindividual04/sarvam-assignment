"""
SQLite-backed search result cache.

The single biggest free-tier quota burner is eval re-runs hitting the same
dataset questions through the same providers. Caching by (provider, query)
with a 24h TTL makes re-runs essentially free on search quota, while keeping
the *first* run honest.

Recency-sensitive queries (intent_origin == "recency_check") bypass the cache —
caching today's news for 24h would defeat the point of that intent. All other
intents are fair game.

The cache is content-addressed by a SHA-256 of the normalized query string,
making it deterministic and trivially auditable. Cache hits are logged so the
trace inspector can flag "served from cache" turns.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Optional

import aiosqlite

from agent.memory import DB_PATH
from agent.models import SearchResult

logger = logging.getLogger(__name__)

_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_S", "86400"))  # 24h default
_DISABLED = os.getenv("SEARCH_CACHE_DISABLED", "0").strip() == "1"


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS search_cache (
    cache_key   TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    query       TEXT NOT NULL,
    results     TEXT NOT NULL,
    cached_at   REAL NOT NULL
);
"""


CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_search_cache_cached_at ON search_cache(cached_at);
"""


def _cache_key(provider: str, query: str) -> str:
    """Stable hash so we can index by it. SHA-256 is overkill but cheap."""
    norm = f"{provider}|{query.strip().lower()}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute(CREATE_TABLE)
    await db.execute(CREATE_INDEX)
    await db.commit()


async def get(provider: str, query: str) -> Optional[list[SearchResult]]:
    """Look up cached results for (provider, query). Returns None on miss
    or when the cache is disabled. Stale entries are returned as None and
    not auto-deleted — the next `put` will overwrite them, keeping the
    hot path single-statement."""
    if _DISABLED:
        return None
    key = _cache_key(provider, query)
    cutoff = time.time() - _TTL_SECONDS
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_table(db)
            db.row_factory = aiosqlite.Row
            row = await db.execute_fetchall(
                "SELECT results FROM search_cache WHERE cache_key = ? AND cached_at >= ?",
                (key, cutoff),
            )
            if not row:
                return None
            data = json.loads(row[0]["results"])
        results = [SearchResult(**item) for item in data]
        logger.info(
            "search_cache HIT provider=%s n=%d query=%r",
            provider, len(results), query[:80],
            extra={"component": "search_cache"},
        )
        return results
    except Exception as exc:  # pragma: no cover — cache must never block the request
        logger.warning("search_cache get failed: %s", exc, extra={"component": "search_cache"})
        return None


async def put(provider: str, query: str, results: list[SearchResult]) -> None:
    """Persist a (provider, query) → results entry with the current timestamp.
    Idempotent: `INSERT OR REPLACE` so re-running the same query with fresh
    upstream data just overwrites the stale row."""
    if _DISABLED or not results:
        return
    key = _cache_key(provider, query)
    try:
        payload = json.dumps([_serialize(r) for r in results])
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_table(db)
            await db.execute(
                "INSERT OR REPLACE INTO search_cache (cache_key, provider, query, results, cached_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, provider, query, payload, time.time()),
            )
            await db.commit()
    except Exception as exc:  # pragma: no cover
        logger.warning("search_cache put failed: %s", exc, extra={"component": "search_cache"})


def _serialize(r: SearchResult) -> dict:
    """SearchResult → JSON-safe dict. Mirrors dataclass field set; explicit
    rather than `asdict` so we control schema drift if SearchResult evolves."""
    return {
        "url": r.url,
        "title": r.title,
        "snippet": r.snippet,
        "domain": r.domain,
        "retrieved_at": r.retrieved_at,
        "raw_content": r.raw_content,
        "intent_origin": r.intent_origin,
        "relevance": r.relevance,
        "relevance_source": r.relevance_source,
    }


def should_bypass(intent: Optional[str]) -> bool:
    """Recency-sensitive intents bypass cache. Cached `recency_check` would
    defeat the whole point of that intent (fetching today's news)."""
    return intent == "recency_check"
