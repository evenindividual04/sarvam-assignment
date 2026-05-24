"""SQLite-backed TTL caches for page fetches and search results.

Two narrow caches, both backed by the same SQLite DB the rest of the app
uses (``agent.memory.DB_PATH``):

  • ``fetched_pages``  — keyed by normalized URL. TTL default 24h. Saves
    Trafilatura + httpx + (Tavily Extract / Jina Reader) round-trips when
    the same URL appears across turns/sessions. Wikipedia/Britannica show
    up over and over, so this is high-ROI.

  • ``search_results`` — keyed by ``(provider, query_norm)``. TTL default
    1h. Makes eval re-runs and dev iteration nearly free, and stops us
    burning Parallel/Tavily quota on identical queries during a debug loop.

Both caches are best-effort: any DB error logs a warning and falls through
to the live fetch. Tests use the ``DB_PATH`` monkeypatch fixture.

We deliberately do NOT cache the final synthesis answer — research queries
should reflect current web data, and a stale "what's the RBI repo rate"
answer is worse than a slow correct one.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import aiosqlite

from agent.memory import DB_PATH
from utils.url_norm import normalize_url

logger = logging.getLogger(__name__)


CREATE_FETCHED_PAGES = """
CREATE TABLE IF NOT EXISTS fetched_pages (
    url_norm     TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    text         TEXT NOT NULL,
    title        TEXT,
    domain       TEXT,
    retrieved_at TEXT NOT NULL,
    cached_at    INTEGER NOT NULL
);
"""

CREATE_SEARCH_CACHE = """
CREATE TABLE IF NOT EXISTS search_results_cache (
    cache_key    TEXT PRIMARY KEY,    -- provider||query_norm
    provider     TEXT NOT NULL,
    query        TEXT NOT NULL,
    results_json TEXT NOT NULL,
    cached_at    INTEGER NOT NULL
);
"""

CREATE_SEARCH_CACHE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_search_cache_cached_at
ON search_results_cache(cached_at);
"""


# Default TTLs (callers can override per-call).
DEFAULT_PAGE_TTL_S = 24 * 60 * 60     # 24h
DEFAULT_SEARCH_TTL_S = 60 * 60        # 1h


async def init_cache_tables() -> None:
    """Create cache tables. Called from agent.memory.init_db()."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_FETCHED_PAGES)
        await db.execute(CREATE_SEARCH_CACHE)
        await db.execute(CREATE_SEARCH_CACHE_INDEX)
        await db.commit()


# ── Page-fetch cache ─────────────────────────────────────────────────────


async def get_cached_page(
    url: str,
    ttl_seconds: int = DEFAULT_PAGE_TTL_S,
) -> Optional[dict[str, Any]]:
    """Return ``{"text", "title", "domain", "retrieved_at"}`` if a fresh row
    exists for this URL, else None. Best-effort: any DB error → None."""
    if not url:
        return None
    key = normalize_url(url)
    if not key:
        return None
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            row = await db.execute_fetchall(
                "SELECT url, text, title, domain, retrieved_at, cached_at "
                "FROM fetched_pages WHERE url_norm = ?",
                (key,),
            )
        if not row:
            return None
        r = row[0]
        if time.time() - r["cached_at"] > ttl_seconds:
            return None
        return {
            "url": r["url"],
            "text": r["text"],
            "title": r["title"],
            "domain": r["domain"],
            "retrieved_at": r["retrieved_at"],
        }
    except Exception as exc:
        logger.debug("page cache get failed: %s", exc, extra={"component": "cache"})
        return None


async def set_cached_page(
    url: str,
    text: str,
    *,
    title: str = "",
    domain: str = "",
    retrieved_at: str = "",
) -> None:
    """Write a page row. No-op on empty url/text. Best-effort."""
    if not url or not text:
        return
    key = normalize_url(url)
    if not key:
        return
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO fetched_pages
                  (url_norm, url, text, title, domain, retrieved_at, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_norm) DO UPDATE SET
                  url=excluded.url,
                  text=excluded.text,
                  title=excluded.title,
                  domain=excluded.domain,
                  retrieved_at=excluded.retrieved_at,
                  cached_at=excluded.cached_at
                """,
                (key, url, text, title or "", domain or "", retrieved_at or "", int(time.time())),
            )
            await db.commit()
    except Exception as exc:
        logger.debug("page cache set failed: %s", exc, extra={"component": "cache"})


# ── Search-result cache ──────────────────────────────────────────────────


def _search_key(provider: str, query: str) -> str:
    """Stable cache key from provider + normalized query."""
    q = (query or "").strip().lower()
    p = (provider or "").strip().lower()
    return f"{p}||{q}"


async def get_cached_search(
    provider: str,
    query: str,
    ttl_seconds: int = DEFAULT_SEARCH_TTL_S,
) -> Optional[list[dict[str, Any]]]:
    """Return a list of raw search-result dicts if a fresh row exists, else None.

    Caller is responsible for adapting dicts back into ``SearchResult`` if
    needed (this module stays serialization-agnostic).
    """
    if not provider or not query:
        return None
    key = _search_key(provider, query)
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            row = await db.execute_fetchall(
                "SELECT results_json, cached_at FROM search_results_cache "
                "WHERE cache_key = ?",
                (key,),
            )
        if not row:
            return None
        r = row[0]
        if time.time() - r["cached_at"] > ttl_seconds:
            return None
        return json.loads(r["results_json"])
    except Exception as exc:
        logger.debug("search cache get failed: %s", exc, extra={"component": "cache"})
        return None


async def set_cached_search(
    provider: str,
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """Write a search-result row. No-op on empty inputs. Best-effort."""
    if not provider or not query or not results:
        return
    key = _search_key(provider, query)
    try:
        payload = json.dumps(results, default=str)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO search_results_cache
                  (cache_key, provider, query, results_json, cached_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                  results_json=excluded.results_json,
                  cached_at=excluded.cached_at
                """,
                (key, provider.lower(), query, payload, int(time.time())),
            )
            await db.commit()
    except Exception as exc:
        logger.debug("search cache set failed: %s", exc, extra={"component": "cache"})


# ── Maintenance ──────────────────────────────────────────────────────────


async def purge_expired(
    *,
    page_ttl_seconds: int = DEFAULT_PAGE_TTL_S,
    search_ttl_seconds: int = DEFAULT_SEARCH_TTL_S,
) -> dict[str, int]:
    """Drop rows older than each TTL. Returns ``{"pages": n, "searches": n}``.
    Cheap to call on startup; not wired into a scheduler by default."""
    now = int(time.time())
    out = {"pages": 0, "searches": 0}
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                "DELETE FROM fetched_pages WHERE cached_at < ?",
                (now - page_ttl_seconds,),
            )
            out["pages"] = cur.rowcount or 0
            cur = await db.execute(
                "DELETE FROM search_results_cache WHERE cached_at < ?",
                (now - search_ttl_seconds,),
            )
            out["searches"] = cur.rowcount or 0
            await db.commit()
    except Exception as exc:
        logger.debug("cache purge failed: %s", exc, extra={"component": "cache"})
    return out
