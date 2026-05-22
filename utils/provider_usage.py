"""
Per-provider daily usage tracking.

Doesn't *increase* your free-tier quota — eliminates the surprise. Before an
eval run you can see "Gemini: 850k/1M tokens used today" and decide whether
to switch synthesizer rather than crash mid-run.

Storage: one row per (provider, UTC date) with counts. Idempotent UPSERT so
concurrent /research turns don't race. Free-tier limits are configured per
provider in `_DAILY_LIMITS`; missing entries mean "unknown limit", which is
honest about the providers (Sarvam, Ollama) where the free tier isn't
documented as a hard number.

Surfaced via `GET /health/providers` (already-existing endpoint).
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from agent.memory import DB_PATH

logger = logging.getLogger(__name__)


# Documented free-tier daily limits. Missing = "unknown / unbounded". Numbers
# come from each provider's documented quota at time of writing (May 2026);
# they're advisory, not enforced — the provider's own 429 is the real ceiling.
_DAILY_LIMITS: dict[str, dict[str, int]] = {
    "gemini":     {"requests": 1500,  "tokens": 1_000_000},  # 2.5 Flash free tier
    "groq":       {"requests": 14400, "tokens": 500_000},    # Llama 3.3 70B free tier
    "openrouter": {"requests": 200,   "tokens": 0},          # 200/day on DeepSeek free models
    "sarvam":     {"requests": 0,     "tokens": 0},          # not publicly documented
    "github":     {"requests": 150,   "tokens": 0},          # GitHub Models GPT-4o-mini
    "cerebras":   {"requests": 14400, "tokens": 1_000_000},  # Cerebras Cloud free tier (Llama 3.1)
    "ollama":     {"requests": 0,     "tokens": 0},          # local: unlimited
    "parallel":   {"requests": 16000, "tokens": 0},          # Base tier — 16k queries
    "tavily":     {"requests": 1000,  "tokens": 0},
    "serper":     {"requests": 2500,  "tokens": 0},
    "cohere":     {"requests": 1000,  "tokens": 0},   # Rerank v3.5 trial: 1000/mo
}


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS provider_usage (
    provider          TEXT NOT NULL,
    usage_date        TEXT NOT NULL,  -- YYYY-MM-DD UTC
    requests          INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    last_updated_at   REAL NOT NULL,
    PRIMARY KEY (provider, usage_date)
);
"""


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute(CREATE_TABLE)
    await db.commit()


async def record(
    provider: str,
    *,
    requests: int = 1,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Increment today's counters for `provider`. Never raises — usage
    tracking must NOT block the request path."""
    if os.getenv("PROVIDER_USAGE_DISABLED", "0").strip() == "1":
        return
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_table(db)
            await db.execute(
                """
                INSERT INTO provider_usage
                    (provider, usage_date, requests, prompt_tokens, completion_tokens, last_updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, usage_date) DO UPDATE SET
                    requests          = requests + excluded.requests,
                    prompt_tokens     = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    last_updated_at   = excluded.last_updated_at
                """,
                (
                    provider.lower(),
                    _today_utc(),
                    requests,
                    prompt_tokens,
                    completion_tokens,
                    datetime.now(timezone.utc).timestamp(),
                ),
            )
            await db.commit()
    except Exception as exc:  # pragma: no cover — never block on telemetry
        logger.warning(
            "provider_usage.record(%s) failed: %s", provider, exc,
            extra={"component": "provider_usage"},
        )
        return
    # Best-effort quota-threshold webhook. Fired in the background so the main
    # request path never waits on alerting. Failures are swallowed by
    # notify_quota_threshold itself.
    try:
        await _maybe_fire_quota_alert(
            provider.lower(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            requests=requests,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("quota alert dispatch skipped: %s", exc)


async def _maybe_fire_quota_alert(
    provider: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    requests: int,
) -> None:
    """Dispatch alerts when today's usage crosses the configured threshold.

    Reads today's totals (post-update) and asks the webhook helper to decide
    whether to POST. The webhook's own cooldown logic prevents duplicate
    alerts within QUOTA_ALERT_COOLDOWN_S.
    """
    limits = _DAILY_LIMITS.get(provider, {})
    token_limit = limits.get("tokens") or 0
    request_limit = limits.get("requests") or 0
    if token_limit <= 0 and request_limit <= 0:
        return
    # Refetch today's row to get accurate totals after the upsert.
    row = await _today_row(provider)
    if row is None:
        return
    from utils.quota_webhook import notify_quota_threshold  # local import: optional
    used_tokens = int(row["prompt_tokens"]) + int(row["completion_tokens"])
    used_requests = int(row["requests"])
    if token_limit > 0:
        asyncio.create_task(
            notify_quota_threshold(provider, used_tokens, token_limit, unit="tokens")
        )
    if request_limit > 0:
        asyncio.create_task(
            notify_quota_threshold(provider, used_requests, request_limit, unit="requests")
        )


async def _today_row(provider: str) -> Optional[aiosqlite.Row]:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM provider_usage WHERE provider = ? AND usage_date = ?",
                (provider, _today_utc()),
            )
            return await cur.fetchone()
    except Exception:
        return None


async def snapshot(date: Optional[str] = None) -> list[dict]:
    """Return today's (or `date`'s) usage rows. Joins documented limits so
    `/health/providers` can show 'used / limit' ratios."""
    date = date or _today_utc()
    out: list[dict] = []
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_table(db)
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT * FROM provider_usage WHERE usage_date = ? ORDER BY provider",
                (date,),
            )
        for r in rows:
            d = dict(r)
            limits = _DAILY_LIMITS.get(d["provider"], {})
            d["request_limit"] = limits.get("requests") or None
            d["token_limit"] = limits.get("tokens") or None
            out.append(d)
    except Exception as exc:  # pragma: no cover
        logger.warning("provider_usage.snapshot failed: %s", exc)
    return out
