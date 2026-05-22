"""Serper Scholar (Google Scholar) wrapper. Uses existing SERPER_API_KEY.
Returns up to ``top_k`` academic results. Supplementary for queries whose
planner-derived ``expected_source_types`` includes ``"academic"``.

Graceful: missing key → ``[]``; HTTP/network failures → ``[]``.

Disable via ``SCHOLAR_DISABLED=1``."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from agent.models import SearchResult

logger = logging.getLogger(__name__)

_SCHOLAR_BASE = "https://google.serper.dev/scholar"


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def fetch_serper_scholar(
    query: str,
    top_k: int = 5,
    timeout_s: float = 8.0,
) -> list[SearchResult]:
    """POST to Serper Scholar; return up to ``top_k`` parsed results.

    Returns an empty list on any failure (no API key, timeout, HTTP error,
    malformed payload, or explicit disable).
    """
    if os.getenv("SCHOLAR_DISABLED", "").strip() in {"1", "true", "True"}:
        logger.debug("Scholar disabled via env", extra={"component": "scholar"})
        return []
    if not query or not query.strip():
        return []
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                _SCHOLAR_BASE,
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query.strip(), "num": top_k},
            )
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
        logger.info("Scholar fetch failed for '%s': %s", query, e, extra={"component": "scholar"})
        return []
    except Exception as e:  # noqa: BLE001
        logger.warning("Scholar unexpected error for '%s': %s", query, e, extra={"component": "scholar"})
        return []

    if resp.status_code != 200:
        logger.info("Scholar returned %d for '%s'", resp.status_code, query, extra={"component": "scholar"})
        return []

    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.info("Scholar non-JSON response for '%s': %s", query, e, extra={"component": "scholar"})
        return []

    organic = data.get("organic") or []
    if not isinstance(organic, list):
        return []

    now = _now_iso()
    results: list[SearchResult] = []
    for idx, item in enumerate(organic[:top_k]):
        if not isinstance(item, dict):
            continue
        url = (item.get("link") or "").strip()
        title = (item.get("title") or "").strip()
        if not url or not title:
            continue
        snippet = (item.get("snippet") or "").strip()
        results.append(
            SearchResult(
                url=url,
                title=title,
                snippet=snippet,
                domain=_domain(url),
                retrieved_at=now,
                raw_content=None,
                intent_origin="academic",
                relevance=max(0.0, 1.0 - 0.1 * idx),
                relevance_source="rank",
            )
        )
    logger.info("Scholar returned %d results", len(results), extra={"component": "scholar"})
    return results
