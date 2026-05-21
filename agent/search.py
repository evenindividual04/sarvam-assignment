"""
Search providers: Parallel (primary) → Tavily (fallback) → Serper (last resort).
All return list[SearchResult]. URL deduplication is applied across all providers.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.models import SearchResult
from utils.provider_adapters import normalize_search_items

logger = logging.getLogger(__name__)

_PARALLEL_BASE = "https://api.parallel.ai/v1"
_SERPER_BASE = "https://google.serper.dev/search"
_MAX_RESULTS = int(os.getenv("AGENT_MAX_SOURCES", "8"))


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedup(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return out


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def _search_parallel(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """Parallel AI — returns AI-native structured excerpts; no separate fetch needed."""
    key = os.getenv("PARALLEL_API_KEY", "")
    if not key:
        return []
    resp = await client.post(
        f"{_PARALLEL_BASE}/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": query, "num_results": _MAX_RESULTS},
        timeout=20.0,
    )
    if resp.status_code == 422:
        # Backward/variant payload compatibility for Parallel API schema changes.
        resp = await client.post(
            f"{_PARALLEL_BASE}/search",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"q": query, "max_results": _MAX_RESULTS},
            timeout=20.0,
        )
    if resp.status_code == 422:
        logger.warning("Parallel rejected request schema for query '%s'", query, extra={"component": "search"})
        return []
    resp.raise_for_status()
    data = resp.json()
    adapted = normalize_search_items("parallel", data, _now(), _domain)
    if not adapted.outcome.ok:
        logger.warning(
            "Parallel adapter normalized mismatch: %s (%s)",
            adapted.outcome.adapter_error_code,
            adapted.outcome.reason,
            extra={"component": "search"},
        )
    results = adapted.results
    logger.info("Parallel returned %d results", len(results), extra={"component": "search"})
    return results


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def _search_tavily(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """Tavily fallback — include_raw_content=True collapses the fetch pipeline."""
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return []
    try:
        from tavily import AsyncTavilyClient
        tavily = AsyncTavilyClient(api_key=key)
        data = await tavily.search(
            query=query,
            search_depth="advanced",
            include_raw_content=True,
            max_results=_MAX_RESULTS,
        )
    except Exception:
        # fallback to direct httpx if tavily-python unavailable
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "search_depth": "advanced",
                "include_raw_content": True,
                "max_results": _MAX_RESULTS,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()

    adapted = normalize_search_items("tavily", data, _now(), _domain)
    if not adapted.outcome.ok:
        logger.warning(
            "Tavily adapter normalized mismatch: %s (%s)",
            adapted.outcome.adapter_error_code,
            adapted.outcome.reason,
            extra={"component": "search"},
        )
    results = adapted.results
    logger.info("Tavily returned %d results", len(results), extra={"component": "search"})
    return results


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def _search_serper(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """Serper last resort — returns snippets only; Trafilatura extracts full content."""
    key = os.getenv("SERPER_API_KEY", "")
    if not key:
        return []
    resp = await client.post(
        _SERPER_BASE,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "num": _MAX_RESULTS},
        timeout=20.0,
    )
    resp.raise_for_status()
    data = resp.json()
    adapted = normalize_search_items("serper", data, _now(), _domain)
    if not adapted.outcome.ok:
        logger.warning(
            "Serper adapter normalized mismatch: %s (%s)",
            adapted.outcome.adapter_error_code,
            adapted.outcome.reason,
            extra={"component": "search"},
        )
    # Serper should remain snippet-only even if adapter sees content keys.
    results = [SearchResult(**{**r.__dict__, "raw_content": None}) for r in adapted.results]
    logger.info("Serper returned %d results", len(results), extra={"component": "search"})
    return results


async def _search_single_query(q: str, client: httpx.AsyncClient) -> list[SearchResult]:
    try:
        results = await _search_parallel(q, client)
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, Exception) as e:
        logger.warning("Parallel failed for query '%s': %s", q, e, extra={"component": "search"})
        results = []
    if not results:
        logger.info("Parallel empty, trying Tavily for: %s", q, extra={"component": "search"})
        try:
            results = await _search_tavily(q, client)
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, Exception) as e:
            logger.warning("Tavily failed for query '%s': %s", q, e, extra={"component": "search"})
            results = []
    if not results:
        logger.info("Tavily empty, trying Serper for: %s", q, extra={"component": "search"})
        try:
            results = await _search_serper(q, client)
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, Exception) as e:
            logger.warning("Serper failed for query '%s': %s", q, e, extra={"component": "search"})
            results = []
    return results

async def search(queries: list[str]) -> list[SearchResult]:
    """
    Run all queries against Parallel → Tavily → Serper with URL deduplication.
    Returns deduplicated list across all queries and all providers.
    """
    all_results: list[SearchResult] = []
    async with httpx.AsyncClient() as client:
        tasks = [_search_single_query(q, client) for q in queries]
        import asyncio
        results_list = await asyncio.gather(*tasks)
        for results in results_list:
            all_results.extend(results)

    deduped = _dedup(all_results)
    logger.info("Total unique results: %d", len(deduped), extra={"component": "search"})
    return deduped
