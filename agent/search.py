"""
Search providers: Parallel (primary) → Tavily (fallback) → Serper (last resort).
All return list[SearchResult]. URL deduplication is applied across all providers.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.models import QueryIntent, SearchResult, TypedQuery
from utils.circuit_breaker import CircuitOpenError, breaker
from utils.provider_adapters import normalize_search_items
# Importing failure_policy registers all breakers at module import time.
from utils import failure_policy as _failure_policy  # noqa: F401

logger = logging.getLogger(__name__)

_PARALLEL_BASE = "https://api.parallel.ai/v1"
_SERPER_BASE = "https://google.serper.dev/search"
_MAX_RESULTS = int(os.getenv("AGENT_MAX_SOURCES", "8"))

# Devanagari Unicode block: U+0900–U+097F. Used for Hindi (and other Indic) detection.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def _detect_language(text: str) -> str:
    """Detect query language. Currently 'hi' for any Devanagari content, else 'en'.

    V3.4 scope: Hindi-only Indic detection. Other Indic scripts can be added later
    without changing the routing contract (callers only care about 'hi' vs not)."""
    if not text:
        return "en"
    return "hi" if _DEVANAGARI_RE.search(text) else "en"


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


@breaker("parallel")
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def _search_parallel(query: str, client: httpx.AsyncClient, num_results: int = _MAX_RESULTS) -> list[SearchResult]:
    """Parallel AI — returns AI-native structured excerpts; no separate fetch needed."""
    key = os.getenv("PARALLEL_API_KEY", "")
    if not key:
        return []
    resp = await client.post(
        f"{_PARALLEL_BASE}/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": query, "num_results": num_results},
        timeout=20.0,
    )
    if resp.status_code == 422:
        # Backward/variant payload compatibility for Parallel API schema changes.
        resp = await client.post(
            f"{_PARALLEL_BASE}/search",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"q": query, "max_results": num_results},
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


@breaker("tavily")
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


@breaker("serper")
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


@breaker("tavily")
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def _search_tavily_news(query: str, client: httpx.AsyncClient, days: int = 30) -> list[SearchResult]:
    """Tavily in news mode for RECENCY_CHECK intent."""
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return []
    try:
        from tavily import AsyncTavilyClient
        tavily = AsyncTavilyClient(api_key=key)
        data = await tavily.search(
            query=query,
            topic="news",
            days=days,
            search_depth="advanced",
            include_raw_content=True,
            max_results=_MAX_RESULTS,
        )
    except Exception:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "topic": "news",
                "days": days,
                "search_depth": "advanced",
                "include_raw_content": True,
                "max_results": _MAX_RESULTS,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
    adapted = normalize_search_items("tavily", data, _now(), _domain)
    return adapted.results


async def _try(coro_factory, label: str, q: str) -> list[SearchResult]:
    try:
        return await coro_factory()
    except CircuitOpenError as e:
        logger.info("%s skipped (breaker open) for query '%s': %s", label, q, e, extra={"component": "search"})
        return []
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, Exception) as e:
        logger.warning("%s failed for query '%s': %s", label, q, e, extra={"component": "search"})
        return []


async def _search_single_query(
    q: str,
    client: httpx.AsyncClient,
    intent: QueryIntent = QueryIntent.PRIMARY,
    language: str | None = None,
) -> list[SearchResult]:
    """Route a query to providers based on intent. Returns results tagged with intent_origin.

    V3.4: when ``language == "hi"`` (Devanagari detected), prefer Parallel only —
    Tavily and Serper return English-heavy results for Hindi queries. If Parallel
    fails (or breaker open), fall through to the standard chain so the turn still
    completes in a degraded mode."""
    results: list[SearchResult] = []

    if language is None:
        language = _detect_language(q)

    if language == "hi":
        results = await _try(lambda: _search_parallel(q, client), "Parallel(hi)", q)
        if not results:
            logger.info(
                "Hindi query Parallel returned empty/breaker-open; falling through to Tavily/Serper",
                extra={"component": "search"},
            )
            results = await _try(lambda: _search_tavily(q, client), "Tavily(hi-fallback)", q)
            if not results:
                results = await _try(lambda: _search_serper(q, client), "Serper(hi-fallback)", q)
        tag = "contradiction_probe" if intent == QueryIntent.CONTRADICTION_PROBE else intent.value
        for r in results:
            if r.intent_origin is None:
                r.intent_origin = tag
        return results

    if intent == QueryIntent.RECENCY_CHECK and os.getenv("TAVILY_API_KEY"):
        results = await _try(lambda: _search_tavily_news(q, client, days=30), "Tavily(news)", q)
        if not results:
            results = await _try(lambda: _search_parallel(q, client), "Parallel", q)
        if not results:
            results = await _try(lambda: _search_serper(q, client), "Serper", q)
    elif intent == QueryIntent.COMPARISON:
        results = await _try(lambda: _search_parallel(q, client, num_results=_MAX_RESULTS + 2), "Parallel", q)
        if not results:
            results = await _try(lambda: _search_tavily(q, client), "Tavily", q)
        if not results:
            results = await _try(lambda: _search_serper(q, client), "Serper", q)
    else:
        # PRIMARY, DEFINITION, CONTRADICTION_PROBE → existing chain
        results = await _try(lambda: _search_parallel(q, client), "Parallel", q)
        if not results:
            results = await _try(lambda: _search_tavily(q, client), "Tavily", q)
        if not results:
            results = await _try(lambda: _search_serper(q, client), "Serper", q)

    tag = "contradiction_probe" if intent == QueryIntent.CONTRADICTION_PROBE else intent.value
    for r in results:
        if r.intent_origin is None:
            r.intent_origin = tag
    return results


def _dedup_preserve_origin(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
    return out


async def search(queries: list[TypedQuery], cancel_token=None) -> list[SearchResult]:
    """
    Run typed queries against the intent-routed provider chain. Returns deduplicated
    list; first intent_origin seen per URL is preserved.
    """
    import asyncio
    if cancel_token is not None and cancel_token.is_set():
        return []
    all_results: list[SearchResult] = []
    async with httpx.AsyncClient() as client:
        tasks = []
        for tq in queries:
            if cancel_token is not None and cancel_token.is_set():
                break
            tasks.append(
                _search_single_query(
                    tq.text, client, tq.intent, language=_detect_language(tq.text)
                )
            )
        results_list = await asyncio.gather(*tasks)
        for results in results_list:
            all_results.extend(results)

    deduped = _dedup_preserve_origin(all_results)
    logger.info("Total unique results: %d", len(deduped), extra={"component": "search"})
    return deduped
