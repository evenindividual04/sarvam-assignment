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
from utils.retry_helpers import wait_retry_after_or_exponential

from agent.models import QueryIntent, SearchResult, TypedQuery
from utils.circuit_breaker import CircuitOpenError, breaker
from utils.provider_adapters import normalize_search_items
# Importing failure_policy registers all breakers at module import time.
from utils import failure_policy as _failure_policy  # noqa: F401

logger = logging.getLogger(__name__)

_PARALLEL_BASE = "https://api.parallel.ai/v1beta"
_SERPER_BASE = "https://google.serper.dev/search"
_MAX_RESULTS = int(os.getenv("AGENT_MAX_SOURCES", "8"))

# Indic script Unicode blocks. Distinguish at the SCRIPT level only —
# Hindi and Marathi share Devanagari and can't be told apart from script alone;
# the eval dataset's explicit `language` field disambiguates them downstream.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")   # Devanagari: Hindi, Marathi, Sanskrit
_TAMIL_RE = re.compile(r"[஀-௿]")         # Tamil
_BENGALI_RE = re.compile(r"[ঀ-৿]")       # Bengali, Assamese


def _detect_language(text: str) -> str:
    """Detect query language. Returns one of: ``hi`` | ``ta`` | ``bn`` | ``en``.

    Delegates to the 3-tier detector in ``utils.lang_detect`` (script → keyword
    → statistical) so romanized-Indic and code-mixed Hinglish queries no longer
    silently route as English. Callers that need the detection method should
    use ``utils.lang_detect.detect_language`` directly; this function preserves
    the original lang-only signature so downstream callsites stay surgical."""
    from utils.lang_detect import detect_language as _ld
    return _ld(text or "")[0]


def _domain(url: str) -> str:
    """Extract a clean, lowercase, port-less domain from a URL.

    Previously `urlparse(url).netloc.replace("www.", "")` which would also
    strip "www." from the middle of a name (apiwww.example.com →
    apiexample.com) and never stripped trailing dot / port.
    """
    try:
        from utils.url_norm import normalize_domain
        return normalize_domain(urlparse(url).netloc)
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
@retry(wait=wait_retry_after_or_exponential(min=2, max=30), stop=stop_after_attempt(3), reraise=True)
async def _search_parallel(query: str, client: httpx.AsyncClient, num_results: int = _MAX_RESULTS) -> list[SearchResult]:
    """Parallel AI — returns AI-native structured excerpts; no separate fetch needed."""
    # V3.9: cache by (provider, query). 24h TTL, intent-aware bypass elsewhere.
    from utils import search_cache
    cached = await search_cache.get("parallel", query)
    if cached is not None:
        return cached
    key = os.getenv("PARALLEL_API_KEY", "")
    if not key:
        return []
    # Parallel v1beta /search schema: objective drives ranking, search_queries
    # is a list (we send one per call to keep cache keys 1:1 with queries),
    # processor selects tier (base|pro|ultra), and results carry multi-excerpt
    # arrays instead of single snippets. See docs.parallel.ai.
    resp = await client.post(
        f"{_PARALLEL_BASE}/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "objective": query,
            "search_queries": [query],
            "processor": os.environ.get("PARALLEL_PROCESSOR", "base"),
            "max_results": num_results,
        },
        timeout=30.0,
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
    await search_cache.put("parallel", query, results)
    logger.info("Parallel returned %d results", len(results), extra={"component": "search"})
    return results


@breaker("tavily")
@retry(wait=wait_retry_after_or_exponential(min=2, max=30), stop=stop_after_attempt(3), reraise=True)
async def _search_tavily(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """Tavily fallback — include_raw_content=True collapses the fetch pipeline."""
    from utils import search_cache
    cached = await search_cache.get("tavily", query)
    if cached is not None:
        return cached
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
    await search_cache.put("tavily", query, results)
    logger.info("Tavily returned %d results", len(results), extra={"component": "search"})
    return results


@breaker("serper")
@retry(wait=wait_retry_after_or_exponential(min=2, max=30), stop=stop_after_attempt(3), reraise=True)
async def _search_serper(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    """Serper last resort — returns snippets only; Trafilatura extracts full content."""
    from utils import search_cache
    cached = await search_cache.get("serper", query)
    if cached is not None:
        return cached
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
    await search_cache.put("serper", query, results)
    logger.info("Serper returned %d results", len(results), extra={"component": "search"})
    return results


@breaker("tavily")
@retry(wait=wait_retry_after_or_exponential(min=2, max=30), stop=stop_after_attempt(3), reraise=True)
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
    time_sensitivity: str = "static",
) -> list[SearchResult]:
    """Route a query to providers based on intent. Returns results tagged with intent_origin.

    V3.4 / FDSE add-on: when ``language`` is any Indic script (``hi``/``ta``/``bn``),
    prefer Parallel only — Tavily and Serper return English-heavy results for
    Indic queries. If Parallel fails (or its breaker is open), fall through to
    the standard chain so the turn still completes in a degraded mode.

    Results are cached for 1h (disable with SEARCH_CACHE_DISABLED=1). The
    cache key combines provider-class + intent + language + time_sensitivity
    so a re-ask of the same query under the same routing returns instantly
    — critical for eval re-runs and dev iteration. We cache the post-routing
    result list (the same list this function would return on a cache miss).
    """
    results: list[SearchResult] = []

    if language is None:
        language = _detect_language(q)

    # Cache lookup. Keyed wider than just `q` so different intents on the
    # same string still get correct routing.
    cache_enabled = os.getenv("SEARCH_CACHE_DISABLED", "").strip() not in {"1", "true", "True"}
    cache_provider_key = f"search:{intent.value}:{language}:{time_sensitivity}"
    if cache_enabled:
        try:
            from utils.cache import get_cached_search
            hit = await get_cached_search(cache_provider_key, q)
            if hit:
                logger.debug(
                    "search cache hit (%s) for %r", cache_provider_key, q,
                    extra={"component": "search"},
                )
                return [SearchResult(**r) for r in hit]
        except Exception as exc:
            logger.debug(
                "search cache lookup error: %s", exc,
                extra={"component": "search"},
            )

    async def _finalize(out: list[SearchResult]) -> list[SearchResult]:
        """Tag intent_origin and persist to cache. Only writes when non-empty
        — caching an empty result list would mask transient provider outages."""
        tag = "contradiction_probe" if intent == QueryIntent.CONTRADICTION_PROBE else intent.value
        for r in out:
            if r.intent_origin is None:
                r.intent_origin = tag
        if cache_enabled and out:
            try:
                from dataclasses import asdict
                from utils.cache import set_cached_search
                await set_cached_search(
                    cache_provider_key, q, [asdict(r) for r in out],
                )
            except Exception as exc:
                logger.debug(
                    "search cache write error: %s", exc,
                    extra={"component": "search"},
                )
        return out

    if language in {"hi", "ta", "bn"}:
        label = f"Parallel({language})"
        results = await _try(lambda: _search_parallel(q, client), label, q)
        if not results:
            logger.info(
                "Indic query (%s) Parallel returned empty/breaker-open; falling through to Tavily/Serper",
                language,
                extra={"component": "search"},
            )
            results = await _try(lambda: _search_tavily(q, client), f"Tavily({language}-fallback)", q)
            if not results:
                results = await _try(lambda: _search_serper(q, client), f"Serper({language}-fallback)", q)
        return await _finalize(results)

    # Phase 1.875: planner-level `time_sensitivity == "live"` forces the
    # Tavily news-mode route regardless of intent, since news is the only
    # provider tuned for the freshest material.
    if time_sensitivity == "live" and os.getenv("TAVILY_API_KEY"):
        results = await _try(lambda: _search_tavily_news(q, client, days=7), "Tavily(news,live)", q)
        if not results:
            results = await _try(lambda: _search_parallel(q, client), "Parallel(live-fallback)", q)
        if not results:
            results = await _try(lambda: _search_serper(q, client), "Serper(live-fallback)", q)
        return await _finalize(results)

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

    return await _finalize(results)


def _dedup_preserve_origin(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
    return out


async def _supplement_with_wikipedia(q: str) -> list[SearchResult]:
    """Wikipedia REST summary as a supplementary source for DEFINITION queries.

    Returns 0 or 1 result; always swallows errors. Tagged with
    ``source_provider`` style ``intent_origin="wikipedia"`` so downstream
    layers and the trace inspector can see provenance."""
    from utils.wikipedia import fetch_wikipedia_summary
    try:
        wiki = await fetch_wikipedia_summary(q)
    except Exception as e:  # noqa: BLE001
        logger.info("Wikipedia supplement failed for '%s': %s", q, e, extra={"component": "search"})
        return []
    if not wiki:
        return []
    return [SearchResult(
        url=wiki["url"],
        title=wiki["title"],
        snippet=wiki["snippet"],
        domain=wiki["domain"],
        retrieved_at=_now(),
        raw_content=None,
        intent_origin="wikipedia",
        relevance=0.95,
        relevance_source="provider",
    )]


async def _supplement_with_scholar(q: str, top_k: int = 3) -> list[SearchResult]:
    """Serper Scholar supplement for academic-typed sub-queries. Errors → []."""
    from utils.scholar import fetch_serper_scholar
    try:
        return await fetch_serper_scholar(q, top_k=top_k)
    except Exception as e:  # noqa: BLE001
        logger.info("Scholar supplement failed for '%s': %s", q, e, extra={"component": "search"})
        return []


async def search(
    queries: list[TypedQuery],
    cancel_token=None,
    time_sensitivity: str = "static",
    urls_by_query: dict[str, list[str]] | None = None,
    domain_blocklist: frozenset[str] | None = None,
    blocklist_drops: list[str] | None = None,
    expected_source_types: list[str] | None = None,
    supplementary_counts: dict[str, int] | None = None,
) -> list[SearchResult]:
    """
    Run typed queries against the intent-routed provider chain. Returns deduplicated
    list; first intent_origin seen per URL is preserved.

    Phase 1.875: if ``urls_by_query`` is provided, it is populated in-place
    with ``{query_text: [url, url, ...]}`` so the orchestrator can compute
    per-TypedQuery evidence gaps after SELECTING.

    Phase 5b: if ``domain_blocklist`` is provided (non-empty), search results
    whose domain matches are dropped *after* dedup so we never attempt to
    fetch low-signal social media URLs. Pass ``frozenset()`` (or omit) to
    disable. Drops are appended to ``blocklist_drops`` for observability —
    the orchestrator stitches this into ``run_metadata["domain_blocklist_drops"]``.
    """
    import asyncio
    if cancel_token is not None and cancel_token.is_set():
        return []
    all_results: list[SearchResult] = []
    wants_academic = "academic" in (expected_source_types or [])
    async with httpx.AsyncClient() as client:
        tasks: list = []
        task_queries: list[TypedQuery] = []
        # Supplementary tasks run in parallel with primary search. Each entry
        # is (tq, kind) so we can attribute results back to a query when
        # populating `urls_by_query`. Kinds: "wikipedia" | "scholar".
        supplement_tasks: list = []
        supplement_meta: list[tuple[TypedQuery, str]] = []
        for tq in queries:
            if cancel_token is not None and cancel_token.is_set():
                break
            tasks.append(
                _search_single_query(
                    tq.text, client, tq.intent,
                    language=_detect_language(tq.text),
                    time_sensitivity=time_sensitivity,
                )
            )
            task_queries.append(tq)
            # Phase 1.875 wiring: SUPPLEMENTARY providers.
            # Wikipedia for DEFINITION-intent sub-queries; Scholar for any
            # sub-query when the plan-level `expected_source_types` contains
            # "academic". Both supplement — they never replace the primary
            # chain — so a failure here is silent.
            if tq.intent == QueryIntent.DEFINITION:
                supplement_tasks.append(_supplement_with_wikipedia(tq.text))
                supplement_meta.append((tq, "wikipedia"))
            if wants_academic:
                supplement_tasks.append(_supplement_with_scholar(tq.text, top_k=3))
                supplement_meta.append((tq, "scholar"))
        # Phase 1.875: use return_exceptions=True so one failing query doesn't
        # abort the whole batch — mirrors the resilient pattern in
        # `agent/extractor.py:91`. We log per-query exceptions and substitute
        # an empty result list so downstream selection still has something to
        # work with from the surviving queries.
        combined = await asyncio.gather(
            *tasks, *supplement_tasks, return_exceptions=True
        )
        primary_count = len(tasks)
        results_list = combined[:primary_count]
        supplement_results = combined[primary_count:]
        for tq, results in zip(task_queries, results_list):
            if isinstance(results, BaseException):
                logger.warning(
                    "Search task failed for query '%s' (intent=%s): %s",
                    tq.text, tq.intent.value, results,
                    extra={"component": "search"},
                )
                if urls_by_query is not None:
                    urls_by_query.setdefault(tq.text, [])
                continue
            if urls_by_query is not None:
                urls_by_query.setdefault(tq.text, []).extend(r.url for r in results)
            all_results.extend(results)

        # Merge supplementary results (Wikipedia / Scholar). Dedup happens
        # below in `_dedup_preserve_origin`; primary-provider results were
        # appended first so they win on URL collisions.
        for (tq, kind), sup_results in zip(supplement_meta, supplement_results):
            if isinstance(sup_results, BaseException):
                logger.info(
                    "Supplement (%s) failed for query '%s': %s",
                    kind, tq.text, sup_results,
                    extra={"component": "search"},
                )
                continue
            if not sup_results:
                continue
            if urls_by_query is not None:
                urls_by_query.setdefault(tq.text, []).extend(r.url for r in sup_results)
            all_results.extend(sup_results)
            if supplementary_counts is not None:
                supplementary_counts[kind] = supplementary_counts.get(kind, 0) + len(sup_results)

    deduped = _dedup_preserve_origin(all_results)

    # Phase 5b: filter blocklisted domains *after* dedup so we never spend an
    # extractor connection on a URL we'd drop later. Skip entirely when the
    # caller passed a falsy/empty blocklist (the "none" disable path).
    if domain_blocklist:
        from utils.source_trust import is_blocked
        filtered: list[SearchResult] = []
        for r in deduped:
            if is_blocked(r.domain or "", domain_blocklist):
                logger.debug(
                    "Blocklisted domain dropped: %s", r.domain,
                    extra={"component": "search"},
                )
                if blocklist_drops is not None:
                    blocklist_drops.append(r.domain or "")
                continue
            filtered.append(r)
        deduped = filtered

    logger.info("Total unique results: %d", len(deduped), extra={"component": "search"})
    return deduped
