"""Forensic-differentiation: LLM-classified source roles.

A single batched Groq call classifies each unique URL in the final chunk pool
into one of a fixed role taxonomy. Results are cached in-memory keyed by
``(url, ROLE_CLASSIFIER_VERSION)`` so repeated runs over the same URL pool skip
the LLM.

Contract:

- One async function, ``classify_source_roles(chunks)`` → ``{url: (role,
  confidence)}``.
- Never raises. On any error/timeout, every input URL gets
  ``("unclassified", 0.0)``.
- Caps the batch at 12 unique URLs (most-relevant first by snippet appearance
  order) to keep the prompt cheap.

This is the ONLY LLM call added by the forensic-events track. All other
forensic event payloads (`hop_evidence`, `source_contribution`) are mechanical.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from typing import Iterable

from agent.models import ContextSnippet

logger = logging.getLogger(__name__)


ROLE_CLASSIFIER_VERSION = "v1"
_VALID_ROLES = frozenset({
    "primary_source",
    "secondary_analysis",
    "statistical",
    "news_event",
    "official",
    "encyclopedic",
    "contradicting",
    "unclassified",
})
_MAX_URLS_PER_CALL = 12
_LLM_TIMEOUT_S = 8.0
_MAX_TOKENS = 400

# Process-local cache. Cleared by tests via _clear_cache().
# FIX 5: bounded LRU — was an unbounded dict that grew forever.
_CACHE_MAX_SIZE = 512
_CACHE: "OrderedDict[tuple[str, str], tuple[str, float]]" = OrderedDict()


def _cache_get(key: tuple[str, str]) -> tuple[str, float] | None:
    val = _CACHE.get(key)
    if val is not None:
        _CACHE.move_to_end(key)
    return val


def _cache_put(key: tuple[str, str], value: tuple[str, float]) -> None:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        _CACHE[key] = value
        return
    if len(_CACHE) >= _CACHE_MAX_SIZE:
        _CACHE.popitem(last=False)
    _CACHE[key] = value


def _clear_cache() -> None:
    """Test hook — drop the in-memory cache."""
    _CACHE.clear()


def _dedupe_chunks(chunks: Iterable[ContextSnippet]) -> list[ContextSnippet]:
    """Keep one chunk per URL, preserving first-seen order."""
    seen: set[str] = set()
    out: list[ContextSnippet] = []
    for c in chunks:
        url = getattr(c, "url", "") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(c)
    return out


def _build_prompt(chunks: list[ContextSnippet]) -> str:
    lines = [
        "You are classifying web sources for a research agent. For each URL, "
        "choose ONE role label from this fixed list. Return strict JSON only.",
        "",
        "Roles:",
        "- primary_source: original document (paper, official statement, raw data)",
        "- secondary_analysis: review article, opinion, explainer",
        "- statistical: contains tables, datasets, or quantitative summaries",
        "- news_event: dated news report of an event",
        "- official: government, regulatory, institutional publication",
        "- encyclopedic: encyclopedia or reference entry",
        "- contradicting: contradicts other sources in the set",
        "- unclassified: doesn't fit any role above",
        "",
        "Sources:",
    ]
    for c in chunks:
        url = getattr(c, "url", "") or ""
        title = (getattr(c, "title", "") or "")[:140]
        domain = getattr(c, "domain", "") or ""
        snippet = (getattr(c, "text", "") or getattr(c, "snippet", "") or "")[:500]
        lines.append(f"URL: {url}")
        lines.append(f"TITLE: {title}")
        lines.append(f"DOMAIN: {domain}")
        lines.append(f"SNIPPET: {snippet}")
        lines.append("---")
    lines.append(
        'Respond with JSON only: {"roles": [{"url": "<url>", "role": '
        '"<one of the roles>", "confidence": <0.0-1.0>}, ...]}'
    )
    return "\n".join(lines)


def _parse_response(raw: str) -> dict[str, tuple[str, float]]:
    """Parse Groq JSON output. Returns {} on any failure."""
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return {}
        data = json.loads(raw[start:end])
        roles = data.get("roles") or []
        out: dict[str, tuple[str, float]] = {}
        for row in roles:
            if not isinstance(row, dict):
                continue
            url = row.get("url")
            role = row.get("role")
            conf = row.get("confidence", 0.0)
            if not isinstance(url, str) or not isinstance(role, str):
                continue
            if role not in _VALID_ROLES:
                role = "unclassified"
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                conf_f = 0.0
            conf_f = max(0.0, min(1.0, conf_f))
            out[url] = (role, conf_f)
        return out
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("source_role parse failed: %s",
                       e, extra={"component": "source_role"})
        return {}


async def classify_source_roles(
    chunks: list[ContextSnippet],
) -> dict[str, tuple[str, float]]:
    """Classify each unique URL in ``chunks`` into a source-role label.

    Returns a mapping ``{url: (role, confidence)}`` for every input URL. URLs
    not classified by the LLM (or any failure path) get
    ``("unclassified", 0.0)``. Never raises.
    """
    if not chunks:
        return {}

    unique = _dedupe_chunks(chunks)
    all_urls = [c.url for c in unique]
    result: dict[str, tuple[str, float]] = {}

    # Cache hits short-circuit the LLM entirely for those URLs.
    uncached: list[ContextSnippet] = []
    for c in unique:
        cached = _cache_get((c.url, ROLE_CLASSIFIER_VERSION))
        if cached is not None:
            result[c.url] = cached
        else:
            uncached.append(c)

    if not uncached:
        return result

    # Cap batch — anything beyond is reported as unclassified (so we keep one
    # LLM call per run, never multiple round-trips).
    batch = uncached[:_MAX_URLS_PER_CALL]
    overflow = uncached[_MAX_URLS_PER_CALL:]
    for c in overflow:
        result[c.url] = ("unclassified", 0.0)

    prompt = _build_prompt(batch)

    raw: str | None = None
    try:
        from utils.provider_router import call_groq
        raw = await asyncio.wait_for(
            call_groq(prompt, max_tokens=_MAX_TOKENS),
            timeout=_LLM_TIMEOUT_S,
        )
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
        logger.warning(
            "source_role classify failed: %s",
            e, extra={"component": "source_role"},
        )
        raw = None

    parsed: dict[str, tuple[str, float]] = {}
    if raw:
        parsed = _parse_response(raw)

    # FIX: the LLM frequently echoes URLs with a trailing slash, scheme
    # change, or missing fragment. A naive `parsed.get(c.url, ...)` lookup
    # then misses and the source gets marked "unclassified" even though the
    # LLM successfully classified it — this was the "1-char difference but
    # still got flagged as a bad source" symptom users saw in the trace.
    # Build a normalized index and look up against that, with the raw URL
    # as a secondary fallback.
    from utils.url_norm import normalize_url
    parsed_by_norm = {normalize_url(u): v for u, v in parsed.items()}

    for c in batch:
        entry = (
            parsed.get(c.url)
            or parsed_by_norm.get(normalize_url(c.url))
            or ("unclassified", 0.0)
        )
        result[c.url] = entry
        # Only cache real classifications, not failure-path fallbacks. This
        # lets a flaky run retry on the next turn instead of pinning every URL
        # to "unclassified" forever.
        if entry[0] != "unclassified" or entry[1] > 0.0:
            _cache_put((c.url, ROLE_CLASSIFIER_VERSION), entry)

    # Final sweep — guarantee every requested URL is present.
    for url in all_urls:
        result.setdefault(url, ("unclassified", 0.0))
    return result
