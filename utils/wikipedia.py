"""Wikipedia REST API client. Free, no key. Used as SUPPLEMENTARY for
definition-intent queries — runs in parallel with primary search providers
and blends results. Falls through to None on any error.

The Wikipedia REST `summary` endpoint returns a short extract plus the canonical
page URL. We auto-route to language-specific subdomains based on the query
script so Hindi / Tamil / Bengali queries hit `hi.wikipedia.org` etc.

Disable via `WIKIPEDIA_DISABLED=1`."""
from __future__ import annotations

import logging
import os
import re
from typing import Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Script detection mirrors agent.search but is duplicated here to avoid a
# circular import (search imports nothing from utils.wikipedia by design;
# the orchestrator wires the two together).
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")  # Hindi, Marathi, Sanskrit
_TAMIL_RE = re.compile(r"[஀-௿]")
_BENGALI_RE = re.compile(r"[ঀ-৿]")


def _detect_lang_for_wikipedia(text: str) -> str:
    """Pick a Wikipedia subdomain code from the 3-tier detector.

    Delegates to ``utils.lang_detect`` so Hinglish queries hit
    ``hi.wikipedia.org`` instead of being silently routed to en.wikipedia."""
    if not text:
        return "en"
    from utils.lang_detect import detect_language as _ld
    return _ld(text)[0]


async def fetch_wikipedia_summary(
    query: str,
    lang: Optional[str] = None,
    timeout_s: float = 5.0,
) -> Optional[dict]:
    """Hit Wikipedia REST `summary` endpoint.

    Returns ``{"title", "url", "snippet", "domain": "wikipedia.org"}`` or
    ``None`` on any failure (404, network error, non-JSON, disabled).
    """
    if os.getenv("WIKIPEDIA_DISABLED", "").strip() in {"1", "true", "True"}:
        logger.debug("Wikipedia disabled via env", extra={"component": "wikipedia"})
        return None
    if not query or not query.strip():
        return None
    if lang is None:
        lang = _detect_lang_for_wikipedia(query)

    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(query.strip())}"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "deep-research-agent/1.0 (sarvam-fdse-assignment)",
                },
            )
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
        logger.info("Wikipedia fetch failed for '%s': %s", query, e, extra={"component": "wikipedia"})
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Wikipedia unexpected error for '%s': %s", query, e, extra={"component": "wikipedia"})
        return None

    if resp.status_code != 200:
        logger.debug("Wikipedia returned %d for '%s'", resp.status_code, query, extra={"component": "wikipedia"})
        return None

    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.info("Wikipedia non-JSON response for '%s': %s", query, e, extra={"component": "wikipedia"})
        return None

    title = data.get("title")
    extract = data.get("extract") or ""
    page_url = (
        data.get("content_urls", {})
        .get("desktop", {})
        .get("page")
    )
    if not title or not page_url:
        return None

    return {
        "title": title,
        "url": page_url,
        "snippet": extract,
        "domain": "wikipedia.org",
    }
