"""
Async URL content extractor.
- Single httpx.AsyncClient per instance (initialized in __init__)
- asyncio.Semaphore(3) gates concurrent fetches
- Skip if SearchResult.raw_content already populated
- Catches asyncio.TimeoutError, httpx.TimeoutException, httpx.ConnectError
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from typing import Optional
from urllib.parse import quote, urlparse

import httpx
import trafilatura
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.models import SearchResult

logger = logging.getLogger(__name__)

# Minimum useful extraction length. Below this we assume Trafilatura got a
# JS-shell / paywall / blocked page and try the fallback chain.
_MIN_USEFUL_CHARS = 200
_TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
_JINA_READER_BASE = "https://r.jina.ai/"


# S2 fix: SSRF guard. Search results come from third-party APIs and could
# contain file://, ftp://, cloud metadata IPs (169.254.169.254), or
# RFC-1918 private-range addresses pointing at internal services. We block
# all of these before httpx ever opens a connection.
#
# DNS-rebinding-style attacks (resolving a public hostname to a private IP)
# are NOT mitigated here — that requires a custom socket resolver, which is
# out of scope for a 2-day project. Document this in the audit.
_METADATA_HOSTS = {"metadata", "metadata.google.internal", "169.254.169.254"}


def _is_safe_url(url: str) -> bool:
    """Return True iff `url` is safe to fetch from a user-supplied source.
    Blocks non-http(s) schemes, private/loopback/link-local IP literals, and
    well-known cloud metadata endpoints."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    # IP-literal check: ipaddress.ip_address rejects hostnames with ValueError.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Hostname is not an IP literal; we allow it. See module docstring re:
        # DNS-resolution-based SSRF.
        pass
    else:
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    if host.lower() in _METADATA_HOSTS:
        return False
    return True


_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


class Extractor:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers=_HEADERS,
            follow_redirects=True,
        )
        self._semaphore = asyncio.Semaphore(3)
        # V3.9: fallback telemetry. Persisted into
        # ``run_metadata["extraction_fallbacks"]`` by the orchestrator so eval
        # runs can quantify how often Trafilatura was insufficient.
        self.fallback_counts: dict[str, int] = {"tavily": 0, "jina": 0}
        # Provenance for opened-but-unreachable pages — assignment line 50
        # wants metadata retained even when fetch fails. Orchestrator copies
        # this into run_metadata["unreachable_pages"] so the UI and eval can
        # distinguish "we tried and failed" from "we never tried".
        self.fetch_failures: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def extract(self, result: SearchResult) -> Optional[str]:
        """Return extracted text or None on failure. Respects semaphore."""
        if result.raw_content:
            return result.raw_content

        async with self._semaphore:
            return await self._fetch_and_extract(result.url)

    async def _tavily_extract(self, url: str, timeout_s: float = 10.0) -> Optional[str]:
        """POST to Tavily Extract API. Returns text or None on any error.

        Defense-in-depth: re-check ``_is_safe_url`` here even though the
        caller has already gated. Tavily fetches the URL on their backend,
        so a redirect-based SSRF would otherwise bypass our local guard.
        """
        if not _is_safe_url(url):
            logger.debug("Tavily extract: refusing unsafe URL %s", url,
                         extra={"component": "extractor"})
            return None
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return None
        payload = {
            "api_key": api_key,
            "urls": [url],
            "include_raw_content": True,
        }
        try:
            resp = await self._client.post(
                _TAVILY_EXTRACT_URL,
                json=payload,
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Tavily extract failed for %s: %s", url, exc,
                         extra={"component": "extractor"})
            return None
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            return None
        first = results[0] if isinstance(results[0], dict) else {}
        text = first.get("raw_content") or first.get("content")
        if isinstance(text, str) and text.strip():
            return text
        return None

    async def _jina_read(self, url: str, timeout_s: float = 10.0) -> Optional[str]:
        """GET https://r.jina.ai/{url}. Free tier, no API key. Returns markdown.

        Two guards apply here:
          1. ``JINA_READER_DISABLED=1`` env opt-out — for operators who don't
             want any URLs disclosed to Jina's infrastructure.
          2. ``_is_safe_url`` defense-in-depth. Jina's backend re-resolves
             the URL on its own infrastructure, so our local check is the
             user-facing trust boundary against private-IP-literal SSRF.
             (DNS rebinding / open-redirect into a metadata IP is still
             out of scope — see ``_is_safe_url`` docstring.)
        """
        if os.getenv("JINA_READER_DISABLED", "").strip() in {"1", "true", "True", "yes"}:
            return None
        if not _is_safe_url(url):
            logger.debug("Jina read: refusing unsafe URL %s", url,
                         extra={"component": "extractor"})
            return None
        try:
            target = _JINA_READER_BASE + quote(url, safe=":/?&=%#")
            resp = await self._client.get(
                target,
                timeout=timeout_s,
                headers={"User-Agent": _HEADERS["User-Agent"]},
            )
            resp.raise_for_status()
            text = resp.text
        except Exception as exc:
            logger.debug("Jina read failed for %s: %s", url, exc,
                         extra={"component": "extractor"})
            return None
        if isinstance(text, str) and text.strip():
            return text
        return None

    async def _extract_with_fallbacks(
        self,
        url: str,
        html_text: str,
    ) -> Optional[str]:
        """Three-tier extraction: Trafilatura → Tavily Extract → Jina Reader.

        Always returns the best-effort result. Never raises.
        """
        primary = trafilatura.extract(
            html_text,
            favor_precision=True,
            output_format="txt",
            include_comments=False,
        )
        if primary and len(primary) >= _MIN_USEFUL_CHARS:
            return primary

        # Tier 2: Tavily Extract (uses existing TAVILY_API_KEY)
        if os.getenv("TAVILY_API_KEY", "").strip():
            tavily_text = await self._tavily_extract(url)
            if tavily_text and len(tavily_text) >= _MIN_USEFUL_CHARS:
                logger.info("Extraction fallback: tavily for %s", url,
                            extra={"component": "extractor"})
                self.fallback_counts["tavily"] += 1
                return tavily_text

        # Tier 3: Jina AI Reader (no key, ~1M tokens/mo free)
        jina_text = await self._jina_read(url)
        if jina_text:
            logger.info("Extraction fallback: jina for %s", url,
                        extra={"component": "extractor"})
            self.fallback_counts["jina"] += 1
            return jina_text

        # All fallbacks failed — surface whatever (possibly short) text we got.
        return primary or None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def _fetch_and_extract(self, url: str) -> Optional[str]:
        # S2 fix: SSRF guard — refuse non-http(s) schemes and private/metadata IPs.
        if not _is_safe_url(url):
            logger.warning(
                "Refusing unsafe URL (SSRF guard) %s", url,
                extra={"component": "extractor"},
            )
            self.fetch_failures[url] = "blocked_unsafe_url"
            return None
        # Page-fetch cache hit: Wikipedia / Britannica / news domains recur
        # often across turns and sessions. Skip the httpx + Trafilatura round
        # trip when we already have a fresh extraction. Disabled by setting
        # PAGE_CACHE_DISABLED=1 (eval/CI may want fresh fetches).
        if os.getenv("PAGE_CACHE_DISABLED", "").strip() not in {"1", "true", "True"}:
            try:
                from utils.cache import get_cached_page
                hit = await get_cached_page(url)
                if hit and hit.get("text"):
                    logger.debug(
                        "page cache hit %s", url, extra={"component": "extractor"},
                    )
                    return hit["text"]
            except Exception as exc:
                logger.debug(
                    "page cache lookup error %s: %s", url, exc,
                    extra={"component": "extractor"},
                )
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            html = resp.text
        except asyncio.TimeoutError:
            logger.warning("asyncio.TimeoutError fetching %s", url, extra={"component": "extractor"})
            self.fetch_failures[url] = "timeout"
            return None
        except httpx.TimeoutException:
            logger.warning("httpx.TimeoutException fetching %s", url, extra={"component": "extractor"})
            self.fetch_failures[url] = "timeout"
            return None
        except httpx.ConnectError:
            logger.warning("httpx.ConnectError fetching %s", url, extra={"component": "extractor"})
            self.fetch_failures[url] = "connection_error"
            return None
        except httpx.HTTPStatusError as e:
            status = getattr(e.response, "status_code", "unknown")
            logger.warning("HTTP %s fetching %s", status, url, extra={"component": "extractor"})
            self.fetch_failures[url] = f"http_{status}"
            return None
        except Exception as e:
            logger.warning("Fetch failed %s: %s", url, e, extra={"component": "extractor"})
            self.fetch_failures[url] = f"error:{type(e).__name__}"
            return None

        text = await self._extract_with_fallbacks(url, html)
        if not text:
            self.fetch_failures[url] = "empty_extraction"
            return None
        # Persist for the next caller. Best-effort; ignores DB errors.
        if os.getenv("PAGE_CACHE_DISABLED", "").strip() not in {"1", "true", "True"}:
            try:
                from utils.cache import set_cached_page
                from datetime import datetime, timezone
                from urllib.parse import urlparse
                from utils.url_norm import normalize_domain
                await set_cached_page(
                    url,
                    text,
                    domain=normalize_domain(urlparse(url).netloc),
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as exc:
                logger.debug(
                    "page cache write failed %s: %s", url, exc,
                    extra={"component": "extractor"},
                )
        return text

    async def extract_all(self, results: list[SearchResult], cancel_token=None) -> dict[str, Optional[str]]:
        """Extract all URLs concurrently (semaphore limits to 3 parallel)."""
        if cancel_token is not None and cancel_token.is_set():
            return {}
        tasks = []
        scheduled: list[SearchResult] = []
        for r in results:
            if cancel_token is not None and cancel_token.is_set():
                break
            tasks.append(self.extract(r))
            scheduled.append(r)
        texts = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, Optional[str]] = {}
        for r, t in zip(scheduled, texts):
            if isinstance(t, Exception):
                logger.warning("Extract error for %s: %s", r.url, t, extra={"component": "extractor"})
                out[r.url] = None
            else:
                out[r.url] = t
        return out
