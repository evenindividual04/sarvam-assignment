"""
Async URL content extractor.
- Single httpx.AsyncClient per instance (initialized in __init__)
- asyncio.Semaphore(3) gates concurrent fetches
- Skip if SearchResult.raw_content already populated
- Catches asyncio.TimeoutError, httpx.TimeoutException, httpx.ConnectError
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx
import trafilatura

from agent.models import SearchResult

logger = logging.getLogger(__name__)


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

    async def aclose(self) -> None:
        await self._client.aclose()

    async def extract(self, result: SearchResult) -> Optional[str]:
        """Return extracted text or None on failure. Respects semaphore."""
        if result.raw_content:
            return result.raw_content

        async with self._semaphore:
            return await self._fetch_and_extract(result.url)

    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)), reraise=True)
    async def _fetch_and_extract(self, url: str) -> Optional[str]:
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            html = resp.text
        except asyncio.TimeoutError:
            logger.warning("asyncio.TimeoutError fetching %s", url, extra={"component": "extractor"})
            return None
        except httpx.TimeoutException:
            logger.warning("httpx.TimeoutException fetching %s", url, extra={"component": "extractor"})
            return None
        except httpx.ConnectError:
            logger.warning("httpx.ConnectError fetching %s", url, extra={"component": "extractor"})
            return None
        except Exception as e:
            logger.warning("Fetch failed %s: %s", url, e, extra={"component": "extractor"})
            return None

        text = trafilatura.extract(
            html,
            favor_precision=True,
            output_format="txt",
            include_comments=False,
        )
        return text or None

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
