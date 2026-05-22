"""
Tenacity wait strategy that honors HTTP `Retry-After`.

Most LLM providers (Gemini, Groq, OpenAI, Anthropic, Cerebras) and search
providers (Tavily, Serper) include a `Retry-After` header on 429 responses
telling you exactly how long to wait. The default `wait_exponential` ignores
this and may retry too fast — burning a second 429 against your daily quota.

This module provides `wait_retry_after_or_exponential` which:
  1. Inspects the exception for an attached `httpx.Response`.
  2. If the response is a 429 with a `Retry-After` header (either integer
     seconds or HTTP-date format), waits exactly that long.
  3. Otherwise falls back to exponential backoff with the same caps the
     previous decorators used.

Usage:
    @retry(
        wait=wait_retry_after_or_exponential(min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_provider(...): ...
"""
from __future__ import annotations

import logging
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import httpx
from tenacity import RetryCallState
from tenacity.wait import wait_base, wait_exponential

logger = logging.getLogger(__name__)


def _parse_retry_after(value: str) -> float | None:
    """Per RFC 9110: Retry-After is either an integer (delta-seconds) or
    an HTTP-date. Returns the wait in seconds, or None on parse failure."""
    if not value:
        return None
    s = value.strip()
    # Integer delta-seconds (most common; Gemini, Groq, Tavily use this).
    try:
        delta = float(s)
        if delta >= 0:
            return delta
    except ValueError:
        pass
    # HTTP-date (rare but spec-compliant; OpenAI uses this sometimes).
    try:
        dt = parsedate_to_datetime(s)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def _response_from_exception(exc: BaseException | None) -> httpx.Response | None:
    """Tenacity wraps the raising exception in RetryCallState.outcome.exception().
    We dig out the httpx.Response when the exception is an HTTPStatusError or
    when it has a `.response` attribute (matches openai-python, google-genai)."""
    if exc is None:
        return None
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response
    resp = getattr(exc, "response", None)
    if isinstance(resp, httpx.Response):
        return resp
    return None


class wait_retry_after_or_exponential(wait_base):
    """Hybrid waiter: respect `Retry-After` on 429, exponential otherwise.

    Honest about its limits — many providers send 429 *without* a header
    (Sarvam, Groq under burst), and some send it on non-429 errors. We treat
    the header as advisory only when status == 429, falling back to
    exponential in all other cases so the contract stays predictable.

    The exponential floor (`min`) prevents a server with a tiny Retry-After
    value (e.g. 0) from triggering a hot-loop; the ceiling (`max`) caps
    pathologically large server values that would deadlock a request.
    """

    def __init__(self, min: float = 2.0, max: float = 30.0, multiplier: float = 1.0) -> None:
        self._exp = wait_exponential(multiplier=multiplier, min=min, max=max)
        self._min = min
        self._max = max

    def __call__(self, retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None and outcome.failed else None
        resp = _response_from_exception(exc)
        if resp is not None and resp.status_code == 429:
            hdr = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            parsed = _parse_retry_after(hdr) if hdr else None
            if parsed is not None:
                # Clamp to [min, max] so a misconfigured server can't lock us out.
                wait = max(self._min, min(parsed, self._max))
                logger.info(
                    "Honoring Retry-After=%s on 429; waiting %.1fs",
                    hdr, wait,
                )
                return wait
        # Default path: exponential backoff (existing behavior).
        return self._exp(retry_state)
