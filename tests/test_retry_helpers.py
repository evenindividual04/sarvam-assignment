"""Smoke tests for utils.retry_helpers — Retry-After parsing and waiter."""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from utils.retry_helpers import (
    _parse_retry_after,
    _response_from_exception,
    wait_retry_after_or_exponential,
)


def test_parse_retry_after_seconds():
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after("0") == 0.0
    assert _parse_retry_after("30.5") == 30.5


def test_parse_retry_after_negative_rejected():
    assert _parse_retry_after("-5") is None


def test_parse_retry_after_invalid():
    assert _parse_retry_after("") is None
    assert _parse_retry_after("not-a-number") is None


def test_parse_retry_after_http_date():
    # Fixed date format per RFC 7231; should yield a delta near or above 0.
    delta = _parse_retry_after("Wed, 21 Oct 2099 07:28:00 GMT")
    assert delta is not None and delta > 0


def test_response_from_httpx_status_error():
    resp = httpx.Response(429, headers={"Retry-After": "7"})
    exc = httpx.HTTPStatusError("rate-limited", request=MagicMock(), response=resp)
    got = _response_from_exception(exc)
    assert got is resp


def test_response_from_exception_none():
    assert _response_from_exception(None) is None
    assert _response_from_exception(ValueError("nope")) is None


def test_wait_honors_retry_after_on_429():
    """When the wrapped exception is a 429 with Retry-After, return that wait."""
    resp = httpx.Response(429, headers={"Retry-After": "12"})
    exc = httpx.HTTPStatusError("rate-limited", request=MagicMock(), response=resp)

    waiter = wait_retry_after_or_exponential(min=2.0, max=30.0)
    state = MagicMock()
    state.outcome = MagicMock(failed=True)
    state.outcome.exception.return_value = exc
    assert waiter(state) == 12.0


def test_wait_clamps_retry_after_to_max():
    """A pathologically large Retry-After should be clamped to `max`."""
    resp = httpx.Response(429, headers={"Retry-After": "99999"})
    exc = httpx.HTTPStatusError("rate-limited", request=MagicMock(), response=resp)
    waiter = wait_retry_after_or_exponential(min=2.0, max=30.0)
    state = MagicMock()
    state.outcome = MagicMock(failed=True)
    state.outcome.exception.return_value = exc
    assert waiter(state) == 30.0


def test_wait_falls_back_to_exponential_when_no_429():
    """Non-429 errors should fall through to exponential backoff."""
    resp = httpx.Response(500)
    exc = httpx.HTTPStatusError("server-err", request=MagicMock(), response=resp)
    waiter = wait_retry_after_or_exponential(min=2.0, max=30.0)
    state = MagicMock()
    state.outcome = MagicMock(failed=True)
    state.outcome.exception.return_value = exc
    state.attempt_number = 1
    val = waiter(state)
    # Exponential wait is always within [min, max]
    assert 2.0 <= val <= 30.0
