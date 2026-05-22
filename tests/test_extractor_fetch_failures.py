"""A5: Extractor.fetch_failures records reason per opened-but-unreachable URL.

Assignment line 50 wants metadata retained even when a fetch fails. The
orchestrator copies extractor.fetch_failures into
run_metadata["unreachable_pages"] so the UI shows a ⚠️ chip and eval can
quantify retrieval robustness.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest


@pytest.fixture
def ex():
    from agent.extractor import Extractor

    e = Extractor()
    yield e
    asyncio.run(e.aclose())


def test_fetch_failures_starts_empty(ex):
    assert ex.fetch_failures == {}


def test_fetch_failures_records_timeout(ex, monkeypatch):
    async def _raise(*_a, **_kw):
        raise httpx.TimeoutException("simulated")

    monkeypatch.setattr(ex._client, "get", _raise)
    out = asyncio.run(ex._fetch_and_extract("https://example.com/slow"))
    assert out is None
    assert ex.fetch_failures.get("https://example.com/slow") == "timeout"


def test_fetch_failures_records_connect_error(ex, monkeypatch):
    async def _raise(*_a, **_kw):
        raise httpx.ConnectError("dns")

    monkeypatch.setattr(ex._client, "get", _raise)
    out = asyncio.run(ex._fetch_and_extract("https://nope.invalid/"))
    assert out is None
    assert ex.fetch_failures.get("https://nope.invalid/") == "connection_error"


def test_fetch_failures_records_http_status(ex, monkeypatch):
    class _Resp:
        status_code = 403

        def raise_for_status(self):
            raise httpx.HTTPStatusError("forbidden", request=None, response=self)

        @property
        def text(self):
            return ""

    async def _get(*_a, **_kw):
        return _Resp()

    monkeypatch.setattr(ex._client, "get", _get)
    out = asyncio.run(ex._fetch_and_extract("https://blocked.example/x"))
    assert out is None
    assert ex.fetch_failures.get("https://blocked.example/x") == "http_403"


def test_fetch_failures_records_unsafe_url(ex):
    out = asyncio.run(ex._fetch_and_extract("file:///etc/passwd"))
    assert out is None
    assert ex.fetch_failures.get("file:///etc/passwd") == "blocked_unsafe_url"


def test_fetch_failures_records_empty_extraction(ex, monkeypatch):
    """Server responds 200 but extractor returns nothing → empty_extraction."""

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        @property
        def text(self):
            return "<html></html>"

    async def _get(*_a, **_kw):
        return _Resp()

    async def _no_text(*_a, **_kw):
        return None

    monkeypatch.setattr(ex._client, "get", _get)
    monkeypatch.setattr(ex, "_extract_with_fallbacks", _no_text)
    out = asyncio.run(ex._fetch_and_extract("https://empty.example/"))
    assert out is None
    assert ex.fetch_failures.get("https://empty.example/") == "empty_extraction"
