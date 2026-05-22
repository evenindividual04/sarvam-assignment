"""V3.9: three-tier extraction fallback chain in agent/extractor.py.

Trafilatura (primary) → Tavily Extract → Jina AI Reader. Each tier only
fires when the previous returns <200 chars (or errors out for Tavily/Jina).
Telemetry is accumulated on the Extractor instance for run_metadata.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent.extractor import Extractor


@pytest.fixture
def ex():
    e = Extractor()
    yield e
    asyncio.run(e.aclose())


def test_extraction_uses_trafilatura_when_sufficient(ex, monkeypatch):
    """Trafilatura returns >=200 chars → fallbacks never called."""
    long_text = "x " * 300  # 600 chars
    monkeypatch.setattr("agent.extractor.trafilatura.extract", lambda *a, **kw: long_text)
    tavily_mock = AsyncMock(return_value="should not be called")
    jina_mock = AsyncMock(return_value="should not be called")
    monkeypatch.setattr(ex, "_tavily_extract", tavily_mock)
    monkeypatch.setattr(ex, "_jina_read", jina_mock)

    out = asyncio.run(ex._extract_with_fallbacks("https://example.com/x", "<html>...</html>"))

    assert out == long_text
    tavily_mock.assert_not_awaited()
    jina_mock.assert_not_awaited()
    assert ex.fallback_counts == {"tavily": 0, "jina": 0}


def test_extraction_falls_back_to_tavily(ex, monkeypatch):
    """Trafilatura <200 chars + TAVILY_API_KEY → Tavily used."""
    monkeypatch.setattr("agent.extractor.trafilatura.extract", lambda *a, **kw: "tiny")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    tavily_text = "y " * 300
    monkeypatch.setattr(ex, "_tavily_extract", AsyncMock(return_value=tavily_text))
    jina_mock = AsyncMock(return_value="unused")
    monkeypatch.setattr(ex, "_jina_read", jina_mock)

    out = asyncio.run(ex._extract_with_fallbacks("https://example.com/x", "<html/>"))

    assert out == tavily_text
    jina_mock.assert_not_awaited()
    assert ex.fallback_counts["tavily"] == 1
    assert ex.fallback_counts["jina"] == 0


def test_extraction_falls_back_to_jina_when_tavily_fails(ex, monkeypatch):
    """Tavily returns None / raises → Jina is tried next."""
    monkeypatch.setattr("agent.extractor.trafilatura.extract", lambda *a, **kw: "")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(ex, "_tavily_extract", AsyncMock(return_value=None))
    jina_text = "z " * 300
    monkeypatch.setattr(ex, "_jina_read", AsyncMock(return_value=jina_text))

    out = asyncio.run(ex._extract_with_fallbacks("https://example.com/x", "<html/>"))

    assert out == jina_text
    assert ex.fallback_counts["tavily"] == 0
    assert ex.fallback_counts["jina"] == 1


def test_extraction_returns_trafilatura_partial_when_all_fail(ex, monkeypatch):
    """All fallbacks fail → return whatever (possibly short) primary text."""
    short_text = "tiny snippet"
    monkeypatch.setattr("agent.extractor.trafilatura.extract", lambda *a, **kw: short_text)
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(ex, "_tavily_extract", AsyncMock(return_value=None))
    monkeypatch.setattr(ex, "_jina_read", AsyncMock(return_value=None))

    out = asyncio.run(ex._extract_with_fallbacks("https://example.com/x", "<html/>"))

    assert out == short_text
    assert ex.fallback_counts == {"tavily": 0, "jina": 0}


def test_extraction_no_tavily_key_skips_to_jina(ex, monkeypatch):
    """TAVILY_API_KEY unset → Tavily skipped, Jina is the only fallback."""
    monkeypatch.setattr("agent.extractor.trafilatura.extract", lambda *a, **kw: "")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tavily_mock = AsyncMock(return_value="never called")
    monkeypatch.setattr(ex, "_tavily_extract", tavily_mock)
    jina_text = "j " * 300
    monkeypatch.setattr(ex, "_jina_read", AsyncMock(return_value=jina_text))

    out = asyncio.run(ex._extract_with_fallbacks("https://example.com/x", "<html/>"))

    assert out == jina_text
    tavily_mock.assert_not_awaited()
    assert ex.fallback_counts["jina"] == 1


def test_tavily_extract_returns_none_on_http_error(ex, monkeypatch):
    """_tavily_extract swallows HTTP errors and returns None."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    async def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(ex._client, "post", boom)
    out = asyncio.run(ex._tavily_extract("https://example.com/x"))
    assert out is None


def test_jina_read_returns_none_on_error(ex, monkeypatch):
    """_jina_read swallows errors and returns None."""
    async def boom(*a, **kw):
        raise RuntimeError("dns failure")

    monkeypatch.setattr(ex._client, "get", boom)
    out = asyncio.run(ex._jina_read("https://example.com/x"))
    assert out is None
