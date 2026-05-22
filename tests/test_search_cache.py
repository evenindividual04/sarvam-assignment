"""Smoke tests for utils.search_cache — covers public API surface."""
from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
from pathlib import Path

import pytest

from agent.models import SearchResult


def _make_result(url: str = "https://example.com/x") -> SearchResult:
    return SearchResult(
        url=url, title="t", snippet="s", domain="example.com",
        retrieved_at="2026-05-20T00:00:00+00:00", raw_content=None,
        intent_origin="primary", relevance=0.5, relevance_source="bm25",
    )


@pytest.fixture
def isolated_db(monkeypatch):
    """Point the cache at a temp sqlite file so tests don't share state."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    monkeypatch.setattr("agent.memory.DB_PATH", tmp.name)
    monkeypatch.setattr("utils.search_cache.DB_PATH", tmp.name)
    yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


def test_miss_returns_none(isolated_db):
    import utils.search_cache as sc
    result = asyncio.run(sc.get("parallel", "nothing cached yet"))
    assert result is None


def test_put_then_get_hit(isolated_db):
    import utils.search_cache as sc
    res = [_make_result()]
    asyncio.run(sc.put("parallel", "test query", res))
    got = asyncio.run(sc.get("parallel", "test query"))
    assert got is not None
    assert len(got) == 1
    assert got[0].url == "https://example.com/x"


def test_disabled_via_env(isolated_db, monkeypatch):
    """SEARCH_CACHE_DISABLED=1 should turn the cache into a no-op."""
    monkeypatch.setenv("SEARCH_CACHE_DISABLED", "1")
    import utils.search_cache as sc
    importlib.reload(sc)
    asyncio.run(sc.put("parallel", "q", [_make_result()]))
    assert asyncio.run(sc.get("parallel", "q")) is None
    # Restore default for other tests
    monkeypatch.setenv("SEARCH_CACHE_DISABLED", "0")
    importlib.reload(sc)


def test_should_bypass_recency_check():
    from utils.search_cache import should_bypass
    assert should_bypass("recency_check") is True
    assert should_bypass("primary") is False
    assert should_bypass(None) is False


def test_cache_key_deterministic():
    from utils.search_cache import _cache_key
    a = _cache_key("parallel", "Hello World")
    b = _cache_key("parallel", "  hello world  ")
    assert a == b  # normalized: lowercase + strip
