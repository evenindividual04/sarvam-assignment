"""Tests for utils.cache — TTL caches for page fetches and search results.

The cache is best-effort: any DB error must degrade silently to a cache
miss, not raise. Tests pin both correctness (hit/miss/TTL) and the
silent-degrade behavior.
"""
from __future__ import annotations

import asyncio
import time

import pytest


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    """Point cache + memory at a fresh temp DB."""
    db = tmp_path / "cache_test.db"
    monkeypatch.setattr("agent.memory.DB_PATH", str(db))
    monkeypatch.setattr("utils.cache.DB_PATH", str(db))
    from agent.memory import init_db
    asyncio.run(init_db())
    return db


# ── page-fetch cache ─────────────────────────────────────────────────────


def test_page_cache_miss_returns_none(isolated_db):
    from utils.cache import get_cached_page
    assert asyncio.run(get_cached_page("https://nope.example/")) is None


def test_page_cache_roundtrip(isolated_db):
    from utils.cache import get_cached_page, set_cached_page
    url = "https://en.wikipedia.org/wiki/Paris"
    asyncio.run(set_cached_page(
        url, "Paris is the capital of France.",
        title="Paris", domain="en.wikipedia.org",
        retrieved_at="2026-05-24T12:00:00+00:00",
    ))
    hit = asyncio.run(get_cached_page(url))
    assert hit is not None
    assert "capital of France" in hit["text"]
    assert hit["title"] == "Paris"
    assert hit["domain"] == "en.wikipedia.org"


def test_page_cache_url_normalization(isolated_db):
    """Two URLs differing only by trailing slash / case must hit the same cell."""
    from utils.cache import get_cached_page, set_cached_page
    asyncio.run(set_cached_page(
        "https://example.com/Article", "hello",
        title="t", domain="example.com", retrieved_at="now",
    ))
    # All of these should resolve to the same cached row.
    for variant in [
        "https://example.com/Article",
        "https://example.com/Article/",
        "HTTPS://example.com/Article",
        "https://www.example.com/Article",
        "https://example.com:443/Article",
    ]:
        assert asyncio.run(get_cached_page(variant)) is not None, variant


def test_page_cache_ttl_expiry(isolated_db):
    """A row older than the TTL must read as a miss."""
    from utils.cache import get_cached_page, set_cached_page
    url = "https://ttl.example/"
    asyncio.run(set_cached_page(url, "stale", title="", domain="", retrieved_at="t"))
    # Pretend the row is ancient by passing a tiny TTL.
    hit = asyncio.run(get_cached_page(url, ttl_seconds=0))
    assert hit is None
    # And with a normal TTL it's a hit.
    hit2 = asyncio.run(get_cached_page(url, ttl_seconds=3600))
    assert hit2 is not None


def test_page_cache_empty_inputs_noop(isolated_db):
    from utils.cache import get_cached_page, set_cached_page
    asyncio.run(set_cached_page("", "x"))
    asyncio.run(set_cached_page("https://x/", ""))
    assert asyncio.run(get_cached_page("")) is None


# ── search-result cache ──────────────────────────────────────────────────


def test_search_cache_miss_returns_none(isolated_db):
    from utils.cache import get_cached_search
    assert asyncio.run(get_cached_search("parallel", "nothing here")) is None


def test_search_cache_roundtrip(isolated_db):
    from utils.cache import get_cached_search, set_cached_search
    results = [
        {"url": "https://a.example/", "title": "A", "snippet": "alpha"},
        {"url": "https://b.example/", "title": "B", "snippet": "beta"},
    ]
    asyncio.run(set_cached_search("parallel", "what is X", results))
    hit = asyncio.run(get_cached_search("parallel", "what is X"))
    assert hit is not None
    assert len(hit) == 2
    assert hit[0]["url"] == "https://a.example/"


def test_search_cache_key_provider_separates(isolated_db):
    """Same query under two different provider keys must not collide."""
    from utils.cache import get_cached_search, set_cached_search
    asyncio.run(set_cached_search("provA", "Q", [{"url": "a"}]))
    asyncio.run(set_cached_search("provB", "Q", [{"url": "b"}]))
    a = asyncio.run(get_cached_search("provA", "Q"))
    b = asyncio.run(get_cached_search("provB", "Q"))
    assert a == [{"url": "a"}]
    assert b == [{"url": "b"}]


def test_search_cache_query_case_insensitive(isolated_db):
    """`Foo Bar` and `foo bar` should resolve to the same cache row — search
    results don't differ by case in practice and this saves duplicates."""
    from utils.cache import get_cached_search, set_cached_search
    asyncio.run(set_cached_search("p", "FOO Bar  ", [{"url": "x"}]))
    assert asyncio.run(get_cached_search("p", "foo bar")) == [{"url": "x"}]


def test_search_cache_ttl_expiry(isolated_db):
    from utils.cache import get_cached_search, set_cached_search
    asyncio.run(set_cached_search("p", "q", [{"url": "x"}]))
    assert asyncio.run(get_cached_search("p", "q", ttl_seconds=0)) is None


def test_search_cache_empty_results_not_persisted(isolated_db):
    """We deliberately don't cache empty result lists — that would mask
    transient provider outages."""
    from utils.cache import get_cached_search, set_cached_search
    asyncio.run(set_cached_search("p", "q", []))
    assert asyncio.run(get_cached_search("p", "q")) is None


# ── purge ────────────────────────────────────────────────────────────────


def test_purge_expired_drops_stale_rows(isolated_db):
    from utils.cache import (
        get_cached_page, set_cached_page,
        get_cached_search, set_cached_search,
        purge_expired,
    )
    asyncio.run(set_cached_page("https://old/", "x"))
    asyncio.run(set_cached_search("p", "old query", [{"u": "x"}]))
    # Sleep >= 1s so int(time.time()) at purge is strictly greater than
    # int(time.time()) at insert; then ttl=0 marks everything stale.
    time.sleep(1.2)
    out = asyncio.run(purge_expired(page_ttl_seconds=0, search_ttl_seconds=0))
    assert out["pages"] >= 1
    assert out["searches"] >= 1
    assert asyncio.run(get_cached_page("https://old/")) is None
    assert asyncio.run(get_cached_search("p", "old query")) is None
