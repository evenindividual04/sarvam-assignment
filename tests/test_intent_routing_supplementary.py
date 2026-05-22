"""Verifies that search() invokes the supplementary Wikipedia/Scholar paths
based on intent and expected_source_types, alongside (not in place of) the
primary provider chain."""
from __future__ import annotations

import asyncio

import pytest

from agent import search as search_mod
from agent.models import QueryIntent, SearchResult, TypedQuery


def _primary(q, *args, **kwargs):
    async def _f():
        return [SearchResult(
            url="https://primary.example.com/a", title="primary",
            snippet="s", domain="primary.example.com",
            retrieved_at="2026-01-01T00:00:00+00:00",
        )]
    return _f()


def test_definition_intent_calls_wikipedia(monkeypatch):
    called: list[str] = []

    async def fake_wiki(q):
        called.append(q)
        return [SearchResult(
            url="https://en.wikipedia.org/wiki/Foo", title="Foo",
            snippet="extract", domain="wikipedia.org",
            retrieved_at="2026-01-01T00:00:00+00:00",
            intent_origin="wikipedia",
        )]

    async def fake_scholar(q, top_k=3):
        called.append(f"scholar:{q}")
        return []

    async def fake_single(q, client, intent, language=None, time_sensitivity="static"):
        return [SearchResult(
            url="https://primary.example.com/a", title="primary",
            snippet="s", domain="primary.example.com",
            retrieved_at="2026-01-01T00:00:00+00:00",
        )]

    monkeypatch.setattr(search_mod, "_supplement_with_wikipedia", fake_wiki)
    monkeypatch.setattr(search_mod, "_supplement_with_scholar", fake_scholar)
    monkeypatch.setattr(search_mod, "_search_single_query", fake_single)

    counts: dict[str, int] = {}
    out = asyncio.run(search_mod.search(
        [TypedQuery(text="define foo", intent=QueryIntent.DEFINITION)],
        supplementary_counts=counts,
    ))
    assert "define foo" in called
    assert counts.get("wikipedia") == 1
    # Both primary and wiki results should be present, primary first.
    urls = [r.url for r in out]
    assert "https://primary.example.com/a" in urls
    assert "https://en.wikipedia.org/wiki/Foo" in urls


def test_academic_expected_type_calls_scholar(monkeypatch):
    called: list[str] = []

    async def fake_wiki(q):
        called.append(f"wiki:{q}")
        return []

    async def fake_scholar(q, top_k=3):
        called.append(f"scholar:{q}")
        return [SearchResult(
            url="https://arxiv.org/abs/1", title="P", snippet="abstract",
            domain="arxiv.org", retrieved_at="2026-01-01T00:00:00+00:00",
            intent_origin="academic",
        )]

    async def fake_single(q, client, intent, language=None, time_sensitivity="static"):
        return []

    monkeypatch.setattr(search_mod, "_supplement_with_wikipedia", fake_wiki)
    monkeypatch.setattr(search_mod, "_supplement_with_scholar", fake_scholar)
    monkeypatch.setattr(search_mod, "_search_single_query", fake_single)

    counts: dict[str, int] = {}
    out = asyncio.run(search_mod.search(
        [TypedQuery(text="attention mechanism", intent=QueryIntent.PRIMARY)],
        expected_source_types=["academic"],
        supplementary_counts=counts,
    ))
    assert "scholar:attention mechanism" in called
    assert counts.get("scholar") == 1
    assert any(r.intent_origin == "academic" for r in out)


def test_no_supplements_when_intent_and_types_dont_match(monkeypatch):
    """PRIMARY intent with empty expected_source_types → no supplement calls."""
    called: list[str] = []

    async def fake_wiki(q):
        called.append("wiki")
        return []

    async def fake_scholar(q, top_k=3):
        called.append("scholar")
        return []

    async def fake_single(q, client, intent, language=None, time_sensitivity="static"):
        return []

    monkeypatch.setattr(search_mod, "_supplement_with_wikipedia", fake_wiki)
    monkeypatch.setattr(search_mod, "_supplement_with_scholar", fake_scholar)
    monkeypatch.setattr(search_mod, "_search_single_query", fake_single)

    asyncio.run(search_mod.search(
        [TypedQuery(text="anything", intent=QueryIntent.PRIMARY)],
        expected_source_types=[],
    ))
    assert called == []


def test_wikipedia_failure_does_not_break_search(monkeypatch):
    """Supplement raising should be swallowed; primary results survive."""
    async def fake_wiki(q):
        raise RuntimeError("boom")

    async def fake_scholar(q, top_k=3):
        return []

    async def fake_single(q, client, intent, language=None, time_sensitivity="static"):
        return [SearchResult(
            url="https://primary.example.com/a", title="t", snippet="s",
            domain="primary.example.com",
            retrieved_at="2026-01-01T00:00:00+00:00",
        )]

    monkeypatch.setattr(search_mod, "_supplement_with_wikipedia", fake_wiki)
    monkeypatch.setattr(search_mod, "_supplement_with_scholar", fake_scholar)
    monkeypatch.setattr(search_mod, "_search_single_query", fake_single)

    out = asyncio.run(search_mod.search(
        [TypedQuery(text="define foo", intent=QueryIntent.DEFINITION)],
    ))
    assert len(out) == 1
    assert out[0].url == "https://primary.example.com/a"
