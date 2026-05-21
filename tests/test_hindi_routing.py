"""V3.4 — Hindi eval subset: Devanagari detection + language-aware search routing."""
from __future__ import annotations

import asyncio

import pytest

from agent import search as search_mod
from agent.models import QueryIntent, SearchResult, TypedQuery


def test_devanagari_query_detected_as_hindi():
    assert search_mod._detect_language("भारत में रिज़र्व बैंक") == "hi"
    assert search_mod._detect_language("भारतीय रिज़र्व बैंक (RBI) की वर्तमान रेपो दर") == "hi"
    # English-only query
    assert search_mod._detect_language("What is the RBI repo rate?") == "en"
    # Empty
    assert search_mod._detect_language("") == "en"


def test_hindi_query_routes_to_parallel_only(monkeypatch):
    """Hindi queries should hit Parallel first and skip Tavily/Serper while Parallel succeeds."""
    call_order: list[str] = []

    async def fake_parallel(q, client, num_results=None):
        call_order.append("parallel")
        return [SearchResult(
            url="https://hi.example.com/a", title="t", snippet="s",
            domain="hi.example.com", retrieved_at="2026-01-01T00:00:00+00:00",
        )]

    async def fake_tavily(q, client):
        call_order.append("tavily")
        return []

    async def fake_serper(q, client):
        call_order.append("serper")
        return []

    monkeypatch.setattr(search_mod, "_search_parallel", fake_parallel)
    monkeypatch.setattr(search_mod, "_search_tavily", fake_tavily)
    monkeypatch.setattr(search_mod, "_search_serper", fake_serper)

    out = asyncio.run(search_mod.search([
        TypedQuery(text="भारत में रिज़र्व बैंक की रेपो दर", intent=QueryIntent.PRIMARY)
    ]))
    assert call_order == ["parallel"]
    assert len(out) == 1


def test_hindi_query_falls_through_when_parallel_breaker_open(monkeypatch):
    """If Parallel returns empty (breaker open / failure), Hindi routing degrades to Tavily/Serper."""
    call_order: list[str] = []

    async def fake_parallel(q, client, num_results=None):
        call_order.append("parallel")
        return []

    async def fake_tavily(q, client):
        call_order.append("tavily")
        return [SearchResult(
            url="https://tav.example.com/a", title="t", snippet="s",
            domain="tav.example.com", retrieved_at="2026-01-01T00:00:00+00:00",
        )]

    async def fake_serper(q, client):
        call_order.append("serper")
        return []

    monkeypatch.setattr(search_mod, "_search_parallel", fake_parallel)
    monkeypatch.setattr(search_mod, "_search_tavily", fake_tavily)
    monkeypatch.setattr(search_mod, "_search_serper", fake_serper)

    out = asyncio.run(search_mod.search([
        TypedQuery(text="भारत के वित्त मंत्री कौन हैं?", intent=QueryIntent.PRIMARY)
    ]))
    assert call_order == ["parallel", "tavily"]
    assert len(out) == 1
    assert out[0].domain == "tav.example.com"


def test_english_query_unchanged_routing(monkeypatch):
    """English queries must still follow the standard PRIMARY chain (Parallel → Tavily → Serper)."""
    call_order: list[str] = []

    async def fake_parallel(q, client, num_results=None):
        call_order.append("parallel")
        return [SearchResult(
            url="https://en.example.com/a", title="t", snippet="s",
            domain="en.example.com", retrieved_at="2026-01-01T00:00:00+00:00",
        )]

    async def fake_tavily(q, client):
        call_order.append("tavily")
        return []

    async def fake_serper(q, client):
        call_order.append("serper")
        return []

    monkeypatch.setattr(search_mod, "_search_parallel", fake_parallel)
    monkeypatch.setattr(search_mod, "_search_tavily", fake_tavily)
    monkeypatch.setattr(search_mod, "_search_serper", fake_serper)

    out = asyncio.run(search_mod.search([
        TypedQuery(text="What is the RBI repo rate?", intent=QueryIntent.PRIMARY)
    ]))
    assert call_order == ["parallel"]
    assert len(out) == 1


def test_synthesizer_prompt_includes_hindi_instruction():
    """The synthesizer system prompt must instruct the model to respond in Hindi
    when the query is in Devanagari (V3.4 contract)."""
    from utils.prompt_registry import PROMPT_REGISTRY

    sys_prompt = PROMPT_REGISTRY["synthesizer"]["system"]
    lower = sys_prompt.lower()
    assert "hindi" in lower
    assert "devanagari" in lower
    # Citation contract must remain intact even with the new language block.
    assert "[doc_n]" in lower or "doc_n" in lower
