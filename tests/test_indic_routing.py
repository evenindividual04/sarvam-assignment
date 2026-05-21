"""V3.4 + FDSE add-on — Multi-Indic eval subset:
Devanagari / Tamil / Bengali detection + language-aware search routing.

Existing Hindi tests preserved verbatim; new tests extend coverage to Tamil
and Bengali. Marathi shares Devanagari with Hindi, so script-level detection
returns ``hi`` and the dataset's explicit language tag handles disambiguation."""
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


# ── FDSE add-on: Tamil / Bengali / Marathi ────────────────────────────────


def test_language_code_returned_per_script():
    """Script-level detection must return distinct codes for hi/ta/bn/en."""
    assert search_mod._detect_language("தமிழ்நாட்டின் தலைநகர்") == "ta"
    assert search_mod._detect_language("ভারতের রাজধানী") == "bn"
    assert search_mod._detect_language("भारत की राजधानी") == "hi"
    assert search_mod._detect_language("capital of India") == "en"


def test_tamil_query_detected():
    assert search_mod._detect_language("இந்தியாவின் தலைநகர் எது?") == "ta"


def test_bengali_query_detected():
    assert search_mod._detect_language("পশ্চিমবঙ্গের রাজধানী কী?") == "bn"


def test_marathi_query_detected_as_devanagari_not_marathi_specific():
    """Marathi shares Devanagari with Hindi — script-level detection cannot
    distinguish the two. The dataset's explicit `language` field handles it."""
    assert search_mod._detect_language("महाराष्ट्राची राजधानी कोणती आहे?") == "hi"


def test_tamil_query_routes_to_parallel_first(monkeypatch):
    call_order: list[str] = []

    async def fake_parallel(q, client, num_results=None):
        call_order.append("parallel")
        return [SearchResult(
            url="https://ta.example.com/a", title="t", snippet="s",
            domain="ta.example.com", retrieved_at="2026-01-01T00:00:00+00:00",
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
        TypedQuery(text="தமிழ்நாட்டின் தலைநகர் எது?", intent=QueryIntent.PRIMARY)
    ]))
    assert call_order == ["parallel"]
    assert len(out) == 1


def test_bengali_query_routes_to_parallel_first(monkeypatch):
    call_order: list[str] = []

    async def fake_parallel(q, client, num_results=None):
        call_order.append("parallel")
        return [SearchResult(
            url="https://bn.example.com/a", title="t", snippet="s",
            domain="bn.example.com", retrieved_at="2026-01-01T00:00:00+00:00",
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
        TypedQuery(text="ভারতের রাজধানী কী?", intent=QueryIntent.PRIMARY)
    ]))
    assert call_order == ["parallel"]
    assert len(out) == 1


def test_tamil_query_falls_through_when_parallel_empty(monkeypatch):
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
        TypedQuery(text="இந்தியாவின் தற்போதைய பிரதமர் யார்?", intent=QueryIntent.PRIMARY)
    ]))
    assert call_order == ["parallel", "tavily"]
    assert len(out) == 1


def test_synthesizer_prompt_covers_all_indic_scripts():
    """The v5_indic prompt must explicitly mention Tamil, Bengali, and Marathi
    in addition to Hindi/Devanagari."""
    from utils.prompt_registry import PROMPT_REGISTRY

    sys_prompt = PROMPT_REGISTRY["synthesizer"]["system"].lower()
    assert "tamil" in sys_prompt
    assert "bengali" in sys_prompt
    assert "marathi" in sys_prompt
    assert "devanagari" in sys_prompt


def test_synthesizer_prompt_id_bumped_to_v5_indic():
    from utils.prompt_registry import PROMPT_REGISTRY

    assert PROMPT_REGISTRY["synthesizer"]["id"] == "synth_v5_indic"
