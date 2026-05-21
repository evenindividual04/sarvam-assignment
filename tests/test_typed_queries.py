"""V2.1 — Typed Query Decomposition tests."""
from __future__ import annotations

import asyncio
from collections import defaultdict

import httpx
import pytest
from pydantic import ValidationError

from agent import search as search_mod
from agent.context_engine import score_chunk, select_with_diversity
from agent.models import (
    ContextSnippet,
    PlannerOutput,
    QueryIntent,
    SearchResult,
    TypedQuery,
)
from utils import provider_router


def _mk_snippet(domain: str, score: float, intent_origin: str | None = None, tokens: int = 100) -> ContextSnippet:
    snip = ContextSnippet(
        doc_id="x",
        url=f"https://{domain}/{score}",
        title="t",
        domain=domain,
        text="text",
        snippet="text",
        token_count=tokens,
        retrieved_at="2026-01-01T00:00:00+00:00",
        intent_origin=intent_origin,
    )
    snip.final_score = score
    return snip


def test_planner_output_schema_accepts_typed_queries():
    out = PlannerOutput(
        strategy="s",
        queries=[
            TypedQuery(text="primary q", intent=QueryIntent.PRIMARY),
            TypedQuery(text="criticism of X", intent=QueryIntent.CONTRADICTION_PROBE, rationale="adversarial"),
        ],
    )
    assert out.queries[1].intent == QueryIntent.CONTRADICTION_PROBE
    assert out.queries[1].rationale == "adversarial"


def test_planner_output_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        TypedQuery(text="q", intent="bogus_intent")  # type: ignore[arg-type]


def test_plan_fallback_on_parse_failure_returns_single_primary():
    out = provider_router.parse_planner_output("totally not json", "the original query")
    assert out.strategy == "Direct retrieval fallback"
    assert len(out.queries) == 1
    assert out.queries[0].text == "the original query"
    assert out.queries[0].intent == QueryIntent.PRIMARY


def test_parse_planner_rejects_bad_intent_with_fallback():
    raw = '{"strategy":"s","queries":[{"text":"q","intent":"not_a_real_intent"}]}'
    out = provider_router.parse_planner_output(raw, "fallback")
    assert out.strategy == "Direct retrieval fallback"
    assert out.queries[0].text == "fallback"


def test_search_routes_recency_check_to_tavily_first_when_available(monkeypatch):
    call_order: list[str] = []

    async def fake_tavily_news(q, client, days=30):
        call_order.append("tavily_news")
        return [SearchResult(
            url="https://news.example.com/a", title="t", snippet="s",
            domain="news.example.com", retrieved_at="2026-01-01T00:00:00+00:00",
        )]

    async def fake_parallel(q, client, num_results=None):
        call_order.append("parallel")
        return []

    async def fake_tavily(q, client):
        call_order.append("tavily")
        return []

    async def fake_serper(q, client):
        call_order.append("serper")
        return []

    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    monkeypatch.setattr(search_mod, "_search_tavily_news", fake_tavily_news)
    monkeypatch.setattr(search_mod, "_search_parallel", fake_parallel)
    monkeypatch.setattr(search_mod, "_search_tavily", fake_tavily)
    monkeypatch.setattr(search_mod, "_search_serper", fake_serper)

    out = asyncio.run(search_mod.search([
        TypedQuery(text="latest GDP 2026", intent=QueryIntent.RECENCY_CHECK)
    ]))
    assert call_order[0] == "tavily_news"
    assert len(out) == 1
    assert out[0].intent_origin == "recency_check"


def test_search_tags_intent_origin_on_contradiction_probe_results(monkeypatch):
    async def fake_parallel(q, client, num_results=None):
        return [SearchResult(
            url="https://example.com/criticism", title="t", snippet="s",
            domain="example.com", retrieved_at="2026-01-01T00:00:00+00:00",
        )]

    async def fake_tavily(q, client):
        return []

    async def fake_serper(q, client):
        return []

    monkeypatch.setattr(search_mod, "_search_parallel", fake_parallel)
    monkeypatch.setattr(search_mod, "_search_tavily", fake_tavily)
    monkeypatch.setattr(search_mod, "_search_serper", fake_serper)

    out = asyncio.run(search_mod.search([
        TypedQuery(text="criticism of X", intent=QueryIntent.CONTRADICTION_PROBE)
    ]))
    assert len(out) == 1
    assert out[0].intent_origin == "contradiction_probe"


def test_score_chunk_diversity_boost_for_contradiction_probe_is_bounded():
    # diversity already 1.0 (no prior same-domain chunks) → stays at 1.0
    snip_full = _mk_snippet("a.com", 0.0, intent_origin="contradiction_probe")
    score_chunk(snip_full, 1.0, [1.0], {"a.com": 0}, intent_origin="contradiction_probe")
    assert snip_full.diversity_score == pytest.approx(1.0)

    # diversity 0.5 (one prior same-domain) → 0.5 * 1.25 = 0.625
    snip_half = _mk_snippet("b.com", 0.0, intent_origin="contradiction_probe")
    score_chunk(snip_half, 1.0, [1.0], {"b.com": 1}, intent_origin="contradiction_probe")
    assert snip_half.diversity_score == pytest.approx(0.625)


def test_selection_allows_3_per_domain_for_contradiction_probe_chunks():
    chunks = [
        _mk_snippet("a.com", 1.0, intent_origin="contradiction_probe"),
        _mk_snippet("a.com", 0.9, intent_origin="contradiction_probe"),
        _mk_snippet("a.com", 0.8, intent_origin="contradiction_probe"),
        _mk_snippet("a.com", 0.7, intent_origin="contradiction_probe"),
    ]
    # default max_per_domain is 2 — contradiction_probe override should yield 3.
    selected = select_with_diversity(chunks, max_tokens=10_000, max_per_domain=2)
    assert len(selected) == 3
    # Non-probe chunks still respect the 2-cap.
    plain = [
        _mk_snippet("a.com", 1.0),
        _mk_snippet("a.com", 0.9),
        _mk_snippet("a.com", 0.8),
    ]
    selected_plain = select_with_diversity(plain, max_tokens=10_000, max_per_domain=2)
    assert len(selected_plain) == 2


def test_planner_output_persisted_to_run_metadata_json(monkeypatch):
    """Smoke check: parse_planner_output round-trips through the typed schema
    and would serialize cleanly into run_metadata['planner_output']."""
    raw = (
        '{"strategy":"compare and probe",'
        '"queries":[{"text":"primary q","intent":"primary"},'
        '{"text":"criticism of q","intent":"contradiction_probe","rationale":"adversarial"}]}'
    )
    out = provider_router.parse_planner_output(raw, "fallback")
    serialized = {
        "strategy": out.strategy,
        "queries": [
            {"text": q.text, "intent": q.intent.value, "rationale": q.rationale}
            for q in out.queries
        ],
    }
    assert serialized["strategy"] == "compare and probe"
    assert serialized["queries"][1]["intent"] == "contradiction_probe"
    assert serialized["queries"][1]["rationale"] == "adversarial"
    # Must be JSON-serializable
    import json as _json
    _json.dumps(serialized)
