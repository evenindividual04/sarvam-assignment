from __future__ import annotations

import asyncio
from unittest.mock import patch

from utils.provider_adapters import normalize_search_items
from utils.prompt_registry import PROMPT_REGISTRY, prompt_id
from utils.failure_policy import FailurePolicy


def _domain(url: str) -> str:
    return url.split('/')[2]


def test_adapter_normalizes_valid_payload():
    payload = {"results": [{"url": "https://a.com/x", "title": "A", "snippet": "S"}]}
    out = normalize_search_items("parallel", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert out.outcome.ok
    assert len(out.results) == 1
    assert out.results[0].domain == "a.com"


def test_adapter_flags_schema_mismatch():
    payload = {"results": {"url": "https://a.com/x"}}
    out = normalize_search_items("parallel", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert not out.outcome.ok
    assert out.outcome.adapter_error_code == "parallel_schema_mismatch"


# ── SearchResult.relevance ────────────────────────────────────────────────
#
# Per current API docs (May 2026): Tavily exposes a `score` field per result;
# Parallel and Serper do not. The adapter must:
#   1. Use Tavily's `score` directly, clamped to [0,1], tagged "provider".
#   2. Derive rank-based scores for Parallel/Serper, tagged "rank".
#   3. Set `relevance_source` consistently so downstream scoring can weight
#      provider-supplied (absolute) signals differently from rank-derived
#      (relative) ones.


def test_tavily_score_field_is_used_as_provider_relevance():
    payload = {"results": [
        {"url": "https://a.com/x", "title": "A", "snippet": "S", "score": 0.92},
        {"url": "https://b.com/y", "title": "B", "snippet": "S", "score": 0.34},
    ]}
    out = normalize_search_items("tavily", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert out.outcome.ok
    assert out.results[0].relevance == 0.92
    assert out.results[0].relevance_source == "provider"
    assert out.results[1].relevance == 0.34
    assert out.results[1].relevance_source == "provider"


def test_tavily_score_clamped_to_zero_one():
    payload = {"results": [
        {"url": "https://a.com/x", "title": "A", "snippet": "S", "score": 1.42},
        {"url": "https://b.com/y", "title": "B", "snippet": "S", "score": -0.1},
    ]}
    out = normalize_search_items("tavily", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert out.results[0].relevance == 1.0  # clamped down
    assert out.results[1].relevance == 0.0  # clamped up


def test_parallel_uses_rank_derived_relevance():
    payload = {"results": [
        {"url": "https://a.com/1", "title": "A", "snippet": "S"},
        {"url": "https://b.com/2", "title": "B", "snippet": "S"},
        {"url": "https://c.com/3", "title": "C", "snippet": "S"},
        {"url": "https://d.com/4", "title": "D", "snippet": "S"},
    ]}
    out = normalize_search_items("parallel", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert out.outcome.ok
    # All tagged "rank" since Parallel doesn't expose a per-result score.
    assert all(r.relevance_source == "rank" for r in out.results)
    # First result is 1.0; monotonically decreasing.
    rels = [r.relevance for r in out.results]
    assert rels[0] == 1.0
    assert rels == sorted(rels, reverse=True)
    # Last result still has a positive bound (formula: 1 - (i / N)).
    assert rels[-1] > 0


def test_serper_uses_rank_derived_relevance():
    # Serper uses `organic` key and `link` instead of `url`.
    payload = {"organic": [
        {"link": "https://a.com/1", "title": "A", "snippet": "S", "position": 1},
        {"link": "https://b.com/2", "title": "B", "snippet": "S", "position": 2},
    ]}
    out = normalize_search_items("serper", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert out.outcome.ok
    assert all(r.relevance_source == "rank" for r in out.results)
    assert out.results[0].relevance == 1.0
    assert out.results[1].relevance == 0.5


def test_tavily_missing_score_falls_back_to_rank():
    # Defensive: if Tavily ever returns an item without `score`, don't crash —
    # fall back to rank like the rank-only providers.
    payload = {"results": [
        {"url": "https://a.com/x", "title": "A", "snippet": "S"},
    ]}
    out = normalize_search_items("tavily", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert out.results[0].relevance == 1.0
    assert out.results[0].relevance_source == "rank"


def test_prompt_registry_ids_nonempty():
    assert prompt_id("planner")
    assert prompt_id("synthesizer")
    assert "id" in PROMPT_REGISTRY["conflict_detection"]


def test_failure_policy_defaults_parsed():
    policy = FailurePolicy()
    assert policy.plan_timeout_s > 0
    assert policy.search_timeout_s > 0
    assert policy.max_retries_per_provider >= 1


# ────────────────────────────────────────────────────────────────────────────
# Phase 1.875 — Agent Flow Polish
# ────────────────────────────────────────────────────────────────────────────

def test_gather_with_return_exceptions():
    """Regression: if one search task raises, surviving queries still return."""
    import httpx
    from agent import search as search_mod
    from agent.models import QueryIntent, SearchResult, TypedQuery

    queries = [
        TypedQuery(text="ok-query", intent=QueryIntent.PRIMARY),
        TypedQuery(text="fail-query", intent=QueryIntent.DEFINITION),
    ]

    async def _fake_single(q, client, intent=QueryIntent.PRIMARY, language=None, time_sensitivity="static"):
        if q == "fail-query":
            raise RuntimeError("simulated provider crash")
        return [SearchResult(
            url="https://ok.example/x",
            title="OK",
            snippet="snip",
            domain="ok.example",
            retrieved_at="2026-05-01T00:00:00+00:00",
            intent_origin=intent.value,
        )]

    async def _run():
        with patch.object(search_mod, "_search_single_query", _fake_single):
            return await search_mod.search(queries)

    results = asyncio.run(_run())
    # Surviving query's result must come through despite the other raising.
    assert any(r.url == "https://ok.example/x" for r in results)


def test_planner_output_enriched_schema_parses():
    """All Phase 1.875 fields round-trip through parse_planner_output."""
    from utils.provider_router import parse_planner_output
    raw = (
        '{"strategy":"s","confidence":"high","time_sensitivity":"live",'
        '"expected_source_types":["academic","official"],"difficulty":"hard",'
        '"ambiguity_flag":true,"success_criteria":["bullet 1","bullet 2"],'
        '"queries":[{"text":"q","intent":"primary"}]}'
    )
    out = parse_planner_output(raw, "fallback")
    assert out.time_sensitivity == "live"
    assert "academic" in out.expected_source_types
    assert "official" in out.expected_source_types
    assert out.difficulty == "hard"
    assert out.ambiguity_flag is True
    assert out.success_criteria == ["bullet 1", "bullet 2"]


def test_planner_output_missing_fields_fallback_to_defaults():
    """Old-shape JSON (no enriched fields) still parses with safe defaults."""
    from utils.provider_router import parse_planner_output
    raw = '{"strategy":"s","confidence":"medium","queries":[{"text":"q","intent":"primary"}]}'
    out = parse_planner_output(raw, "fallback")
    assert out.time_sensitivity == "static"
    assert out.expected_source_types == []
    assert out.difficulty == "medium"
    assert out.ambiguity_flag is False
    assert out.success_criteria == []


def test_planner_output_invalid_enum_values_fall_back():
    """Garbage values for enriched enums fall back to defaults — no raise."""
    from utils.provider_router import parse_planner_output
    raw = (
        '{"strategy":"s","time_sensitivity":"chaos","difficulty":"trivial",'
        '"expected_source_types":["spam","academic"],"ambiguity_flag":"yes",'
        '"success_criteria":"not-a-list","queries":[{"text":"q","intent":"primary"}]}'
    )
    out = parse_planner_output(raw, "fallback")
    assert out.time_sensitivity == "static"
    assert out.difficulty == "medium"
    # only the valid value survives the filter
    assert out.expected_source_types == ["academic"]
    # non-bool is rejected; non-list success_criteria becomes []
    assert out.ambiguity_flag is False
    assert out.success_criteria == []


def test_verify_numeric_grounding_basic():
    """Numeric tokens present in cited snippets are flagged grounded=True."""
    from agent.citation_guard import CitationGuard
    guard = CitationGuard()
    answer = "The repo rate is 5.50% as of 2026."
    doc_map = {"doc_1": ("Title", "https://example.com", "example.com")}
    snippet_lookup = {"doc_1": "Per the latest RBI notification, the repo rate stands at 5.50% as of 2026."}
    out = guard.verify_numeric_grounding(answer, doc_map, snippet_lookup)
    assert out["total"] >= 2
    assert out["grounded"] == out["total"]
    assert out["numeric_grounding_ratio"] == 1.0


def test_verify_numeric_grounding_hallucinated():
    """A number not present in any snippet is flagged grounded=False."""
    from agent.citation_guard import CitationGuard
    guard = CitationGuard()
    answer = "The figure is 99.99% according to sources."
    doc_map = {"doc_1": ("Title", "https://example.com", "example.com")}
    snippet_lookup = {"doc_1": "The actual figure is 5.50% per the report."}
    out = guard.verify_numeric_grounding(answer, doc_map, snippet_lookup)
    flagged = [a for a in out["audit"] if a["token"].startswith("99")]
    assert flagged and not flagged[0]["grounded"]
    assert out["numeric_grounding_ratio"] < 1.0


# ────────────────────────────────────────────────────────────────────────────
# Parallel v1beta /search schema (excerpts: list[str], objective + search_queries)
# ────────────────────────────────────────────────────────────────────────────


def test_parallel_v1beta_excerpts_are_concatenated_into_snippet():
    payload = {"results": [
        {
            "url": "https://india.gov.in/capital",
            "title": "Capital of India",
            "excerpts": [
                "New Delhi is the capital of India.",
                "It became the capital in 1911.",
                "Located in the National Capital Territory.",
                "Fourth excerpt should be ignored.",
            ],
        }
    ]}
    out = normalize_search_items("parallel", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert out.outcome.ok
    assert len(out.results) == 1
    r = out.results[0]
    assert r.url == "https://india.gov.in/capital"
    assert r.title == "Capital of India"
    # First three excerpts joined; the fourth is dropped.
    assert "New Delhi" in r.snippet
    assert "1911" in r.snippet
    assert "National Capital Territory" in r.snippet
    assert "Fourth excerpt" not in r.snippet
    # raw_content carries the full set for downstream extraction.
    assert "Fourth excerpt" in (r.raw_content or "")
    # Per CLAUDE.md, Parallel does not return a per-result score.
    assert r.relevance_source == "rank"


def test_parallel_v1beta_request_body_uses_new_schema(monkeypatch):
    """Ensure _search_parallel POSTs {objective, search_queries, processor, max_results}."""
    import httpx
    from agent import search as search_mod

    captured: dict = {}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    async def _fake_post(self, url, headers=None, json=None, timeout=None):  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResp()

    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    async def _run():
        async with httpx.AsyncClient() as c:
            # Bypass cache to force the network path.
            from utils import search_cache
            async def _noget(*a, **k): return None
            async def _noput(*a, **k): return None
            monkeypatch.setattr(search_cache, "get", _noget)
            monkeypatch.setattr(search_cache, "put", _noput)
            return await search_mod._search_parallel("capital of India", c, num_results=5)

    asyncio.run(_run())
    assert captured["url"].endswith("/v1beta/search")
    body = captured["json"]
    assert "objective" in body
    assert body["objective"] == "capital of India"
    assert body["search_queries"] == ["capital of India"]
    assert body["processor"] == "base"
    assert body["max_results"] == 5


def test_cerebras_404_logs_helpful_warning(monkeypatch, caplog):
    """A 404 from Cerebras logs a warning naming the working models."""
    import logging
    from utils import provider_router

    class _Resp:
        status_code = 404

    class _NotFound(Exception):
        def __init__(self):
            super().__init__(
                "Error code: 404 - Model llama-3.3-70b does not exist "
                "or you do not have access to it."
            )
            self.status_code = 404
            self.response = _Resp()

    class _FakeChat:
        async def create(self, **kwargs):
            raise _NotFound()

    class _FakeCompletions:
        def __init__(self):
            self.completions = _FakeChat()

    class _FakeClient:
        def __init__(self, **_):
            self.chat = _FakeCompletions()

    monkeypatch.setenv("CEREBRAS_API_KEY", "test")
    monkeypatch.setenv("CEREBRAS_MODEL", "llama-3.3-70b")
    # The Cerebras key rotator caches its pool at module-load time, so a
    # runtime monkeypatch.setenv doesn't reach it. Stub next_key() directly
    # so the 404 path executes against the fake client.
    monkeypatch.setattr(provider_router._CEREBRAS_ROTATOR, "next_key", lambda: "test")
    monkeypatch.setattr(provider_router._CEREBRAS_ROTATOR, "mark_success", lambda *_a, **_kw: None)
    monkeypatch.setattr(provider_router._CEREBRAS_ROTATOR, "mark_throttled", lambda *_a, **_kw: None)
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)

    with caplog.at_level(logging.WARNING, logger=provider_router.logger.name):
        async def _run():
            try:
                await provider_router._plan_with_cerebras("hello")
            except Exception:
                pass
        asyncio.run(_run())

    msg_blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "llama3.1-8b" in msg_blob
    assert "cloud.cerebras.ai/models" in msg_blob


def test_parse_numeric_tokens_separates_year_from_number():
    """Year tokens parse as kind='year', not 'number'."""
    from agent.citation_guard import parse_numeric_tokens
    toks = parse_numeric_tokens("In 2026 the rate hit 5.50%.")
    kinds = {t["token"]: t["kind"] for t in toks}
    assert kinds.get("2026") == "year"
    assert kinds.get("5.50%") == "number"
