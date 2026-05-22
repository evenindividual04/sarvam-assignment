"""V3.9: Cohere Rerank augmentation in agent/context_engine.py.

Cohere sits between FlashRank and 5-factor scoring. Gated on COHERE_API_KEY,
English-only routing, fails soft (returns None → fall back to FlashRank).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent.models import ContextSnippet
from utils.cohere_rerank import rerank_with_cohere


def _mk(doc_id: str, text: str = "lorem ipsum dolor sit amet") -> ContextSnippet:
    return ContextSnippet(
        doc_id=doc_id,
        url=f"https://example.com/{doc_id}",
        title=doc_id,
        domain="example.com",
        text=text,
        snippet=text[:60],
        token_count=8,
        retrieved_at="2026-05-20T00:00:00+00:00",
    )


class _MockResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_cohere_rerank_reorders_chunks(monkeypatch):
    """POST returns shuffled order → snippets come back in that order."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    chunks = [_mk("a"), _mk("b"), _mk("c")]
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.1},
        ]
    }

    async def fake_post(*a, **kw):
        return _MockResp(payload)

    with patch("httpx.AsyncClient.post", new=fake_post):
        out = asyncio.run(rerank_with_cohere("q", chunks, top_k=3))

    assert out is not None
    assert [c.doc_id for c in out] == ["c", "a", "b"]
    assert getattr(out[0], "cohere_relevance", None) == 0.9


def test_cohere_client_singleton_reused(monkeypatch):
    """Two consecutive rerank calls must reuse a single ``httpx.AsyncClient``
    rather than spinning up a new connection pool each time."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    import utils.cohere_rerank as cr

    # Reset the module singleton so this test starts clean.
    async def _reset():
        await cr.close_cohere_client()

    asyncio.run(_reset())

    payload = {"results": [{"index": 0, "relevance_score": 0.9}]}
    post_calls: list[object] = []

    async def fake_post(self, *a, **kw):
        post_calls.append(self)
        return _MockResp(payload)

    with patch("httpx.AsyncClient.post", new=fake_post):
        out1 = asyncio.run(rerank_with_cohere("q", [_mk("a")], top_k=1))
        client_after_first = cr._CLIENT
        out2 = asyncio.run(rerank_with_cohere("q2", [_mk("b")], top_k=1))
        client_after_second = cr._CLIENT

    assert out1 is not None and out2 is not None
    assert client_after_first is not None
    assert client_after_first is client_after_second, (
        "Cohere httpx client should be reused across calls"
    )
    asyncio.run(_reset())


def test_cohere_rerank_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    out = asyncio.run(rerank_with_cohere("q", [_mk("a")]))
    assert out is None


def test_cohere_rerank_timeout_returns_none(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    async def fake_post(*a, **kw):
        raise asyncio.TimeoutError()

    with patch("httpx.AsyncClient.post", new=fake_post):
        out = asyncio.run(rerank_with_cohere("q", [_mk("a"), _mk("b")]))
    assert out is None


def test_cohere_rerank_empty_chunks_returns_none(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    out = asyncio.run(rerank_with_cohere("q", []))
    assert out is None


def test_cohere_rerank_malformed_response_returns_none(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    async def fake_post(*a, **kw):
        return _MockResp({"unexpected": "shape"})

    with patch("httpx.AsyncClient.post", new=fake_post):
        out = asyncio.run(rerank_with_cohere("q", [_mk("a"), _mk("b")]))
    assert out is None


# ── Context engine integration ────────────────────────────────────────────


def test_context_engine_non_english_skipped(monkeypatch):
    """Devanagari query → Cohere skipped, reranker_used = flashrank."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    from agent import context_engine

    sink: dict = {}
    token = context_engine.set_reranker_sink(sink)
    try:
        chunks = [_mk("a", "भारत की राजधानी"), _mk("b", "नई दिल्ली")]
        cohere_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("utils.cohere_rerank.rerank_with_cohere", cohere_mock)
        asyncio.run(context_engine.rank_and_select_async("भारत की राजधानी क्या है", chunks))
        cohere_mock.assert_not_awaited()
        assert sink.get("reranker_used") in {"flashrank", "none"}
    finally:
        context_engine.reset_reranker_sink(token)


def test_context_engine_uses_cohere_when_available(monkeypatch):
    """English query + COHERE_API_KEY set + cohere returns ordering →
    reranker_used = "cohere"."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    from agent import context_engine

    chunks = [_mk("a", "the capital of france is paris"),
              _mk("b", "the capital of germany is berlin")]

    async def fake_cohere(query, candidates, top_k=10, timeout_s=8.0):
        return list(reversed(candidates))

    monkeypatch.setattr("utils.cohere_rerank.rerank_with_cohere", fake_cohere)

    sink: dict = {}
    token = context_engine.set_reranker_sink(sink)
    try:
        out = asyncio.run(context_engine.rank_and_select_async(
            "what is the capital of france?", chunks,
        ))
    finally:
        context_engine.reset_reranker_sink(token)

    assert sink.get("reranker_used") == "cohere"
    assert len(out) >= 1


def test_context_engine_falls_back_to_flashrank_without_key(monkeypatch):
    """No COHERE_API_KEY → reranker_used = "flashrank" (or "none")."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    from agent import context_engine

    chunks = [_mk("a", "capital france paris"),
              _mk("b", "capital germany berlin")]

    sink: dict = {}
    token = context_engine.set_reranker_sink(sink)
    try:
        asyncio.run(context_engine.rank_and_select_async(
            "what is the capital of france?", chunks,
        ))
    finally:
        context_engine.reset_reranker_sink(token)

    assert sink.get("reranker_used") in {"flashrank", "none"}
