"""Tests for utils.scholar — supplementary Serper Scholar wrapper."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from utils import scholar as scholar_mod


def _run(coro):
    return asyncio.run(coro)


def _patched_client(monkeypatch, responder):
    real_cls = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(responder)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(scholar_mod.httpx, "AsyncClient", factory)


def test_fetch_serper_scholar_basic(monkeypatch):
    """Happy path: returns parsed SearchResult list with academic intent."""
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.delenv("SCHOLAR_DISABLED", raising=False)

    payload = {
        "organic": [
            {"title": "Paper A", "link": "https://arxiv.org/abs/1234.5678", "snippet": "abstract A"},
            {"title": "Paper B", "link": "https://www.nature.com/articles/xyz", "snippet": "abstract B"},
            {"title": "Paper C", "link": "https://scholar.example.org/c", "snippet": "abstract C"},
        ]
    }

    def responder(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://google.serper.dev/scholar"
        assert request.headers.get("X-API-KEY") == "test-key"
        return httpx.Response(200, json=payload)

    _patched_client(monkeypatch, responder)
    out = _run(scholar_mod.fetch_serper_scholar("transformer attention", top_k=3))
    assert len(out) == 3
    assert out[0].title == "Paper A"
    assert out[0].intent_origin == "academic"
    assert out[0].domain == "arxiv.org"
    assert out[1].domain == "nature.com"  # www. stripped
    assert out[0].relevance > out[2].relevance  # rank-based descending


def test_fetch_serper_scholar_no_api_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("SCHOLAR_DISABLED", raising=False)
    out = _run(scholar_mod.fetch_serper_scholar("anything"))
    assert out == []


def test_fetch_serper_scholar_timeout(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.delenv("SCHOLAR_DISABLED", raising=False)

    def responder(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    _patched_client(monkeypatch, responder)
    out = _run(scholar_mod.fetch_serper_scholar("anything"))
    assert out == []


def test_fetch_serper_scholar_disabled(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setenv("SCHOLAR_DISABLED", "1")
    out = _run(scholar_mod.fetch_serper_scholar("anything"))
    assert out == []


def test_fetch_serper_scholar_http_error(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.delenv("SCHOLAR_DISABLED", raising=False)

    def responder(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    _patched_client(monkeypatch, responder)
    out = _run(scholar_mod.fetch_serper_scholar("anything"))
    assert out == []


def test_fetch_serper_scholar_malformed_payload(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.delenv("SCHOLAR_DISABLED", raising=False)

    def responder(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"organic": "not-a-list"})

    _patched_client(monkeypatch, responder)
    out = _run(scholar_mod.fetch_serper_scholar("anything"))
    assert out == []
