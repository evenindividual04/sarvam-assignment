"""Tests for utils.wikipedia — supplementary REST client for DEFINITION queries.

We use httpx's MockTransport rather than monkeypatching httpx.AsyncClient
directly so we exercise the real request path (URL construction, headers,
JSON parsing) and only mock the network."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from utils import wikipedia as wiki_mod


def _run(coro):
    return asyncio.run(coro)


def _mock_transport(responder):
    return httpx.MockTransport(responder)


def _patched_client(monkeypatch, responder):
    """Replace httpx.AsyncClient with one bound to a MockTransport."""
    real_cls = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = _mock_transport(responder)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(wiki_mod.httpx, "AsyncClient", factory)


def test_fetch_wikipedia_summary_known_entity(monkeypatch):
    """Happy path: 200 + Wikipedia JSON → returns parsed dict."""
    payload = {
        "title": "Reserve Bank of India",
        "extract": "The Reserve Bank of India is India's central bank.",
        "content_urls": {
            "desktop": {"page": "https://en.wikipedia.org/wiki/Reserve_Bank_of_India"}
        },
    }

    def responder(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "en.wikipedia.org"
        assert "/page/summary/" in request.url.path
        return httpx.Response(200, json=payload)

    _patched_client(monkeypatch, responder)
    out = _run(wiki_mod.fetch_wikipedia_summary("Reserve Bank of India"))
    assert out is not None
    assert out["title"] == "Reserve Bank of India"
    assert out["url"].startswith("https://en.wikipedia.org/wiki/")
    assert "central bank" in out["snippet"]
    assert out["domain"] == "wikipedia.org"


def test_fetch_wikipedia_summary_404(monkeypatch):
    def responder(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"type": "not_found"})

    _patched_client(monkeypatch, responder)
    out = _run(wiki_mod.fetch_wikipedia_summary("ThisShouldNotExistXYZ"))
    assert out is None


def test_fetch_wikipedia_summary_timeout(monkeypatch):
    def responder(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    _patched_client(monkeypatch, responder)
    out = _run(wiki_mod.fetch_wikipedia_summary("anything"))
    assert out is None


def test_fetch_wikipedia_summary_hindi_routing(monkeypatch):
    """Devanagari query must hit hi.wikipedia.org."""
    seen_hosts: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        return httpx.Response(200, json={
            "title": "भारतीय रिज़र्व बैंक",
            "extract": "भारतीय रिज़र्व बैंक भारत का केंद्रीय बैंक है।",
            "content_urls": {"desktop": {"page": "https://hi.wikipedia.org/wiki/भारतीय_रिज़र्व_बैंक"}},
        })

    _patched_client(monkeypatch, responder)
    out = _run(wiki_mod.fetch_wikipedia_summary("भारतीय रिज़र्व बैंक"))
    assert out is not None
    assert seen_hosts == ["hi.wikipedia.org"]
    assert out["url"].startswith("https://hi.wikipedia.org/")


def test_wikipedia_disabled_via_env(monkeypatch):
    monkeypatch.setenv("WIKIPEDIA_DISABLED", "1")
    # No httpx mock needed — function must short-circuit before any network.
    out = _run(wiki_mod.fetch_wikipedia_summary("anything"))
    assert out is None


def test_fetch_wikipedia_summary_non_json(monkeypatch):
    def responder(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>", headers={"content-type": "text/html"})

    _patched_client(monkeypatch, responder)
    out = _run(wiki_mod.fetch_wikipedia_summary("anything"))
    assert out is None


def test_fetch_wikipedia_summary_empty_query():
    out = _run(wiki_mod.fetch_wikipedia_summary(""))
    assert out is None
