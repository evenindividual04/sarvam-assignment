"""Tests that the provider_router Gemini call sites use the shared KeyRotator
and mark keys throttled on 429 with the right Retry-After value.

Mirrors tests/test_provider_router_groq_rotation.py."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fresh_router(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "AIzaSy_k1,AIzaSy_k2,AIzaSy_k3")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import utils.provider_router as pr
    pr._GEMINI_ROTATOR.reload()
    return pr


def _install_fake_genai(monkeypatch, *, raise_429: bool = False,
                        retry_after: str | None = None) -> dict:
    """Install a fake `google.genai` package that records the api_key used
    to construct genai.Client(...) and either streams 'ok' or raises a 429."""
    captured: dict = {"keys": []}

    google_pkg = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    types_mod.GenerateContentConfig = GenerateContentConfig

    class _Chunk:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage_metadata = types.SimpleNamespace(
                prompt_token_count=1, candidates_token_count=1
            )

    class _StreamErr(Exception):
        def __init__(self, msg: str, code: int = 429, response=None) -> None:
            super().__init__(msg)
            self.code = code
            self.response = response

    async def _stream_ok():
        # async generator yielding one chunk
        for t in ["hello "]:
            yield _Chunk(t)

    class _AsyncModels:
        async def generate_content_stream(self, **kwargs):
            if raise_429:
                resp = MagicMock()
                resp.status_code = 429
                resp.headers = {"Retry-After": retry_after} if retry_after else {}
                raise _StreamErr("429 RESOURCE_EXHAUSTED", code=429, response=resp)
            return _stream_ok()

        async def generate_content(self, **kwargs):
            if raise_429:
                resp = MagicMock()
                resp.status_code = 429
                resp.headers = {"Retry-After": retry_after} if retry_after else {}
                raise _StreamErr("429 RESOURCE_EXHAUSTED", code=429, response=resp)
            return types.SimpleNamespace(text='{"strategy":"x","queries":[]}')

    class _Aio:
        models = _AsyncModels()

    class Client:
        def __init__(self, api_key: str) -> None:
            captured["keys"].append(api_key)
            self.aio = _Aio()

    genai_mod.Client = Client
    google_pkg.genai = genai_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    return captured


@pytest.mark.asyncio
async def test_gemini_synth_uses_rotator_key(fresh_router, monkeypatch):
    """_synthesize_gemini should pull api_key from the rotator (round-robin)."""
    captured = _install_fake_genai(monkeypatch, raise_429=False)

    async def _noop(*a, **kw): return None
    monkeypatch.setattr("utils.provider_usage.record", _noop)

    # Drain three streams.
    async def _drain():
        out = []
        async for chunk in fresh_router._synthesize_gemini("p"):
            out.append(chunk)
        return out

    await _drain()
    await _drain()
    await _drain()
    assert captured["keys"] == ["AIzaSy_k1", "AIzaSy_k2", "AIzaSy_k3"]


@pytest.mark.asyncio
async def test_gemini_synth_on_rate_limit_marks_throttled(fresh_router, monkeypatch):
    """A 429 with Retry-After=11 should call mark_throttled with retry_after_s=11."""
    _install_fake_genai(monkeypatch, raise_429=True, retry_after="11")

    calls: list[tuple[str, float | None]] = []
    real_mark = fresh_router._GEMINI_ROTATOR.mark_throttled

    def spy(key, retry_after_s=None):
        calls.append((key, retry_after_s))
        return real_mark(key, retry_after_s=retry_after_s)

    monkeypatch.setattr(fresh_router._GEMINI_ROTATOR, "mark_throttled", spy)

    with pytest.raises(Exception):
        async for _ in fresh_router._synthesize_gemini("p"):
            pass

    assert len(calls) >= 1
    key, ra = calls[0]
    assert key in ("AIzaSy_k1", "AIzaSy_k2", "AIzaSy_k3")
    assert ra == 11.0


@pytest.mark.asyncio
async def test_gemini_structured_planner_uses_rotator(fresh_router, monkeypatch):
    """_plan_with_gemini_structured should also pull from the rotator."""
    captured = _install_fake_genai(monkeypatch, raise_429=False)
    out = await fresh_router._plan_with_gemini_structured("prompt")
    assert out  # non-empty JSON-ish text
    assert len(captured["keys"]) == 1
    assert captured["keys"][0] in ("AIzaSy_k1", "AIzaSy_k2", "AIzaSy_k3")
