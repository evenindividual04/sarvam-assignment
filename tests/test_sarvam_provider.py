"""Item 1 (FDSE add-ons) — Sarvam Model API provider tests.

These mock the OpenAI SDK client so no live key is needed. The Sarvam endpoint
is OpenAI-compatible, so the test surface focuses on:
  - base_url + api_key wiring
  - streaming delta plumbing into the (text, prompt_tokens, completion_tokens) tuple
  - sarvam-m → sarvam-30b model-level fallback on first-call failure
  - SYNTH_PROVIDER env dispatch
  - circuit breaker opening after threshold failures
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import pytest

from utils import provider_router


class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


async def _fake_stream(chunks: list[str]) -> AsyncIterator[_FakeChunk]:
    for c in chunks:
        yield _FakeChunk(c)
    yield _FakeChunk(None)  # final chunk with no delta


class _FakeCompletions:
    def __init__(self, capture: dict, chunks: list[str], raise_exc: Exception | None) -> None:
        self._capture = capture
        self._chunks = chunks
        self._raise = raise_exc

    async def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self._capture.update(kwargs)
        if self._raise is not None:
            raise self._raise
        return _fake_stream(self._chunks)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeAsyncOpenAI:
    """Captures init args for assertion; returns fake chat.completions.create."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        _FakeAsyncOpenAI.last_init = kwargs
        self.chat = _FakeChat(
            _FakeCompletions(
                _FakeAsyncOpenAI.captured_call,
                _FakeAsyncOpenAI.next_chunks,
                _FakeAsyncOpenAI.next_raise,
            )
        )


def _reset_fake() -> None:
    _FakeAsyncOpenAI.last_init = {}
    _FakeAsyncOpenAI.captured_call = {}
    _FakeAsyncOpenAI.next_chunks = []
    _FakeAsyncOpenAI.next_raise = None


@pytest.fixture(autouse=True)
def _patch_openai(monkeypatch):
    _reset_fake()
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setenv("SARVAM_API_KEY", "sk_test")
    # Reset any prior SARVAM_MODEL pinning leaking between tests.
    monkeypatch.delenv("SARVAM_MODEL", raising=False)
    monkeypatch.delenv("SYNTH_PROVIDER", raising=False)
    yield


async def _collect(gen) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    async for chunk in gen:
        out.append(chunk)
    return out


def test_sarvam_provider_uses_correct_base_url():
    _FakeAsyncOpenAI.next_chunks = ["hi"]

    asyncio.run(_collect(provider_router._synthesize_sarvam("question")))

    assert _FakeAsyncOpenAI.last_init["base_url"] == "https://api.sarvam.ai/v1"
    assert _FakeAsyncOpenAI.last_init["api_key"] == "sk_test"
    assert _FakeAsyncOpenAI.captured_call["model"] == "sarvam-m"
    # System + user messages always present.
    roles = [m["role"] for m in _FakeAsyncOpenAI.captured_call["messages"]]
    assert roles == ["system", "user"]


def test_sarvam_provider_streams_chunks():
    _FakeAsyncOpenAI.next_chunks = ["The ", "answer ", "is 42."]

    chunks = asyncio.run(_collect(provider_router._synthesize_sarvam("q")))

    text = "".join(c[0] for c in chunks)
    assert "The answer is 42." in text
    # Final sentinel chunk (empty text + 0/0 tokens) always emitted.
    assert chunks[-1] == ("", 0, 0)


def test_sarvam_falls_back_to_30b_on_m_error(monkeypatch):
    # First create() raises; second succeeds. Sequence the fakes.
    call_models: list[str] = []

    class _Recording(_FakeCompletions):
        def __init__(self) -> None:  # type: ignore[no-untyped-def]
            super().__init__({}, [], None)

        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            call_models.append(kwargs["model"])
            if kwargs["model"] == "sarvam-m":
                raise RuntimeError("sarvam-m unavailable on this account")
            return _fake_stream(["Hello"])

    rec = _Recording()

    class _RecOpenAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.chat = _FakeChat(rec)

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecOpenAI)

    chunks = asyncio.run(_collect(provider_router._synthesize_sarvam("q")))
    text = "".join(c[0] for c in chunks)
    assert "Hello" in text
    assert call_models == ["sarvam-m", "sarvam-30b"]


def test_synth_provider_env_dispatches_to_sarvam(monkeypatch):
    monkeypatch.setenv("SYNTH_PROVIDER", "sarvam")
    _FakeAsyncOpenAI.next_chunks = ["Sarvam reply."]

    async def go():
        out = []
        async for chunk in provider_router.synthesize(
            query="capital of India?",
            context_xml="<documents></documents>",
            doc_map={},
            history_text="",
            conflict_note=None,
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    text = "".join(c[0] for c in chunks)
    assert "Sarvam reply." in text
    # Verify dispatch landed on Sarvam (not Gemini): captured model proves it.
    assert _FakeAsyncOpenAI.captured_call["model"] == "sarvam-m"


def test_circuit_breaker_opens_on_sarvam_failures(monkeypatch):
    """4 consecutive failures should open the 'sarvam' breaker."""
    from utils.circuit_breaker import CircuitOpenError, get_breaker
    from utils import failure_policy  # noqa: F401  (registers breakers)

    cb = get_breaker()
    # Reset Sarvam breaker state to a clean closed (the registry persists
    # across tests in the same process).
    if "sarvam" in cb._providers:
        cb._providers["sarvam"].failures.clear()
        cb._providers["sarvam"].state = "closed"
        cb._providers["sarvam"].opened_at = 0.0

    # Force both sarvam-m AND sarvam-30b to raise so one synth attempt counts
    # as one breaker failure.
    class _AlwaysFail(_FakeCompletions):
        def __init__(self) -> None:  # type: ignore[no-untyped-def]
            super().__init__({}, [], None)

        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            # asyncio.TimeoutError is in the breaker's counted-failure set.
            raise asyncio.TimeoutError("upstream slow")

    fail = _AlwaysFail()

    class _FailOpenAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.chat = _FakeChat(fail)

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", _FailOpenAI)

    async def run_n(n: int) -> list[Exception | None]:
        errs: list[Exception | None] = []
        for _ in range(n):
            try:
                async for _chunk in provider_router._synthesize_sarvam("q"):
                    pass
                errs.append(None)
            except Exception as e:
                errs.append(e)
        return errs

    # Threshold is 4; the 4th attempt counts as the failure that opens the
    # breaker, so we need at least a 5th attempt to observe CircuitOpenError.
    errs = asyncio.run(run_n(7))
    # The first failures should be RuntimeError; once threshold (4) is hit,
    # subsequent attempts must raise CircuitOpenError without dispatching.
    assert any(isinstance(e, CircuitOpenError) for e in errs), errs


def test_synthesize_default_is_gemini(monkeypatch):
    """Default SYNTH_PROVIDER (unset) must not dispatch to Sarvam."""
    monkeypatch.delenv("SYNTH_PROVIDER", raising=False)

    # Patch out gemini & openrouter to no-op generators so we don't make real
    # network calls; we just want to confirm sarvam path is NOT taken.
    sarvam_called = {"yes": False}

    async def fake_gemini(prompt):
        yield ("gemini-out", 0, 0)

    async def fake_sarvam(prompt):
        sarvam_called["yes"] = True
        yield ("nope", 0, 0)

    monkeypatch.setattr(provider_router, "_synthesize_gemini", fake_gemini)
    monkeypatch.setattr(provider_router, "_synthesize_sarvam", fake_sarvam)

    async def go():
        async for _ in provider_router.synthesize(
            query="q", context_xml="<documents/>", doc_map={}, history_text="",
        ):
            pass

    asyncio.run(go())
    assert sarvam_called["yes"] is False
