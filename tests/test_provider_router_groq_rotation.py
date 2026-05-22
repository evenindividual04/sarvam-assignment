"""Tests that the provider_router Groq call sites use the shared KeyRotator
and mark keys throttled on 429 with the right Retry-After value."""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# These tests exercise the rotator wiring inside provider_router. We need to
# import the module fresh AFTER setting env vars so the module-level
# _GROQ_ROTATOR loads our test keys.
@pytest.fixture
def fresh_router(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_aaa,gsk_bbb,gsk_ccc")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # If provider_router already imported, force its rotator to reload.
    import utils.provider_router as pr
    pr._GROQ_ROTATOR.reload()
    return pr


def _fake_groq_sdk(*, response_text: str = "ok", raise_rate_limit: bool = False,
                   retry_after: str | None = None) -> types.ModuleType:
    """Build a fake `groq` module with AsyncGroq + RateLimitError that records
    which api_key it was constructed with."""
    fake = types.ModuleType("groq")

    class RateLimitError(Exception):
        def __init__(self, msg: str, response=None) -> None:
            super().__init__(msg)
            self.response = response

    fake.RateLimitError = RateLimitError
    captured: dict = {"keys": []}

    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [FakeChoice(content)]
            self.usage = types.SimpleNamespace(prompt_tokens=0, completion_tokens=0)

    class FakeCompletions:
        async def create(self, **kwargs):
            if raise_rate_limit:
                resp = MagicMock()
                resp.headers = {"Retry-After": retry_after} if retry_after else {}
                raise RateLimitError("429", response=resp)
            return FakeResponse(response_text)

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class AsyncGroq:
        def __init__(self, api_key: str) -> None:
            captured["keys"].append(api_key)
            self.chat = FakeChat()

    fake.AsyncGroq = AsyncGroq
    fake._captured = captured  # type: ignore[attr-defined]
    return fake


@pytest.mark.asyncio
async def test_call_groq_uses_rotator_key(fresh_router, monkeypatch):
    """call_groq() should pull its api_key from the rotator (round-robin)."""
    fake = _fake_groq_sdk(response_text="hello")
    monkeypatch.setitem(sys.modules, "groq", fake)
    # provider_usage.record is awaited inside call_groq; stub it out.
    async def _noop(*a, **kw): return None
    monkeypatch.setattr("utils.provider_usage.record", _noop)

    out1 = await fresh_router.call_groq("p1")
    out2 = await fresh_router.call_groq("p2")
    out3 = await fresh_router.call_groq("p3")
    assert out1 == out2 == out3 == "hello"
    # Three distinct keys were used in round-robin.
    used = fake._captured["keys"]
    assert used == ["gsk_aaa", "gsk_bbb", "gsk_ccc"]


@pytest.mark.asyncio
async def test_call_groq_on_rate_limit_marks_throttled(fresh_router, monkeypatch):
    """A 429 with Retry-After=7 should call mark_throttled with retry_after_s=7."""
    fake = _fake_groq_sdk(raise_rate_limit=True, retry_after="7")
    monkeypatch.setitem(sys.modules, "groq", fake)

    # Spy on mark_throttled.
    calls: list[tuple[str, float | None]] = []
    real_mark = fresh_router._GROQ_ROTATOR.mark_throttled

    def spy(key, retry_after_s=None):
        calls.append((key, retry_after_s))
        return real_mark(key, retry_after_s=retry_after_s)

    monkeypatch.setattr(fresh_router._GROQ_ROTATOR, "mark_throttled", spy)
    # Also disable tenacity retry so the assertion is deterministic — wrap the
    # raw client call only once. Tenacity sees RateLimitError as a non-retry
    # exception (it retries on tenacity.RetryError chain), so a single
    # invocation should re-raise on first failure; but to keep this test
    # focused, just verify mark_throttled was called at least once with the
    # right Retry-After.
    with pytest.raises(Exception):
        await fresh_router.call_groq("p1")

    assert len(calls) >= 1
    key, ra = calls[0]
    assert key in ("gsk_aaa", "gsk_bbb", "gsk_ccc")
    assert ra == 7.0
