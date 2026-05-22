"""Tests for utils/circuit_breaker.py."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from tenacity import retry, stop_after_attempt, wait_fixed

from utils import circuit_breaker as cb_mod
from utils.circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    breaker,
    get_breaker,
    _is_counted_failure,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _fresh_breaker() -> CircuitBreaker:
    """Return the singleton with all state wiped (tests run sequentially)."""
    cb = get_breaker()
    cb._providers.clear()
    cb._configs.clear()
    cb._save_event = None
    return cb


def _fake_time(monkeypatch, start: float = 1000.0):
    """Patch time.time inside circuit_breaker to return a controllable value."""
    state = {"now": start}

    def _now():
        return state["now"]

    monkeypatch.setattr(cb_mod.time, "time", _now)
    return state


def _http_status(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://example.com")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"status {code}", request=req, response=resp)


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_breaker_starts_closed():
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=3))
    assert cb.state("p") == "closed"
    await cb.before("p")  # should not raise


@pytest.mark.asyncio
async def test_rapid_failures_open_circuit(monkeypatch):
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=3, window_s=60, open_duration_s=30))
    _fake_time(monkeypatch)
    for _ in range(3):
        await cb.on_failure("p", httpx.TimeoutException("timeout"))
    assert cb.state("p") == "open"
    with pytest.raises(CircuitOpenError) as exc:
        await cb.before("p")
    assert exc.value.provider == "p"


@pytest.mark.asyncio
async def test_open_duration_elapsed_transitions_to_half_open(monkeypatch):
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=2, window_s=60, open_duration_s=30))
    clock = _fake_time(monkeypatch, start=100.0)
    await cb.on_failure("p", asyncio.TimeoutError())
    await cb.on_failure("p", asyncio.TimeoutError())
    assert cb.state("p") == "open"
    clock["now"] = 100.0 + 31.0
    await cb.before("p")
    assert cb.state("p") == "half_open"


@pytest.mark.asyncio
async def test_half_open_success_transitions_to_closed(monkeypatch):
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=2, window_s=60, open_duration_s=30))
    clock = _fake_time(monkeypatch, start=100.0)
    await cb.on_failure("p", asyncio.TimeoutError())
    await cb.on_failure("p", asyncio.TimeoutError())
    clock["now"] = 131.0
    await cb.before("p")  # → half_open, inflight probe
    await cb.on_success("p")
    assert cb.state("p") == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_transitions_back_to_open(monkeypatch):
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=2, window_s=60, open_duration_s=30))
    clock = _fake_time(monkeypatch, start=100.0)
    await cb.on_failure("p", asyncio.TimeoutError())
    await cb.on_failure("p", asyncio.TimeoutError())
    clock["now"] = 131.0
    await cb.before("p")  # half_open probe
    clock["now"] = 132.0
    await cb.on_failure("p", asyncio.TimeoutError())
    assert cb.state("p") == "open"
    # opened_at reset to the new time
    assert cb._providers["p"].opened_at == 132.0


@pytest.mark.asyncio
async def test_failures_outside_window_dont_count(monkeypatch):
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=3, window_s=10, open_duration_s=30))
    clock = _fake_time(monkeypatch, start=0.0)
    await cb.on_failure("p", asyncio.TimeoutError())
    await cb.on_failure("p", asyncio.TimeoutError())
    clock["now"] = 20.0
    await cb.on_failure("p", asyncio.TimeoutError())
    # Only 1 failure in current window
    assert cb.state("p") == "closed"


@pytest.mark.asyncio
async def test_isolation_between_providers(monkeypatch):
    cb = _fresh_breaker()
    cb.register("groq", BreakerConfig(threshold=2))
    cb.register("gemini", BreakerConfig(threshold=2))
    _fake_time(monkeypatch)
    await cb.on_failure("groq", asyncio.TimeoutError())
    await cb.on_failure("groq", asyncio.TimeoutError())
    assert cb.state("groq") == "open"
    assert cb.state("gemini") == "closed"
    await cb.before("gemini")  # not blocked


@pytest.mark.asyncio
async def test_uncounted_exception_does_not_increment():
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=2))

    @breaker("p")
    async def fn():
        raise ValueError("not counted")

    for _ in range(5):
        with pytest.raises(ValueError):
            await fn()
    assert cb.state("p") == "closed"


@pytest.mark.asyncio
async def test_4xx_other_than_429_does_not_count():
    assert not _is_counted_failure(_http_status(400))
    assert not _is_counted_failure(_http_status(404))
    assert not _is_counted_failure(_http_status(422))


@pytest.mark.asyncio
async def test_429_counts_as_failure():
    assert _is_counted_failure(_http_status(429))


@pytest.mark.asyncio
async def test_500_counts_as_failure():
    assert _is_counted_failure(_http_status(500))
    assert _is_counted_failure(_http_status(503))


@pytest.mark.asyncio
async def test_breaker_doesnt_double_count_tenacity_retries():
    """One tenacity-exhausted call must count as ONE breaker failure, not N."""
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=3))

    attempts = {"n": 0}

    @breaker("p")
    @retry(wait=wait_fixed(0), stop=stop_after_attempt(3), reraise=True)
    async def fn():
        attempts["n"] += 1
        raise httpx.TimeoutException("nope")

    # Two outer calls (each Tenacity-retries 3× internally = 6 attempts total)
    # but only 2 logical failures from the breaker's view.
    with pytest.raises(httpx.TimeoutException):
        await fn()
    with pytest.raises(httpx.TimeoutException):
        await fn()

    assert attempts["n"] == 6  # tenacity retries fired
    # Threshold is 3 → 2 failures keeps breaker closed
    assert cb.state("p") == "closed"
    assert len(cb._providers["p"].failures) == 2


@pytest.mark.asyncio
async def test_event_persister_called_on_transitions(monkeypatch):
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=2))
    _fake_time(monkeypatch)

    persisted: list[dict] = []

    async def fake_save(event: dict) -> None:
        persisted.append(event)

    cb.set_event_persister(fake_save)
    await cb.on_failure("p", asyncio.TimeoutError())
    await cb.on_failure("p", asyncio.TimeoutError())
    assert len(persisted) == 1
    e = persisted[0]
    assert e["provider"] == "p"
    assert e["from_state"] == "closed"
    assert e["to_state"] == "open"
    assert "threshold_breached" in e["reason"]
    assert "event_id" in e


@pytest.mark.asyncio
async def test_event_persister_failure_doesnt_break_transition(monkeypatch):
    cb = _fresh_breaker()
    cb.register("p", BreakerConfig(threshold=2))
    _fake_time(monkeypatch)

    async def boom(_event):
        raise RuntimeError("db down")

    cb.set_event_persister(boom)
    await cb.on_failure("p", asyncio.TimeoutError())
    await cb.on_failure("p", asyncio.TimeoutError())
    # Transition still occurred despite persister failure
    assert cb.state("p") == "open"


def test_circuit_open_error_provider_attribute():
    e = CircuitOpenError("groq")
    assert e.provider == "groq"
    assert "groq" in str(e)


@pytest.mark.asyncio
async def test_search_skips_parallel_when_breaker_open(monkeypatch):
    """When parallel breaker is open, _search_single_query falls through to tavily."""
    from agent import search as search_mod
    from agent.models import QueryIntent, SearchResult

    cb = _fresh_breaker()
    cb.register("parallel", BreakerConfig(threshold=1))
    cb.register("tavily", BreakerConfig(threshold=5))
    cb.register("serper", BreakerConfig(threshold=5))
    _fake_time(monkeypatch)
    # Force parallel breaker open
    await cb.on_failure("parallel", asyncio.TimeoutError())
    assert cb.state("parallel") == "open"

    parallel_called = {"n": 0}
    tavily_called = {"n": 0}

    async def fake_parallel(*args, **kwargs):
        parallel_called["n"] += 1
        return []

    async def fake_tavily(*args, **kwargs):
        tavily_called["n"] += 1
        return [
            SearchResult(
                url="https://t.com",
                title="t",
                snippet="s",
                domain="t.com",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        ]

    async def fake_serper(*args, **kwargs):
        return []

    # Wrap the fakes with the same breaker decorator the originals used
    monkeypatch.setattr(search_mod, "_search_parallel", breaker("parallel")(fake_parallel))
    monkeypatch.setattr(search_mod, "_search_tavily", breaker("tavily")(fake_tavily))
    monkeypatch.setattr(search_mod, "_search_serper", breaker("serper")(fake_serper))

    async with httpx.AsyncClient() as client:
        results = await search_mod._search_single_query("q", client, QueryIntent.PRIMARY)

    assert parallel_called["n"] == 0  # breaker short-circuited
    assert tavily_called["n"] == 1
    assert len(results) == 1


@pytest.mark.asyncio
async def test_synthesize_falls_back_to_openrouter_when_gemini_breaker_open(monkeypatch):
    """When gemini breaker is open, synthesize() routes to openrouter."""
    from utils import provider_router

    cb = _fresh_breaker()
    cb.register("gemini", BreakerConfig(threshold=1))
    cb.register("openrouter", BreakerConfig(threshold=5))
    _fake_time(monkeypatch)
    await cb.on_failure("gemini", asyncio.TimeoutError())
    assert cb.state("gemini") == "open"

    gemini_calls = {"n": 0}
    openrouter_calls = {"n": 0}

    @breaker("gemini")
    async def fake_gemini(prompt):
        gemini_calls["n"] += 1
        yield ("g", 0, 0)

    @breaker("openrouter")
    async def fake_openrouter(prompt):
        openrouter_calls["n"] += 1
        yield ("from-openrouter", 0, 0)

    monkeypatch.setattr(provider_router, "_synthesize_gemini", fake_gemini)
    monkeypatch.setattr(provider_router, "_synthesize_openrouter", fake_openrouter)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    # V3.9: ensure providers BETWEEN gemini and openrouter in the chain are
    # skipped via missing-key pre-flight, so we still land on openrouter
    # under this test's hypothesis.
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    chunks = []
    async for c in provider_router.synthesize(
        query="q",
        context_xml="<context/>",
        doc_map={},
        history_text="",
    ):
        chunks.append(c)

    assert gemini_calls["n"] == 0  # breaker blocked it
    assert openrouter_calls["n"] == 1
    assert any("openrouter" in c[0] for c in chunks)


@pytest.mark.asyncio
async def test_synthesize_returns_error_string_when_both_open(monkeypatch):
    """When both gemini AND openrouter breakers are open, synthesize yields an error string."""
    from utils import provider_router

    cb = _fresh_breaker()
    cb.register("gemini", BreakerConfig(threshold=1))
    cb.register("openrouter", BreakerConfig(threshold=1))
    _fake_time(monkeypatch)
    await cb.on_failure("gemini", asyncio.TimeoutError())
    await cb.on_failure("openrouter", asyncio.TimeoutError())

    @breaker("gemini")
    async def fake_gemini(prompt):
        yield ("g", 0, 0)

    @breaker("openrouter")
    async def fake_openrouter(prompt):
        yield ("o", 0, 0)

    monkeypatch.setattr(provider_router, "_synthesize_gemini", fake_gemini)
    monkeypatch.setattr(provider_router, "_synthesize_openrouter", fake_openrouter)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    # V3.9: exhaust the rest of the chain so the error string is the final
    # outcome — sarvam, cerebras, ollama all need to be unavailable too.
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    # Ollama has no key, so kill its base URL to force an unreachable error.
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1/v1")

    chunks = []
    async for c in provider_router.synthesize(
        query="q", context_xml="<c/>", doc_map={}, history_text=""
    ):
        chunks.append(c)

    assert len(chunks) == 1
    assert "unavailable" in chunks[0][0].lower()


@pytest.mark.asyncio
async def test_json_decode_error_counts_as_failure():
    assert _is_counted_failure(json.JSONDecodeError("bad", "doc", 0))
