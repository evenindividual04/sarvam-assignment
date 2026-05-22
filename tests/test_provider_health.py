"""Smoke tests for utils.provider_health — probe specs, key resolution,
snapshot serialization."""
from __future__ import annotations

import pytest

from utils.provider_health import (
    HealthSnapshot,
    ProviderProbe,
    _has_required_key,
    _PROBE_SPECS,
    _REQUIRED_KEYS,
)


def test_probe_specs_cover_known_providers():
    """All probe-spec providers must have a key entry. Prevents drift where
    a new probe is added without its required-key declaration."""
    spec_names = {name for name, _, _ in _PROBE_SPECS}
    for n in spec_names:
        assert n in _REQUIRED_KEYS, f"missing _REQUIRED_KEYS entry for {n}"


def test_has_required_key_for_keyless_provider():
    """Ollama has no API key and should always probe (server reachability
    being its real signal)."""
    assert _has_required_key("ollama") is True


def test_has_required_key_missing(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    assert _has_required_key("parallel") is False


def test_has_required_key_present(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "dummy-key")
    assert _has_required_key("parallel") is True


def test_has_required_key_unknown_provider():
    assert _has_required_key("not-a-real-provider") is False


def test_snapshot_to_dict_round_trip():
    snap = HealthSnapshot(
        checked_at=1716000000.0,
        overall="ok",
        providers=[
            ProviderProbe(
                name="parallel", role="search", status="ok",
                latency_ms=42, detail="",
            ),
        ],
    )
    d = snap.to_dict()
    assert d["overall"] == "ok"
    assert d["providers"][0]["name"] == "parallel"
    assert d["providers"][0]["latency_ms"] == 42
    assert "cache_ttl_s" in d


def test_provider_probe_is_frozen():
    p = ProviderProbe(name="x", role="search", status="ok")
    with pytest.raises(Exception):
        p.name = "y"  # type: ignore[misc]


# ── Smart-polling: TTL cache + invalidate() ────────────────────────────────


@pytest.mark.asyncio
async def test_get_health_returns_cached_within_ttl(monkeypatch):
    """Second call inside the TTL must return the same snapshot object and
    NOT re-run probes. This is the load-bearing behavior — without it, every
    dashboard poll hits real provider APIs and burns quota."""
    import utils.provider_health as ph

    monkeypatch.setattr(ph, "HEALTH_CACHE_TTL_S", 90)
    ph.invalidate()  # reset module-level cache

    call_count = {"n": 0}

    async def fake_run():
        call_count["n"] += 1
        import time as _t
        return ph.HealthSnapshot(checked_at=_t.time(), overall="ok", providers=[])

    monkeypatch.setattr(ph, "_run_all_probes", fake_run)

    snap1 = await ph.get_health()
    snap2 = await ph.get_health()
    assert call_count["n"] == 1
    assert snap1 is snap2


@pytest.mark.asyncio
async def test_invalidate_forces_reprobe(monkeypatch):
    """invalidate() must clear the cache so the next get_health() re-runs
    probes, even within the TTL — this is the lazy-on-demand probing path
    triggered by orchestrator 429 detection."""
    import utils.provider_health as ph

    ph.invalidate()
    call_count = {"n": 0}

    async def fake_run():
        call_count["n"] += 1
        import time as _t
        return ph.HealthSnapshot(checked_at=_t.time(), overall="ok", providers=[])

    monkeypatch.setattr(ph, "_run_all_probes", fake_run)

    await ph.get_health()
    ph.invalidate()  # whole-cache invalidate
    await ph.get_health()
    assert call_count["n"] == 2


def test_invalidate_provider_marks_stale(monkeypatch):
    """invalidate(provider) keeps the snapshot but flags stale=True, and adds
    the provider to the force-reprobe set."""
    import time as _t
    import utils.provider_health as ph

    ph.invalidate()
    ph._cache = ph.HealthSnapshot(
        checked_at=_t.time(), overall="ok",
        providers=[ProviderProbe(name="gemini", role="synth", status="ok")],
    )
    ph._next_probe_at["gemini"] = _t.time() + 9999  # would block re-probe

    ph.invalidate("gemini")

    assert ph._cache is not None
    assert ph._cache.stale is True
    assert "gemini" in ph._force_reprobe
    assert "gemini" not in ph._next_probe_at


def test_parse_retry_after():
    from utils.provider_health import _parse_retry_after
    assert _parse_retry_after("30") == 30.0
    assert _parse_retry_after(" 12.5 ") == 12.5
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("Tue, 01 Jan 2030") is None
    assert _parse_retry_after("") is None
