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
