"""V3.8 — retrieval mode resolver tests.

Covers the three-way decision tree:
   requested ∈ {auto, hybrid, lexical} × vec_available ∈ {True, False}
"""
from __future__ import annotations

import pytest

from utils.retrieval_mode import (
    EffectiveRetrievalMode,
    RetrievalMode,
    requested_mode,
    resolve,
)


# ── resolve() ──────────────────────────────────────────────────────────────

def test_auto_with_vec_resolves_to_hybrid():
    eff = resolve(RetrievalMode.AUTO, vec_available=True)
    assert eff.effective == RetrievalMode.HYBRID
    assert eff.reason is None
    assert not eff.is_fallback
    assert eff.is_hybrid


def test_auto_without_vec_falls_back_to_lexical():
    eff = resolve(RetrievalMode.AUTO, vec_available=False)
    assert eff.effective == RetrievalMode.LEXICAL
    assert eff.reason == "sqlite_extensions_unavailable"
    assert eff.is_fallback
    assert not eff.is_hybrid


def test_hybrid_without_vec_raises_for_fail_loud_contract():
    # The whole point of `hybrid` mode is to refuse to silently degrade.
    # Eval / CI runs depend on this contract.
    with pytest.raises(RuntimeError, match="hybrid"):
        resolve(RetrievalMode.HYBRID, vec_available=False)


def test_hybrid_with_vec_runs_hybrid():
    eff = resolve(RetrievalMode.HYBRID, vec_available=True)
    assert eff.effective == RetrievalMode.HYBRID
    assert eff.reason is None


def test_lexical_ignores_vec_capability():
    # Forced lexical never tries hybrid even if vec is available.
    for vec in (True, False):
        eff = resolve(RetrievalMode.LEXICAL, vec_available=vec)
        assert eff.effective == RetrievalMode.LEXICAL
        assert eff.reason is None


# ── requested_mode() env resolution ───────────────────────────────────────

def test_default_is_auto_when_no_env_set(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    monkeypatch.delenv("HYBRID_RETRIEVAL", raising=False)
    assert requested_mode() == RetrievalMode.AUTO


def test_retrieval_mode_env_wins(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("HYBRID_RETRIEVAL", "0")  # legacy should be ignored
    assert requested_mode() == RetrievalMode.HYBRID


def test_legacy_hybrid_retrieval_alias_1(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("HYBRID_RETRIEVAL", "1")
    assert requested_mode() == RetrievalMode.HYBRID


def test_legacy_hybrid_retrieval_alias_0(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("HYBRID_RETRIEVAL", "0")
    assert requested_mode() == RetrievalMode.LEXICAL


def test_unknown_mode_value_warns_and_falls_back_to_auto(monkeypatch, caplog):
    monkeypatch.setenv("RETRIEVAL_MODE", "magic")
    with caplog.at_level("WARNING"):
        assert requested_mode() == RetrievalMode.AUTO
    assert any("Unknown RETRIEVAL_MODE" in r.message for r in caplog.records)


# ── EffectiveRetrievalMode.to_metadata() ──────────────────────────────────

def test_to_metadata_shape_for_fallback():
    eff = EffectiveRetrievalMode(
        requested=RetrievalMode.AUTO,
        effective=RetrievalMode.LEXICAL,
        reason="sqlite_extensions_unavailable",
    )
    meta = eff.to_metadata()
    assert meta == {
        "requested": "auto",
        "effective": "lexical",
        "reason": "sqlite_extensions_unavailable",
    }


def test_to_metadata_shape_for_no_fallback():
    eff = EffectiveRetrievalMode(
        requested=RetrievalMode.HYBRID,
        effective=RetrievalMode.HYBRID,
        reason=None,
    )
    meta = eff.to_metadata()
    assert meta["reason"] is None
    assert meta["requested"] == meta["effective"] == "hybrid"
