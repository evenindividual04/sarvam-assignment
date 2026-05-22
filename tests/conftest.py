"""Shared test fixtures.

Provider-routing additions (Goal 1–6) introduced auto-detect behavior keyed on
whether ``OPENROUTER_API_KEY`` / ``CEREBRAS_API_KEY`` are set. Tests should
opt in to those code paths explicitly via ``monkeypatch.setenv(...)``, so we
clear them by default to keep legacy-test behavior identical regardless of
what's in the developer's ``.env``.
"""
from __future__ import annotations

import os

import pytest


_PROVIDER_OPT_IN_KEYS = (
    "OPENROUTER_API_KEY",
    "CEREBRAS_API_KEY",
    "SARVAM_API_KEY",
    "SARVAM_INDIC_AUTO",
    "PLANNER_PROVIDER",
    "CLAIM_VERIFIER_PROVIDER",
    "SYNTH_PROVIDER",
)


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch):
    """Unset provider-routing env vars for every test unless the test sets them.

    The legacy test suite predates Cerebras / DeepSeek / Sarvam-auto routing
    and asserts on behavior that assumed those keys were absent. Tests that
    exercise the new behavior set the relevant env vars explicitly with
    ``monkeypatch.setenv(...)`` — those will override this clear and run the
    new code path.
    """
    for k in _PROVIDER_OPT_IN_KEYS:
        if k in os.environ:
            monkeypatch.delenv(k, raising=False)
