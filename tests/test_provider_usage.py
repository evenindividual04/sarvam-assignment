"""Smoke tests for utils.provider_usage — counter increments + limit join."""
from __future__ import annotations

import asyncio
import pytest

from utils.provider_usage import _DAILY_LIMITS, _today_utc


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    db = tmp_path / "db.sqlite"
    monkeypatch.setattr("agent.memory.DB_PATH", str(db))
    monkeypatch.setattr("utils.provider_usage.DB_PATH", str(db))
    from agent.memory import init_db
    asyncio.run(init_db())
    return db


def test_today_utc_format():
    today = _today_utc()
    assert len(today) == 10  # YYYY-MM-DD
    assert today[4] == "-" and today[7] == "-"


def test_known_providers_have_limits():
    """Audit: documented providers all carry a limits entry. Catches drift
    where a new provider lands without quota tracking."""
    for p in ["gemini", "groq", "parallel", "tavily", "serper"]:
        assert p in _DAILY_LIMITS
        assert "requests" in _DAILY_LIMITS[p]


def test_record_then_snapshot_increments(isolated_db):
    from utils import provider_usage
    asyncio.run(provider_usage.record("gemini", requests=1,
                                      prompt_tokens=100, completion_tokens=20))
    rows = asyncio.run(provider_usage.snapshot())
    gemini = next((r for r in rows if r["provider"] == "gemini"), None)
    assert gemini is not None
    assert gemini["requests"] == 1
    assert gemini["prompt_tokens"] == 100
    assert gemini["completion_tokens"] == 20
    # Documented free-tier limit joined in.
    assert gemini["request_limit"] == 1500


def test_record_aggregates(isolated_db):
    """Two record() calls for the same provider/day should sum, not replace."""
    from utils import provider_usage
    asyncio.run(provider_usage.record("groq", requests=2))
    asyncio.run(provider_usage.record("groq", requests=3))
    rows = asyncio.run(provider_usage.snapshot())
    groq = next((r for r in rows if r["provider"] == "groq"), None)
    assert groq is not None
    assert groq["requests"] == 5


def test_record_disabled_via_env(isolated_db, monkeypatch):
    """PROVIDER_USAGE_DISABLED=1 should bypass the increment."""
    monkeypatch.setenv("PROVIDER_USAGE_DISABLED", "1")
    from utils import provider_usage
    asyncio.run(provider_usage.record("gemini", requests=5))
    rows = asyncio.run(provider_usage.snapshot())
    assert not any(r["provider"] == "gemini" for r in rows)


def test_snapshot_empty_returns_empty_list(isolated_db):
    from utils import provider_usage
    rows = asyncio.run(provider_usage.snapshot())
    assert rows == []
