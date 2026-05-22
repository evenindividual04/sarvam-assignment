"""Tests for utils.quota_webhook."""
from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from utils import quota_webhook
from utils.quota_webhook import notify_quota_threshold, reset_cooldown_state


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("QUOTA_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("QUOTA_ALERT_THRESHOLD", raising=False)
    monkeypatch.delenv("QUOTA_ALERT_COOLDOWN_S", raising=False)
    reset_cooldown_state()
    yield
    reset_cooldown_state()


class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class _FakeClient:
    """Records POSTs without making real HTTP calls."""

    captured: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        _FakeClient.captured.append({"url": url, "json": json})
        return _FakeResponse(200)


@pytest.fixture
def fake_client(monkeypatch):
    _FakeClient.captured = []
    monkeypatch.setattr(quota_webhook.httpx, "AsyncClient", _FakeClient)
    return _FakeClient


def test_webhook_fires_above_threshold(monkeypatch, fake_client):
    monkeypatch.setenv("QUOTA_WEBHOOK_URL", "https://example.com/hook")
    asyncio.run(notify_quota_threshold("groq", used=850_000, limit=1_000_000))
    assert len(fake_client.captured) == 1
    body = fake_client.captured[0]["json"]
    assert body["provider"] == "groq"
    assert body["used"] == 850_000
    assert body["limit"] == 1_000_000
    assert body["unit"] == "tokens"
    assert 0.84 < body["fraction"] < 0.86
    assert "85" in body["text"]


def test_webhook_skipped_below_threshold(monkeypatch, fake_client):
    monkeypatch.setenv("QUOTA_WEBHOOK_URL", "https://example.com/hook")
    asyncio.run(notify_quota_threshold("groq", used=500_000, limit=1_000_000))
    assert fake_client.captured == []


def test_webhook_cooldown_prevents_duplicate_alerts(monkeypatch, fake_client):
    monkeypatch.setenv("QUOTA_WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.setenv("QUOTA_ALERT_COOLDOWN_S", "3600")
    asyncio.run(notify_quota_threshold("groq", used=900_000, limit=1_000_000))
    asyncio.run(notify_quota_threshold("groq", used=910_000, limit=1_000_000))
    assert len(fake_client.captured) == 1


def test_webhook_no_url_skips_silently(monkeypatch, fake_client):
    # QUOTA_WEBHOOK_URL deliberately unset.
    asyncio.run(notify_quota_threshold("groq", used=900_000, limit=1_000_000))
    assert fake_client.captured == []


def test_webhook_timeout_logged(monkeypatch, caplog):
    monkeypatch.setenv("QUOTA_WEBHOOK_URL", "https://example.com/hook")

    class _HangingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            raise httpx.TimeoutException("simulated timeout")

    monkeypatch.setattr(quota_webhook.httpx, "AsyncClient", _HangingClient)
    caplog.set_level(logging.WARNING, logger="utils.quota_webhook")
    # Must not raise.
    asyncio.run(notify_quota_threshold("groq", used=900_000, limit=1_000_000))
    assert any("quota_webhook POST failed" in rec.message for rec in caplog.records)


def test_webhook_zero_limit_skips(monkeypatch, fake_client):
    monkeypatch.setenv("QUOTA_WEBHOOK_URL", "https://example.com/hook")
    asyncio.run(notify_quota_threshold("sarvam", used=1, limit=0))
    assert fake_client.captured == []


def test_webhook_custom_threshold(monkeypatch, fake_client):
    monkeypatch.setenv("QUOTA_WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.setenv("QUOTA_ALERT_THRESHOLD", "0.95")
    asyncio.run(notify_quota_threshold("groq", used=850_000, limit=1_000_000))
    assert fake_client.captured == []
    asyncio.run(notify_quota_threshold("groq", used=960_000, limit=1_000_000))
    assert len(fake_client.captured) == 1
