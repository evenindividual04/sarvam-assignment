"""S2 security fix: SSRF guard in agent/extractor.py.

The extractor receives URLs from third-party search APIs that we don't
control. Without a guard, a malicious or compromised search result could
target file://, ftp://, cloud metadata endpoints, or internal IPs.
"""
from __future__ import annotations

from agent.extractor import _is_safe_url


def test_is_safe_url_http_ok():
    assert _is_safe_url("http://example.com/page") is True


def test_is_safe_url_https_ok():
    assert _is_safe_url("https://example.com/page?q=1") is True


def test_is_safe_url_file_scheme_blocked():
    assert _is_safe_url("file:///etc/passwd") is False


def test_is_safe_url_ftp_blocked():
    assert _is_safe_url("ftp://example.com/file") is False


def test_is_safe_url_gopher_blocked():
    assert _is_safe_url("gopher://example.com/") is False


def test_is_safe_url_localhost_blocked():
    # Loopback by name resolves at fetch-time, not blocked by IP guard, but
    # the explicit 127.0.0.1 literal IS blocked.
    assert _is_safe_url("http://127.0.0.1/admin") is False
    assert _is_safe_url("http://127.1.2.3/") is False


def test_is_safe_url_ipv6_loopback_blocked():
    assert _is_safe_url("http://[::1]/") is False


def test_is_safe_url_private_ip_rfc1918_blocked():
    assert _is_safe_url("http://10.0.0.1/") is False
    assert _is_safe_url("http://192.168.1.1/") is False
    assert _is_safe_url("http://172.16.0.5/") is False


def test_is_safe_url_link_local_blocked():
    assert _is_safe_url("http://169.254.1.1/") is False


def test_is_safe_url_metadata_endpoint_blocked():
    # AWS/GCP/Azure cloud metadata.
    assert _is_safe_url("http://169.254.169.254/latest/meta-data/") is False
    assert _is_safe_url("http://metadata.google.internal/computeMetadata/v1/") is False
    assert _is_safe_url("http://metadata/") is False


def test_is_safe_url_invalid_url_blocked():
    assert _is_safe_url("not a url") is False
    assert _is_safe_url("") is False
    assert _is_safe_url("http://") is False


def test_is_safe_url_unspecified_blocked():
    assert _is_safe_url("http://0.0.0.0/") is False


# ── Phase 6 follow-up: Jina + Tavily fallback SSRF guards ─────────────────


def test_jina_skipped_for_private_ip_url(monkeypatch):
    """Jina Reader must NOT be invoked for URLs that fail the SSRF guard.
    Even though Jina re-resolves URLs on its own infrastructure, our local
    guard is the user-facing trust boundary."""
    import asyncio
    from agent.extractor import Extractor

    monkeypatch.delenv("JINA_READER_DISABLED", raising=False)
    ext = Extractor()
    # Sentinel: if httpx is touched, the test fails. We monkeypatch the
    # client.get to raise so any call surfaces immediately.
    called: list[str] = []

    async def must_not_call(*args, **kwargs):
        called.append("get")
        raise AssertionError("Jina should be skipped for private IP URL")

    monkeypatch.setattr(ext._client, "get", must_not_call, raising=True)

    async def _run():
        result = await ext._jina_read("http://10.0.0.1/foo")
        await ext.aclose()
        return result

    out = asyncio.run(_run())
    assert out is None
    assert called == [], "Jina HTTP client must NOT be invoked"


def test_jina_skipped_when_disabled(monkeypatch):
    """JINA_READER_DISABLED=1 disables Jina even for safe URLs."""
    import asyncio
    from agent.extractor import Extractor

    monkeypatch.setenv("JINA_READER_DISABLED", "1")
    ext = Extractor()

    async def must_not_call(*args, **kwargs):
        raise AssertionError("Jina should be disabled by env opt-out")

    monkeypatch.setattr(ext._client, "get", must_not_call, raising=True)

    async def _run():
        result = await ext._jina_read("https://example.com/page")
        await ext.aclose()
        return result

    assert asyncio.run(_run()) is None


def test_tavily_extract_skipped_for_private_ip_url(monkeypatch):
    """Tavily Extract must also re-check the SSRF guard. Tavily's backend
    would otherwise fetch a private-IP target on our behalf."""
    import asyncio
    from agent.extractor import Extractor

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    ext = Extractor()
    called: list[str] = []

    async def must_not_call(*args, **kwargs):
        called.append("post")
        raise AssertionError("Tavily should be skipped for private IP URL")

    monkeypatch.setattr(ext._client, "post", must_not_call, raising=True)

    async def _run():
        result = await ext._tavily_extract("http://192.168.1.1/admin")
        await ext.aclose()
        return result

    assert asyncio.run(_run()) is None
    assert called == []
