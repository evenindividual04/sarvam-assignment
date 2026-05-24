"""Centralized URL/domain normalization.

Different parts of the code learned to normalize URLs independently:
- citation_guard had its own `_normalize_for_dedupe`
- source_role compared LLM-echoed URLs by exact string (silent failure when
  the LLM added a trailing slash)
- source_trust matched domains with an `endswith(s)` rule that over-matched
  when `s` lacked a leading dot (`notgov.in` matched `gov.in`)
- search._domain used `.replace("www.", "")` which substitutes anywhere in
  the netloc (`apiwww.example.com` → `apiexample.com`)

This module centralizes all of those so future bugs stay in one place.

Two public functions:
- ``normalize_url(url)`` — for URL equality / cache keys / LLM-echo match
- ``normalize_domain(domain_or_netloc)`` — for trust-tier matching
"""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

# Standard ports we strip from the host so :80/:443 don't break matching.
_DEFAULT_PORTS: dict[str, str] = {"http": "80", "https": "443"}


def normalize_url(url: str) -> str:
    """Return a canonical lowercase URL safe for equality, cache keys, and
    LLM-echo matching.

    Rules (in order):
      1. strip leading/trailing whitespace
      2. drop the fragment (#...)
      3. drop default ports (:80 for http, :443 for https)
      4. strip a leading `www.` from the host (whole token, not substring)
      5. strip a single trailing slash from the path when the path is non-empty
      6. lowercase scheme + host (path/query case is preserved — case-sensitive
         on most servers)
      7. drop a trailing dot from the host (`example.com.` → `example.com`)

    On any parsing failure, returns the input lower-cased and stripped so the
    function never raises.
    """
    if not url:
        return ""
    raw = url.strip()
    try:
        parts = urlsplit(raw)
        scheme = parts.scheme.lower()
        host = parts.hostname or ""
        host = host.rstrip(".").lower()
        if host.startswith("www."):
            host = host[4:]
        port = parts.port
        if port is not None and str(port) == _DEFAULT_PORTS.get(scheme):
            port = None
        netloc = host
        if port is not None:
            netloc = f"{host}:{port}"
        # userinfo (user:pass@) is dropped intentionally — it's rare in
        # research URLs and almost always a leak indicator if present.
        path = parts.path
        if path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")
        query = parts.query
        # Fragment dropped.
        return urlunsplit((scheme, netloc, path, query, ""))
    except (ValueError, AttributeError):
        return raw.lower()


def normalize_domain(domain_or_netloc: str) -> str:
    """Return a canonical lowercase domain string for trust-tier matching.

    Accepts either a bare domain (``example.com``) or a URL netloc
    (``www.example.com:8080``). Strips ``www.`` only when it is the leading
    label, strips trailing dots, drops port. Never raises.
    """
    if not domain_or_netloc:
        return ""
    d = domain_or_netloc.strip().lower()
    # Drop scheme if a full URL slipped in
    if "://" in d:
        d = d.split("://", 1)[1]
    # Drop path / query / fragment
    for sep in ("/", "?", "#"):
        if sep in d:
            d = d.split(sep, 1)[0]
    # Drop port
    if ":" in d:
        d = d.split(":", 1)[0]
    # Drop trailing and leading dots
    d = d.strip(".")
    # Strip leading www. token (not substring)
    if d.startswith("www."):
        d = d[4:]
    return d
