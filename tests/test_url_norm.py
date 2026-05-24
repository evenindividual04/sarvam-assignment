"""Tests for the shared URL/domain normalizer that fixes the source-
classification "1-character difference" bug.

The bug had three layered causes:
  A1: source_role.py matched LLM-echoed URLs by exact string, so a trailing
      slash from the LLM dropped the classification.
  A2: source_trust.py's third matching condition `d.endswith(s)` over-matched
      when the tier-list entry lacked a leading dot, so `notgov.in` matched
      `gov.in`.
  A3: search._domain used `.replace("www.", "")` which strips "www." from
      anywhere in the netloc, not just the leading label.

These tests pin those down so they can't regress.
"""
from __future__ import annotations

import pytest

from utils.url_norm import normalize_url, normalize_domain
from utils.source_trust import trust_for
from agent.search import _domain


# ── normalize_url ────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    # Trailing slash
    ("https://www.britannica.com/place/Paris",
     "https://www.britannica.com/place/Paris/"),
    # Fragment
    ("https://en.wikipedia.org/wiki/Paris",
     "https://en.wikipedia.org/wiki/Paris#History"),
    # Scheme + host case
    ("HTTPS://EN.WIKIPEDIA.ORG/wiki/Paris",
     "https://en.wikipedia.org/wiki/Paris"),
    # Leading www
    ("https://example.com/x",
     "https://www.example.com/x"),
    # Default port
    ("https://example.com/x",
     "https://example.com:443/x"),
    # Trailing dot in host (canonical DNS)
    ("https://example.com/x",
     "https://example.com./x"),
    # Whitespace
    ("https://example.com/x",
     "  https://example.com/x  "),
])
def test_normalize_url_collapses_equivalents(a, b):
    assert normalize_url(a) == normalize_url(b)


def test_normalize_url_preserves_path_case():
    # Some servers are case-sensitive on path. Don't lowercase that.
    a = normalize_url("https://example.com/CamelCase/Path")
    assert "/CamelCase/Path" in a


def test_normalize_url_empty_safe():
    assert normalize_url("") == ""
    assert normalize_url("   ") == ""


def test_normalize_url_garbage_does_not_raise():
    # Bad input must not crash classification.
    out = normalize_url("not a url at all")
    assert isinstance(out, str)


# ── normalize_domain ────────────────────────────────────────────────────

@pytest.mark.parametrize("inp,expected", [
    ("example.com", "example.com"),
    ("EXAMPLE.COM", "example.com"),
    ("www.example.com", "example.com"),
    ("example.com.", "example.com"),       # trailing dot
    ("example.com:8080", "example.com"),   # port
    ("https://www.example.com/path", "example.com"),
    ("apiwww.example.com", "apiwww.example.com"),  # NOT stripped — fix A3
    ("", ""),
])
def test_normalize_domain(inp, expected):
    assert normalize_domain(inp) == expected


# ── source_trust.trust_for (fix A2) ─────────────────────────────────────

def test_trust_for_exact_gov_in():
    score, tier = trust_for("pib.gov.in")
    assert tier == "tier_1_primary"
    assert score == 1.00


def test_trust_for_subdomain_gov_in():
    # www.pib.gov.in should also match via the dotted-suffix rule
    score, tier = trust_for("www.pib.gov.in")
    assert tier == "tier_1_primary"


def test_trust_for_does_not_overmatch_substring():
    """Regression A2: `notgov.in` must NOT match `gov.in` even though its
    string ends with `gov.in`. The old code used `endswith(s)` directly
    which let any suffix-similar adversarial domain hijack tier_1_primary."""
    score, tier = trust_for("notgov.in")
    assert tier == "unknown"
    assert score == 0.70


def test_trust_for_does_not_overmatch_concatenation():
    """`somegovernment.com` and `wikipedialookalike.org` must not promote."""
    assert trust_for("somegovernment.com")[1] == "unknown"
    assert trust_for("wikipedialookalike.org")[1] == "unknown"


def test_trust_for_handles_dirty_input():
    # Trailing dot / port / www must not break the lookup.
    assert trust_for("WWW.WIKIPEDIA.ORG.")[1] == "tier_2_reference"
    assert trust_for("en.wikipedia.org:443")[1] == "tier_2_reference"


# ── search._domain (fix A3) ─────────────────────────────────────────────

def test_search_domain_strips_only_leading_www():
    """Regression A3: the old `.replace("www.", "")` would substitute
    anywhere — apiwww.example.com → apiexample.com. Now only the leading
    `www.` token is removed."""
    assert _domain("https://www.example.com/x") == "example.com"
    assert _domain("https://apiwww.example.com/x") == "apiwww.example.com"


def test_search_domain_drops_port_and_trailing_dot():
    assert _domain("https://example.com:8080/x") == "example.com"
    assert _domain("https://example.com./x") == "example.com"
