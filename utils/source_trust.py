"""
Deterministic source trust prior.

Tiered, bounded, additive contribution to chunk scoring.
Python module (not JSON) so the tiers are import-time validated, mypy-aware,
and diff-reviewable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustTier:
    name: str
    score: float
    domains: tuple[str, ...]  # suffix or exact match


TRUST_TIERS: tuple[TrustTier, ...] = (
    TrustTier(
        "tier_1_primary",
        1.00,
        (".gov", ".edu", ".ac.uk", "nih.gov", "europa.eu", "who.int",
         "imf.org", "worldbank.org",
         # India: official / government / regulator. `gov.in` covers
         # all `*.gov.in` subdomains via trust_for's suffix logic.
         "rbi.org.in", "sebi.gov.in", "gov.in", "pib.gov.in", "mygov.in",
         "niti.gov.in", "mea.gov.in", "india.gov.in", "meity.gov.in",
         "dpiit.gov.in"),
    ),
    TrustTier(
        "tier_2_reference",
        0.90,
        ("nature.com", "science.org", "arxiv.org", "ncbi.nlm.nih.gov",
         "wikipedia.org", "ieee.org", "acm.org",
         # India: research / think tanks / Sarvam easter egg.
         "sarvam.ai", "research.sarvam.ai", "iitb.ac.in", "iitm.ac.in",
         "iitd.ac.in", "iitkgp.ac.in", "iitk.ac.in", "iisc.ac.in",
         "idfcinstitute.org", "prsindia.org", "orfonline.org"),
    ),
    TrustTier(
        "tier_3_journalism",
        0.80,
        ("reuters.com", "apnews.com", "bloomberg.com", "ft.com", "wsj.com",
         "economist.com", "bbc.com", "bbc.co.uk", "nytimes.com",
         # India: high-quality Indian press.
         "indianexpress.com", "thehindu.com", "livemint.com",
         "business-standard.com", "hindustantimes.com", "scroll.in",
         "theprint.in", "moneycontrol.com",
         "economictimes.indiatimes.com", "timesofindia.indiatimes.com"),
    ),
    TrustTier(
        "tier_4_mid",
        0.65,
        ("techcrunch.com", "theverge.com", "wired.com", "forbes.com",
         "businessinsider.com"),
    ),
    TrustTier(
        "tier_5_low",
        0.45,
        ("medium.com", "substack.com", "quora.com", "blogspot.com",
         "wordpress.com"),
    ),
)

DEFAULT_TRUST: float = 0.70
DEFAULT_TIER: str = "unknown"

# Phase 5b: default social-media domain blocklist.
# These domains tend to produce low-signal, conversation-style results that
# dilute deep-research synthesis. Listed lowercase; subdomain matching is
# inherited via `is_blocked` (e.g. `old.reddit.com` matches `reddit.com`).
#
# CHANGELOG NOTE: introducing this is a behavior change — queries that
# previously surfaced reddit/twitter/x/tiktok/quora/instagram/facebook/pinterest
# results will now silently drop them at search-result and chunk-selection
# stages. Users can extend or fully disable via RETRIEVAL_DOMAIN_BLOCKLIST
# (env or per-request override): comma-separated extra domains, or the
# literal "none" to disable the default blocklist entirely.
BLOCKLIST: frozenset[str] = frozenset({
    "reddit.com", "twitter.com", "x.com", "tiktok.com",
    "quora.com", "instagram.com", "facebook.com", "pinterest.com",
})


def is_blocked(
    domain: str,
    custom_blocklist: frozenset[str] = frozenset(),
) -> bool:
    """Return True if `domain` (or any parent) is in BLOCKLIST ∪ custom_blocklist.

    Matching mirrors `trust_for` — exact match OR suffix match on a dotted
    boundary, so `reddit.com` blocks `old.reddit.com` but not `notreddit.com`.

    Note: the orchestrator passes `RuntimeConfig.domain_blocklist` (which
    already represents the effective set — BLOCKLIST ∪ env extras, or empty
    when the user passed `RETRIEVAL_DOMAIN_BLOCKLIST=none`). To honor the
    "none" disable path, callers should skip this function entirely when
    `RuntimeConfig.domain_blocklist` is empty.
    """
    if not domain:
        return False
    effective = BLOCKLIST | custom_blocklist
    if not effective:
        return False
    from utils.url_norm import normalize_domain
    d = normalize_domain(domain)
    if not d:
        return False
    for target in effective:
        t = target.lower().lstrip(".")
        if not t:
            continue
        if d == t or d.endswith("." + t):
            return True
    return False


def trust_for(domain: str) -> tuple[float, str]:
    """Return (score, tier_name) for a domain. Unknown → (0.70, 'unknown').

    Matching is exact OR suffix-on-dot-boundary so `pib.gov.in` matches
    `gov.in` but `notgov.in` does not. Previously the third condition
    `d.endswith(s)` over-matched whenever `s` lacked a leading dot
    (`notgov.in`.endswith(`gov.in`) is True), silently promoting
    bogus-but-suffix-similar domains to tier_1_primary.
    """
    from utils.url_norm import normalize_domain
    d = normalize_domain(domain)
    if not d:
        return DEFAULT_TRUST, DEFAULT_TIER
    for tier in TRUST_TIERS:
        for s in tier.domains:
            target = s.lstrip(".")
            if not target:
                continue
            if d == target or d.endswith("." + target):
                return tier.score, tier.name
    return DEFAULT_TRUST, DEFAULT_TIER


def trust_weight_enabled() -> bool:
    """Env-var ablation hook for eval comparisons."""
    return os.environ.get("SOURCE_TRUST_DISABLED", "0") != "1"
