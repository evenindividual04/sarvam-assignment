"""
V2.3 — Deterministic source trust prior.

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
         "imf.org", "worldbank.org", "rbi.org.in"),
    ),
    TrustTier(
        "tier_2_reference",
        0.90,
        ("nature.com", "science.org", "arxiv.org", "ncbi.nlm.nih.gov",
         "wikipedia.org", "ieee.org", "acm.org"),
    ),
    TrustTier(
        "tier_3_journalism",
        0.80,
        ("reuters.com", "apnews.com", "bloomberg.com", "ft.com", "wsj.com",
         "economist.com", "bbc.com", "bbc.co.uk", "nytimes.com"),
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


def trust_for(domain: str) -> tuple[float, str]:
    """Return (score, tier_name) for a domain. Unknown → (0.70, 'unknown')."""
    if not domain:
        return DEFAULT_TRUST, DEFAULT_TIER
    d = domain.lower().lstrip(".")
    for tier in TRUST_TIERS:
        for s in tier.domains:
            target = s.lstrip(".")
            if d == target or d.endswith("." + target) or d.endswith(s):
                return tier.score, tier.name
    return DEFAULT_TRUST, DEFAULT_TIER


def trust_weight_enabled() -> bool:
    """Env-var ablation hook for eval comparisons."""
    return os.environ.get("SOURCE_TRUST_DISABLED", "0") != "1"
