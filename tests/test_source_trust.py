"""
V2.3 — deterministic source trust prior tests.
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import aiosqlite
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import memory
from agent.context_engine import score_chunk
from agent.models import ContextSnippet
from utils.source_trust import (
    DEFAULT_TIER,
    DEFAULT_TRUST,
    trust_for,
    trust_weight_enabled,
)


def _snippet(domain: str, trust_score: float = 0.7, trust_tier: str = "unknown",
             text: str = "sample", token_count: int = 50,
             intent_origin: str | None = None) -> ContextSnippet:
    return ContextSnippet(
        doc_id="doc_1",
        url=f"https://{domain}/x",
        title="T",
        domain=domain,
        text=text,
        snippet=text[:50],
        token_count=token_count,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        trust_score=trust_score,
        trust_tier=trust_tier,
        intent_origin=intent_origin,
    )


# ── trust_for ──────────────────────────────────────────────────────────────

def test_trust_for_known_gov_domain_returns_tier_1():
    score, tier = trust_for("nasa.gov")
    assert score == 1.00
    assert tier == "tier_1_primary"


def test_trust_for_exact_match_priority():
    score, tier = trust_for("rbi.org.in")
    assert tier == "tier_1_primary"
    assert score == 1.00


def test_trust_for_unknown_domain_returns_default_neutral():
    score, tier = trust_for("randomblog.xyz")
    assert score == DEFAULT_TRUST == 0.70
    assert tier == DEFAULT_TIER == "unknown"


def test_trust_for_medium_returns_tier_5_low():
    score, tier = trust_for("medium.com")
    assert score == 0.45
    assert tier == "tier_5_low"


def test_trust_for_empty_domain_returns_default():
    score, tier = trust_for("")
    assert score == DEFAULT_TRUST
    assert tier == DEFAULT_TIER


def test_trust_for_subdomain_inherits_tier():
    score, tier = trust_for("blog.medium.com")
    assert tier == "tier_5_low"
    assert score == 0.45


# ── score_chunk additive math ──────────────────────────────────────────────

def test_score_chunk_additive_trust_does_not_dominate_zero_relevance():
    """Max trust on zero-relevance/zero-recency/zero-diversity chunk → final ≤ 0.20."""
    snip = _snippet("nasa.gov", trust_score=1.0, trust_tier="tier_1_primary")
    # Force zero recency by faking an extremely old retrieved_at
    snip.retrieved_at = "1970-01-01T00:00:00+00:00"
    # diversity = 1/(1+99) ≈ 0.01
    domain_counts: dict[str, int] = defaultdict(int)
    domain_counts["nasa.gov"] = 99
    score = score_chunk(snip, bm25_score=0.0, all_bm25_scores=[1.0, 1.0],
                        url_domain_count=domain_counts)
    # 0.50*0 + 0.15*~0 + 0.15*0.01 + 0.20*1.0 ≈ 0.2015 — bounded near 0.20
    assert score <= 0.21
    assert score >= 0.19


def test_score_chunk_weights_sum_to_1():
    """All factors == 1.0 → final == 1.0."""
    snip = _snippet("nasa.gov", trust_score=1.0, trust_tier="tier_1_primary")
    domain_counts: dict[str, int] = defaultdict(int)  # diversity = 1/(1+0) = 1
    score = score_chunk(snip, bm25_score=1.0, all_bm25_scores=[1.0],
                        url_domain_count=domain_counts)
    # recency on a just-now timestamp ≈ 1.0
    assert math.isclose(score, 1.0, abs_tol=1e-3)


# ── persistence ────────────────────────────────────────────────────────────

def test_turn_context_persists_trust_score_per_row(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", str(db_file))

    async def _run() -> float | None:
        await memory.init_db()
        await memory.create_session("s1", datetime.now(timezone.utc).isoformat())
        turn_id = str(uuid.uuid4())
        # Need a turn row for FK (REFERENCES turns(turn_id))
        from agent.models import Turn
        turn = Turn(
            turn_id=turn_id, session_id="s1", query="q",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await memory.save_turn(turn)
        snip = _snippet("nature.com", trust_score=0.90, trust_tier="tier_2_reference")
        snip.doc_id = "doc_1"
        await memory.save_turn_context(turn_id, [snip])

        async with aiosqlite.connect(str(db_file)) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT trust_score FROM turn_context WHERE turn_id = ?", (turn_id,)
            )
        return rows[0]["trust_score"] if rows else None

    persisted = asyncio.run(_run())
    assert persisted is not None
    assert math.isclose(persisted, 0.90, abs_tol=1e-6)


# ── ablation flag ──────────────────────────────────────────────────────────

def test_source_trust_disabled_env_flag_zeroes_weight(monkeypatch):
    monkeypatch.setenv("SOURCE_TRUST_DISABLED", "1")
    assert trust_weight_enabled() is False

    # In ablation mode: weights are 0.6/0.2/0.2; trust must not influence the result.
    snip_high = _snippet("nasa.gov", trust_score=1.0)
    snip_low = _snippet("nasa.gov", trust_score=0.45)
    snip_high.retrieved_at = snip_low.retrieved_at  # same recency
    counts: dict[str, int] = defaultdict(int)
    s_high = score_chunk(snip_high, 0.5, [1.0], counts)
    s_low = score_chunk(snip_low, 0.5, [1.0], counts.copy())
    assert math.isclose(s_high, s_low, abs_tol=1e-9)

    # And sanity check: enabled mode shows divergence
    monkeypatch.setenv("SOURCE_TRUST_DISABLED", "0")
    s_high2 = score_chunk(snip_high, 0.5, [1.0], defaultdict(int))
    s_low2 = score_chunk(snip_low, 0.5, [1.0], defaultdict(int))
    assert s_high2 > s_low2


# ── V2.1 × V2.3 composition ────────────────────────────────────────────────

def test_contradiction_probe_diversity_boost_still_applies_with_trust():
    """V2.1 diversity bump (×1.25) composes with V2.3 additive trust contribution."""
    # Two identical chunks; one tagged contradiction_probe, one not.
    snip_probe = _snippet("nature.com", trust_score=0.90,
                          intent_origin="contradiction_probe")
    snip_plain = _snippet("nature.com", trust_score=0.90)
    counts: dict[str, int] = defaultdict(int)
    counts["nature.com"] = 1  # so diversity = 0.5 → probe boost to 0.625

    s_probe = score_chunk(snip_probe, 0.5, [1.0], counts,
                          intent_origin="contradiction_probe")
    s_plain = score_chunk(snip_plain, 0.5, [1.0], counts)
    # Probe chunk should win on diversity component
    assert s_probe > s_plain
    # Both must include the trust contribution (0.20 * 0.90 = 0.18 baseline)
    # Lower bound: 0.50*0.5 + 0.20*0.90 = 0.43; both should clear it.
    assert s_probe > 0.43
    assert s_plain > 0.43
