"""
Phase 5b — default social-media domain blocklist tests.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.context_engine import (
    rank_and_select,
    reset_domain_blocklist,
    set_domain_blocklist,
)
from agent.models import ContextSnippet
from agent.orchestrator import RuntimeConfig
from utils.source_trust import BLOCKLIST, is_blocked


# ── is_blocked ─────────────────────────────────────────────────────────────


def test_is_blocked_exact_match():
    assert is_blocked("reddit.com", BLOCKLIST) is True


def test_is_blocked_subdomain():
    assert is_blocked("old.reddit.com", BLOCKLIST) is True


def test_is_blocked_deep_subdomain():
    assert is_blocked("a.b.c.twitter.com", BLOCKLIST) is True


def test_is_blocked_unrelated_domain():
    assert is_blocked("nytimes.com", BLOCKLIST) is False


def test_is_blocked_substring_not_suffix():
    # `notreddit.com` ends with "reddit.com" by suffix but not on a dotted
    # boundary — must NOT be blocked.
    assert is_blocked("notreddit.com", BLOCKLIST) is False


def test_is_blocked_custom_addition():
    custom = frozenset({"badsite.example"})
    assert is_blocked("badsite.example", custom) is True
    assert is_blocked("sub.badsite.example", custom) is True


def test_is_blocked_empty_domain():
    assert is_blocked("", BLOCKLIST) is False


def test_is_blocked_default_blocklist_applied_when_no_custom():
    # Even passing the default empty frozenset, default BLOCKLIST still matches.
    assert is_blocked("reddit.com", frozenset()) is True


# ── RuntimeConfig.from_overrides ───────────────────────────────────────────


def test_runtime_config_default_blocklist_is_BLOCKLIST(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_DOMAIN_BLOCKLIST", raising=False)
    cfg = RuntimeConfig.from_overrides(None)
    assert cfg.domain_blocklist == BLOCKLIST


def test_runtime_config_parses_blocklist_override(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_DOMAIN_BLOCKLIST", raising=False)
    cfg = RuntimeConfig.from_overrides({
        "RETRIEVAL_DOMAIN_BLOCKLIST": "foo.com,bar.com"
    })
    # Extends — doesn't replace
    assert "foo.com" in cfg.domain_blocklist
    assert "bar.com" in cfg.domain_blocklist
    assert "reddit.com" in cfg.domain_blocklist  # default still present
    assert cfg.domain_blocklist == BLOCKLIST | frozenset({"foo.com", "bar.com"})


def test_runtime_config_blocklist_none_disables(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_DOMAIN_BLOCKLIST", raising=False)
    cfg = RuntimeConfig.from_overrides({"RETRIEVAL_DOMAIN_BLOCKLIST": "none"})
    assert cfg.domain_blocklist == frozenset()


def test_runtime_config_blocklist_env_override(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DOMAIN_BLOCKLIST", "envonly.com")
    cfg = RuntimeConfig.from_overrides(None)
    assert "envonly.com" in cfg.domain_blocklist
    assert "reddit.com" in cfg.domain_blocklist


def test_runtime_config_as_dict_blocklist_is_sorted_list(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_DOMAIN_BLOCKLIST", raising=False)
    cfg = RuntimeConfig.from_overrides(None)
    d = cfg.as_dict()
    assert isinstance(d["domain_blocklist"], list)
    assert d["domain_blocklist"] == sorted(BLOCKLIST)


# ── context_engine integration ─────────────────────────────────────────────


def _snippet(domain: str, text: str = "the quick brown fox jumps over") -> ContextSnippet:
    return ContextSnippet(
        doc_id="doc_1",
        url=f"https://{domain}/path",
        title="t",
        domain=domain,
        text=text,
        snippet=text[:50],
        token_count=10,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        trust_score=0.7,
        trust_tier="unknown",
    )


def test_context_engine_drops_blocklisted_snippets():
    chunks = [
        _snippet("reddit.com", "alpha beta gamma topic discussion"),
        _snippet("old.reddit.com", "another reddit comment thread"),
        _snippet("nytimes.com", "alpha beta gamma topic news article"),
    ]
    drops: list[str] = []
    tokens = set_domain_blocklist(BLOCKLIST, drops)
    try:
        selected = rank_and_select("alpha beta gamma topic", chunks, max_tokens=5000)
    finally:
        reset_domain_blocklist(tokens)

    assert any(s.domain == "nytimes.com" for s in selected)
    assert not any(s.domain.endswith("reddit.com") for s in selected)
    # Drops were recorded (both reddit chunks)
    assert "reddit.com" in drops
    assert "old.reddit.com" in drops


def test_context_engine_no_drop_when_blocklist_empty():
    chunks = [_snippet("reddit.com", "alpha beta gamma topic")]
    drops: list[str] = []
    tokens = set_domain_blocklist(frozenset(), drops)
    try:
        selected = rank_and_select("alpha beta gamma topic", chunks, max_tokens=5000)
    finally:
        reset_domain_blocklist(tokens)
    # Disabled → reddit chunk survives, no drops recorded.
    assert any(s.domain == "reddit.com" for s in selected)
    assert drops == []
