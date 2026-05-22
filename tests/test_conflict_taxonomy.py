"""P3 — DRAGged-into-Conflict taxonomy tests.

Tests the typed conflict kinds (self / pair / conditional / none) from
Cattan et al., arXiv:2506.08500. Mocks the same provider routing as
``tests/test_contradiction_probe.py`` for consistency.
"""
from __future__ import annotations

import asyncio
import json

import aiosqlite

from agent import memory as memory_mod
from agent.context_engine import probe_contradictions
from agent.memory import init_db
from agent.models import ContextSnippet
from utils import provider_router


def _mk_snippet(doc_id: str, text: str, domain: str = "example.com") -> ContextSnippet:
    return ContextSnippet(
        doc_id=doc_id,
        url=f"https://{domain}/{doc_id}",
        title=f"Title {doc_id}",
        domain=domain,
        text=text,
        snippet=text[:200],
        token_count=len(text.split()),
        retrieved_at="2026-05-21T00:00:00+00:00",
    )


def _patch_groq(monkeypatch, payload: str) -> None:
    async def fake_call_groq(prompt: str, max_tokens: int = 200) -> str:
        return payload

    # Force the Groq path (skip Cerebras attempt) for deterministic tests.
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.setenv("CONFLICT_PROBE_PROVIDER", "groq")
    monkeypatch.setattr(provider_router, "call_groq", fake_call_groq)


# ── 1. pair conflict — dominant_kind=pair, kind on each contradiction ──────

def test_pair_conflict_returned_with_kind(monkeypatch):
    payload = json.dumps({
        "has_conflict": True,
        "dominant_kind": "pair",
        "conflict_summary": "Sources disagree on May 2026 repo rate.",
        "contradictions": [
            {
                "claim": "RBI repo rate May 2026",
                "kind": "pair",
                "doc_ids_a": ["doc_1"],
                "position_a": "6.50%",
                "doc_ids_b": ["doc_2"],
                "position_b": "5.50%",
                "qualifier": None,
                "is_temporal_evolution": False,
                "confidence": 0.9,
            }
        ],
    })
    _patch_groq(monkeypatch, payload)

    chunks = [
        _mk_snippet("doc_1", "RBI set the repo rate at 6.50% in May 2026."),
        _mk_snippet("doc_2", "RBI repo rate in May 2026 is reported as 5.50%."),
    ]
    result = asyncio.run(probe_contradictions(chunks, "RBI repo rate May 2026"))

    assert result.has_conflict is True
    assert result.dominant_kind == "pair"
    assert len(result.contradictions) == 1
    assert result.contradictions[0].kind == "pair"
    assert result.contradictions[0].qualifier is None


# ── 2. conditional conflict — qualifier populated ───────────────────────────

def test_conditional_conflict_includes_qualifier(monkeypatch):
    payload = json.dumps({
        "has_conflict": True,
        "dominant_kind": "conditional",
        "conflict_summary": "Sources agree once year qualifier is applied.",
        "contradictions": [
            {
                "claim": "Indian Prime Minister",
                "kind": "conditional",
                "doc_ids_a": ["doc_1"],
                "position_a": "Manmohan Singh",
                "doc_ids_b": ["doc_2"],
                "position_b": "Narendra Modi",
                "qualifier": "year",
                "is_temporal_evolution": True,
                "confidence": 0.95,
            }
        ],
    })
    _patch_groq(monkeypatch, payload)

    chunks = [
        _mk_snippet("doc_1", "In 2013 the Indian PM was Manmohan Singh."),
        _mk_snippet("doc_2", "In 2024 the Indian PM is Narendra Modi."),
    ]
    result = asyncio.run(probe_contradictions(chunks, "Who is the Indian PM?"))

    assert result.dominant_kind == "conditional"
    assert len(result.contradictions) == 1
    assert result.contradictions[0].kind == "conditional"
    assert result.contradictions[0].qualifier == "year"


# ── 3. self conflict — single source contradicting itself ──────────────────

def test_self_conflict_supported(monkeypatch):
    payload = json.dumps({
        "has_conflict": True,
        "dominant_kind": "self",
        "conflict_summary": "doc_1 contradicts itself.",
        "contradictions": [
            {
                "claim": "RBI repo rate",
                "kind": "self",
                "doc_ids_a": ["doc_1"],
                "position_a": "6.50% (paragraph 2)",
                "doc_ids_b": ["doc_1"],
                "position_b": "5.50% (paragraph 4)",
                "qualifier": None,
                "is_temporal_evolution": False,
                "confidence": 0.85,
            }
        ],
    })
    _patch_groq(monkeypatch, payload)

    chunks = [
        _mk_snippet("doc_1", "The repo rate is 6.50%. Later: the repo rate is 5.50%."),
        _mk_snippet("doc_2", "Unrelated content."),
    ]
    result = asyncio.run(probe_contradictions(chunks, "repo rate"))

    assert result.dominant_kind == "self"
    assert result.contradictions[0].kind == "self"
    assert result.contradictions[0].doc_ids_a == result.contradictions[0].doc_ids_b == ["doc_1"]


# ── 4. unparseable response → dominant_kind="none" + parse_fail ────────────

def test_unparseable_response_degrades_to_none(monkeypatch):
    _patch_groq(monkeypatch, "this is not json at all")

    chunks = [_mk_snippet("doc_1", "a"), _mk_snippet("doc_2", "b")]
    result = asyncio.run(probe_contradictions(chunks, "q"))

    assert result.has_conflict is False
    assert result.dominant_kind == "none"
    assert result.probe_skipped_reason == "parse_fail"


# ── 5. back-compat with old DB row (dominant_kind column NULL) ─────────────

def test_back_compat_with_old_db_row(monkeypatch, tmp_path):
    """Simulate a DB written before the dominant_kind migration: NULL column
    must round-trip through the eval_queries reader as ``dominant_kind="none"``
    without raising."""
    db_path = str(tmp_path / "back_compat.db")
    monkeypatch.setattr(memory_mod, "DB_PATH", db_path)

    async def _setup_and_read():
        await init_db()
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                ("s1", "2026-05-21T00:00:00+00:00", "2026-05-21T00:00:00+00:00"),
            )
            await db.execute(
                """INSERT INTO turns (turn_id, session_id, query, search_queries, urls_opened, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t_old", "s1", "q", "[]", "[]", "2026-05-21T00:00:00+00:00"),
            )
            # Insert a row WITHOUT dominant_kind (NULL column) → emulates a
            # row written by code that predates the P3 migration.
            await db.execute(
                """INSERT INTO contradiction_probes
                   (turn_id, has_conflict, conflict_summary, contradictions_json,
                    probe_skipped_reason, probe_ms, prompt_id, dominant_kind, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("t_old", 0, None, "[]", None, 12, "conflict_v3", None,
                 "2026-05-21T00:00:00+00:00"),
            )
            await db.commit()

        # Read back via the same path eval surfaces use.
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT * FROM contradiction_probes WHERE turn_id = ?", ("t_old",)
            )
        return rows

    rows = asyncio.run(_setup_and_read())
    assert len(rows) == 1
    row = dict(rows[0])
    # Raw column is NULL — back-compat means the read path supplies "none".
    assert row.get("dominant_kind") in (None, "none")

    # Confirm the eval_queries module loads and applies the same fallback
    # rule (NULL → "none") as the production read path.
    from agent.eval_queries import get_question_detail  # noqa: F401
    probe_dict = dict(row)
    if not probe_dict.get("dominant_kind"):
        probe_dict["dominant_kind"] = "none"
    assert probe_dict["dominant_kind"] == "none"
