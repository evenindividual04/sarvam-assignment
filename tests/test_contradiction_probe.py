"""V2.2 — Cross-Source Contradiction Probe tests."""
from __future__ import annotations

import asyncio
import json

import aiosqlite
import pytest

from agent import context_engine
from agent import memory as memory_mod
from agent.context_engine import probe_contradictions
from agent.memory import init_db, save_contradiction_probe
from agent.models import (
    ClaimContradiction,
    ConflictResult,
    ContextSnippet,
)
from utils import provider_router
from utils.provider_router import _build_disagreement_block


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


# ── 1. typed contradictions returned on conflicting rates ────────────────────

def test_probe_returns_typed_contradictions_on_conflicting_rates(monkeypatch):
    payload = json.dumps({
        "has_conflict": True,
        "conflict_summary": "Two sources state different repo rates for May 2026.",
        "contradictions": [
            {
                "claim": "RBI repo rate in May 2026",
                "doc_ids_a": ["doc_1"],
                "position_a": "6.50%",
                "doc_ids_b": ["doc_2"],
                "position_b": "5.50%",
                "is_temporal_evolution": False,
                "confidence": 0.92,
            }
        ],
    })

    async def fake_call_groq(prompt: str, max_tokens: int = 200) -> str:
        return payload

    monkeypatch.setattr(provider_router, "call_groq", fake_call_groq)

    chunks = [
        _mk_snippet("doc_1", "RBI set the repo rate at 6.50% in May 2026.", "rbi.org.in"),
        _mk_snippet("doc_2", "RBI repo rate in May 2026 is reported as 5.50%.", "example.com"),
    ]
    result = asyncio.run(probe_contradictions(chunks, "What is the RBI repo rate in May 2026?"))

    assert result.has_conflict is True
    assert len(result.contradictions) == 1
    assert result.contradictions[0].is_temporal_evolution is False
    assert result.probe_skipped_reason is None


# ── 2. temporal evolution correctly marked + no disagreement block ───────────

def test_probe_marks_temporal_evolution_not_contradiction(monkeypatch):
    payload = json.dumps({
        "has_conflict": True,
        "conflict_summary": None,
        "contradictions": [
            {
                "claim": "RBI repo rate",
                "doc_ids_a": ["doc_1"],
                "position_a": "6.50% in 2024",
                "doc_ids_b": ["doc_2"],
                "position_b": "5.50% in 2026",
                "is_temporal_evolution": True,
                "confidence": 0.95,
            }
        ],
    })

    async def fake_call_groq(prompt: str, max_tokens: int = 200) -> str:
        return payload

    monkeypatch.setattr(provider_router, "call_groq", fake_call_groq)

    chunks = [
        _mk_snippet("doc_1", "RBI repo rate was 6.50% in 2024."),
        _mk_snippet("doc_2", "RBI repo rate is 5.50% in 2026."),
    ]
    result = asyncio.run(probe_contradictions(chunks, "RBI repo rate trajectory"))
    assert result.contradictions[0].is_temporal_evolution is True

    # Synthesizer must NOT receive a disagreement block when all are temporal
    block = _build_disagreement_block(result)
    assert block == ""


# ── 3. timeout yields skipped reason ────────────────────────────────────────

def test_probe_timeout_yields_skipped_reason(monkeypatch):
    from utils.failure_policy import POLICY
    monkeypatch.setattr(POLICY, "probe_timeout_s", 0.05)

    async def slow_call_groq(prompt: str, max_tokens: int = 200) -> str:
        await asyncio.sleep(1.0)
        return "{}"

    monkeypatch.setattr(provider_router, "call_groq", slow_call_groq)

    chunks = [
        _mk_snippet("doc_1", "Some claim."),
        _mk_snippet("doc_2", "Another claim."),
    ]
    result = asyncio.run(probe_contradictions(chunks, "q"))
    assert result.has_conflict is False
    assert result.probe_skipped_reason == "timeout"


# ── 4. fewer than 2 chunks ──────────────────────────────────────────────────

def test_probe_skipped_when_fewer_than_2_chunks(monkeypatch):
    async def should_not_be_called(prompt: str, max_tokens: int = 200) -> str:
        raise AssertionError("LLM should not be invoked with <2 chunks")

    monkeypatch.setattr(provider_router, "call_groq", should_not_be_called)

    result = asyncio.run(probe_contradictions([_mk_snippet("doc_1", "only one")], "q"))
    assert result.has_conflict is False
    assert result.probe_skipped_reason == "lt_2_chunks"


# ── 5. parse failure ─────────────────────────────────────────────────────────

def test_probe_parse_failure_returns_skipped_reason(monkeypatch):
    async def malformed(prompt: str, max_tokens: int = 200) -> str:
        return "this is not json at all"

    monkeypatch.setattr(provider_router, "call_groq", malformed)

    chunks = [_mk_snippet("doc_1", "a"), _mk_snippet("doc_2", "b")]
    result = asyncio.run(probe_contradictions(chunks, "q"))
    assert result.has_conflict is False
    assert result.probe_skipped_reason == "parse_fail"


# ── 6. synthesizer receives disagreement block only when non-temporal ────────

def test_synthesizer_receives_disagreement_block_only_when_non_temporal():
    real_conflict = ConflictResult(
        has_conflict=True,
        contradictions=[
            ClaimContradiction(
                claim="repo rate",
                doc_ids_a=["doc_1"],
                position_a="6.50%",
                doc_ids_b=["doc_2"],
                position_b="5.50%",
                is_temporal_evolution=False,
                confidence=0.9,
            )
        ],
    )
    block = _build_disagreement_block(real_conflict)
    assert "<cross_source_disagreement>" in block
    assert "Sources disagree" in block
    assert "doc_1" in block and "doc_2" in block

    temporal_only = ConflictResult(
        has_conflict=True,
        contradictions=[
            ClaimContradiction(
                claim="repo rate",
                doc_ids_a=["doc_1"],
                position_a="6.50% in 2024",
                doc_ids_b=["doc_2"],
                position_b="5.50% in 2026",
                is_temporal_evolution=True,
                confidence=0.95,
            )
        ],
    )
    assert _build_disagreement_block(temporal_only) == ""

    no_conflict = ConflictResult(has_conflict=False)
    assert _build_disagreement_block(no_conflict) == ""


# ── Tier C: Cerebras-first routing ──────────────────────────────────────────


def test_conflict_detection_uses_cerebras_when_available(monkeypatch):
    """CEREBRAS_API_KEY set + auto → Cerebras tried first, Groq not called."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    monkeypatch.setenv("CONFLICT_PROBE_PROVIDER", "auto")

    payload = json.dumps({"has_conflict": False, "contradictions": []})

    async def fake_cerebras(prompt: str, max_tokens: int = 200) -> str:
        return payload

    async def must_not_call_groq(prompt: str, max_tokens: int = 200) -> str:
        raise AssertionError("Groq should not be invoked when Cerebras works")

    monkeypatch.setattr(provider_router, "call_cerebras", fake_cerebras, raising=True)
    monkeypatch.setattr(provider_router, "call_groq", must_not_call_groq, raising=True)

    chunks = [
        _mk_snippet("doc_1", "claim a"),
        _mk_snippet("doc_2", "claim b"),
    ]
    from agent.context_engine import last_conflict_probe_provider

    async def _run():
        r = await probe_contradictions(chunks, "q")
        return r, last_conflict_probe_provider()

    result, prov = asyncio.run(_run())
    assert result.has_conflict is False
    assert prov == "cerebras"


def test_conflict_detection_falls_back_to_groq_on_cerebras_error(monkeypatch):
    """Cerebras raises → Groq is invoked + provider stamped as ``groq``."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    monkeypatch.setenv("CONFLICT_PROBE_PROVIDER", "auto")

    async def broken_cerebras(prompt: str, max_tokens: int = 200) -> str:
        raise RuntimeError("cerebras down")

    payload = json.dumps({"has_conflict": False, "contradictions": []})

    async def fake_groq(prompt: str, max_tokens: int = 200) -> str:
        return payload

    monkeypatch.setattr(provider_router, "call_cerebras", broken_cerebras, raising=True)
    monkeypatch.setattr(provider_router, "call_groq", fake_groq, raising=True)

    chunks = [
        _mk_snippet("doc_1", "a"),
        _mk_snippet("doc_2", "b"),
    ]
    from agent.context_engine import last_conflict_probe_provider

    async def _run():
        r = await probe_contradictions(chunks, "q")
        return r, last_conflict_probe_provider()

    result, prov = asyncio.run(_run())
    assert result.probe_skipped_reason is None
    assert prov == "groq"


def test_probe_provider_is_per_task_under_concurrent_gather(monkeypatch):
    """Two concurrent probes (one picks Cerebras, one picks Groq) must each
    see their OWN provider via the ContextVar. The old module-level global
    would race: whichever setter ran last would win for *both* readers."""
    from agent.context_engine import last_conflict_probe_provider

    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")

    no_conflict = json.dumps({"has_conflict": False, "contradictions": []})

    async def cerebras_slow(prompt: str, max_tokens: int = 200) -> str:
        await asyncio.sleep(0.05)  # let the other task's groq attempt race
        return no_conflict

    async def cerebras_raise(prompt: str, max_tokens: int = 200) -> str:
        # Used by the groq-bound task. Force fallback into Groq.
        raise RuntimeError("cerebras unavailable for this task")

    async def groq_fast(prompt: str, max_tokens: int = 200) -> str:
        return no_conflict

    chunks = [_mk_snippet("doc_1", "a"), _mk_snippet("doc_2", "b")]

    # We monkeypatch globally and switch behavior per-task using a flag in
    # the prompt text (q-cere vs q-groq).
    async def call_cerebras_dispatch(prompt: str, max_tokens: int = 200) -> str:
        if "q-groq" in prompt:
            return await cerebras_raise(prompt, max_tokens)
        return await cerebras_slow(prompt, max_tokens)

    monkeypatch.setattr(provider_router, "call_cerebras", call_cerebras_dispatch, raising=True)
    monkeypatch.setattr(provider_router, "call_groq", groq_fast, raising=True)

    async def _task(tag: str) -> str:
        await probe_contradictions(chunks, tag)
        return last_conflict_probe_provider()

    async def _drive():
        return await asyncio.gather(_task("q-cere"), _task("q-groq"))

    cere_prov, groq_prov = asyncio.run(_drive())
    assert cere_prov == "cerebras", f"expected cerebras, got {cere_prov}"
    assert groq_prov == "groq", f"expected groq, got {groq_prov}"


# ── 7. one row per turn persisted ───────────────────────────────────────────

def test_probe_persists_row_per_turn(monkeypatch, tmp_path):
    db_path = str(tmp_path / "probe_test.db")
    monkeypatch.setattr(memory_mod, "DB_PATH", db_path)

    async def _setup_and_save():
        await init_db()
        # Insert a fake session + turn so the FK reference is satisfiable.
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                ("s1", "2026-05-21T00:00:00+00:00", "2026-05-21T00:00:00+00:00"),
            )
            await db.execute(
                """INSERT INTO turns (turn_id, session_id, query, search_queries, urls_opened, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t1", "s1", "q", "[]", "[]", "2026-05-21T00:00:00+00:00"),
            )
            await db.commit()

        result = ConflictResult(
            has_conflict=True,
            conflict_summary="summary",
            contradictions=[
                ClaimContradiction(
                    claim="c", doc_ids_a=["doc_1"], position_a="p1",
                    doc_ids_b=["doc_2"], position_b="p2",
                    is_temporal_evolution=False, confidence=0.7,
                )
            ],
        )
        await save_contradiction_probe("t1", result, probe_ms=42, prompt_id="conflict_v3")

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT * FROM contradiction_probes WHERE turn_id = ?", ("t1",)
            )
        return rows

    rows = asyncio.run(_setup_and_save())
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["has_conflict"] == 1
    assert row["conflict_summary"] == "summary"
    assert row["probe_ms"] == 42
    assert row["prompt_id"] == "conflict_v3"
    payload = json.loads(row["contradictions_json"])
    assert len(payload) == 1
    assert payload[0]["claim"] == "c"
