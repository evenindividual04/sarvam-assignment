"""V3.1 — Hybrid RRF retrieval (BM25 + dense fusion via sqlite-vec)."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from agent import memory
from agent.context_engine import (
    chunk,
    hybrid_retrieval_enabled,
    rank_and_select,
    reciprocal_rank_fusion,
)
from agent.models import ContextSnippet, SearchResult


# ── RRF formula ────────────────────────────────────────────────────────────

def test_rrf_formula_basic():
    # Single ranking: doc at rank 1 → 1/(60+1).
    out = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert out[0][0] == "a"
    assert math.isclose(out[0][1], 1 / 61, rel_tol=1e-9)
    assert math.isclose(out[1][1], 1 / 62, rel_tol=1e-9)
    assert math.isclose(out[2][1], 1 / 63, rel_tol=1e-9)


def test_rrf_handles_partial_overlap():
    # Two rankings; "b" appears in both → highest fused score.
    out = reciprocal_rank_fusion([["a", "b", "c"], ["d", "b", "e"]], k=60)
    fused = dict(out)
    assert "b" in fused
    # b score: 1/62 + 1/62 = ~0.0322; a only in one list at rank 1: 1/61.
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["d"]


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[]], k=60) == []


# ── chunk_embeddings table ────────────────────────────────────────────────

def test_chunk_embeddings_table_created_on_init_db(tmp_path, monkeypatch):
    db_file = tmp_path / "vec.db"
    monkeypatch.setattr(memory, "DB_PATH", str(db_file))

    async def _run():
        await memory.init_db()
        # If sqlite-vec is available, table exists. If not, _VEC_AVAILABLE is False;
        # skip the check in that case (the helper is what we're really testing).
        if not memory._VEC_AVAILABLE:
            pytest.skip("sqlite-vec not available on this build")
        import aiosqlite, sqlite_vec
        async with aiosqlite.connect(str(db_file)) as db:
            await db.enable_load_extension(True)
            await db.load_extension(sqlite_vec.loadable_path())
            rows = await db.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE name = 'chunk_embeddings'"
            )
        return rows

    rows = asyncio.run(_run())
    assert rows, "chunk_embeddings virtual table should exist after init_db"


# ── Hybrid path gating ────────────────────────────────────────────────────

def _mk_snippet(doc_id: str, text: str, domain: str = "example.com") -> ContextSnippet:
    return ContextSnippet(
        doc_id=doc_id, url=f"https://{domain}/{doc_id}",
        title=doc_id, domain=domain, text=text, snippet=text[:200],
        token_count=len(text.split()), retrieved_at=datetime.now(timezone.utc).isoformat(),
    )


def test_lexical_mode_skips_embedder(monkeypatch):
    """V3.8: explicit `RETRIEVAL_MODE=lexical` must not call the embedder
    regardless of capability — this is the ablation / forced-baseline path.
    (Note: the old test asserted the *default* was off; under V3.8 the default
    is `auto`, which uses hybrid when sqlite-vec is available. To assert "no
    embedding" you now have to opt out explicitly.)"""
    monkeypatch.delenv("HYBRID_RETRIEVAL", raising=False)
    monkeypatch.setenv("RETRIEVAL_MODE", "lexical")
    assert hybrid_retrieval_enabled() is False
    chunks = [
        _mk_snippet("d1", "the quick brown fox jumps over the lazy dog"),
        _mk_snippet("d2", "machine learning models embed text into dense vectors"),
    ]
    called = {"n": 0}

    def _spy(texts):
        called["n"] += 1
        return [[0.0] * 384 for _ in texts]

    with patch("agent.embedder._embed_sync", side_effect=_spy):
        out = rank_and_select("brown fox", chunks, max_tokens=4000)

    assert called["n"] == 0, "embedder must NOT be called in RETRIEVAL_MODE=lexical"
    assert out, "BM25 path should still return chunks"


def test_hybrid_retrieval_enabled_calls_embed_and_fuses(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRID_RETRIEVAL", "1")
    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "h.db"))
    assert hybrid_retrieval_enabled() is True

    chunks = [
        _mk_snippet("d1", "the quick brown fox jumps over the lazy dog"),
        _mk_snippet("d2", "machine learning models embed text into dense vectors"),
        _mk_snippet("d3", "a brown fox sprints through the meadow at dawn"),
    ]
    called = {"n": 0}

    def _fake_embed(texts):
        called["n"] += 1
        # Deterministic toy embeddings; query similar to d1 and d3.
        out = []
        for t in texts:
            v = [0.0] * 384
            v[0] = 1.0 if "fox" in t else 0.0
            v[1] = 1.0 if "vectors" in t else 0.0
            out.append(v)
        return out

    async def _run():
        await memory.init_db()
        if not memory._VEC_AVAILABLE:
            pytest.skip("sqlite-vec not available")
        with patch("agent.embedder._embed_sync", side_effect=_fake_embed):
            # rank_and_select uses asyncio.run() internally for the hybrid path;
            # must be invoked from a thread with no running event loop.
            return await asyncio.to_thread(
                rank_and_select, "brown fox", chunks, 4000,
            )

    out = asyncio.run(_run())
    assert called["n"] >= 1, "embedder should be invoked when hybrid is enabled"
    assert out, "Hybrid path must still return selected chunks"


def test_hybrid_retrieval_falls_back_on_embed_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRID_RETRIEVAL", "1")
    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "fail.db"))

    chunks = [
        _mk_snippet("d1", "the quick brown fox"),
        _mk_snippet("d2", "lorem ipsum dolor sit amet"),
    ]

    def _boom(_texts):
        raise RuntimeError("embedder exploded")

    async def _run():
        await memory.init_db()
        with patch("agent.embedder._embed_sync", side_effect=_boom):
            return await asyncio.to_thread(
                rank_and_select, "brown fox", chunks, 4000,
            )

    out = asyncio.run(_run())
    # On failure, hybrid path returns None → BM25 path executes → non-empty.
    assert out, "Must gracefully degrade to BM25 when embedding fails"


# ── Real embed smoke ──────────────────────────────────────────────────────

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("fastembed"),
    reason="fastembed not installed",
)
def test_real_embed_returns_384_dim():
    from agent.embedder import EMBEDDING_DIM, embed_batch
    out = asyncio.run(embed_batch(["hello world"]))
    assert out and len(out) == 1
    assert len(out[0]) == EMBEDDING_DIM == 384


# ── P0-5: async entry point ───────────────────────────────────────────────
# Regression coverage for the asyncio.run-inside-to_thread bug. The orchestrator
# now drives hybrid retrieval through `rank_and_select_async`, which keeps the
# async DB I/O in the caller's event loop. The sync `rank_and_select` retains
# back-compat for legacy callers but safely skips hybrid when a loop is active.


@pytest.mark.asyncio
async def test_rank_and_select_async_runs_inside_event_loop(tmp_path, monkeypatch):
    """Async entry point completes cleanly when invoked from a running loop —
    proving the hybrid path no longer trips `RuntimeError: This event loop is
    already running`."""
    monkeypatch.setenv("HYBRID_RETRIEVAL", "1")
    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "async.db"))
    from agent.context_engine import rank_and_select_async

    await memory.init_db()
    chunks = [
        _mk_snippet("d1", "the quick brown fox jumps over the lazy dog"),
        _mk_snippet("d2", "machine learning embeds text into dense vectors"),
        _mk_snippet("d3", "a brown fox sprints through the meadow"),
    ]

    def _fake_embed(texts):
        return [[1.0 if "fox" in t else 0.0] + [0.0] * 383 for t in texts]

    with patch("agent.embedder._embed_sync", side_effect=_fake_embed):
        out = await rank_and_select_async("brown fox", chunks, max_tokens=4000)
    assert out, "Async hybrid path must return selected chunks"


@pytest.mark.asyncio
async def test_rank_and_select_async_does_not_call_asyncio_run(tmp_path, monkeypatch):
    """If `asyncio.run` is invoked anywhere on the hybrid path, it would raise
    inside the test's running loop. Patching it to raise unconditionally proves
    the async path never reaches it."""
    monkeypatch.setenv("HYBRID_RETRIEVAL", "1")
    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "noasyncrun.db"))
    from agent import context_engine
    from agent.context_engine import rank_and_select_async

    await memory.init_db()
    chunks = [
        _mk_snippet("d1", "alpha beta gamma delta"),
        _mk_snippet("d2", "epsilon zeta eta theta"),
    ]

    def _fake_embed(texts):
        return [[float(len(t) % 7)] + [0.0] * 383 for t in texts]

    def _boom(*_a, **_kw):
        raise AssertionError("asyncio.run must not be called on async hybrid path")

    with patch("agent.embedder._embed_sync", side_effect=_fake_embed), \
         patch.object(context_engine.asyncio, "run", side_effect=_boom):
        out = await rank_and_select_async("alpha beta", chunks, max_tokens=4000)
    assert out is not None


def test_sync_rank_and_select_skips_hybrid_when_loop_running(monkeypatch):
    """Legacy sync `rank_and_select` must NOT trigger the hybrid path when a
    loop is already running on the current thread — instead it should fall
    back to BM25-only retrieval silently. This prevents the original
    RuntimeError surface area for any straggler call site."""
    monkeypatch.setenv("HYBRID_RETRIEVAL", "1")
    chunks = [
        _mk_snippet("d1", "the quick brown fox"),
        _mk_snippet("d2", "lorem ipsum dolor"),
    ]

    embed_calls = {"n": 0}

    def _spy(texts):
        embed_calls["n"] += 1
        return [[0.0] * 384 for _ in texts]

    async def _run_inside_loop():
        with patch("agent.embedder._embed_sync", side_effect=_spy):
            return rank_and_select("brown fox", chunks, max_tokens=4000)

    out = asyncio.run(_run_inside_loop())
    assert out, "BM25 fallback must still return chunks"
    assert embed_calls["n"] == 0, (
        "Hybrid path must be skipped when called from a running event loop"
    )
