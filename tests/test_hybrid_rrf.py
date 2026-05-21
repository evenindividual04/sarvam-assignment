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


def test_hybrid_retrieval_disabled_uses_bm25_only(monkeypatch):
    monkeypatch.delenv("HYBRID_RETRIEVAL", raising=False)
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

    assert called["n"] == 0, "embedder must NOT be called when HYBRID_RETRIEVAL is unset"
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
