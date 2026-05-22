"""
Context pipeline:
  BM25 pre-filter → top 30 chunks
  FlashRank cross-encoder rerank → top 10
  3-factor scoring (0.6 relevance + 0.2 recency + 0.2 diversity) → final selection
  format_context_xml() → (xml_string, doc_map)
  probe_contradictions() → typed ConflictResult via Groq
"""
from __future__ import annotations

import json
import html
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

from rank_bm25 import BM25Okapi
try:
    from flashrank import Ranker, RerankRequest
    _FLASHRANK_AVAILABLE = True
except ImportError:
    _FLASHRANK_AVAILABLE = False

from agent.models import ClaimContradiction, ContextSnippet, ConflictResult, SearchResult
from utils.failure_policy import POLICY
from utils.prompt_registry import PROMPT_REGISTRY
from utils.source_trust import is_blocked, trust_for, trust_weight_enabled
from utils.token_counter import count_tokens

import asyncio
import contextvars
import pydantic

logger = logging.getLogger(__name__)

# Phase 1.875: per-turn live-time-sensitivity override. When set to True,
# `score_chunk` doubles the recency weight (0.15 → 0.30) at the expense of
# relevance for that turn only. Propagates through `asyncio.to_thread`.
_TIME_SENSITIVITY_LIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "context_engine.time_sensitivity_live", default=False
)


def set_time_sensitivity_live(live: bool) -> contextvars.Token:
    return _TIME_SENSITIVITY_LIVE.set(bool(live))


def reset_time_sensitivity_live(token: contextvars.Token) -> None:
    _TIME_SENSITIVITY_LIVE.reset(token)


# Phase 5b: per-request domain blocklist propagated into rank_and_select via
# contextvar so we don't churn every call signature. Empty == disabled.
# Drops sink (a mutable list) is also stashed contextually so the engine can
# record dropped domains into `run_metadata["domain_blocklist_drops"]`.
_DOMAIN_BLOCKLIST: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "context_engine.domain_blocklist", default=frozenset()
)
_BLOCKLIST_DROPS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "context_engine.blocklist_drops", default=None
)

# V3.9: per-request sink for reranker telemetry — set by orchestrator before
# the SELECTING stage so it can read back which reranker actually ran
# ("cohere" | "flashrank" | "none") without changing return signatures.
_RERANKER_SINK: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "context_engine.reranker_sink", default=None
)


def set_reranker_sink(sink: dict | None) -> contextvars.Token:
    return _RERANKER_SINK.set(sink)


def reset_reranker_sink(token: contextvars.Token) -> None:
    _RERANKER_SINK.reset(token)


def _record_reranker(name: str) -> None:
    sink = _RERANKER_SINK.get()
    if sink is not None:
        sink["reranker_used"] = name


def set_domain_blocklist(
    blocklist: frozenset[str],
    drops_sink: list[str] | None = None,
) -> tuple[contextvars.Token, contextvars.Token]:
    return (
        _DOMAIN_BLOCKLIST.set(blocklist),
        _BLOCKLIST_DROPS.set(drops_sink),
    )


def reset_domain_blocklist(tokens: tuple[contextvars.Token, contextvars.Token]) -> None:
    _DOMAIN_BLOCKLIST.reset(tokens[0])
    _BLOCKLIST_DROPS.reset(tokens[1])


def _apply_blocklist(chunks: list[ContextSnippet]) -> list[ContextSnippet]:
    """Filter out chunks whose domain matches the per-request blocklist.

    Defensive: search already drops blocklisted URLs, but a chunk can also
    arrive from cache, prior turns, or a future provider that bypasses the
    search filter. Re-checking here is cheap (O(n)) and means snippets never
    reach the synthesizer regardless of how they got into the chunk pool.
    """
    blocklist = _DOMAIN_BLOCKLIST.get()
    if not blocklist:
        return chunks
    sink = _BLOCKLIST_DROPS.get()
    kept: list[ContextSnippet] = []
    for c in chunks:
        if is_blocked(c.domain or "", blocklist):
            logger.debug(
                "Blocklisted domain dropped: %s", c.domain,
                extra={"component": "context_engine"},
            )
            if sink is not None:
                sink.append(c.domain or "")
            continue
        kept.append(c)
    return kept

_CHUNK_SIZE_WORDS = int(int(os.getenv("CHUNK_SIZE", "500")) * 0.75)
_CHUNK_OVERLAP = 50
_BM25_TOP_N = 30
_RERANK_TOP_N = 10
_MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))


# ── Chunking ───────────────────────────────────────────────────────────────

def chunk(result: SearchResult, text: str) -> list[ContextSnippet]:
    """Split text into ~CHUNK_SIZE token chunks with 50-token overlap."""
    words = text.split()
    if not words:
        return []

    chunks: list[ContextSnippet] = []
    start = 0
    doc_idx = 0
    trust_score, trust_tier = trust_for(result.domain or "")
    while start < len(words):
        end = start + _CHUNK_SIZE_WORDS
        segment = " ".join(words[start:end])
        tok_count = count_tokens(segment)
        chunks.append(ContextSnippet(
            doc_id=f"doc_{len(chunks)+1}",
            url=result.url,
            title=result.title,
            domain=result.domain,
            text=segment,
            snippet=segment[:200],
            token_count=tok_count,
            retrieved_at=result.retrieved_at,
            intent_origin=result.intent_origin,
            trust_score=trust_score,
            trust_tier=trust_tier,
            provider_relevance=result.relevance,
            provider_relevance_source=result.relevance_source,
        ))
        start = end - _CHUNK_OVERLAP
        doc_idx += 1

    return chunks


# ── Scoring ────────────────────────────────────────────────────────────────

def score_chunk(
    chunk: ContextSnippet,
    bm25_score: float,
    all_bm25_scores: list[float],
    url_domain_count: dict[str, int],
    intent_origin: str | None = None,
) -> float:
    """
    Five-signal additive score (provider-relevance added in this revision):
        final = 0.50*relevance + 0.15*recency + 0.15*diversity + 0.20*trust
                + 0.05 * provider_relevance_bonus
    The provider bonus sits *on top* of the 1.0 base (i.e. cap is 1.05 before
    intent boosts), kept deliberately small so search-engine ordering can
    *break ties* and *nudge* the BM25 ranking without overpowering it. We do
    not give it primary weight because:
      - For rank-derived providers (Parallel, Serper) the signal is purely
        positional — already implicit in result order.
      - For score-providers (Tavily) the absolute scale is uncalibrated to
        our BM25 + FlashRank pipeline.
    Provider="provider" (absolute) is used as-is; provider="rank" is halved
    so it contributes at most 0.025 — pure provenance-aware weighting.

    When SOURCE_TRUST_DISABLED=1 (ablation), trust weight is zeroed and the
    remaining factors get the original 0.6/0.2/0.2 weighting (provider bonus
    is dropped in that mode too, for clean ablation comparability).
    """
    # 1. Relevance: normalize BM25 score to [0, 1]
    max_score = max(all_bm25_scores) if all_bm25_scores else 1.0
    relevance = bm25_score / max_score if max_score > 0 else 0.0

    # 2. Recency: gentle exponential decay over days since retrieval
    try:
        retrieved = datetime.fromisoformat(chunk.retrieved_at.replace("Z", "+00:00"))
        if retrieved.tzinfo is None:
            retrieved = retrieved.replace(tzinfo=timezone.utc)
        days_old = max(0, (datetime.now(timezone.utc) - retrieved).days)
    except Exception:
        days_old = 0
    recency = math.exp(-0.001 * days_old)

    # 3. Diversity: penalise domains already heavily represented
    domain_count = url_domain_count.get(chunk.domain, 0)
    diversity = 1.0 / (1.0 + domain_count)

    # V2.1: bounded diversity boost for contradiction_probe chunks
    if intent_origin == "contradiction_probe":
        diversity = min(1.0, diversity * 1.25)

    # 4. Trust: tiered source prior (additive, not multiplicative)
    trust = chunk.trust_score

    # 5. Provider relevance bonus (V3.7) — halved when rank-derived since the
    # signal is purely positional and we don't want to amplify search-result
    # ordering past what BM25/FlashRank already captured.
    pr = chunk.provider_relevance
    if pr is None:
        provider_bonus = 0.0
    else:
        pr_clamped = max(0.0, min(1.0, pr))
        provider_bonus = pr_clamped if chunk.provider_relevance_source == "provider" else pr_clamped * 0.5

    chunk.recency_score = recency
    chunk.diversity_score = diversity

    # Phase 1.875: when this turn was planned with time_sensitivity="live",
    # double the recency weight (taken proportionally from relevance) so the
    # selector prefers fresh material. Trust + diversity + provider bonus
    # stay constant — they don't model freshness.
    live = _TIME_SENSITIVITY_LIVE.get()
    if trust_weight_enabled():
        if live:
            return (
                0.35 * relevance
                + 0.30 * recency
                + 0.15 * diversity
                + 0.20 * trust
                + 0.05 * provider_bonus
            )
        return (
            0.50 * relevance
            + 0.15 * recency
            + 0.15 * diversity
            + 0.20 * trust
            + 0.05 * provider_bonus
        )
    # Ablation mode: redistribute the trust weight back to the original 3 factors;
    # drop provider bonus for clean ablation comparability.
    if live:
        return 0.45 * relevance + 0.35 * recency + 0.20 * diversity
    return 0.60 * relevance + 0.20 * recency + 0.20 * diversity


# ── Domain-diversity-aware selection ──────────────────────────────────────

def select_with_diversity(
    chunks: list[ContextSnippet],
    max_tokens: int = 6400,
    max_per_domain: int = 2,
) -> list[ContextSnippet]:
    """
    Select highest-scoring chunks respecting:
    - max_per_domain cap (default 2) — source diversity
    - max_tokens budget
    Expects chunks already have final_score set.
    """
    logger.debug("Context selection path=%s lambda=%s", "legacy_diversity", "n/a")
    domain_counts: dict[str, int] = defaultdict(int)
    selected: list[ContextSnippet] = []
    token_count = 0

    for c in sorted(chunks, key=lambda x: x.final_score, reverse=True):
        # V2.1: contradiction_probe chunks may exceed default cap (3 per domain).
        cap = 3 if c.intent_origin == "contradiction_probe" else max_per_domain
        if domain_counts[c.domain] >= cap:
            continue
        if token_count + c.token_count <= max_tokens:
            selected.append(c)
            domain_counts[c.domain] += 1
            token_count += c.token_count
            continue
        # Phase 1.75: adaptive score-ranked drop. Rather than skipping `c`,
        # try evicting the lowest-scored already-selected chunk if doing so
        # frees enough budget AND `c.final_score` is strictly higher. Capped
        # at 1 swap per insertion to avoid thrash.
        if not selected:
            continue
        lowest = min(selected, key=lambda s: s.final_score)
        if c.final_score > lowest.final_score and (
            token_count - lowest.token_count + c.token_count <= max_tokens
        ):
            logger.debug(
                "Adaptive eviction: dropped score=%.3f for score=%.3f",
                lowest.final_score, c.final_score,
            )
            selected.remove(lowest)
            domain_counts[lowest.domain] -= 1
            token_count -= lowest.token_count
            selected.append(c)
            domain_counts[c.domain] += 1
            token_count += c.token_count

    return selected


# ── V3.1 / V3.8: Hybrid RRF (BM25 + dense), capability-aware ──────────────

from utils.retrieval_mode import effective_mode_for_request


def hybrid_retrieval_enabled() -> bool:
    """Resolve the effective retrieval mode for this request and return
    True iff hybrid (BM25 ⊕ sqlite-vec RRF) should run.

    Delegates to `utils.retrieval_mode.effective_mode_for_request`, which
    handles: per-request override (RuntimeConfig from settings page) →
    `RETRIEVAL_MODE` env → legacy `HYBRID_RETRIEVAL` alias → default
    ``auto`` (use hybrid when sqlite-vec loads, lexical fallback otherwise).
    """
    from agent import memory  # lazy to avoid circular import at module load
    eff = effective_mode_for_request(memory._VEC_AVAILABLE)
    return eff.is_hybrid


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60,
) -> list[tuple[str, float]]:
    """Standard RRF: score(doc) = sum over rankings of 1 / (k + rank).
    Returns descending list of (doc_id, score)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)  # 1-indexed rank per RRF convention
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _has_running_loop() -> bool:
    """Detect whether the current thread has an active asyncio event loop.

    Used by the legacy sync `rank_and_select` to skip the hybrid path safely
    when called from inside a running loop — `asyncio.run` would otherwise
    raise `RuntimeError: This event loop is already running`. The async
    orchestrator entry point (`rank_and_select_async`) is the correct path
    for that case; sync callers that hit this branch silently fall back to
    BM25-only retrieval.
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _hybrid_precompute(
    query: str,
    all_chunks: list,
) -> tuple[list[str], list[float], list[list[float]]] | None:
    """CPU-bound: assign unique chunk_ids, embed query + chunks. No I/O, no DB.

    Returns (unique_ids, query_embedding, chunk_embeddings) or None on failure.
    Safe to call from a worker thread (no asyncio dependency).
    """
    try:
        from agent.embedder import _embed_sync as _sync_embed

        import uuid as _uuid
        unique_ids: list[str] = []
        run_prefix = _uuid.uuid4().hex[:8]
        for i, c in enumerate(all_chunks):
            cid = f"h-{run_prefix}-{i}"
            unique_ids.append(cid)
            try:
                c._hybrid_chunk_id = cid  # type: ignore[attr-defined]
            except Exception:
                pass

        texts = [c.text for c in all_chunks]
        embeds = _sync_embed([query] + texts)
        if not embeds or len(embeds) < 2:
            return None
        return unique_ids, embeds[0], embeds[1:]
    except Exception as exc:
        logger.warning("Hybrid precompute failed: %s", exc,
                       extra={"component": "context_engine"})
        return None


def _hybrid_fuse(
    unique_ids: list[str],
    bm25_top_indices: list[int],
    vec_hits: list[tuple[str, float]],
) -> list[int]:
    """Sync RRF fusion. Returns top indices into `unique_ids`."""
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
    bm25_ranking = [unique_ids[i] for i in bm25_top_indices]
    vec_ranking = [cid for cid, _ in vec_hits]
    fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
    return [id_to_idx[cid] for cid, _ in fused if cid in id_to_idx][:_BM25_TOP_N]


async def _hybrid_shortlist_async(
    query: str,
    all_chunks: list,
    bm25_top_indices: list[int],
) -> tuple[list[int], list[str]] | None:
    """Async hybrid shortlist — runs in the caller's event loop.

    Splits work along the sync/async seam: CPU-bound embedding via
    `asyncio.to_thread`, async DB writes/reads via `await`, then a final sync
    RRF fusion. This is the correct entry point from inside a running event
    loop (e.g. the orchestrator's `SELECTING` phase).
    """
    try:
        from agent import memory
        pre = await asyncio.to_thread(_hybrid_precompute, query, all_chunks)
        if pre is None:
            return None
        unique_ids, q_emb, chunk_embs = pre
        await memory.save_chunk_embeddings(unique_ids, chunk_embs)
        vec_hits = await memory.vector_search(q_emb, unique_ids, top_k=_BM25_TOP_N)
        if not vec_hits:
            return None
        return _hybrid_fuse(unique_ids, bm25_top_indices, vec_hits), unique_ids
    except Exception as exc:
        logger.warning("Hybrid RRF path failed, falling back to BM25: %s", exc,
                       extra={"component": "context_engine"})
        return None


def _hybrid_shortlist(
    query: str,
    all_chunks: list,
    bm25_top_indices: list[int],
) -> tuple[list[int], list[str]] | None:
    """Sync wrapper for legacy callers (no running event loop).

    When a running loop is detected, returns None so the caller falls back
    to BM25-only — preventing `RuntimeError: This event loop is already
    running`. Async callers must use `_hybrid_shortlist_async` instead.
    """
    if _has_running_loop():
        logger.debug(
            "Hybrid sync path skipped: running event loop detected. "
            "Caller should use rank_and_select_async / _hybrid_shortlist_async.",
            extra={"component": "context_engine"},
        )
        return None
    try:
        from agent import memory
        pre = _hybrid_precompute(query, all_chunks)
        if pre is None:
            return None
        unique_ids, q_emb, chunk_embs = pre

        async def _persist_and_search():
            await memory.save_chunk_embeddings(unique_ids, chunk_embs)
            return await memory.vector_search(q_emb, unique_ids, top_k=_BM25_TOP_N)

        vec_hits = asyncio.run(_persist_and_search())
        if not vec_hits:
            return None
        return _hybrid_fuse(unique_ids, bm25_top_indices, vec_hits), unique_ids
    except Exception as exc:
        logger.warning("Hybrid RRF path failed, falling back to BM25: %s", exc,
                       extra={"component": "context_engine"})
        return None


# ── BM25 + FlashRank pipeline ──────────────────────────────────────────────

def _bm25_prefilter(
    query: str, all_chunks: list[ContextSnippet],
) -> tuple[list[int], list[float]]:
    """Shared BM25 step. Returns (top_indices, all_scores)."""
    tokenized = [c.text.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in indexed[:_BM25_TOP_N]], list(scores)


def _finalize_selection(
    query: str,
    all_chunks: list[ContextSnippet],
    top_indices: list[int],
    scores: list[float],
    max_tokens: int,
    preranked_chunks: list[ContextSnippet] | None = None,
) -> list[ContextSnippet]:
    """Shared FlashRank + 3-factor scoring + diversity selection (sync).

    If ``preranked_chunks`` is provided, FlashRank is bypassed and that ordering
    is used directly (e.g. when Cohere Rerank already produced the top-K).
    """
    top_chunks = [all_chunks[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    if preranked_chunks is not None:
        # Honour caller-supplied ordering (Cohere). Recover BM25 scores so the
        # 5-factor relevance term remains comparable across paths.
        bm25_by_id = {id(c): s for c, s in zip(top_chunks, top_scores)}
        top_chunks = preranked_chunks[:_RERANK_TOP_N]
        top_scores = [bm25_by_id.get(id(c), 0.0) for c in top_chunks]
        domain_counts: dict[str, int] = defaultdict(int)
        for c in top_chunks:
            domain_counts[c.domain] += 1
        for c, s in zip(top_chunks, top_scores):
            c.bm25_score = s
            c.final_score = score_chunk(c, s, top_scores, domain_counts, intent_origin=c.intent_origin)
        return select_with_diversity(top_chunks, max_tokens=max_tokens)

    # Step 2: FlashRank cross-encoder rerank → top 10
    if _FLASHRANK_AVAILABLE and len(top_chunks) > 1:
        try:
            ranker = Ranker()
            passages = [{"id": i, "text": c.text} for i, c in enumerate(top_chunks)]
            req = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(req)
            reranked_ids = [r["id"] for r in results[:_RERANK_TOP_N]]
            top_chunks = [top_chunks[i] for i in reranked_ids]
            top_scores = [r.get("score", top_scores[i]) for i, r in enumerate(results[:_RERANK_TOP_N])]
        except Exception as e:
            logger.warning("FlashRank rerank failed, using BM25 order: %s", e)
            top_chunks = top_chunks[:_RERANK_TOP_N]
            top_scores = top_scores[:_RERANK_TOP_N]
    else:
        top_chunks = top_chunks[:_RERANK_TOP_N]
        top_scores = top_scores[:_RERANK_TOP_N]

    # Step 3: 3-factor scoring
    domain_counts: dict[str, int] = defaultdict(int)
    for c in top_chunks:
        domain_counts[c.domain] += 1

    for c, s in zip(top_chunks, top_scores):
        c.bm25_score = s
        c.final_score = score_chunk(c, s, top_scores, domain_counts, intent_origin=c.intent_origin)

    # Step 4: diversity-aware final selection
    return select_with_diversity(top_chunks, max_tokens=max_tokens)


async def rank_and_select_async(
    query: str,
    all_chunks: list[ContextSnippet],
    max_tokens: int = 6400,
) -> list[ContextSnippet]:
    """Async pipeline: hybrid prep in event loop, CPU work via to_thread.

    Use this from inside a running event loop (e.g. the orchestrator) so the
    async DB work for hybrid retrieval shares the caller's loop / pooling
    instead of spinning up a private `asyncio.run` per call.
    """
    if not all_chunks:
        return []
    all_chunks = _apply_blocklist(all_chunks)
    if not all_chunks:
        return []

    bm25_top_indices, scores = await asyncio.to_thread(
        _bm25_prefilter, query, all_chunks
    )

    top_indices = bm25_top_indices
    if hybrid_retrieval_enabled() and len(all_chunks) > 1:
        fused = await _hybrid_shortlist_async(query, all_chunks, bm25_top_indices)
        if fused is not None:
            top_indices, _ = fused

    # V3.9: Cohere Rerank augmentation. English-only, gated on COHERE_API_KEY.
    # Falls back silently to FlashRank in `_finalize_selection`.
    preranked: list[ContextSnippet] | None = None
    if os.getenv("COHERE_API_KEY", "").strip() and len(top_indices) > 1:
        from agent.search import _detect_language  # lazy: avoid cycle at import
        if _detect_language(query) == "en":
            from utils.cohere_rerank import rerank_with_cohere
            candidates = [all_chunks[i] for i in top_indices]
            cohere_ordered = await rerank_with_cohere(query, candidates, top_k=_RERANK_TOP_N)
            if cohere_ordered:
                preranked = cohere_ordered
                _record_reranker("cohere")
    if preranked is None:
        _record_reranker("flashrank" if _FLASHRANK_AVAILABLE else "none")

    return await asyncio.to_thread(
        _finalize_selection, query, all_chunks, top_indices, scores, max_tokens,
        preranked,
    )


def rank_and_select(
    query: str,
    all_chunks: list[ContextSnippet],
    max_tokens: int = 6400,
) -> list[ContextSnippet]:
    """Sync pipeline: BM25 → FlashRank → 3-factor score → diversity selection.

    Backward-compat entry point for callers without an event loop. When a
    running loop is detected, the hybrid path is skipped (see
    `_hybrid_shortlist`); use `rank_and_select_async` to keep hybrid active
    in that case.
    """
    if not all_chunks:
        return []

    # Phase 5b: drop blocklisted domains before any scoring work.
    all_chunks = _apply_blocklist(all_chunks)
    if not all_chunks:
        return []

    # Step 1: BM25 pre-filter → top 30
    bm25_top_indices, scores = _bm25_prefilter(query, all_chunks)

    # V3.1: optional hybrid RRF fusion. Failure path returns None → BM25-only.
    top_indices = bm25_top_indices
    if hybrid_retrieval_enabled() and len(all_chunks) > 1:
        fused = _hybrid_shortlist(query, all_chunks, bm25_top_indices)
        if fused is not None:
            top_indices, _ = fused

    return _finalize_selection(query, all_chunks, top_indices, scores, max_tokens)


def _shortlist_chunks(
    query: str,
    all_chunks: list[ContextSnippet],
) -> tuple[list[ContextSnippet], list[float]]:
    """Shared BM25 + optional rerank shortlist for fair selector comparison."""
    tokenized = [c.text.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in indexed[:_BM25_TOP_N]]
    top_chunks = [all_chunks[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    if _FLASHRANK_AVAILABLE and len(top_chunks) > 1:
        try:
            ranker = Ranker()
            passages = [{"id": i, "text": c.text} for i, c in enumerate(top_chunks)]
            req = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(req)
            reranked_ids = [r["id"] for r in results[:_RERANK_TOP_N]]
            top_chunks = [top_chunks[i] for i in reranked_ids]
            top_scores = [r.get("score", top_scores[i]) for i, r in enumerate(results[:_RERANK_TOP_N])]
        except Exception:
            top_chunks = top_chunks[:_RERANK_TOP_N]
            top_scores = top_scores[:_RERANK_TOP_N]
    else:
        top_chunks = top_chunks[:_RERANK_TOP_N]
        top_scores = top_scores[:_RERANK_TOP_N]
    return top_chunks, top_scores


def _novelty(a: ContextSnippet, b: ContextSnippet) -> float:
    """Simple token-set dissimilarity in [0, 1] for MMR novelty."""
    a_tokens = set(a.text.lower().split()[:120])
    b_tokens = set(b.text.lower().split()[:120])
    if not a_tokens or not b_tokens:
        return 1.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens) or 1
    return 1.0 - (inter / union)


def rank_and_select_mmr(
    query: str,
    all_chunks: list[ContextSnippet],
    max_tokens: int = 6400,
    max_per_domain: int = 2,
    mmr_lambda: float = _MMR_LAMBDA,
) -> list[ContextSnippet]:
    """MMR selector on the same shortlist as heuristic selector."""
    logger.debug("Context selection path=%s lambda=%s", "mmr", mmr_lambda)
    if not all_chunks:
        return []
    # Phase 5b: drop blocklisted domains before shortlist work.
    all_chunks = _apply_blocklist(all_chunks)
    if not all_chunks:
        return []
    shortlist, scores = _shortlist_chunks(query, all_chunks)
    if not shortlist:
        return []

    max_score = max(scores) if scores else 1.0
    relevance = [(s / max_score) if max_score > 0 else 0.0 for s in scores]
    paired = list(zip(shortlist, relevance))

    selected: list[ContextSnippet] = []
    remaining = paired[:]
    domain_counts: dict[str, int] = defaultdict(int)
    token_count = 0

    while remaining:
        best_idx = None
        best_value = float("-inf")
        best_overflow_idx = None
        best_overflow_value = float("-inf")
        for idx, (cand, rel) in enumerate(remaining):
            if domain_counts[cand.domain] >= max_per_domain:
                continue
            if not selected:
                mmr_score = rel
            else:
                max_sim = max(1.0 - _novelty(cand, s) for s in selected)
                mmr_score = (mmr_lambda * rel) - ((1.0 - mmr_lambda) * max_sim)
            if token_count + cand.token_count > max_tokens:
                # Track best overflow candidate for adaptive eviction below.
                if mmr_score > best_overflow_value:
                    best_overflow_value = mmr_score
                    best_overflow_idx = idx
                continue
            if mmr_score > best_value:
                best_value = mmr_score
                best_idx = idx
        if best_idx is not None:
            chosen, rel = remaining.pop(best_idx)
            chosen.bm25_score = rel
            chosen.final_score = best_value
            selected.append(chosen)
            domain_counts[chosen.domain] += 1
            token_count += chosen.token_count
            continue
        # Phase 1.75: adaptive score-ranked drop for MMR path. Try evicting
        # the lowest-scored already-selected snippet to make room for the
        # best overflow candidate. Capped at 1 swap per outer iteration.
        if best_overflow_idx is None or not selected:
            break
        cand, _rel = remaining[best_overflow_idx]
        lowest = min(selected, key=lambda s: s.final_score)
        if best_overflow_value > lowest.final_score and (
            token_count - lowest.token_count + cand.token_count <= max_tokens
        ):
            logger.debug(
                "Adaptive eviction (mmr): dropped score=%.3f for score=%.3f",
                lowest.final_score, best_overflow_value,
            )
            chosen, rel = remaining.pop(best_overflow_idx)
            selected.remove(lowest)
            domain_counts[lowest.domain] -= 1
            token_count -= lowest.token_count
            chosen.bm25_score = rel
            chosen.final_score = best_overflow_value
            selected.append(chosen)
            domain_counts[chosen.domain] += 1
            token_count += chosen.token_count
        else:
            break
    return selected


# ── Context XML formatting ─────────────────────────────────────────────────

def _reorder_u_shape(selected: list[ContextSnippet]) -> list[ContextSnippet]:
    """U-shape reordering for Lost-in-the-Middle mitigation.

    Liu et al. 2023, "Lost in the Middle: How Language Models Use Long Contexts" —
    LLMs attend strongest at the start and end of the context window. We place
    the top-ranked snippet first and the second-ranked snippet last so the
    middle holds lower-ranked items. No-op for lists shorter than 3.
    """
    if len(selected) < 3:
        return selected
    top1 = selected[0]
    top2 = selected[1]
    middle = selected[2:]
    return [top1, *middle, top2]


def _iso_utc(value: str) -> str:
    """Best-effort ISO-8601 UTC normalisation. Returns input unchanged if unparseable."""
    if not value:
        return ""
    try:
        # Already ISO? trust it.
        if "T" in value and (value.endswith("Z") or "+" in value[10:]):
            return value
        # Fallback: parse + render.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (ValueError, TypeError):
        return value


def format_context_xml(
    chunks: list[ContextSnippet],
) -> tuple[str, dict[str, tuple[str, str, str]]]:
    """
    Returns (xml_string, doc_map).
    doc_map: {"doc_1": (title, url, domain)}

    Phase 1.75:
      - U-shape ordering at injection time (Liu et al. 2023). The persisted
        ``Turn.doc_map`` and snippet ranks still reflect the U-shape output
        order, but ``<relevance_score>`` carries the true editorial score so
        audit fidelity is preserved.
      - Each ``<document>`` carries ``<retrieved_at>``, ``<rank>``, and
        ``<relevance_score>`` sub-elements (Vectara NAACL 2025: metadata
        enrichment lifts QA accuracy).
    """
    ordered = _reorder_u_shape(chunks)
    doc_map: dict[str, tuple[str, str, str]] = {}
    parts = ["<context>"]
    for i, c in enumerate(ordered, start=1):
        doc_id = f"doc_{i}"
        c.doc_id = doc_id
        doc_map[doc_id] = (c.title or c.domain, c.url, c.domain)
        safe_url = html.escape(c.url, quote=True)
        safe_title = html.escape(c.title or "", quote=True)
        safe_domain = html.escape(c.domain or "", quote=True)
        safe_text = html.escape(c.text or "")
        safe_retrieved = html.escape(_iso_utc(c.retrieved_at or ""), quote=False)
        relevance = float(getattr(c, "final_score", 0.0) or 0.0)
        parts.append(
            f'  <document id="{doc_id}" url="{safe_url}" title="{safe_title}" domain="{safe_domain}">\n'
            f"    <retrieved_at>{safe_retrieved}</retrieved_at>\n"
            f"    <rank>{i}</rank>\n"
            f"    <relevance_score>{relevance:.3f}</relevance_score>\n"
            f"    {safe_text}\n"
            f"  </document>"
        )
    parts.append("</context>")
    return "\n".join(parts), doc_map


# ── Contradiction probe ────────────────────────────────────────────────────

# ContextVar so concurrent ``probe_contradictions`` coroutines each see
# their own provider attribution. The setter writes inside the same task
# context that the reader is in, so values propagate back to the orchestrator
# after ``await`` returns (unlike a child task's context).
_LAST_PROBE_PROVIDER_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "context_engine.last_probe_provider", default="groq"
)


def last_conflict_probe_provider() -> str:
    """Return the provider used for the most recent ``probe_contradictions``
    invocation. Stamped into ``run_metadata["conflict_probe_provider"]``."""
    return _LAST_PROBE_PROVIDER_VAR.get()


async def probe_contradictions(chunks: list[ContextSnippet], query: str) -> ConflictResult:
    """
    First-class pipeline stage: LLM call returning typed ClaimContradiction entries.
    Distinguishes real contradictions from temporal evolution at the prompt level.
    Never raises — every failure mode maps to a probe_skipped_reason.

    Provider selection via ``CONFLICT_PROBE_PROVIDER`` env (default ``auto``):
      - ``auto``: Cerebras when key present + prompt fits in ~6K tokens, else Groq.
      - ``cerebras``: force Cerebras (errors fall back to Groq).
      - ``groq``: force Groq.
    """
    if len(chunks) < 2:
        return ConflictResult(has_conflict=False, probe_skipped_reason="lt_2_chunks")

    sources = "\n\n".join(
        f"[{c.doc_id or f'doc_{i+1}'} | {c.domain}]: {c.text[:500]}"
        for i, c in enumerate(chunks[:8])
    )
    prompt = PROMPT_REGISTRY["conflict_v3"]["template"].format(query=query, sources=sources)

    pref = os.environ.get("CONFLICT_PROBE_PROVIDER", "auto").lower().strip()
    raw: str | None = None
    chosen = "groq"

    # Try Cerebras first when allowed
    if pref in ("auto", "cerebras") and os.environ.get("CEREBRAS_API_KEY"):
        if count_tokens(prompt) <= 6000:
            try:
                from utils.provider_router import call_cerebras
                raw = await asyncio.wait_for(
                    call_cerebras(prompt, max_tokens=600),
                    timeout=POLICY.probe_timeout_s,
                )
                chosen = "cerebras"
            except asyncio.TimeoutError:
                logger.warning(
                    "Probe Cerebras timeout — falling back to Groq",
                    extra={"component": "context_engine"},
                )
                raw = None
            except Exception as e:
                logger.warning(
                    "Probe Cerebras failed (%s) — falling back to Groq", e,
                    extra={"component": "context_engine"},
                )
                raw = None

    if raw is None and pref != "cerebras":
        try:
            from utils.provider_router import call_groq
            raw = await asyncio.wait_for(
                call_groq(prompt, max_tokens=600),
                timeout=POLICY.probe_timeout_s,
            )
            chosen = "groq"
        except asyncio.TimeoutError:
            logger.warning("Probe timeout", extra={"component": "context_engine"})
            _LAST_PROBE_PROVIDER_VAR.set(chosen)
            return ConflictResult(has_conflict=False, probe_skipped_reason="timeout")
        except Exception as e:
            logger.warning("Probe LLM call failed: %s", e, extra={"component": "context_engine"})
            _LAST_PROBE_PROVIDER_VAR.set(chosen)
            return ConflictResult(has_conflict=False, probe_skipped_reason="parse_fail")

    if raw is None:
        # Cerebras-forced and failed.
        _LAST_PROBE_PROVIDER_VAR.set(chosen)
        return ConflictResult(has_conflict=False, probe_skipped_reason="parse_fail")

    _LAST_PROBE_PROVIDER_VAR.set(chosen)

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return ConflictResult(has_conflict=False, probe_skipped_reason="parse_fail")
    try:
        data = json.loads(raw[start:end])
        result = ConflictResult(
            has_conflict=bool(data.get("has_conflict", False)),
            conflict_summary=data.get("conflict_summary"),
            contradictions=[ClaimContradiction(**c) for c in (data.get("contradictions") or [])],
        )
        return result
    except (json.JSONDecodeError, pydantic.ValidationError, TypeError, ValueError) as e:
        logger.warning("Probe parse failed: %s", e, extra={"component": "context_engine"})
        return ConflictResult(has_conflict=False, probe_skipped_reason="parse_fail")


async def detect_conflicts(chunks: list[ContextSnippet], query: str) -> ConflictResult:
    """Backward-compat shim. Prefer probe_contradictions for new code."""
    return await probe_contradictions(chunks, query)
