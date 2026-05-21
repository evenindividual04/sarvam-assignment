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
from utils.source_trust import trust_for, trust_weight_enabled
from utils.token_counter import count_tokens

import asyncio
import pydantic

logger = logging.getLogger(__name__)

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
    Four-factor additive score (V2.3):
        final = 0.50*relevance + 0.15*recency + 0.15*diversity + 0.20*trust
    Weights sum to 1.0. Trust contribution is bounded — even max-trust cannot
    promote a chunk with near-zero relevance.

    When SOURCE_TRUST_DISABLED=1 (ablation), trust weight is zeroed and the
    remaining three factors get the original 0.6/0.2/0.2 weighting.

    When intent_origin == "contradiction_probe", diversity is multiplied by 1.25
    (capped at 1.0) BEFORE additive scoring to help adversarial chunks survive.
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

    chunk.recency_score = recency
    chunk.diversity_score = diversity

    if trust_weight_enabled():
        return 0.50 * relevance + 0.15 * recency + 0.15 * diversity + 0.20 * trust
    # Ablation mode: redistribute the trust weight back to the original 3 factors
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
    domain_counts: dict[str, int] = defaultdict(int)
    selected: list[ContextSnippet] = []
    token_count = 0

    for c in sorted(chunks, key=lambda x: x.final_score, reverse=True):
        # V2.1: contradiction_probe chunks may exceed default cap (3 per domain).
        cap = 3 if c.intent_origin == "contradiction_probe" else max_per_domain
        if domain_counts[c.domain] >= cap:
            continue
        if token_count + c.token_count > max_tokens:
            continue
        selected.append(c)
        domain_counts[c.domain] += 1
        token_count += c.token_count

    return selected


# ── V3.1: Hybrid RRF (BM25 + dense) ────────────────────────────────────────

import contextvars

# Per-request override for HYBRID_RETRIEVAL. The orchestrator sets this from
# the resolved `RuntimeConfig` at the start of each turn (Option D); it
# cleanly resets at task scope so concurrent /research calls don't bleed.
_hybrid_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "hybrid_override", default=None,
)


def set_hybrid_override(value: bool | None) -> contextvars.Token:
    """Bind a per-request hybrid-retrieval flag. Returns a Token to reset with."""
    return _hybrid_override.set(value)


def reset_hybrid_override(token: contextvars.Token) -> None:
    _hybrid_override.reset(token)


def hybrid_retrieval_enabled() -> bool:
    """Per-request override (Option D) > env var HYBRID_RETRIEVAL > off."""
    ov = _hybrid_override.get()
    if ov is not None:
        return ov
    return os.getenv("HYBRID_RETRIEVAL", "0").strip() == "1"


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


def _hybrid_shortlist(
    query: str,
    all_chunks: list,
    bm25_top_indices: list[int],
) -> tuple[list[int], list[str]] | None:
    """Embed query + chunks, persist to sqlite-vec, run vector KNN, fuse with BM25
    via RRF. Returns (top_indices, chunk_ids) into all_chunks for the top-30 fused
    list, or None on any failure (caller falls back to BM25 path).

    Sync; called from within the asyncio.to_thread worker that drives the whole
    pipeline. Drives async memory helpers via a private event loop.
    """
    try:
        from agent import embedder, memory
        from agent.embedder import _embed_sync as _sync_embed

        import uuid as _uuid
        # Per-turn unique chunk_ids; stamp onto chunk for later cleanup by
        # orchestrator. Avoids collisions across results that share doc_N ids.
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
        q_emb, chunk_embs = embeds[0], embeds[1:]

        async def _persist_and_search():
            await memory.save_chunk_embeddings(unique_ids, chunk_embs)
            return await memory.vector_search(q_emb, unique_ids, top_k=_BM25_TOP_N)

        vec_hits = asyncio.run(_persist_and_search())
        if not vec_hits:
            return None
        id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
        bm25_ranking = [unique_ids[i] for i in bm25_top_indices]
        vec_ranking = [cid for cid, _ in vec_hits]
        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
        top_indices = [id_to_idx[cid] for cid, _ in fused if cid in id_to_idx][:_BM25_TOP_N]
        return top_indices, unique_ids
    except Exception as exc:
        logger.warning("Hybrid RRF path failed, falling back to BM25: %s", exc,
                       extra={"component": "context_engine"})
        return None


# ── BM25 + FlashRank pipeline ──────────────────────────────────────────────

def rank_and_select(
    query: str,
    all_chunks: list[ContextSnippet],
    max_tokens: int = 6400,
) -> list[ContextSnippet]:
    """Full pipeline: BM25 → FlashRank → 3-factor score → diversity selection."""
    if not all_chunks:
        return []

    # Step 1: BM25 pre-filter → top 30
    tokenized = [c.text.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    bm25_top_indices = [i for i, _ in indexed[:_BM25_TOP_N]]

    # V3.1: optional hybrid RRF fusion. Failure path returns None → BM25-only.
    top_indices = bm25_top_indices
    if hybrid_retrieval_enabled() and len(all_chunks) > 1:
        fused = _hybrid_shortlist(query, all_chunks, bm25_top_indices)
        if fused is not None:
            top_indices, _ = fused

    top_chunks = [all_chunks[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

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
        for idx, (cand, rel) in enumerate(remaining):
            if domain_counts[cand.domain] >= max_per_domain:
                continue
            if token_count + cand.token_count > max_tokens:
                continue
            if not selected:
                mmr_score = rel
            else:
                max_sim = max(1.0 - _novelty(cand, s) for s in selected)
                mmr_score = (mmr_lambda * rel) - ((1.0 - mmr_lambda) * max_sim)
            if mmr_score > best_value:
                best_value = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        chosen, rel = remaining.pop(best_idx)
        chosen.bm25_score = rel
        chosen.final_score = best_value
        selected.append(chosen)
        domain_counts[chosen.domain] += 1
        token_count += chosen.token_count
    return selected


# ── Context XML formatting ─────────────────────────────────────────────────

def format_context_xml(
    chunks: list[ContextSnippet],
) -> tuple[str, dict[str, tuple[str, str, str]]]:
    """
    Returns (xml_string, doc_map).
    doc_map: {"doc_1": (title, url, domain)}
    Each chunk assigned a sequential doc_N id.
    """
    doc_map: dict[str, tuple[str, str, str]] = {}
    parts = ["<context>"]
    for i, c in enumerate(chunks, start=1):
        doc_id = f"doc_{i}"
        c.doc_id = doc_id
        doc_map[doc_id] = (c.title or c.domain, c.url, c.domain)
        safe_url = html.escape(c.url, quote=True)
        safe_title = html.escape(c.title or "", quote=True)
        safe_domain = html.escape(c.domain or "", quote=True)
        safe_text = html.escape(c.text or "")
        parts.append(
            f'  <document id="{doc_id}" url="{safe_url}" title="{safe_title}" domain="{safe_domain}">\n'
            f"    {safe_text}\n"
            f"  </document>"
        )
    parts.append("</context>")
    return "\n".join(parts), doc_map


# ── Contradiction probe ────────────────────────────────────────────────────

async def probe_contradictions(chunks: list[ContextSnippet], query: str) -> ConflictResult:
    """
    First-class pipeline stage: Groq call returning typed ClaimContradiction entries.
    Distinguishes real contradictions from temporal evolution at the prompt level.
    Never raises — every failure mode maps to a probe_skipped_reason.
    """
    if len(chunks) < 2:
        return ConflictResult(has_conflict=False, probe_skipped_reason="lt_2_chunks")

    sources = "\n\n".join(
        f"[{c.doc_id or f'doc_{i+1}'} | {c.domain}]: {c.text[:500]}"
        for i, c in enumerate(chunks[:8])
    )
    prompt = PROMPT_REGISTRY["conflict_v3"]["template"].format(query=query, sources=sources)

    try:
        from utils.provider_router import call_groq
        raw = await asyncio.wait_for(
            call_groq(prompt, max_tokens=600),
            timeout=POLICY.probe_timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("Probe timeout", extra={"component": "context_engine"})
        return ConflictResult(has_conflict=False, probe_skipped_reason="timeout")
    except Exception as e:
        logger.warning("Probe LLM call failed: %s", e, extra={"component": "context_engine"})
        return ConflictResult(has_conflict=False, probe_skipped_reason="parse_fail")

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
