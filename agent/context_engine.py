"""
Context pipeline:
  BM25 pre-filter → top 30 chunks
  FlashRank cross-encoder rerank → top 10
  3-factor scoring (0.6 relevance + 0.2 recency + 0.2 diversity) → final selection
  format_context_xml() → (xml_string, doc_map)
  detect_conflicts() → ConflictResult via Groq
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

from agent.models import ContextSnippet, ConflictResult, SearchResult
from utils.token_counter import count_tokens

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
    Three-factor score: 0.6 * relevance + 0.2 * recency + 0.2 * diversity.
    Weights sum to 1.0. No division by zero anywhere.
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

    # 4. Credibility (Domain reputation)
    reputable_suffixes = ('.edu', '.gov', 'wikipedia.org', 'nature.com', 'ncbi.nlm.nih.gov', 'arxiv.org', '.ac.uk')
    credibility = 1.2 if chunk.domain and any(chunk.domain.endswith(s) for s in reputable_suffixes) else 1.0

    chunk.recency_score = recency
    chunk.diversity_score = diversity
    return (0.6 * relevance + 0.2 * recency + 0.2 * diversity) * credibility


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
    top_indices = [i for i, _ in indexed[:_BM25_TOP_N]]
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


# ── Conflict detection ─────────────────────────────────────────────────────

async def detect_conflicts(chunks: list[ContextSnippet], query: str) -> ConflictResult:
    """Cheap Groq call (~150 tokens) to detect contradictory factual claims."""
    if len(chunks) < 2:
        return ConflictResult(has_conflict=False)

    preview = "\n\n".join([
        f"[Source {i+1} from {c.domain}]: {c.text[:300]}"
        for i, c in enumerate(chunks[:6])
    ])
    prompt = f"""Question: {query}
Sources:
{preview}

Do any sources make CONTRADICTORY factual claims about the same entity?
Look for: different numbers, dates, opposite conclusions.

Respond ONLY in JSON: {{"has_conflict": bool, "conflict_summary": "one sentence or null"}}"""

    try:
        from utils.provider_router import call_groq
        raw = await call_groq(prompt, max_tokens=100)
        # Parse JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            return ConflictResult(
                has_conflict=bool(data.get("has_conflict", False)),
                conflict_summary=data.get("conflict_summary"),
            )
    except Exception as e:
        logger.warning("Conflict detection failed: %s", e, extra={"component": "context_engine"})

    return ConflictResult(has_conflict=False)
