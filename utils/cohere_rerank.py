"""
Cohere Rerank (rerank-v3.5) augmentation layer.

Optional reranker that sits BETWEEN FlashRank (lightweight cross-encoder) and
the 5-factor scoring pass in `agent/context_engine.py`. Gated on the
`COHERE_API_KEY` env var — if unset, the helper returns None and the engine
falls back to FlashRank's ordering.

Design choices:
- Never raise. Any failure (missing key, timeout, malformed response, quota
  exhaustion) returns None so the caller silently falls back to FlashRank.
- English-only routing — Cohere rerank-v3.5 nominally supports 100+ languages,
  but quality drops for low-resource Indic scripts and we'd rather not burn
  the 1k/mo free quota on Devanagari/Tamil/Bengali queries where FlashRank +
  BM25 already perform reasonably.
- Truncate chunk text to 2000 chars on the wire — Cohere counts tokens
  per-document and the free quota dries up fast if we send full chunks.
- 8s timeout — Cohere's p95 latency for ~30 docs is ~1-2s; 8s is generous and
  matches the surrounding select-stage budget without breaching it.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from agent.models import ContextSnippet

logger = logging.getLogger(__name__)

_COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
_COHERE_MODEL = "rerank-v3.5"
_MAX_DOC_CHARS = 2000

# Module-level singleton httpx client to avoid spinning up a new connection
# pool (and re-doing the TLS handshake) on every rerank call. Lazy-init on
# first use; close via :func:`close_cohere_client` on graceful shutdown.
_CLIENT: Optional[httpx.AsyncClient] = None


def _get_cohere_client() -> httpx.AsyncClient:
    """Return the module-level lazy-init ``httpx.AsyncClient``."""
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.AsyncClient()
    return _CLIENT


async def close_cohere_client() -> None:
    """Close the module-level client. Safe no-op if not initialized."""
    global _CLIENT
    if _CLIENT is not None and not _CLIENT.is_closed:
        try:
            await _CLIENT.aclose()
        except Exception:  # pragma: no cover — defensive
            pass
    _CLIENT = None


async def rerank_with_cohere(
    query: str,
    chunks: list[ContextSnippet],
    top_k: int = 10,
    timeout_s: float = 8.0,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[list[ContextSnippet]]:
    """Reorder `chunks` using Cohere Rerank. Return None on any failure.

    Quota-aware: records a request against `provider_usage` on success so
    `/health/providers` can surface the 1000/mo free-tier usage.
    """
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if not api_key:
        return None
    if not chunks:
        return None

    documents = [(c.text or "")[:_MAX_DOC_CHARS] for c in chunks]
    payload = {
        "model": _COHERE_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }
    headers = {
        "Authorization": f"bearer {api_key}",
        "Content-Type": "application/json",
    }

    http_client = client if client is not None else _get_cohere_client()
    try:
        resp = await http_client.post(
            _COHERE_RERANK_URL,
            json=payload,
            headers=headers,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # never raise into the selection path
        logger.warning(
            "Cohere rerank failed (%s); falling back to FlashRank",
            exc, extra={"component": "cohere_rerank"},
        )
        return None

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        return None

    reordered: list[ContextSnippet] = []
    for item in results:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            rel = item.get("relevance_score")
            if isinstance(rel, (int, float)):
                # Stash on the snippet for downstream telemetry; harmless
                # if the attribute doesn't exist on the dataclass.
                try:
                    chunk.cohere_relevance = float(rel)  # type: ignore[attr-defined]
                except Exception:
                    pass
            reordered.append(chunk)
    if not reordered:
        return None

    # Best-effort usage telemetry. Never block on it.
    try:
        from utils import provider_usage

        await provider_usage.record("cohere", requests=1)
    except Exception:
        pass

    return reordered
