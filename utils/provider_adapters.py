"""Typed provider adapter contracts for schema normalization + validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.models import SearchResult


@dataclass
class AdapterOutcome:
    ok: bool
    adapter_error_code: str | None = None
    reason: str | None = None


@dataclass
class SearchAdapterResult:
    results: list[SearchResult]
    outcome: AdapterOutcome


def _extract_provider_relevance(
    provider: str, item: dict[str, Any]
) -> tuple[float | None, str | None]:
    """Per-provider provider-supplied relevance extraction.

    Researched via current API docs (May 2026):
    - **Tavily** exposes `score` (float 0-1) per result. Treat as authoritative.
    - **Parallel** returns results pre-ranked but does NOT expose a per-result
      score — confidence/score fields are on the Task API, not Search. Use
      rank-derived fallback.
    - **Serper** scrapes Google SERPs; per-result `position` is rank, no score.
      Use rank-derived fallback.
    """
    if provider == "tavily":
        raw = item.get("score")
        if raw is None:
            return None, None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, None
        # Clamp into [0,1] for safety — Tavily sometimes returns slight overflows.
        return max(0.0, min(1.0, v)), "provider"
    # Parallel + Serper: no provider score; caller will fill from rank.
    return None, None


def normalize_search_items(
    provider: str,
    payload: dict[str, Any],
    now_iso: str,
    domain_fn,
) -> SearchAdapterResult:
    """Normalize provider payload into canonical SearchResult list, including
    a [0,1] relevance signal stamped with its provenance (provider vs rank).
    """
    items = payload.get("results") or payload.get("data") or payload.get("organic") or []
    if not isinstance(items, list):
        return SearchAdapterResult([], AdapterOutcome(False, f"{provider}_schema_mismatch", "items-not-list"))

    # First pass: extract structural fields without computing rank-based relevance
    # (we need to know the kept-count to denominate the rank score).
    raw_rows: list[tuple[dict, float | None, str | None]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link") or ""
        if not isinstance(url, str) or not url.strip():
            continue
        rel, src = _extract_provider_relevance(provider, item)
        raw_rows.append((item, rel, src))

    out: list[SearchResult] = []
    kept = len(raw_rows)
    for i, (item, rel, src) in enumerate(raw_rows):
        # Rank-derived fallback. Formula: 1.0 - (i / max(kept, 1)), so the
        # first result scores 1.0 and the worst-kept result scores 1/kept.
        # Bounded; never zero (the last result still carries a tiny weight).
        if rel is None:
            rel = round(1.0 - (i / max(kept, 1)), 4)
            src = "rank"

        # Parallel v1beta returns multi-excerpt arrays; older providers and
        # the legacy Parallel shape return single snippet/content strings.
        # Normalize both into a flat snippet by concatenating up to the first
        # three excerpts. Falls back to single-string fields when no array.
        excerpts = item.get("excerpts")
        snippet_val = ""
        if isinstance(excerpts, list) and excerpts:
            snippet_val = " ".join(str(e) for e in excerpts[:3] if isinstance(e, str))
        else:
            cand = item.get("snippet") or item.get("content") or ""
            if isinstance(cand, str):
                snippet_val = cand

        raw_content_val = (
            item.get("raw_content")
            or item.get("content")
            or item.get("text")
            or item.get("excerpt")
        )
        if raw_content_val is None and isinstance(excerpts, list) and excerpts:
            raw_content_val = "\n\n".join(str(e) for e in excerpts if isinstance(e, str))

        out.append(
            SearchResult(
                url=item["url"] if "url" in item else item.get("link", ""),
                title=item.get("title", "") if isinstance(item.get("title", ""), str) else "",
                snippet=snippet_val,
                domain=domain_fn(item.get("url") or item.get("link", "")),
                retrieved_at=now_iso,
                raw_content=raw_content_val,
                relevance=rel,
                relevance_source=src,
            )
        )

    if not out and items:
        return SearchAdapterResult([], AdapterOutcome(False, f"{provider}_schema_mismatch", "no-valid-items"))
    return SearchAdapterResult(out, AdapterOutcome(True))
