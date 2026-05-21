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


def normalize_search_items(
    provider: str,
    payload: dict[str, Any],
    now_iso: str,
    domain_fn,
) -> SearchAdapterResult:
    """Normalize provider payload into canonical SearchResult list."""
    items = payload.get("results") or payload.get("data") or payload.get("organic") or []
    if not isinstance(items, list):
        return SearchAdapterResult([], AdapterOutcome(False, f"{provider}_schema_mismatch", "items-not-list"))

    out: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link") or ""
        if not isinstance(url, str) or not url.strip():
            continue
        out.append(
            SearchResult(
                url=url,
                title=item.get("title", "") if isinstance(item.get("title", ""), str) else "",
                snippet=(item.get("snippet") or item.get("content") or "") if isinstance((item.get("snippet") or item.get("content") or ""), str) else "",
                domain=domain_fn(url),
                retrieved_at=now_iso,
                raw_content=item.get("raw_content") or item.get("content") or item.get("text") or item.get("excerpt"),
            )
        )

    if not out and items:
        return SearchAdapterResult([], AdapterOutcome(False, f"{provider}_schema_mismatch", "no-valid-items"))
    return SearchAdapterResult(out, AdapterOutcome(True))
