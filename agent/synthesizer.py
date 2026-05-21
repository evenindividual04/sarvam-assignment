"""
Thin wrapper around provider_router.synthesize().
Collects streaming chunks and returns (full_text, prompt_tokens, completion_tokens).
Also exposes an async streaming interface for the orchestrator.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


async def stream_synthesis(
    query: str,
    context_xml: str,
    doc_map: dict,
    history_text: str = "",
    conflict_note: Optional[str] = None,
) -> AsyncIterator[tuple[str, int, int]]:
    """Proxy to provider_router.synthesize(). Yields (text, prompt_tok, completion_tok)."""
    from utils.provider_router import synthesize
    async for chunk in synthesize(query, context_xml, doc_map, history_text, conflict_note):
        yield chunk
