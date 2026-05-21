"""
Thin wrapper around provider_router.synthesize().
Collects streaming chunks and returns (full_text, prompt_tokens, completion_tokens).
Also exposes an async streaming interface for the orchestrator.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from utils.cancellation import CancellationToken, OperationCancelledError

logger = logging.getLogger(__name__)


async def stream_synthesis(
    query: str,
    context_xml: str,
    doc_map: dict,
    history_text: str = "",
    conflict_note: Optional[str] = None,
    conflict_result=None,
    cancel_token: Optional[CancellationToken] = None,
) -> AsyncIterator[tuple[str, int, int]]:
    """Proxy to provider_router.synthesize(). Yields (text, prompt_tok, completion_tok).

    If cancel_token fires between chunks, stop yielding silently (the orchestrator
    catches OperationCancelledError at its top level for persistence)."""
    from utils.provider_router import synthesize
    async for chunk in synthesize(
        query, context_xml, doc_map, history_text, conflict_note, conflict_result
    ):
        yield chunk
        if cancel_token is not None and cancel_token.is_set():
            return
