"""In-process cancellation primitive: a registry of asyncio.Event-backed tokens
keyed by turn_id. Used by the FastAPI /research endpoint to abort the orchestrator
pipeline on client disconnect or explicit cancel call."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class OperationCancelledError(Exception):
    """Raised by cooperative checks inside the orchestrator pipeline when cancellation has been requested."""
    pass


@dataclass
class CancellationToken:
    asyncio_event: asyncio.Event = field(default_factory=asyncio.Event)

    def is_set(self) -> bool:
        return self.asyncio_event.is_set()

    def check(self) -> None:
        if self.is_set():
            raise OperationCancelledError()

    def cancel(self) -> None:
        self.asyncio_event.set()

    async def wait(self) -> None:
        await self.asyncio_event.wait()


class CancellationRegistry:
    """In-process registry. Server-singleton; not shared across workers."""

    def __init__(self) -> None:
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = asyncio.Lock()

    async def register(self, turn_id: str) -> CancellationToken:
        async with self._lock:
            tok = CancellationToken()
            self._tokens[turn_id] = tok
            return tok

    async def cancel(self, turn_id: str) -> bool:
        async with self._lock:
            tok = self._tokens.get(turn_id)
            if not tok:
                return False
            tok.cancel()
            logger.info("turn_cancelled", extra={"turn_id": turn_id})
            return True

    async def release(self, turn_id: str) -> None:
        async with self._lock:
            self._tokens.pop(turn_id, None)


_REGISTRY = CancellationRegistry()


def get_registry() -> CancellationRegistry:
    return _REGISTRY
