"""In-process cancellation primitive: a registry of asyncio.Event-backed tokens
keyed by turn_id. Used by the FastAPI /research endpoint to abort the orchestrator
pipeline on client disconnect or explicit cancel call.

Phase 2: extended with an `approval_event` + `approved_payload` so the
orchestrator can pause between PLANNING and SEARCHING for human-in-the-loop
plan editing. We deliberately reuse the cancellation token rather than minting
a parallel `ApprovalToken` class so we keep one lifecycle and one registry —
cancellation MUST interrupt a pending approval wait.
"""
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
    # Phase 2: approval-gate primitives. `approval_event` is set by the
    # /research/approve/{turn_id} endpoint (or by `cancel()` so the wait
    # unblocks promptly). `approved_payload` carries the edited sub_queries
    # (or None if the user approved unchanged).
    approval_event: asyncio.Event = field(default_factory=asyncio.Event)
    approved_payload: dict | None = None
    # Phase 2: state of the approval flow — observed by /research/approve
    # for idempotency (409 on second call) and by the SSE disconnect-watcher
    # to suspend polling while paused.
    approval_status: str = "idle"  # "idle" | "waiting" | "approved" | "cancelled" | "timeout"
    # Phase 2: when True, the SSE disconnect-watcher in main.py skips its
    # `request.is_disconnected()` poll so a momentary SSE drop during a long
    # approval pause does not auto-cancel the turn.
    paused: bool = False
    # S1 fix: bind the owning session_id so /cancel and /approve can verify
    # the caller actually owns this turn. Without this, any client that
    # observes the turn_id in the first SSE frame can hijack the turn.
    session_id: str | None = None

    def is_set(self) -> bool:
        return self.asyncio_event.is_set()

    def check(self) -> None:
        if self.is_set():
            raise OperationCancelledError()

    def cancel(self) -> None:
        self.asyncio_event.set()
        # Unblock any pending approval wait so the orchestrator can exit promptly.
        # Leave `approved_payload` as None — `wait_for_approval` distinguishes
        # the cancellation path by checking `asyncio_event.is_set()`.
        if not self.approval_event.is_set():
            self.approval_event.set()

    async def wait(self) -> None:
        await self.asyncio_event.wait()

    async def wait_for_approval(
        self, timeout: float = 300.0
    ) -> tuple[bool, dict | None]:
        """Block until the approval endpoint resolves the gate, cancellation
        fires, or `timeout` seconds elapse.

        Returns:
            (approved: bool, edited_payload: dict | None)
              - approved=True iff /research/approve was called (payload may be
                None for "accept plan as-is").
              - approved=False on cancellation OR timeout. Caller distinguishes
                via `self.is_set()` (cancelled) vs not-set (timeout).
        """
        self.approval_status = "waiting"
        try:
            await asyncio.wait_for(self.approval_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.approval_status = "timeout"
            return False, None

        if self.is_set():
            self.approval_status = "cancelled"
            return False, None

        self.approval_status = "approved"
        return True, self.approved_payload


class CancellationRegistry:
    """In-process registry. Server-singleton; not shared across workers."""

    def __init__(self) -> None:
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = asyncio.Lock()

    async def register(self, turn_id: str, session_id: str | None = None) -> CancellationToken:
        async with self._lock:
            tok = CancellationToken(session_id=session_id)
            self._tokens[turn_id] = tok
            return tok

    async def get(self, turn_id: str, session_id: str | None = None) -> CancellationToken | None:
        """Phase 2: lookup without mutating registry state — used by
        /research/approve to resolve the approval gate.

        S1 fix: when `session_id` is provided, only return the token if it
        matches the session bound at registration. Mismatches return None so
        callers can map to 404 (no information disclosure)."""
        async with self._lock:
            tok = self._tokens.get(turn_id)
            if tok is None:
                return None
            if session_id is not None and tok.session_id is not None and tok.session_id != session_id:
                return None
            return tok

    async def cancel(self, turn_id: str, session_id: str | None = None) -> bool:
        """S1 fix: when `session_id` is provided, only cancel if it matches
        the session bound at registration. Mismatch returns False (caller
        maps to 404)."""
        async with self._lock:
            tok = self._tokens.get(turn_id)
            if not tok:
                return False
            if session_id is not None and tok.session_id is not None and tok.session_id != session_id:
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
