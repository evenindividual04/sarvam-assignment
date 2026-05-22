"""Per-provider circuit breaker.

Tracks failures per provider over a sliding window; opens the circuit after a
threshold; auto-routes callers via CircuitOpenError. Tenacity retries are
expected to live INSIDE the wrapped function — one Tenacity-exhausted call
counts as one breaker failure.

Note on async safety: state lives in module-level dicts keyed by provider. In
the FastAPI app process, all turns share one event loop — state IS shared
across concurrent turns, so the breaker correctly trips for everyone when an
upstream provider goes down. For the eval CLI (sequential turns in one loop)
behavior is identical. Lock-free reads are acceptable because writes
(state transitions) are async-cooperative — no preemption between read+write.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import json as _json
import logging
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Awaitable, Callable, Literal

import httpx

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when the breaker is open and refuses to dispatch."""

    def __init__(self, provider: str):
        super().__init__(f"Circuit open for provider: {provider}")
        self.provider = provider


@dataclass
class ProviderHealth:
    name: str
    failures: collections.deque = field(default_factory=lambda: collections.deque(maxlen=20))
    state: Literal["closed", "open", "half_open"] = "closed"
    opened_at: float = 0.0
    half_open_inflight: bool = False


@dataclass(frozen=True)
class BreakerConfig:
    threshold: int
    window_s: float = 60.0
    open_duration_s: float = 30.0


class CircuitBreaker:
    def __init__(self) -> None:
        self._providers: dict[str, ProviderHealth] = {}
        self._configs: dict[str, BreakerConfig] = {}
        self._lock = asyncio.Lock()
        self._save_event: Callable[[dict], Awaitable[None]] | None = None

    def register(self, name: str, config: BreakerConfig) -> None:
        self._configs[name] = config
        self._providers.setdefault(name, ProviderHealth(name=name))

    def set_event_persister(self, fn: Callable[[dict], Awaitable[None]]) -> None:
        self._save_event = fn

    def reset(self, name: str | None = None) -> None:
        """Reset breaker state. Used by tests."""
        if name is None:
            self._providers = {n: ProviderHealth(name=n) for n in self._configs}
        elif name in self._providers:
            self._providers[name] = ProviderHealth(name=name)

    async def before(self, name: str) -> None:
        async with self._lock:
            health = self._providers.setdefault(name, ProviderHealth(name=name))
            cfg = self._configs.get(name, BreakerConfig(threshold=5))
            now = time.time()
            if health.state == "open":
                if now - health.opened_at >= cfg.open_duration_s:
                    await self._transition(health, "open", "half_open", "open_duration_elapsed")
                    health.half_open_inflight = True
                else:
                    raise CircuitOpenError(name)
            elif health.state == "half_open":
                if health.half_open_inflight:
                    raise CircuitOpenError(name)
                health.half_open_inflight = True

    async def on_success(self, name: str) -> None:
        async with self._lock:
            health = self._providers.get(name)
            if not health:
                return
            if health.state == "half_open":
                await self._transition(health, "half_open", "closed", "probe_success")
            health.failures.clear()
            health.half_open_inflight = False

    async def on_failure(self, name: str, exc: BaseException) -> None:
        async with self._lock:
            health = self._providers.setdefault(name, ProviderHealth(name=name))
            cfg = self._configs.get(name, BreakerConfig(threshold=5))
            now = time.time()
            while health.failures and now - health.failures[0] > cfg.window_s:
                health.failures.popleft()
            health.failures.append(now)
            if health.state == "half_open":
                await self._transition(health, "half_open", "open", f"probe_failed:{type(exc).__name__}")
                health.opened_at = now
                health.half_open_inflight = False
            elif health.state == "closed" and len(health.failures) >= cfg.threshold:
                await self._transition(health, "closed", "open", f"threshold_breached:{type(exc).__name__}")
                health.opened_at = now

    async def _transition(
        self, health: ProviderHealth, from_state: str, to_state: str, reason: str
    ) -> None:
        health.state = to_state
        logger.info(
            "circuit_transition",
            extra={
                "provider": health.name,
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
            },
        )
        if self._save_event:
            try:
                await self._save_event(
                    {
                        "event_id": str(uuid.uuid4()),
                        "provider": health.name,
                        "from_state": from_state,
                        "to_state": to_state,
                        "reason": reason,
                    }
                )
            except Exception as e:
                logger.warning(f"failed to persist circuit event: {e}")

    def state(self, name: str) -> str:
        return self._providers.get(name, ProviderHealth(name=name)).state


_BREAKER = CircuitBreaker()


def get_breaker() -> CircuitBreaker:
    return _BREAKER


_COUNTED_EXC_TYPES = (
    httpx.TimeoutException,
    asyncio.TimeoutError,
    _json.JSONDecodeError,
)


def _is_counted_failure(exc: BaseException) -> bool:
    if isinstance(exc, CircuitOpenError):
        return False
    if isinstance(exc, _COUNTED_EXC_TYPES):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    name = type(exc).__name__
    if name in ("ResourceExhausted", "RateLimitError", "APITimeoutError", "InternalServerError"):
        return True
    return False


def breaker(provider_name: str) -> Callable:
    """Decorator wrapping an async function with circuit-breaker logic.

    Tenacity @retry must remain INSIDE this decorator so the retry-exhausted
    exception counts as exactly one logical failure.
    """
    cb = get_breaker()

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.isasyncgenfunction(fn):
            @wraps(fn)
            async def gen_wrapper(*args: Any, **kwargs: Any):
                await cb.before(provider_name)
                try:
                    async for item in fn(*args, **kwargs):
                        yield item
                except BaseException as e:
                    if _is_counted_failure(e):
                        await cb.on_failure(provider_name, e)
                    raise
                else:
                    await cb.on_success(provider_name)
            return gen_wrapper

        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await cb.before(provider_name)
            try:
                result = await fn(*args, **kwargs)
            except BaseException as e:
                if _is_counted_failure(e):
                    await cb.on_failure(provider_name, e)
                raise
            else:
                await cb.on_success(provider_name)
                return result

        return wrapper

    return decorator
