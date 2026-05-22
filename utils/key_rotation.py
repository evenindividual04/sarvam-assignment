"""Multi-key round-robin with 429-aware failover.

Generic helper for providers that benefit from running across multiple free-tier
API keys (Groq today; extensible to Tavily, Serper, etc).

Env var convention:
- Single key (legacy): ``GROQ_API_KEY=gsk_...``
- Multi-key (new):     ``GROQ_API_KEYS=gsk_aaa,gsk_bbb,gsk_ccc``

Behavior:
- If the multi-key env var is set, the rotator round-robins across those keys.
- If only the single-key env var is set, the rotator returns just that key.
- If both are set, the multi-key var wins (so operators can flip on multi-key
  without first having to clear the legacy var).
- On 429, callers should call ``mark_throttled(key, retry_after_s=...)`` so the
  key is skipped from rotation for ``retry_after_s`` seconds (or
  ``throttle_window_s`` default, 60s).
- If all keys are throttled, ``next_key()`` returns the least-recently-throttled
  key (the caller will retry after the underlying provider's backoff completes).
- Env is lazily re-read every ``KEY_ROTATION_REFRESH_S`` seconds (default 60s)
  so adding/removing keys via the HF Spaces secrets UI takes effect without
  a container restart. Set ``KEY_ROTATION_REFRESH_S=0`` to disable.
- When ``KEY_ROTATION_PERSIST=1``, throttle state survives restart by being
  persisted to SQLite (only SHA256 hashes of keys are stored, never raw keys).

Public API:
    rotator = KeyRotator("groq", legacy_var="GROQ_API_KEY", multi_var="GROQ_API_KEYS")
    key = rotator.next_key()
    rotator.mark_throttled(key, retry_after_s=10)
    rotator.mark_success(key)
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


_DEFAULT_ENV_REFRESH_INTERVAL_S = 60.0
_THROTTLE_GC_MAX_AGE_S = 600.0  # 10 minutes


_CREATE_THROTTLE_TABLE = """
CREATE TABLE IF NOT EXISTS key_throttle_state (
    provider TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    throttled_until_unix REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (provider, key_hash)
);
"""


def _hash_key(key: str) -> str:
    """SHA256 of an API key. Never store/log the raw key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _persist_enabled() -> bool:
    return os.environ.get("KEY_ROTATION_PERSIST", "0").strip() == "1"


def _db_path() -> str:
    # Late import to avoid a circular import at module load.
    from agent.memory import DB_PATH
    return DB_PATH


def _persist_throttle(provider: str, key_hash: str, throttled_until: float) -> None:
    """Sync write. Never raises."""
    try:
        path = _db_path()
        db_dir = os.path.dirname(path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with sqlite3.connect(path, timeout=2.0) as conn:
            conn.execute(_CREATE_THROTTLE_TABLE)
            conn.execute(
                """
                INSERT INTO key_throttle_state
                    (provider, key_hash, throttled_until_unix, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider, key_hash) DO UPDATE SET
                    throttled_until_unix = excluded.throttled_until_unix,
                    updated_at = excluded.updated_at
                """,
                (
                    provider,
                    key_hash,
                    throttled_until,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
    except Exception as exc:  # pragma: no cover - never block on persistence
        logger.warning(
            "key_rotation persist failed for %s: %s",
            provider,
            exc,
            extra={"component": "key_rotation"},
        )


def _load_persisted_throttles(provider: str) -> dict[str, float]:
    """Return {key_hash: throttled_until_unix} for ``provider``. GCs old rows."""
    out: dict[str, float] = {}
    try:
        path = _db_path()
        if not os.path.exists(path):
            return out
        with sqlite3.connect(path, timeout=2.0) as conn:
            conn.execute(_CREATE_THROTTLE_TABLE)
            cutoff = time.time() - _THROTTLE_GC_MAX_AGE_S
            conn.execute(
                "DELETE FROM key_throttle_state WHERE throttled_until_unix < ?",
                (cutoff,),
            )
            conn.commit()
            cur = conn.execute(
                "SELECT key_hash, throttled_until_unix FROM key_throttle_state "
                "WHERE provider = ?",
                (provider,),
            )
            for key_hash, until in cur.fetchall():
                out[key_hash] = float(until)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "key_rotation rehydrate failed for %s: %s",
            provider,
            exc,
            extra={"component": "key_rotation"},
        )
    return out


def _refresh_interval_s() -> float:
    """Read the refresh interval from env. 0 disables reload."""
    raw = os.environ.get("KEY_ROTATION_REFRESH_S")
    if raw is None or raw.strip() == "":
        return _DEFAULT_ENV_REFRESH_INTERVAL_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_ENV_REFRESH_INTERVAL_S


class KeyRotator:
    """Thread-safe round-robin key picker with per-key throttle tracking."""

    def __init__(
        self,
        provider_name: str,
        legacy_var: str,
        multi_var: str,
        throttle_window_s: float = 60.0,
    ) -> None:
        self.provider_name = provider_name
        self.legacy_var = legacy_var
        self.multi_var = multi_var
        self.throttle_window_s = throttle_window_s
        self._lock = threading.Lock()
        self._index = 0
        self._throttled_until: dict[str, float] = {}
        self._keys: list[str] = []
        self._last_env_check_at: float = 0.0
        # Initial load (also rehydrates persisted throttles if enabled).
        self._refresh_from_env(force=True)
        if _persist_enabled():
            self._rehydrate_throttles_locked_unsafe()

    @staticmethod
    def _load_keys(legacy_var: str, multi_var: str) -> list[str]:
        multi = os.environ.get(multi_var, "").strip()
        if multi:
            keys = [k.strip() for k in multi.split(",") if k.strip()]
            if keys:
                return keys
        single = os.environ.get(legacy_var, "").strip()
        return [single] if single else []

    def _refresh_from_env(self, *, force: bool = False) -> None:
        """Re-read env vars and reconcile ``self._keys``. Caller MUST hold the
        lock (or call during ``__init__`` before any concurrent access)."""
        # Caller-controlled lock semantics: this method is invoked either
        # from __init__ (no contention yet) or from inside ``next_key`` which
        # already holds the lock.
        now = time.time()
        interval = _refresh_interval_s()
        if not force:
            if interval == 0.0:
                return  # reload disabled
            if now - self._last_env_check_at < interval:
                return
        self._last_env_check_at = now
        new_keys = self._load_keys(self.legacy_var, self.multi_var)
        if new_keys == self._keys:
            return
        existing = set(self._keys)
        incoming = set(new_keys)
        removed = existing - incoming
        # Drop throttle entries for removed keys.
        for k in removed:
            self._throttled_until.pop(k, None)
        # Preserve fairness: keep existing keys in their order, then append
        # any newly added keys at the end. This keeps ``_index`` meaningful
        # without a reset.
        preserved = [k for k in self._keys if k in incoming]
        added = [k for k in new_keys if k not in existing]
        self._keys = preserved + added

    def _rehydrate_throttles_locked_unsafe(self) -> None:
        """Read persisted throttles. Caller MUST hold the lock OR be __init__."""
        persisted = _load_persisted_throttles(self.provider_name)
        if not persisted:
            return
        now = time.time()
        for key in self._keys:
            h = _hash_key(key)
            until = persisted.get(h)
            if until is not None and until > now:
                self._throttled_until[key] = until

    def reload(self) -> None:
        """Re-read env vars and reset internal state. Used by tests."""
        with self._lock:
            self._keys = self._load_keys(self.legacy_var, self.multi_var)
            self._index = 0
            self._throttled_until.clear()
            self._last_env_check_at = time.time()

    def has_keys(self) -> bool:
        with self._lock:
            self._refresh_from_env()
            return len(self._keys) > 0

    def key_count(self) -> int:
        with self._lock:
            self._refresh_from_env()
            return len(self._keys)

    def next_key(self) -> Optional[str]:
        """Return next key in round-robin order, skipping throttled keys.

        If every key is currently throttled, returns the least-recently-throttled
        key so the caller can still attempt (and let the underlying SDK's
        Retry-After-aware backoff do the waiting). Returns ``None`` if no keys
        are configured at all.
        """
        with self._lock:
            self._refresh_from_env()
            if not self._keys:
                return None
            now = time.time()
            for _ in range(len(self._keys)):
                key = self._keys[self._index % len(self._keys)]
                self._index += 1
                if self._throttled_until.get(key, 0.0) > now:
                    continue
                return key
            # All throttled — return least-recently-throttled.
            sorted_keys = sorted(
                self._keys, key=lambda k: self._throttled_until.get(k, 0.0)
            )
            return sorted_keys[0]

    def mark_throttled(
        self, key: str, retry_after_s: Optional[float] = None
    ) -> None:
        """Mark ``key`` as throttled. Honors ``retry_after_s`` (from the
        provider's Retry-After header) when present and positive; otherwise
        falls back to ``throttle_window_s``."""
        with self._lock:
            wait = (
                retry_after_s
                if retry_after_s is not None and retry_after_s > 0
                else self.throttle_window_s
            )
            throttled_until = time.time() + wait
            self._throttled_until[key] = throttled_until
            should_persist = _persist_enabled() and key in self._keys
            key_hash = _hash_key(key) if should_persist else None
        # Persist outside the in-memory lock (sync sqlite call). Never raises.
        if should_persist and key_hash is not None:
            _persist_throttle(self.provider_name, key_hash, throttled_until)

    def mark_success(self, key: str) -> None:
        """Clear any throttle state for ``key``."""
        with self._lock:
            self._throttled_until.pop(key, None)

    def snapshot(self) -> dict:
        """Telemetry snapshot: total keys + currently-throttled count."""
        with self._lock:
            now = time.time()
            throttled_now = sum(
                1 for _, t in self._throttled_until.items() if t > now
            )
            return {
                "provider": self.provider_name,
                "total_keys": len(self._keys),
                "throttled_now": throttled_now,
            }
