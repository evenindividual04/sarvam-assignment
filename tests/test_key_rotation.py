"""Tests for utils.key_rotation.KeyRotator."""
from __future__ import annotations

import os
import threading
import time

import pytest

from utils.key_rotation import KeyRotator, _hash_key


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure the relevant env vars are unset before each test, so each
    test sets exactly what it wants."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    monkeypatch.delenv("KEY_ROTATION_REFRESH_S", raising=False)
    monkeypatch.delenv("KEY_ROTATION_PERSIST", raising=False)
    yield


def _make(throttle_window_s: float = 60.0) -> KeyRotator:
    return KeyRotator(
        "groq",
        legacy_var="GROQ_API_KEY",
        multi_var="GROQ_API_KEYS",
        throttle_window_s=throttle_window_s,
    )


def test_single_key_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_only")
    r = _make()
    assert r.key_count() == 1
    assert r.next_key() == "gsk_only"
    # Round-robin with only one key just returns it repeatedly.
    assert r.next_key() == "gsk_only"


def test_multi_key_env_round_robins(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b,gsk_c")
    r = _make()
    assert r.key_count() == 3
    seq = [r.next_key() for _ in range(7)]
    assert seq == ["gsk_a", "gsk_b", "gsk_c", "gsk_a", "gsk_b", "gsk_c", "gsk_a"]


def test_legacy_var_used_when_multi_unset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_legacy")
    # GROQ_API_KEYS is intentionally unset (cleaned by fixture).
    r = _make()
    assert r.key_count() == 1
    assert r.next_key() == "gsk_legacy"


def test_multi_var_takes_precedence_over_legacy_when_both_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_legacy")
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b")
    r = _make()
    assert r.key_count() == 2
    assert r.next_key() == "gsk_a"
    assert r.next_key() == "gsk_b"


def test_throttled_key_skipped_in_rotation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b,gsk_c")
    r = _make()
    r.mark_throttled("gsk_b", retry_after_s=60)
    # gsk_b is skipped — rotation should yield a,c,a,c,...
    seq = [r.next_key() for _ in range(4)]
    assert "gsk_b" not in seq
    assert set(seq) == {"gsk_a", "gsk_c"}


def test_all_throttled_returns_least_recently_throttled(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b,gsk_c")
    r = _make()
    # gsk_a was throttled longest ago (smallest throttled_until value).
    now = time.time() + 60
    r._throttled_until = {
        "gsk_a": now + 5,
        "gsk_b": now + 10,
        "gsk_c": now + 15,
    }
    # All in the future → all throttled → fall back to least-recently-throttled.
    got = r.next_key()
    assert got == "gsk_a"


def test_mark_success_clears_throttle(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b")
    r = _make()
    r.mark_throttled("gsk_a", retry_after_s=60)
    r.mark_success("gsk_a")
    # gsk_a is back in rotation.
    seq = {r.next_key() for _ in range(4)}
    assert seq == {"gsk_a", "gsk_b"}


def test_mark_throttled_honors_retry_after(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b")
    r = _make(throttle_window_s=300.0)
    before = time.time()
    r.mark_throttled("gsk_a", retry_after_s=5.0)
    after = time.time()
    # The throttle window honored is 5s (not the 300s default).
    until = r._throttled_until["gsk_a"]
    assert before + 5.0 - 0.1 <= until <= after + 5.0 + 0.1


def test_snapshot_reports_throttle_count(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b,gsk_c")
    r = _make()
    r.mark_throttled("gsk_a", retry_after_s=60)
    r.mark_throttled("gsk_c", retry_after_s=60)
    snap = r.snapshot()
    assert snap["provider"] == "groq"
    assert snap["total_keys"] == 3
    assert snap["throttled_now"] == 2


def test_no_keys_returns_none():
    # Both env vars unset by fixture.
    r = _make()
    assert r.key_count() == 0
    assert not r.has_keys()
    assert r.next_key() is None


def test_empty_keys_in_multi_var_are_skipped(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,, ,gsk_b,")
    r = _make()
    assert r.key_count() == 2
    seq = [r.next_key() for _ in range(2)]
    assert sorted(seq) == ["gsk_a", "gsk_b"]


# ── Goal 1: lazy env reload ───────────────────────────────────────────────


def test_lazy_reload_picks_up_new_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a")
    monkeypatch.setenv("KEY_ROTATION_REFRESH_S", "0.05")
    r = _make()
    assert r.key_count() == 1
    assert r.next_key() == "gsk_a"
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b")
    time.sleep(0.1)
    assert r.key_count() == 2
    # gsk_b is appended at the end; round-robin reaches it.
    seen = {r.next_key() for _ in range(6)}
    assert "gsk_b" in seen


def test_lazy_reload_drops_removed_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b,gsk_c")
    monkeypatch.setenv("KEY_ROTATION_REFRESH_S", "0.05")
    r = _make()
    r.mark_throttled("gsk_b", retry_after_s=300)
    assert r.key_count() == 3
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_c")
    time.sleep(0.1)
    # Triggers refresh.
    seen = {r.next_key() for _ in range(6)}
    assert "gsk_b" not in seen
    assert seen == {"gsk_a", "gsk_c"}
    # And the removed key's throttle entry has been dropped.
    assert "gsk_b" not in r._throttled_until


def test_lazy_reload_can_be_disabled_via_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a")
    monkeypatch.setenv("KEY_ROTATION_REFRESH_S", "0")
    r = _make()
    assert r.key_count() == 1
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b,gsk_c")
    time.sleep(0.05)
    # Reload disabled — key count must NOT change even after a long wait.
    assert r.key_count() == 1
    seen = {r.next_key() for _ in range(8)}
    assert seen == {"gsk_a"}


def test_lazy_reload_thread_safe(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b")
    monkeypatch.setenv("KEY_ROTATION_REFRESH_S", "0.001")
    r = _make()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(200):
                k = r.next_key()
                assert k in {"gsk_a", "gsk_b", "gsk_c"}
        except Exception as exc:  # pragma: no cover - assert below
            errors.append(exc)

    def env_mutator() -> None:
        try:
            for i in range(20):
                if i % 2 == 0:
                    os.environ["GROQ_API_KEYS"] = "gsk_a,gsk_b,gsk_c"
                else:
                    os.environ["GROQ_API_KEYS"] = "gsk_a,gsk_b"
                time.sleep(0.002)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    threads.append(threading.Thread(target=env_mutator))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ── Goal 3: throttle-state persistence ─────────────────────────────────────


@pytest.fixture
def _temp_db(tmp_path, monkeypatch):
    """Point DB_PATH at a temp file for the duration of one test."""
    db_file = tmp_path / "test_rotator.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    # Reload the agent.memory module so DB_PATH picks up the new env.
    import importlib
    import agent.memory as memory
    importlib.reload(memory)
    yield str(db_file)
    importlib.reload(memory)  # restore default for subsequent tests


def test_throttle_state_persists_to_db_when_enabled(monkeypatch, _temp_db):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_aaa,gsk_bbb")
    monkeypatch.setenv("KEY_ROTATION_PERSIST", "1")
    r1 = _make(throttle_window_s=300.0)
    r1.mark_throttled("gsk_aaa", retry_after_s=300.0)
    # A new instance reads back the throttle state.
    r2 = _make(throttle_window_s=300.0)
    assert r2._throttled_until.get("gsk_aaa", 0.0) > time.time()


def test_throttle_state_disabled_by_default(monkeypatch, _temp_db):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_aaa,gsk_bbb")
    # KEY_ROTATION_PERSIST is NOT set.
    r1 = _make(throttle_window_s=300.0)
    r1.mark_throttled("gsk_aaa", retry_after_s=300.0)
    import sqlite3
    # DB may or may not exist; if it does, the table must be empty for our provider.
    if os.path.exists(_temp_db):
        with sqlite3.connect(_temp_db) as conn:
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM key_throttle_state WHERE provider = ?",
                    ("groq",),
                )
                assert cur.fetchone()[0] == 0
            except sqlite3.OperationalError:
                # Table never created — also acceptable.
                pass


def test_throttle_state_gc_old_entries(monkeypatch, _temp_db):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_aaa")
    monkeypatch.setenv("KEY_ROTATION_PERSIST", "1")
    import sqlite3
    # Manually insert a stale entry from 1 hour ago.
    from utils.key_rotation import _CREATE_THROTTLE_TABLE
    with sqlite3.connect(_temp_db) as conn:
        conn.execute(_CREATE_THROTTLE_TABLE)
        conn.execute(
            "INSERT INTO key_throttle_state VALUES (?, ?, ?, ?)",
            ("groq", _hash_key("gsk_aaa"), time.time() - 3600, "old"),
        )
        conn.commit()
    # Construct a new rotator — should GC the old entry on rehydrate.
    _ = _make()
    with sqlite3.connect(_temp_db) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM key_throttle_state WHERE provider = ?",
            ("groq",),
        )
        assert cur.fetchone()[0] == 0


def test_throttle_state_never_stores_raw_key(monkeypatch, _temp_db):
    raw = "gsk_supersecret_should_never_leak"
    monkeypatch.setenv("GROQ_API_KEYS", raw)
    monkeypatch.setenv("KEY_ROTATION_PERSIST", "1")
    r = _make(throttle_window_s=300.0)
    r.mark_throttled(raw, retry_after_s=300.0)
    import sqlite3
    with sqlite3.connect(_temp_db) as conn:
        cur = conn.execute("SELECT key_hash FROM key_throttle_state")
        rows = cur.fetchall()
    assert rows, "expected at least one row"
    for (key_hash,) in rows:
        assert raw not in key_hash
        assert key_hash == _hash_key(raw)
        assert len(key_hash) == 64  # SHA256 hex
