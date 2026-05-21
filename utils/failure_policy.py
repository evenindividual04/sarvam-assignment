"""Runtime failure budget policy with env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass

from utils.circuit_breaker import BreakerConfig, get_breaker


@dataclass
class FailurePolicy:
    plan_timeout_s: float = float(os.getenv("FAILURE_POLICY_PLAN_TIMEOUT_S", "25"))
    search_timeout_s: float = float(os.getenv("FAILURE_POLICY_SEARCH_TIMEOUT_S", "45"))
    fetch_timeout_s: float = float(os.getenv("FAILURE_POLICY_FETCH_TIMEOUT_S", "60"))
    select_timeout_s: float = float(os.getenv("FAILURE_POLICY_SELECT_TIMEOUT_S", "20"))
    synth_timeout_s: float = float(os.getenv("FAILURE_POLICY_SYNTH_TIMEOUT_S", "90"))
    max_total_turn_time_s: float = float(os.getenv("FAILURE_POLICY_MAX_TOTAL_TURN_TIME_S", "240"))
    max_retries_per_provider: int = int(os.getenv("FAILURE_POLICY_MAX_RETRIES_PER_PROVIDER", "3"))
    probe_timeout_s: float = float(os.getenv("FAILURE_POLICY_PROBE_TIMEOUT_S", "6"))
    # V3.2: hard cap on adaptive retrieval hops (1 = no second hop, 2 = up to one re-search).
    max_hops: int = int(os.getenv("FAILURE_POLICY_MAX_HOPS", "2"))


POLICY = FailurePolicy()


CIRCUIT_THRESHOLDS: dict[str, BreakerConfig] = {
    "groq":          BreakerConfig(threshold=5, window_s=60.0, open_duration_s=30.0),
    "gemini":        BreakerConfig(threshold=5, window_s=60.0, open_duration_s=30.0),
    "openrouter":    BreakerConfig(threshold=5, window_s=60.0, open_duration_s=30.0),
    "github_models": BreakerConfig(threshold=3, window_s=60.0, open_duration_s=30.0),
    "parallel":      BreakerConfig(threshold=4, window_s=60.0, open_duration_s=30.0),
    "tavily":        BreakerConfig(threshold=4, window_s=60.0, open_duration_s=30.0),
    "serper":        BreakerConfig(threshold=4, window_s=60.0, open_duration_s=30.0),
    "sarvam":        BreakerConfig(threshold=4, window_s=60.0, open_duration_s=30.0),
}


def register_all_breakers() -> None:
    cb = get_breaker()
    for name, cfg in CIRCUIT_THRESHOLDS.items():
        cb.register(name, cfg)


# Register at import time so any code path that imports failure_policy gets
# the breakers wired up. Idempotent (setdefault inside register).
register_all_breakers()
