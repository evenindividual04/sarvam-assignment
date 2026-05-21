"""Runtime failure budget policy with env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass


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


POLICY = FailurePolicy()
