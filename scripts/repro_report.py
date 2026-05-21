from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.failure_policy import POLICY
from utils.prompt_registry import PROMPT_REGISTRY


def _git_info() -> dict:
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def _deps() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _env_knobs() -> dict:
    keys = [
        "CONTEXT_SELECTION_STRATEGY",
        "GEMINI_MODEL",
        "HISTORY_RETRIEVAL_LIMIT",
        "FAILURE_POLICY_PLAN_TIMEOUT_S",
        "FAILURE_POLICY_SEARCH_TIMEOUT_S",
        "FAILURE_POLICY_FETCH_TIMEOUT_S",
        "FAILURE_POLICY_SELECT_TIMEOUT_S",
        "FAILURE_POLICY_SYNTH_TIMEOUT_S",
        "FAILURE_POLICY_MAX_TOTAL_TURN_TIME_S",
        "FAILURE_POLICY_MAX_RETRIES_PER_PROVIDER",
    ]
    return {k: os.getenv(k, "<unset>") for k in keys}


def _prompt_ids() -> dict:
    return {k: v["id"] for k, v in PROMPT_REGISTRY.items() if isinstance(v, dict) and "id" in v}


def _smoke_summary(query: str) -> dict:
    try:
        out = subprocess.check_output([sys.executable, "scripts/smoke_run.py", "--query", query], text=True)
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        return {"status": "ok", "output": lines[-6:]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic reproducibility report")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--query", default="What is the current repo rate set by the RBI?")
    args = parser.parse_args()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "deps": _deps(),
        "env_knobs": _env_knobs(),
        "prompt_versions": _prompt_ids(),
        "failure_policy": POLICY.__dict__,
    }

    if args.mode in {"quick", "full"}:
        report["smoke"] = _smoke_summary(args.query)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
