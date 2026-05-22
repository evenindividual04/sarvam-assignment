"""Complete the hybrid leg of the running ablation on a stratified 13-question
subset, then merge with the existing partial hybrid JSONL so the ablation
report sees a single combined hybrid leg.

Reads the stratified subset + ablation_id from /tmp/subset_qids.json (built by
the inline shell snippet that selected the questions). Writes a fresh hybrid
JSONL, then concatenates it onto the existing partial hybrid JSONL.

After this finishes, the ablation report can be regenerated against ablation_id.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXISTING_HYBRID_JSONL = ROOT / "eval/results/eval_20260522_130810.jsonl"
SUBSET_PATH = Path("/tmp/subset_qids.json")


async def main() -> None:
    payload = json.loads(SUBSET_PATH.read_text())
    ablation_id = payload["ablation_id"]
    question_ids = payload["question_ids"]
    print(f"ablation_id:  {ablation_id}")
    print(f"running hybrid on {len(question_ids)} stratified question IDs")
    print(f"  ids: {question_ids}\n")

    os.environ["HYBRID_RETRIEVAL"] = "1"
    os.environ.setdefault("JUDGE_PROVIDER", "cerebras")
    os.environ.setdefault("JUDGE_MODEL", "qwen-3-235b-a22b-instruct-2507")

    from eval.eval_runner import run_eval

    run_ts = await run_eval(
        ablation_id=ablation_id,
        question_ids=question_ids,
        cross_family_judge=False,  # cross-family already sampled in BM25 leg
    )
    print(f"\nsubset hybrid run finished: run_ts={run_ts}")

    new_jsonl = ROOT / f"eval/results/eval_{run_ts}.jsonl"
    if not new_jsonl.exists():
        print(f"WARNING: expected new JSONL at {new_jsonl} not found")
        return

    # Merge: append new rows onto the existing partial hybrid JSONL so the
    # ablation report sees a single combined hybrid leg.
    backup = EXISTING_HYBRID_JSONL.with_suffix(".jsonl.bak")
    shutil.copy(EXISTING_HYBRID_JSONL, backup)
    print(f"backed up existing hybrid JSONL → {backup.name}")

    with open(EXISTING_HYBRID_JSONL, "a") as out, open(new_jsonl) as src:
        rows_appended = 0
        for line in src:
            out.write(line)
            rows_appended += 1
    print(f"appended {rows_appended} new hybrid rows → {EXISTING_HYBRID_JSONL.name}")

    # Regenerate the ablation report
    from eval.ablation_report import generate_report
    await generate_report(ablation_id)
    print(f"\nablation report regenerated for {ablation_id}")


if __name__ == "__main__":
    asyncio.run(main())
