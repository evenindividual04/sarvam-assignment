"""Generate a side-by-side delta report for an ablation run (BM25 vs hybrid).

Both legs share an `ablation_id`. We read all eval_runs rows tagged with it,
split by `retrieval_mode`, and compute per-metric / per-category mean deltas
plus failure-class transitions. Output JSON to eval/results/.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Optional

import aiosqlite

from agent.memory import DB_PATH

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

_METRICS = [
    "faithfulness_score",
    "answer_relevance_score",
    "context_precision_score",
    "citation_integrity_score",
    "claim_precision_score",
    "factual_accuracy_score",
]


def _safe_mean(values: list[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return mean(clean)


async def _load_rows(ablation_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT * FROM eval_runs WHERE ablation_id = ?", (ablation_id,),
        )
    return [dict(r) for r in rows]


async def generate_report(ablation_id: str) -> dict:
    rows = await _load_rows(ablation_id)
    if not rows:
        logger.warning("No eval rows found for ablation_id=%s", ablation_id)
        return {"ablation_id": ablation_id, "error": "no rows found"}

    bm25 = [r for r in rows if r.get("retrieval_mode") == "bm25"]
    hybrid = [r for r in rows if r.get("retrieval_mode") == "hybrid"]

    overall: dict[str, dict] = {}
    for m in _METRICS:
        b = _safe_mean([r.get(m) for r in bm25])
        h = _safe_mean([r.get(m) for r in hybrid])
        overall[m] = {
            "bm25": b,
            "hybrid": h,
            "delta": (h - b) if (b is not None and h is not None) else None,
        }

    # Per-category breakdown
    categories = sorted({r["category"] for r in rows})
    per_cat: dict[str, dict] = {}
    for cat in categories:
        cat_rows = {
            "bm25": [r for r in bm25 if r["category"] == cat],
            "hybrid": [r for r in hybrid if r["category"] == cat],
        }
        per_cat[cat] = {}
        for m in _METRICS:
            b = _safe_mean([r.get(m) for r in cat_rows["bm25"]])
            h = _safe_mean([r.get(m) for r in cat_rows["hybrid"]])
            per_cat[cat][m] = {
                "bm25": b,
                "hybrid": h,
                "delta": (h - b) if (b is not None and h is not None) else None,
            }

    # Per-question diff, indexed by question_id, sorted by Δfaithfulness desc.
    bm25_by_q = {r["question_id"]: r for r in bm25}
    hybrid_by_q = {r["question_id"]: r for r in hybrid}
    per_q: list[dict] = []
    for qid in sorted(set(bm25_by_q) & set(hybrid_by_q)):
        b = bm25_by_q[qid]
        h = hybrid_by_q[qid]
        b_faith = b.get("faithfulness_score")
        h_faith = h.get("faithfulness_score")
        delta = (
            (h_faith - b_faith)
            if (b_faith is not None and h_faith is not None)
            else None
        )
        per_q.append({
            "question_id": qid,
            "category": b.get("category"),
            "bm25_failure_class": b.get("failure_class"),
            "hybrid_failure_class": h.get("failure_class"),
            "delta_faithfulness": delta,
            "delta_context_precision": _delta(b.get("context_precision_score"), h.get("context_precision_score")),
            "delta_claim_precision": _delta(b.get("claim_precision_score"), h.get("claim_precision_score")),
        })
    per_q.sort(
        key=lambda r: (r["delta_faithfulness"] if r["delta_faithfulness"] is not None else 0.0),
        reverse=True,
    )

    # Failure-class transitions
    transitions = {"FAIL_to_PASS": 0, "PASS_to_FAIL": 0, "unchanged": 0, "other": 0}
    for qid in set(bm25_by_q) & set(hybrid_by_q):
        b_cls = bm25_by_q[qid].get("failure_class") or "PASS"
        h_cls = hybrid_by_q[qid].get("failure_class") or "PASS"
        if b_cls == h_cls:
            transitions["unchanged"] += 1
        elif b_cls != "PASS" and h_cls == "PASS":
            transitions["FAIL_to_PASS"] += 1
        elif b_cls == "PASS" and h_cls != "PASS":
            transitions["PASS_to_FAIL"] += 1
        else:
            transitions["other"] += 1

    report = {
        "ablation_id": ablation_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_bm25": len(bm25),
        "n_hybrid": len(hybrid),
        "overall": overall,
        "per_category": per_cat,
        "per_question": per_q,
        "failure_transitions": transitions,
    }

    out_path = _RESULTS_DIR / f"ablation_{ablation_id}.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    _print_summary(report, out_path)
    return report


def _delta(b: Optional[float], h: Optional[float]) -> Optional[float]:
    if b is None or h is None:
        return None
    return h - b


def _print_summary(report: dict, out_path: Path) -> None:
    print(f"\n{'='*72}")
    print(f"ABLATION REPORT  ·  {report['ablation_id']}")
    print(f"{'='*72}")
    print(f"BM25 rows: {report['n_bm25']}   Hybrid rows: {report['n_hybrid']}")
    print(f"\n{'metric':<28}{'bm25':>10}{'hybrid':>10}{'Δ':>10}")
    print("-" * 58)
    for m, vals in report["overall"].items():
        b = vals["bm25"]; h = vals["hybrid"]; d = vals["delta"]
        b_s = f"{b:.3f}" if b is not None else "—"
        h_s = f"{h:.3f}" if h is not None else "—"
        d_s = f"{d:+.3f}" if d is not None else "—"
        print(f"{m:<28}{b_s:>10}{h_s:>10}{d_s:>10}")
    t = report["failure_transitions"]
    print(
        f"\nTransitions: FAIL→PASS {t['FAIL_to_PASS']}   "
        f"PASS→FAIL {t['PASS_to_FAIL']}   unchanged {t['unchanged']}   other {t['other']}"
    )
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if len(sys.argv) < 2:
        print("usage: python eval/ablation_report.py <ablation_id>")
        sys.exit(1)
    asyncio.run(generate_report(sys.argv[1]))
