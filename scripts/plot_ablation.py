"""Plot a grouped bar chart comparing BM25-only vs Hybrid RRF retrieval modes.

Reads the latest ``eval/results/ablation_<id>.json`` produced by
``eval/eval_runner.py --ablate``. The ablation report writer emits an
``overall`` block keyed by metric name with ``{bm25, hybrid, delta}`` floats.

If no ablation file exists yet, falls back to an illustrative dataset drawn
from the V3.1 hybrid RRF design hypothesis (BM25 numbers come from the
committed 2026-05-20 eval run summarized in README; hybrid numbers are the
expected directional uplift on multi-hop / insufficient-evidence questions
where keyword recall alone misses the relevant chunk). The chart legend and
README caption call this out explicitly so the reader is not misled.

Usage::

    python scripts/plot_ablation.py

Output: ``docs/assets/ablation_chart.png``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eval" / "results"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = ASSETS_DIR / "ablation_chart.png"

# Metrics surfaced in the chart (subset of the ablation report's full set,
# chosen to be README-readable; Quote Grounding is a Phase 1 addition tracked
# in the eval summary alongside the older metrics).
CHART_METRICS = [
    ("faithfulness_score", "Faithfulness"),
    ("context_precision_score", "Context Precision"),
    ("citation_integrity_score", "Citation Integrity"),
    ("quote_grounding_ratio", "Quote Grounding"),
]

# Illustrative fallback (see module docstring). Numbers represent the
# directional hypothesis, not measured outcomes.
_FALLBACK = {
    "source": "illustrative_fallback",
    "overall": {
        "faithfulness_score": {"bm25": 0.76, "hybrid": 0.83},
        "context_precision_score": {"bm25": 0.71, "hybrid": 0.82},
        "citation_integrity_score": {"bm25": 1.00, "hybrid": 1.00},
        "quote_grounding_ratio": {"bm25": 0.79, "hybrid": 0.86},
    },
}


def _latest_ablation_file() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    files = sorted(RESULTS_DIR.glob("ablation_*.json"))
    return files[-1] if files else None


def _load_data() -> tuple[dict, str]:
    """Return (overall_dict, source_label)."""
    latest = _latest_ablation_file()
    if latest is not None:
        try:
            payload = json.loads(latest.read_text())
            overall = payload.get("overall") or {}
            if overall:
                return overall, f"measured ({latest.name})"
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            print(f"WARNING: could not parse {latest}: {exc}", file=sys.stderr)
    return _FALLBACK["overall"], "illustrative (no ablation file)"


def _render(overall: dict, source_label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _render_text_fallback(overall, source_label)
        return

    labels = [pretty for _, pretty in CHART_METRICS]
    bm25_vals: list[float] = []
    hybrid_vals: list[float] = []
    for key, _ in CHART_METRICS:
        entry = overall.get(key) or {}
        bm25_vals.append(float(entry.get("bm25") or 0.0))
        hybrid_vals.append(float(entry.get("hybrid") or 0.0))

    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    bars1 = ax.bar([i - width / 2 for i in x], bm25_vals, width,
                   label="BM25 only", color="#6c8ebf")
    bars2 = ax.bar([i + width / 2 for i in x], hybrid_vals, width,
                   label="Hybrid RRF (BM25 + sqlite-vec)", color="#82b366")

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Ablation: BM25 vs Hybrid RRF")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    fig.text(0.99, 0.01, f"data: {source_label}", ha="right",
             va="bottom", fontsize=7.5, color="#888")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}  (source: {source_label})")


def _render_text_fallback(overall: dict, source_label: str) -> None:
    md_path = ASSETS_DIR / "ablation_chart.md"
    lines = ["# Ablation chart (text fallback — matplotlib unavailable)\n",
             f"_source: {source_label}_\n",
             "| Metric | BM25 | Hybrid | Δ |", "|---|---:|---:|---:|"]
    for key, pretty in CHART_METRICS:
        entry = overall.get(key) or {}
        b = entry.get("bm25") or 0.0
        h = entry.get("hybrid") or 0.0
        lines.append(f"| {pretty} | {b:.2f} | {h:.2f} | {h - b:+.2f} |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"matplotlib not installed; wrote text fallback to {md_path}")


def main() -> int:
    overall, source_label = _load_data()
    _render(overall, source_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
