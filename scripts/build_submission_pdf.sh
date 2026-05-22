#!/usr/bin/env bash
# Build submission.pdf from README.md for Sarvam FDSE submission.
# Assignment requires a single PDF containing links + assumptions + deliverables.
#
# Requires: pandoc + a LaTeX engine (mactex / texlive-xetex).
# On macOS:   brew install pandoc; brew install --cask mactex-no-gui
# On Linux:   apt install pandoc texlive-xetex texlive-fonts-recommended
#
# Usage:
#   bash scripts/build_submission_pdf.sh                 # → submission.pdf
#   bash scripts/build_submission_pdf.sh README.md out.pdf
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT="${1:-README.md}"
OUTPUT="${2:-submission.pdf}"

if ! command -v pandoc >/dev/null 2>&1; then
    echo "error: pandoc not found. Install: brew install pandoc"
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "error: $INPUT not found"
    exit 1
fi

ENGINE="xelatex"
if ! command -v "$ENGINE" >/dev/null 2>&1; then
    if command -v wkhtmltopdf >/dev/null 2>&1; then
        ENGINE="wkhtmltopdf"
    elif command -v weasyprint >/dev/null 2>&1; then
        ENGINE="weasyprint"
    else
        echo "warn: no xelatex/wkhtmltopdf/weasyprint; falling back to pandoc default"
        ENGINE=""
    fi
fi

ARGS=(
    "$INPUT"
    -o "$OUTPUT"
    --metadata=title:"Sarvam Deep Research Agent — Submission"
    --metadata=author:"$(git config user.name 2>/dev/null || echo Submission)"
    --metadata=date:"$(date +%Y-%m-%d)"
    --toc
    --toc-depth=2
    -V geometry:margin=0.85in
    -V fontsize=10pt
    -V colorlinks=true
    -V linkcolor=teal
    -V urlcolor=teal
    --highlight-style=tango
)

if [[ -n "$ENGINE" ]]; then
    ARGS+=(--pdf-engine="$ENGINE")
fi

echo "building $OUTPUT from $INPUT (engine: ${ENGINE:-default})..."
pandoc "${ARGS[@]}"
echo "ok → $OUTPUT ($(wc -c < "$OUTPUT" | tr -d ' ') bytes)"
