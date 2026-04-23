#!/usr/bin/env bash
# run_pipeline.sh — full Option-2 pipeline: optimize triggers → run experiments → build tables
#
# Usage (from repo root):
#   bash scripts/run_pipeline.sh
#   bash scripts/run_pipeline.sh --skip-optimize   # skip optimization if artifacts already exist
#   bash scripts/run_pipeline.sh --phase A          # cybersec only
#   bash scripts/run_pipeline.sh --phase B          # bio only
#   bash scripts/run_pipeline.sh --num-trials 3     # trials per condition (default 3)
#
# Logs are written to logs/pipeline_<timestamp>.log in addition to stdout.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PYTHON=".venv/bin/python"
PHASE="all"
NUM_TRIALS=3
SKIP_OPTIMIZE=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-optimize) SKIP_OPTIMIZE=1; shift ;;
        --phase)         PHASE="$2"; shift 2 ;;
        --num-trials)    NUM_TRIALS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Logging
mkdir -p logs results
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pipeline_${TIMESTAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================================"
echo "  RAG poisoning pipeline — $(date)"
echo "  phase=$PHASE  num_trials=$NUM_TRIALS  skip_optimize=$SKIP_OPTIMIZE"
echo "  log: $LOG"
echo "========================================================"

# Step 1 — Trigger optimization
if [[ "$SKIP_OPTIMIZE" -eq 0 ]]; then
    echo ""
    echo ">>> STEP 1: optimize triggers (per-query, Option 2)"
    OPTIMIZE_ARGS="--skip-done"
    if [[ "$PHASE" == "A" ]]; then
        OPTIMIZE_ARGS="$OPTIMIZE_ARGS --corpus cybersec"
    elif [[ "$PHASE" == "B" ]]; then
        OPTIMIZE_ARGS="$OPTIMIZE_ARGS --corpus bio"
    fi
    $PYTHON scripts/optimize_all_triggers.py $OPTIMIZE_ARGS
    echo ">>> STEP 1 done"
else
    echo ""
    echo ">>> STEP 1: skipping trigger optimization (--skip-optimize)"
fi

# Step 2 — Run experiments
echo ""
echo ">>> STEP 2: run experiments (phase=$PHASE, num_trials=$NUM_TRIALS)"
$PYTHON scripts/run_all_experiments.py \
    --phase "$PHASE" \
    --num-trials "$NUM_TRIALS" \
    --output-dir results
echo ">>> STEP 2 done"

# Step 3 — Build result tables
echo ""
echo ">>> STEP 3: build result tables"
$PYTHON -m src.analysis.make_results_table \
    --runs-file results/runs.jsonl \
    | tee results/results_table.txt
$PYTHON -m src.analysis.make_results_table \
    --runs-file results/runs.jsonl \
    --format csv \
    > results/results_table.csv
echo ">>> STEP 3 done"

echo ""
echo "========================================================"
echo "  Pipeline complete — $(date)"
echo "  Results: results/results_table.txt"
echo "           results/results_table.csv"
echo "  Full log: $LOG"
echo "========================================================"
