#!/bin/bash
#SBATCH --job-name=mod7_smoke
#SBATCH --partition=top6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mod7_smoke_%j.log
#SBATCH --error=logs/mod7_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=john.arellano@students.fhnw.ch

# =============================================================================
# MOD7 Smoke Test (Calculon)
# Runs the validation matrix in --quick_test mode against the production
# consensus rule directory. Covers all 6 activations × 1 topology (shallow)
# = 6 cells, 1 seed × 5 datasets each. Total: ~330 FNN trainings.
# Expected wall-clock: 5-15 minutes.
#
# Purpose: confirm MOD7 end-to-end pipeline works before launching the full
# 17-cell sweep. Validates: rule loading, equation compilation, FNN topology
# instantiation, baseline init schemes, Phase B dataset loading, and JSON
# report writing.
# =============================================================================

set -e

cd "$HOME/Applied_Comp_Intel_Project"

echo "=== MOD7 SMOKE TEST ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo ""

# Verify production rule directory has consensus files
PROD_RULES="experiment_modules/generated_files/GA_rule_files"
CONSENSUS_COUNT=$(ls "${PROD_RULES}"/Final_Discovered_Rules_*.txt 2>/dev/null | wc -l)
echo "Consensus files in ${PROD_RULES}: ${CONSENSUS_COUNT}"
if [ "$CONSENSUS_COUNT" -lt 1 ]; then
    echo "ERROR: No consensus files found. Did the aggregation job finish?"
    exit 1
fi

echo ""
echo "=== Launching driver in --quick_test mode (shallow topology only) ==="

conda run --no-capture-output -n aci_project python -u \
    experiment_modules/MOD7_pipeline_driver.py \
    --rule_directory "${PROD_RULES}" \
    --topologies shallow \
    --quick_test

echo ""
echo "=== Verifying outputs ==="
REPORT_DIR="experiment_modules/experimental_results_analysis_visualizations/reports/MOD7_validation_matrix"
if [ ! -d "$REPORT_DIR" ]; then
    echo "ERROR: Report directory not created: $REPORT_DIR"
    exit 1
fi

REPORT_COUNT=$(ls "$REPORT_DIR"/statistical_validation_shallow_*.json 2>/dev/null | wc -l)
echo "JSON reports written: $REPORT_COUNT"
ls -la "$REPORT_DIR"/statistical_validation_shallow_*.json 2>/dev/null || true

if [ "$REPORT_COUNT" -lt 4 ]; then
    echo "WARNING: Fewer than 4 reports (expected 5-6 — 4 if trigonometric_shallow lacks rules)."
    echo "         Inspect logs to confirm this is expected for your data."
fi

echo ""
echo "=== SMOKE TEST COMPLETE: $(date) ==="
