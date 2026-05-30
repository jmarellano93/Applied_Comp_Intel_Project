#!/bin/bash
#SBATCH --job-name=mod7_full_sweep
#SBATCH --partition=top6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=logs/mod7_sweep_%j.log
#SBATCH --error=logs/mod7_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=FHNW_EMAIL_ADDRESS_HERE

# =============================================================================
# MOD7 Full Validation Sweep (Calculon)
# Runs the framework validation matrix against ALL available consensus rule
# files in the production directory.
#
# Expected coverage:
#   - 15 cells at N=3 full GP runs (linear/rectification/squashing/smooth/aggregation
#     × shallow/deep_narrow/funnel — all 15 of these were complete)
#   - 1 cell at N=3 (aggregation_deep_narrow once tonight's jobs finish)
#   - 1 cell at N=3 (trigonometric_shallow — was complete)
#   - 2 cells skipped (trigonometric_deep_narrow and trigonometric_funnel — no rules)
#
# Per cell: 25 Phase B datasets × 5 trial seeds × 11 methods (5 GP + 6 baselines)
#          = 1375 FNN trainings per cell.
# Total:    17 cells × 1375 = 23,375 FNN trainings.
# Estimated wall-clock: 6-12 hours depending on early-stopping rate.
# =============================================================================

set -e

cd "$HOME/Applied_Comp_Intel_Project"

echo "=== MOD7 FULL SWEEP ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo ""

PROD_RULES="experiment_modules/generated_files/GA_rule_files"
CONSENSUS_COUNT=$(ls "${PROD_RULES}"/Final_Discovered_Rules_*.txt 2>/dev/null | wc -l)
echo "Consensus files available: ${CONSENSUS_COUNT}"
if [ "$CONSENSUS_COUNT" -lt 16 ]; then
    echo "WARNING: Expected at least 16 consensus files; found ${CONSENSUS_COUNT}."
    echo "         Proceeding anyway. Cells without consensus will be skipped."
fi
echo ""

echo "=== Launching driver (full sweep, all topologies) ==="

conda run --no-capture-output -n aci_project python -u \
    experiment_modules/MOD7_pipeline_driver.py \
    --rule_directory "${PROD_RULES}"

echo ""
echo "=== Verifying outputs ==="
REPORT_DIR="experiment_modules/experimental_results_analysis_visualizations/reports/MOD7_validation_matrix"
REPORT_COUNT=$(ls "$REPORT_DIR"/statistical_validation_*.json 2>/dev/null | wc -l)
echo "JSON reports written: $REPORT_COUNT (expected: ~17)"
ls -la "$REPORT_DIR"/statistical_validation_*.json 2>/dev/null

echo ""
echo "=== FULL SWEEP COMPLETE: $(date) ==="
