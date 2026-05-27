#!/bin/bash
#SBATCH --job-name=mod8_reporter
#SBATCH --partition=top6
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mod8_%j.log
#SBATCH --error=logs/mod8_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=john.arellano@students.fhnw.ch

# =============================================================================
# MOD8 Statistical Reporter Run (Calculon)
#
# Consumes MOD7's JSON validation artifacts and emits:
#   * Pooled Wilcoxon LaTeX tables (one per cell)        (legacy MOD8 behavior)
#   * Pooled accuracy/epochs/Pareto distribution plots   (legacy MOD8 behavior)
#   * Cluster-stratified Wilcoxon + binomial CSV         (NEW)
#   * Per-cell cluster-stratified LaTeX tables           (NEW)
#   * Cluster diagnostics (silhouette across K=2..5)     (NEW)
#   * Phase B PCA(2) figure colored by cluster           (NEW)
#
# Wall-clock: 5-15 minutes. Single node, no GPU required.
# =============================================================================

set -e

cd "$HOME/Applied_Comp_Intel_Project"

echo "=== MOD8 STATISTICAL REPORTER ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo ""

# Sanity checks
MOD7_DIR="experiment_modules/experimental_results_analysis_visualizations/reports/MOD7_validation_matrix"
JSON_COUNT=$(ls "${MOD7_DIR}"/statistical_validation_*.json 2>/dev/null | wc -l)
echo "MOD7 JSON artifacts available: ${JSON_COUNT}"
if [ "$JSON_COUNT" -lt 1 ]; then
    echo "ERROR: No MOD7 JSONs found. Run mod7_full_sweep.sbatch first."
    exit 1
fi

PHASE_B_CSV="experiment_modules/generated_files/Phase_B_Validation_Datasets.csv"
if [ ! -f "$PHASE_B_CSV" ]; then
    echo "WARNING: Phase B CSV missing at ${PHASE_B_CSV}; cluster-stratified analysis will be skipped."
fi

NORM_CSV="experiment_modules/generated_files/meta_feature_normalization_params.csv"
if [ ! -f "$NORM_CSV" ]; then
    echo "WARNING: Normalization params missing at ${NORM_CSV}; cluster analysis will degrade to fresh Phase-B fit."
fi

echo ""
echo "=== Launching MOD8 ==="

conda run --no-capture-output -n aci_project python -u \
    experiment_modules/MOD8_framework_statistical_reporter.py

echo ""
echo "=== Verifying outputs ==="
EARV="experiment_modules/experimental_results_analysis_visualizations"

echo "Pooled LaTeX tables:"
ls -l "${EARV}/reports/MOD8_statistical_reports/"*.tex 2>/dev/null | wc -l

echo "Pooled distribution plots:"
ls -l "${EARV}/visualizations/MOD8_distributions/"*.png 2>/dev/null | wc -l

echo "Cluster-stratified LaTeX tables:"
ls -l "${EARV}/reports/MOD8_cluster_stratified/Latex_Cluster_"*.tex 2>/dev/null | wc -l

echo "Cluster master CSV exists:"
ls -l "${EARV}/reports/MOD8_cluster_stratified/cluster_master_results.csv" 2>/dev/null || echo "  MISSING"

echo "Cluster diagnostics:"
ls -l "${EARV}/reports/MOD8_cluster_stratified/cluster_diagnostics.txt" 2>/dev/null || echo "  MISSING"

echo "PCA figure:"
ls -l "${EARV}/visualizations/MOD8_cluster_stratified/Phase_B_PCA_clusters.png" 2>/dev/null || echo "  MISSING"

echo ""
echo "=== MOD8 RUN COMPLETE: $(date) ==="
