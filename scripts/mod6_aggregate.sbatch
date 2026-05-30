#!/bin/bash
# ============================================================================
# MOD6 consensus aggregation pass on FHNW Calculon
# ----------------------------------------------------------------------------
# Purpose: After the per-seed job array (mod6_sweep.sbatch) writes 51 per-run
#          files to generated_files/GA_rule_files/per_run_archive/, this job
#          reads them, aggregates per-pair Pareto fronts into consensus fronts
#          via NSGA-II re-sort, and writes the canonical consensus artifacts
#          that MOD7 / MOD9 will glob for downstream.
# ----------------------------------------------------------------------------
# Runtime : ~1-5 minutes (pure file I/O + in-memory DEAP operations; no GP)
# Resources: 2 CPUs, 4 GB RAM, performance partition (lightweight)
# ----------------------------------------------------------------------------
# Submit AFTER the array job, with a dependency so this only runs if every
# per-seed task completed successfully:
#
#   ARRAY_JOB=$(sbatch --parsable scripts/mod6_sweep.sbatch)
#   sbatch --dependency=afterok:${ARRAY_JOB} scripts/mod6_aggregate.sbatch
# ============================================================================

#SBATCH --job-name=mogp_aggregate
#SBATCH --partition=performance
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mod6_aggregate_%j.log
#SBATCH --error=logs/mod6_aggregate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=john.arellano@students.fhnw.ch

echo "============================================================"
echo "MOD6 consensus aggregation pass"
echo "Slurm job  : ${SLURM_JOB_ID}"
echo "Host       : $(hostname)"
echo "Started    : $(date)"
echo "============================================================"

PROJECT_DIR=$HOME/Applied_Comp_Intel_Project
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 2; }

# --aggregate_from_disk: skip GP entirely, read per_run_archive/, write
#                        canonical consensus artifacts for every pair.
# All 6 activations x 3 topologies are considered. Pairs whose consensus
# file already exists are skipped (this protects the pre-existing
# rectification+shallow result from being recomputed).
conda run --no-capture-output -n aci_project python -u experiment_modules/MOD6_om_mogp_engine_final.py \
    --activations linear rectification squashing smooth aggregation trigonometric \
    --topologies shallow deep_narrow funnel \
    --aggregate_from_disk

RC=$?
echo "============================================================"
echo "Finished   : $(date)"
echo "Exit code  : ${RC}"
echo "============================================================"
exit ${RC}
