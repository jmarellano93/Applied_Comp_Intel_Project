#!/bin/bash
# ============================================================================
# MOD6 long-run sweep on FHNW Calculon  (per-seed job array)
# ----------------------------------------------------------------------------
# Author : John M. Arellano
# Layout : 51-task job array  (17 pairs × 3 seeds per pair).
#          The (rectification, shallow) pair is excluded because a consensus
#          artifact for it already exists in generated_files/GA_rule_files/.
#          Each task runs ONE seed for ONE (topology, activation) pair and
#          writes only the per-run file. Consensus aggregation is deferred
#          to a separate dependent job (see mod6_aggregate.sbatch).
# ----------------------------------------------------------------------------
# Per-task estimate    : 4-6 hours (6x safety margin under the 24h limit)
# Per-task resources   : 4 CPUs, 16 GB RAM, top6 partition
# Sweep wall-clock     : ~20-40 hours depending on cluster occupancy
# ----------------------------------------------------------------------------
# Usage:
#   cd ~/Applied_Comp_Intel_Project
#   mkdir -p logs
#
#   # Launch the per-seed array; capture the array job ID for the dependency
#   ARRAY_JOB=$(sbatch --parsable scripts/mod6_sweep.sbatch)
#   echo "Array job: $ARRAY_JOB"
#
#   # Launch the aggregation job to run after the array fully completes
#   sbatch --dependency=afterok:${ARRAY_JOB} scripts/mod6_aggregate.sbatch
#
#   # Monitor
#   squeue -u $USER -t all
#   tail -f logs/mod6_${ARRAY_JOB}_0.log
# ============================================================================

#SBATCH --job-name=mogp_sweep
#SBATCH --partition=top6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --array=0-50
#SBATCH --output=logs/mod6_%A_%a.log
#SBATCH --error=logs/mod6_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=john.arellano@students.fhnw.ch

# ----------------------------------------------------------------------------
# Decode task ID into (activation, topology, seed)
#
# Pair list excludes (rectification, shallow) — already has consensus on disk.
# Total pairs : 17  →  51 tasks = 17 × 3 seeds
#
# task_id  0..50
# pair_idx = task_id // 3     →  index into PAIRS[]   (0..16)
# seed_idx = task_id %  3     →  index into SEEDS[]   (0..2)
# ----------------------------------------------------------------------------

SEEDS=(42 43 44)

PAIRS=(
    "linear,shallow"
    "linear,deep_narrow"
    "linear,funnel"
    "rectification,deep_narrow"
    "rectification,funnel"
    "squashing,shallow"
    "squashing,deep_narrow"
    "squashing,funnel"
    "smooth,shallow"
    "smooth,deep_narrow"
    "smooth,funnel"
    "aggregation,shallow"
    "aggregation,deep_narrow"
    "aggregation,funnel"
    "trigonometric,shallow"
    "trigonometric,deep_narrow"
    "trigonometric,funnel"
)

PAIR_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

PAIR="${PAIRS[$PAIR_IDX]}"
ACTIVATION="${PAIR%%,*}"
TOPOLOGY="${PAIR##*,}"
SEED="${SEEDS[$SEED_IDX]}"

echo "============================================================"
echo "Slurm task       : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Pair             : (${ACTIVATION}, ${TOPOLOGY})"
echo "Seed             : ${SEED}"
echo "Host             : $(hostname)"
echo "CPUs allocated   : ${SLURM_CPUS_PER_TASK}"
echo "Started          : $(date)"
echo "============================================================"

PROJECT_DIR=$HOME/Applied_Comp_Intel_Project
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 2; }

# -u forces unbuffered stdout so the log shows live progress.
# --skip_aggregation: this single-seed job writes per_run_archive/ only.
# --force_rerun: ensure the job doesn't skip itself if a stray consensus file
#                exists (it shouldn't — only rectification+shallow does and
#                that pair is excluded from this array).
conda run --no-capture-output -n aci_project python -u experiment_modules/MOD6_om_mogp_engine_final.py \
    --activations "${ACTIVATION}" \
    --topologies "${TOPOLOGY}" \
    --gp_run_seeds "${SEED}" \
    --population_size 100 \
    --skip_aggregation \
    --force_rerun

RC=$?
echo "============================================================"
echo "Finished         : $(date)"
echo "Exit code        : ${RC}"
echo "============================================================"
exit ${RC}
