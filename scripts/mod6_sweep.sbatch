#!/bin/bash
# ============================================================================
# MOD6 long-run sweep on FHNW Calculon
# ----------------------------------------------------------------------------
# Author : John M. Arellano
# Layout : 18-task job array  (3 topologies × 6 activations)
#          Each task runs 3 random seeds → 54 GP runs total.
# Runtime: ~24–36 h wall-clock (bound by slowest task).
# ----------------------------------------------------------------------------
# Usage:
#   mkdir -p logs                              # one-time, on Calculon
#   sbatch scripts/mod6_sweep.sbatch
#   squeue -u $USER                            # check status
#   tail -f logs/mod6_<JOBID>_<TASK>.log       # follow one task
# ============================================================================

#SBATCH --job-name=mogp_mod6
#SBATCH --partition=top6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-12:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/mod6_%A_%a.log
#SBATCH --error=logs/mod6_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=john.arellano@students.fhnw.ch

# ----------------------------------------------------------------------------
# Decode the array task ID into (activation, topology)
# 6 activations × 3 topologies = 18 pairs
# ----------------------------------------------------------------------------
ACTIVATIONS=(linear rectification squashing smooth aggregation trigonometric)
TOPOLOGIES=(shallow deep_narrow funnel)

ACT_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
TOP_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

ACTIVATION="${ACTIVATIONS[$ACT_IDX]}"
TOPOLOGY="${TOPOLOGIES[$TOP_IDX]}"

# ----------------------------------------------------------------------------
# Pre-flight info (visible in the log file)
# ----------------------------------------------------------------------------
echo "============================================================"
echo "Slurm job array  : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Activation       : ${ACTIVATION}"
echo "Topology         : ${TOPOLOGY}"
echo "Host             : $(hostname)"
echo "CPUs allocated   : ${SLURM_CPUS_PER_TASK}"
echo "Memory allocated : 16G"
echo "Started          : $(date)"
echo "============================================================"

# ----------------------------------------------------------------------------
# Run MOD6 in the project conda environment
# (Adjust the env name and project path if yours differ.)
# ----------------------------------------------------------------------------
PROJECT_DIR=$HOME/Applied_Comp_Intel_Project
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 2; }

conda run --no-capture-output -n aci_project python experiment_modules/MOD6_om_mogp_engine_final.py \
    --activations "${ACTIVATION}" \
    --topologies "${TOPOLOGY}" \
    --n_runs 3 \
    --population_size 100

RC=$?
echo "============================================================"
echo "Finished         : $(date)"
echo "Exit code        : ${RC}"
echo "============================================================"
exit ${RC}
