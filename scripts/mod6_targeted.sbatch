#!/bin/bash
# ============================================================================
# MOD6 targeted re-run for 8 missing (activation, topology, seed) triples
# ----------------------------------------------------------------------------
# After the main sweep (job 145982) had Slurm bookkeeping issues, these 8
# triples didn't produce per-run files. This script re-runs just those.
# ============================================================================

#SBATCH --job-name=mogp_fixup
#SBATCH --partition=top6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/mod6_fixup_%A_%a.log
#SBATCH --error=logs/mod6_fixup_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=FHNW_EMAIL_ADDRESS_HERE

# 8 missing triples — each entry is "activation,topology,seed"
TRIPLES=(
    "aggregation,deep_narrow,43"
    "aggregation,deep_narrow,44"
    "trigonometric,deep_narrow,42"
    "trigonometric,deep_narrow,43"
    "trigonometric,deep_narrow,44"
    "trigonometric,funnel,42"
    "trigonometric,funnel,43"
    "trigonometric,funnel,44"
)

TRIPLE="${TRIPLES[$SLURM_ARRAY_TASK_ID]}"
ACTIVATION=$(echo "$TRIPLE" | cut -d',' -f1)
TOPOLOGY=$(echo "$TRIPLE" | cut -d',' -f2)
SEED=$(echo "$TRIPLE" | cut -d',' -f3)

echo "============================================================"
echo "Targeted re-run task: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Triple   : (${ACTIVATION}, ${TOPOLOGY}, seed=${SEED})"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "============================================================"

PROJECT_DIR=$HOME/Applied_Comp_Intel_Project
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 2; }

conda run --no-capture-output -n aci_project python -u experiment_modules/MOD6_om_mogp_engine_final.py \
    --activations "${ACTIVATION}" \
    --topologies "${TOPOLOGY}" \
    --gp_run_seeds "${SEED}" \
    --population_size 100 \
    --skip_aggregation \
    --force_rerun

RC=$?
echo "============================================================"
echo "Finished : $(date)"
echo "Exit code: ${RC}"
echo "============================================================"
exit ${RC}
