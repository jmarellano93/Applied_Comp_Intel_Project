#!/bin/bash
# ============================================================================
# MOD6 smoke test — run this BEFORE the full sweep
# ----------------------------------------------------------------------------
# Validates that:
#   1. Conda environment activates correctly
#   2. MOD6 imports all its dependencies
#   3. The PyTorch/DEAP/scikit-learn stack actually works on the compute node
#   4. The data cache from MOD3 is reachable
#
# Runs ONE (activation, topology) pair with ONE seed and a small population
# of 20 individuals for 5 generations. Should finish in 20–40 minutes.
# ============================================================================

#SBATCH --job-name=mod6_smoke
#SBATCH --partition=top6
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.log
#SBATCH --error=logs/smoke_%j.err

echo "============================================================"
echo "MOD6 smoke test"
echo "Host  : $(hostname)"
echo "Start : $(date)"
echo "============================================================"

PROJECT_DIR=$HOME/Applied_Comp_Intel_Project
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 2; }

conda run --no-capture-output -n aci_project python experiment_modules/MOD6_om_mogp_engine_final.py \
    --activations rectification \
    --topologies shallow \
    --n_runs 1 \
    --population_size 20 \
    --force_rerun

RC=$?
echo "============================================================"
echo "Finished : $(date)"
echo "Exit code: ${RC}"
echo "============================================================"
exit ${RC}
