#!/bin/bash
# Submit a single MOD6 job for one (activation, topology, seed) triple.
# Usage: ./mod6_single_job.sh <activation> <topology> <seed>
set -e

ACT=$1
TOP=$2
SEED=$3

if [[ -z "$ACT" || -z "$TOP" || -z "$SEED" ]]; then
    echo "Usage: $0 <activation> <topology> <seed>"
    exit 1
fi

sbatch \
    --job-name="mogp_${ACT}_${TOP}_s${SEED}" \
    --partition=top6 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=23:00:00 \
    --output="logs/mod6_single_${ACT}_${TOP}_s${SEED}_%j.log" \
    --error="logs/mod6_single_${ACT}_${TOP}_s${SEED}_%j.err" \
    --mail-type=END,FAIL \
    --mail-user=FHNW_EMAIL_ADDRESS_HERE \
    --wrap="cd \$HOME/Applied_Comp_Intel_Project && conda run --no-capture-output -n aci_project python -u experiment_modules/MOD6_om_mogp_engine_final.py --activations ${ACT} --topologies ${TOP} --gp_run_seeds ${SEED} --population_size 100 --skip_aggregation --force_rerun"
