#!/bin/bash

#SBATCH --time=0-5:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --job-name=stochJump

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload python
module unload cuda
module load cuda/11.2
module load python/anaconda-2021.05

export PYTHONNOUSERSITE=1
export TF_ENABLE_ONEDNN_OPTS=0

if [ -z "${EXPORT_FOLDER:-}" ]; then
    echo "EXPORT_FOLDER must be set." >&2
    exit 2
fi

if [ -z "${XI:-}" ]; then
    echo "XI must be set." >&2
    exit 2
fi

SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-1}}"
N_PATHS="${N_PATHS:-100}"
YEARS="${YEARS:-60}"
DT="${DT:-0.08333333333333333}"
Y0="${Y0:-1.2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPORT_FOLDER}/SimulationResults}"

echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"
echo "Export folder: ${EXPORT_FOLDER}"
echo "Output root: ${OUTPUT_ROOT}"
echo "xi: ${XI}"
echo "seed: ${SEED}"
echo "n_paths: ${N_PATHS}"
echo "years: ${YEARS}"
start_time=$(date +%s)

python -u models/SimulationStochasticJumps.py \
    --export-folder "${EXPORT_FOLDER}" \
    --output-root "${OUTPUT_ROOT}" \
    --xi "${XI}" \
    --seed "${SEED}" \
    --n-paths "${N_PATHS}" \
    --years "${YEARS}" \
    --dt "${DT}" \
    --y0 "${Y0}"

echo "Program ends $(date)"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
