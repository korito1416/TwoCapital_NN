#!/bin/bash

#SBATCH --time=0-6:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=svrd_decomp

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

N_PATHS="${N_PATHS:-512}"
YEARS="${YEARS:-80}"
DT="${DT:-0.08333333333333333}"
SEED="${SEED:-1}"
EPS_LOGR="${EPS_LOGR:-0.0001}"
Y0="${Y0:-1.2}"
BATCH_SIZE="${BATCH_SIZE:-${N_PATHS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPORT_FOLDER}/SVRDDecomposition}"
PROGRESS_EVERY="${PROGRESS_EVERY:-120}"

echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"
echo "Export folder: ${EXPORT_FOLDER}"
echo "Output root: ${OUTPUT_ROOT}"
echo "xi: ${XI}"
echo "seed: ${SEED}"
echo "n_paths: ${N_PATHS}"
echo "years: ${YEARS}"
echo "dt: ${DT}"
start_time=$(date +%s)

python -u compute_svrd_decomposition.py \
    --export-folder "${EXPORT_FOLDER}" \
    --output-root "${OUTPUT_ROOT}" \
    --xi "${XI}" \
    --seed "${SEED}" \
    --n-paths "${N_PATHS}" \
    --years "${YEARS}" \
    --dt "${DT}" \
    --eps-logr "${EPS_LOGR}" \
    --y0 "${Y0}" \
    --batch-size "${BATCH_SIZE}" \
    --progress-every "${PROGRESS_EVERY}"

echo "Program ends $(date)"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
