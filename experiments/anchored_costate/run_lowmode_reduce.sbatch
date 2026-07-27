#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=lowmode_reduce
#SBATCH --output=experiments/anchored_costate/logs/lowmode_reduce_%j.out
#SBATCH --error=experiments/anchored_costate/logs/lowmode_reduce_%j.err

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda || true
module unload python || true
module load cuda/11.2
module load python/anaconda-2021.05

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/experiments/anchored_costate/results/lowmode_map}"
GRID="${GRID:-32}"
DTYPE="${DTYPE:-float32}"

mkdir -p "${OUTPUT_ROOT}" experiments/anchored_costate/logs

echo "Reduce starts $(date)"
echo "Output root: ${OUTPUT_ROOT}"

python -u experiments/anchored_costate/collect_lowmode_results.py \
    --root "${OUTPUT_ROOT}" \
    --out_csv "${OUTPUT_ROOT}/lowmode_summary.csv" \
    --out_json "${OUTPUT_ROOT}/lowmode_summary.json" \
    --best_path "${OUTPUT_ROOT}/best_path.txt"

BEST_DIR="$(tr -d '\n' < "${OUTPUT_ROOT}/best_path.txt")"
if [[ -z "${BEST_DIR}" ]]; then
    echo "No completed lowmode run found; skipping spectral diagnostics."
    exit 0
fi

echo "Best dir: ${BEST_DIR}"

python -u experiments/anchored_costate/spectral_diagnostics.py \
    --grid "${GRID}" \
    --dtype "${DTYPE}" \
    --new_dir "${BEST_DIR}" \
    --logxi -2.995732273553991 \
    --out_dir "${OUTPUT_ROOT}/spectral_best_lowxi"

python -u experiments/anchored_costate/spectral_diagnostics.py \
    --grid "${GRID}" \
    --dtype "${DTYPE}" \
    --new_dir "${BEST_DIR}" \
    --logxi -2.302585092994046 \
    --out_dir "${OUTPUT_ROOT}/spectral_best_xi0p1"

echo "Reduce ends $(date)"
