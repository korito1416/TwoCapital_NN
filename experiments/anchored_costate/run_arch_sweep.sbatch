#!/bin/bash

#SBATCH --time=0-12:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=anch_arch

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda || true
module unload python || true
module load cuda/11.2
module load python/anaconda-2021.05

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-256}"
VALID_BATCHES="${VALID_BATCHES:-2}"
VALID_SEED="${VALID_SEED:-12345}"
LOG_EVERY="${LOG_EVERY:-250}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-float32}"
VARIANTS="${VARIANTS:-all}"
ROOT_CONTROL_INIT="${ROOT_CONTROL_INIT:-surrogate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/experiments/anchored_costate/results/arch_sweep}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Program starts $(date)"
echo "Steps: ${STEPS}"
echo "Variants: ${VARIANTS}"
echo "Output root: ${OUTPUT_ROOT}"

python -u experiments/anchored_costate/run_arch_sweep.py \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --valid_batch_size "${VALID_BATCH_SIZE}" \
    --valid_batches "${VALID_BATCHES}" \
    --valid_seed "${VALID_SEED}" \
    --log_every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --variants "${VARIANTS}" \
    --root_control_init "${ROOT_CONTROL_INIT}" \
    --output_root "${OUTPUT_ROOT}" \
    ${EXTRA_ARGS}

echo "Program ends $(date)"
