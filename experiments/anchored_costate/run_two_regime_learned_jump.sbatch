#!/bin/bash

#SBATCH --time=0-10:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=two_jump_min

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda || true
module unload python || true
module load cuda/11.2
module load python/anaconda-2021.05

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

JUMP_ROBUST_MODE="${JUMP_ROBUST_MODE:-residual}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-256}"
VALID_BATCHES="${VALID_BATCHES:-2}"
VALID_SEED="${VALID_SEED:-12345}"
LOG_EVERY="${LOG_EVERY:-500}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-float32}"
LR_VALUE="${LR_VALUE:-4e-4}"
LR_CONTROL="${LR_CONTROL:-4e-4}"
LR_JUMP="${LR_JUMP:-4e-4}"
JUMP_STEPS="${JUMP_STEPS:-1}"
JUMP_DUALITY_WEIGHT="${JUMP_DUALITY_WEIGHT:-1.0}"
ROOT_CONTROL_INIT="${ROOT_CONTROL_INIT:-surrogate}"
EXPORT_ROOT="${EXPORT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/experiments/anchored_costate/results/two_regime_learned_jump}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
EXPORT_DIR="${EXPORT_DIR:-${EXPORT_ROOT}/${JUMP_ROBUST_MODE}_seed${SEED}_${STAMP}}"
mkdir -p "${EXPORT_DIR}"

echo "Program starts $(date)"
echo "Jump robust mode: ${JUMP_ROBUST_MODE}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Validation: ${VALID_BATCHES} x ${VALID_BATCH_SIZE}"
echo "Seed: ${SEED}"
echo "Dtype: ${DTYPE}"
echo "Export: ${EXPORT_DIR}"
echo "Extra args: ${EXTRA_ARGS}"

python -u experiments/anchored_costate/train_two_regime_learned_jump.py \
    --jump_robust_mode "${JUMP_ROBUST_MODE}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --valid_batch_size "${VALID_BATCH_SIZE}" \
    --valid_batches "${VALID_BATCHES}" \
    --valid_seed "${VALID_SEED}" \
    --log_every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --lr_value "${LR_VALUE}" \
    --lr_control "${LR_CONTROL}" \
    --lr_jump "${LR_JUMP}" \
    --jump_steps "${JUMP_STEPS}" \
    --jump_duality_weight "${JUMP_DUALITY_WEIGHT}" \
    --root_control_init "${ROOT_CONTROL_INIT}" \
    --export "${EXPORT_DIR}" \
    ${EXTRA_ARGS}

echo "Program ends $(date)"
