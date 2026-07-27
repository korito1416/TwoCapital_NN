#!/bin/bash

#SBATCH --time=0-08:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=cst_rob_root

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda || true
module unload python || true
module load cuda/11.2
module load python/anaconda-2021.05

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ROBUST_MODE="${ROBUST_MODE:-residual}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-512}"
VALID_BATCHES="${VALID_BATCHES:-2}"
VALID_SEED="${VALID_SEED:-12345}"
LOG_EVERY="${LOG_EVERY:-500}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-float32}"
LR_COSTATE="${LR_COSTATE:-4e-4}"
LR_CONTROL="${LR_CONTROL:-4e-4}"
LR_ROBUST="${LR_ROBUST:-4e-4}"
ROBUST_STEPS="${ROBUST_STEPS:-1}"
CONTROL_STEPS="${CONTROL_STEPS:-1}"
LAMBDA_CURL="${LAMBDA_CURL:-1.0}"
DUALITY_WEIGHT="${DUALITY_WEIGHT:-1.0}"
ROBUST_GAP_WEIGHT="${ROBUST_GAP_WEIGHT:-0.0}"
QUAD_POINTS="${QUAD_POINTS:-8}"
WIDTH="${WIDTH:-32}"
LAYERS="${LAYERS:-4}"
ROBUST_WIDTH="${ROBUST_WIDTH:-64}"
ROBUST_DEPTH="${ROBUST_DEPTH:-3}"
EXPORT_ROOT="${EXPORT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/experiments/anchored_costate/results/costate_robust_root}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
EXPORT_DIR="${EXPORT_DIR:-${EXPORT_ROOT}/${ROBUST_MODE}_seed${SEED}_${STAMP}}"
mkdir -p "${EXPORT_DIR}"

echo "Program starts $(date)"
echo "Robust mode: ${ROBUST_MODE}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Validation: ${VALID_BATCHES} x ${VALID_BATCH_SIZE}"
echo "Seed: ${SEED}"
echo "Dtype: ${DTYPE}"
echo "Export: ${EXPORT_DIR}"
echo "Extra args: ${EXTRA_ARGS}"

python -u experiments/anchored_costate/costate_robust_root.py \
    --robust_mode "${ROBUST_MODE}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --valid_batch_size "${VALID_BATCH_SIZE}" \
    --valid_batches "${VALID_BATCHES}" \
    --valid_seed "${VALID_SEED}" \
    --log_every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --lr_costate "${LR_COSTATE}" \
    --lr_control "${LR_CONTROL}" \
    --lr_robust "${LR_ROBUST}" \
    --robust_steps "${ROBUST_STEPS}" \
    --control_steps "${CONTROL_STEPS}" \
    --lambda_curl "${LAMBDA_CURL}" \
    --duality_weight "${DUALITY_WEIGHT}" \
    --robust_gap_weight "${ROBUST_GAP_WEIGHT}" \
    --quad_points "${QUAD_POINTS}" \
    --width "${WIDTH}" \
    --layers "${LAYERS}" \
    --robust_width "${ROBUST_WIDTH}" \
    --robust_depth "${ROBUST_DEPTH}" \
    --export "${EXPORT_DIR}" \
    ${EXTRA_ARGS}

echo "Program ends $(date)"
