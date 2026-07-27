#!/bin/bash

#SBATCH --time=0-12:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=anchored_hjb

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda || true
module unload python || true
module load cuda/11.2
module load python/anaconda-2021.05

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MODE="${MODE:-two}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-512}"
VALID_BATCHES="${VALID_BATCHES:-2}"
LOG_EVERY="${LOG_EVERY:-500}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-float32}"
LR_VALUE="${LR_VALUE:-4e-4}"
LR_CONTROL="${LR_CONTROL:-4e-4}"
ROOT_CONTROL_INIT="${ROOT_CONTROL_INIT:-surrogate}"
EXPORT_ROOT="${EXPORT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/experiments/anchored_costate/results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
EXPORT_DIR="${EXPORT_DIR:-${EXPORT_ROOT}/${MODE}_seed${SEED}_${STAMP}}"
mkdir -p "${EXPORT_DIR}"

echo "Program starts $(date)"
echo "Mode: ${MODE}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Validation: ${VALID_BATCHES} x ${VALID_BATCH_SIZE}"
echo "Seed: ${SEED}"
echo "Dtype: ${DTYPE}"
echo "Export: ${EXPORT_DIR}"
echo "Extra args: ${EXTRA_ARGS}"

COMMON_ARGS=(
    --steps "${STEPS}"
    --batch_size "${BATCH_SIZE}"
    --valid_batch_size "${VALID_BATCH_SIZE}"
    --valid_batches "${VALID_BATCHES}"
    --log_every "${LOG_EVERY}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --lr_value "${LR_VALUE}"
    --lr_control "${LR_CONTROL}"
    --export "${EXPORT_DIR}"
)

case "${MODE}" in
    root)
        python -u experiments/anchored_costate/train_root_recentered.py \
            "${COMMON_ARGS[@]}" \
            --control_init "${ROOT_CONTROL_INIT}" \
            ${EXTRA_ARGS}
        ;;
    two)
        TWO_ARGS=()
        if [[ "${COUPLE_ANCHOR_GRAD:-0}" == "1" ]]; then
            TWO_ARGS+=(--couple_anchor_grad)
        fi
        if [[ "${COUPLE_NEIGHBOR_GRAD:-0}" == "1" ]]; then
            TWO_ARGS+=(--couple_neighbor_grad)
        fi
        if [[ "${FREEZE_ROOT_CONTROLS:-0}" == "1" ]]; then
            TWO_ARGS+=(--freeze_root_controls)
        fi
        python -u experiments/anchored_costate/train_two_regime_joint.py \
            "${COMMON_ARGS[@]}" \
            --root_control_init "${ROOT_CONTROL_INIT}" \
            "${TWO_ARGS[@]}" \
            ${EXTRA_ARGS}
        ;;
    *)
        echo "Unknown MODE=${MODE}; expected root or two" >&2
        exit 2
        ;;
esac

echo "Program ends $(date)"
