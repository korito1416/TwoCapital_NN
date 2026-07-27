#!/bin/bash

#SBATCH --time=0-12:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --job-name=analytic_cost
#SBATCH --output=experiments/anchored_costate/logs/analytic_cost_%A_%a.out
#SBATCH --error=experiments/anchored_costate/logs/analytic_cost_%A_%a.err

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda || true
module unload python || true
module load cuda/11.2
module load python/anaconda-2021.05

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STEPS="${STEPS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
COSTATE_BATCH_SIZE="${COSTATE_BATCH_SIZE:-256}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-512}"
VALID_BATCHES="${VALID_BATCHES:-2}"
VALID_SEED="${VALID_SEED:-12345}"
LOG_EVERY="${LOG_EVERY:-250}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-float32}"
SLICE_GRID="${SLICE_GRID:-24}"
SLICE_CHUNK_SIZE="${SLICE_CHUNK_SIZE:-144}"
CONTROL_STEPS="${CONTROL_STEPS:-2}"
LR_VALUE="${LR_VALUE:-3e-5}"
LR_CONTROL="${LR_CONTROL:-5e-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/experiments/anchored_costate/results/analytic_costate}"
WARM_START_DIR="${WARM_START_DIR:-experiments/anchored_costate/results/lowmode_map_20260701_122130/dc1_rms025_foc025_seed0_task0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
    0)
        RUN_NAME="q_mild"
        CONFIG_ARGS="--costate_mode q --pre_costate_weight 0.25 --slice_costate_weight 0.25 --slice_costate_low_weight 0.25"
        ;;
    1)
        RUN_NAME="both_mild"
        CONFIG_ARGS="--costate_mode both --pre_costate_weight 0.25 --slice_costate_weight 0.25 --slice_costate_low_weight 0.25"
        ;;
    2)
        RUN_NAME="both_strong"
        CONFIG_ARGS="--costate_mode both --pre_costate_weight 0.50 --slice_costate_weight 0.50 --slice_costate_low_weight 0.50"
        ;;
    3)
        RUN_NAME="p_focus"
        CONFIG_ARGS="--costate_mode p --pre_costate_weight 0.25 --slice_costate_weight 0.25 --slice_costate_low_weight 0.50"
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
        exit 2
        ;;
esac

EXPORT_DIR="${OUTPUT_ROOT}/${RUN_NAME}_seed${SEED}_task${TASK_ID}"
mkdir -p "${OUTPUT_ROOT}" experiments/anchored_costate/logs

echo "Program starts $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
echo "Run: ${RUN_NAME}"
echo "Output: ${EXPORT_DIR}"
echo "Config: ${CONFIG_ARGS}"

python -u experiments/anchored_costate/train_two_regime_analytic_costate.py \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --costate_batch_size "${COSTATE_BATCH_SIZE}" \
    --valid_batch_size "${VALID_BATCH_SIZE}" \
    --valid_batches "${VALID_BATCHES}" \
    --valid_seed "${VALID_SEED}" \
    --log_every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --lr_value "${LR_VALUE}" \
    --lr_control "${LR_CONTROL}" \
    --slice_grid "${SLICE_GRID}" \
    --slice_chunk_size "${SLICE_CHUNK_SIZE}" \
    --control_steps "${CONTROL_STEPS}" \
    --warm_start_dir "${WARM_START_DIR}" \
    --export "${EXPORT_DIR}" \
    ${CONFIG_ARGS} \
    ${EXTRA_ARGS}

echo "Program ends $(date)"
