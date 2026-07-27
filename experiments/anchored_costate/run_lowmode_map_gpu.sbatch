#!/bin/bash

#SBATCH --time=0-12:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --job-name=lowmode_map
#SBATCH --output=experiments/anchored_costate/logs/lowmode_map_%A_%a.out
#SBATCH --error=experiments/anchored_costate/logs/lowmode_map_%A_%a.err

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda || true
module unload python || true
module load cuda/11.2
module load python/anaconda-2021.05

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-512}"
VALID_BATCHES="${VALID_BATCHES:-2}"
VALID_SEED="${VALID_SEED:-12345}"
LOG_EVERY="${LOG_EVERY:-250}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-float32}"
SLICE_GRID="${SLICE_GRID:-16}"
SLICE_CHUNK_SIZE="${SLICE_CHUNK_SIZE:-128}"
CONTROL_STEPS="${CONTROL_STEPS:-2}"
LR_VALUE="${LR_VALUE:-1e-4}"
LR_CONTROL="${LR_CONTROL:-1e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/experiments/anchored_costate/results/lowmode_map}"
WARM_START_DIR="${WARM_START_DIR:-experiments/anchored_costate/results/precision_push_batch256/seed0_20000steps_20260630_210314/fedctrl_long_base}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
    0)
        RUN_NAME="dc1_rms025_foc025"
        CONFIG_ARGS="--slice_dc_weight 1.0 --slice_rms_weight 0.25 --slice_foc_weight 0.25 --slice_low_weight 0.0 --root_weight 0.25"
        ;;
    1)
        RUN_NAME="dc2_rms025_foc025"
        CONFIG_ARGS="--slice_dc_weight 2.0 --slice_rms_weight 0.25 --slice_foc_weight 0.25 --slice_low_weight 0.0 --root_weight 0.25"
        ;;
    2)
        RUN_NAME="dc1_rms05_foc025"
        CONFIG_ARGS="--slice_dc_weight 1.0 --slice_rms_weight 0.50 --slice_foc_weight 0.25 --slice_low_weight 0.0 --root_weight 0.25"
        ;;
    3)
        RUN_NAME="dc1_rms025_foc1"
        CONFIG_ARGS="--slice_dc_weight 1.0 --slice_rms_weight 0.25 --slice_foc_weight 1.00 --slice_low_weight 0.0 --root_weight 0.25"
        ;;
    4)
        RUN_NAME="dc1_low025_rms025_foc025"
        CONFIG_ARGS="--slice_dc_weight 1.0 --slice_rms_weight 0.25 --slice_foc_weight 0.25 --slice_low_weight 0.25 --low_radius 2.0 --root_weight 0.25"
        ;;
    5)
        RUN_NAME="prefocus_root01_dc1_rms05_foc05"
        CONFIG_ARGS="--slice_dc_weight 1.0 --slice_rms_weight 0.50 --slice_foc_weight 0.50 --slice_low_weight 0.0 --root_weight 0.10 --root_control_weight 0.10"
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
echo "Torch probe:"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
echo "Run: ${RUN_NAME}"
echo "Output: ${EXPORT_DIR}"
echo "Config: ${CONFIG_ARGS}"

python -u experiments/anchored_costate/train_two_regime_lowmode.py \
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
    --slice_grid "${SLICE_GRID}" \
    --slice_chunk_size "${SLICE_CHUNK_SIZE}" \
    --control_steps "${CONTROL_STEPS}" \
    --warm_start_dir "${WARM_START_DIR}" \
    --export "${EXPORT_DIR}" \
    ${CONFIG_ARGS} \
    ${EXTRA_ARGS}

echo "Program ends $(date)"
