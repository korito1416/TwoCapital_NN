#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=hardened_pdpt
#SBATCH --output=/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/job-outs/hardened_pdpt_%j.out

set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module load parallel
module unload cuda || true
module unload python || true
module load cuda/11.2
module load python/anaconda-2021.05

# SCRATCH output (NEVER output_001)
JOB_NAME="${JOB_NAME:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_hardened_scratch/hardened_pdpt}"
PRETRAINED_FOLDER="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_ITERATIONS="${NUM_ITERATIONS:-150000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
NUM_NEURONS="${NUM_NEURONS:-32}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}"
HIDDEN_LAYER_ACTIVATIONS="${HIDDEN_LAYER_ACTIVATIONS:-swish,tanh,tanh,softplus}"
OUTPUT_LAYER_ACTIVATIONS="${OUTPUT_LAYER_ACTIVATIONS:-softplus,custom,custom,softplus}"
LEARNING_RATE_SCHEDULE_TYPE="${LEARNING_RATE_SCHEDULE_TYPE:-warmup_cosine}"
LEARNING_RATES="${LEARNING_RATES:-10e-6,10e-4}"
TECH_JUMP_INTENSITY_SCALE="${TECH_JUMP_INTENSITY_SCALE:-1.0}"
TECH_JUMP_PROBABILITY="${TECH_JUMP_PROBABILITY:-1.0}"

# validation + hardening knobs (env-overridable in the hardened main)
export VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-1024}"
export VALIDATION_BATCHES="${VALIDATION_BATCHES:-4}"
export VALIDATION_CONTROL_WEIGHT="${VALIDATION_CONTROL_WEIGHT:-5.0}"
export GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
export ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-0}"
export COSTATE_SUP_WEIGHT="${COSTATE_SUP_WEIGHT:-0.5}"
export RELATIVE_RESIDUAL="${RELATIVE_RESIDUAL:-1}"
export SCALE_EMA="${SCALE_EMA:-0.99}"
export C_FLOOR="${C_FLOOR:-1e-3}"
export XI_CURRICULUM="${XI_CURRICULUM:-1}"
export LOGXI_CURRICULUM_START="${LOGXI_CURRICULUM_START:-4.0}"
export XI_CURRICULUM_FRAC="${XI_CURRICULUM_FRAC:-0.3}"

mkdir -p "${JOB_NAME}/PostDamagePostTech" "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/job-outs"

echo "Hardened PDPT retrain starts $(date)"
echo "Output: ${JOB_NAME}"
echo "Warm-start: ${PRETRAINED_FOLDER}"
echo "Iters: ${NUM_ITERATIONS}  COSTATE_SUP_WEIGHT=${COSTATE_SUP_WEIGHT} XI_CURRICULUM=${XI_CURRICULUM}"

python models/PostDamagePostTech_hardened.py \
    "$JOB_NAME" \
    "$BATCH_SIZE" \
    "$NUM_ITERATIONS" \
    "$PRETRAINED_FOLDER" \
    "$LOGGING_FREQUENCY" \
    "$LEARNING_RATES" \
    "$HIDDEN_LAYER_ACTIVATIONS" \
    "$OUTPUT_LAYER_ACTIVATIONS" \
    "$NUM_HIDDEN_LAYERS" \
    "$NUM_NEURONS" \
    "$LEARNING_RATE_SCHEDULE_TYPE" \
    "hardened_pdpt" \
    "$TECH_JUMP_INTENSITY_SCALE" \
    "$TECH_JUMP_PROBABILITY"

echo "Hardened PDPT retrain ends $(date)"
