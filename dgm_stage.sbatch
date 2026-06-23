#!/bin/bash

#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=dgm_stage

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_dgm_001}"
MODEL_DIR="${MODEL_DIR:-models_dgm}"
FOLDERNAME="${FOLDERNAME:?FOLDERNAME must be set}"
JOB_NAME="${JOB_NAME:-${OUTPUT_ROOT}/${FOLDERNAME}}"
PRETRAINED_FOLDER="${PRETRAINED_FOLDER:-None}"

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
NUM_NEURONS="${NUM_NEURONS:-32}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-3}"
HIDDEN_LAYER_ACTIVATIONS="${HIDDEN_LAYER_ACTIVATIONS:-swish,tanh,tanh,softplus}"
OUTPUT_LAYER_ACTIVATIONS="${OUTPUT_LAYER_ACTIVATIONS:-softplus,custom,custom,softplus}"
LEARNING_RATE_SCHEDULE_TYPE="${LEARNING_RATE_SCHEDULE_TYPE:-warmup_cosine}"
LEARNING_RATES="${LEARNING_RATES:-1e-5,1e-4}"
TECH_JUMP_INTENSITY_SCALE="${TECH_JUMP_INTENSITY_SCALE:-1.0}"
TECH_JUMP_PROBABILITY="${TECH_JUMP_PROBABILITY:-0.04}"
LOGXI_MIN="${LOGXI_MIN:--3.0}"
LOGXI_MAX="${LOGXI_MAX:-5.0}"

VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-1024}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-2}"
VALIDATION_CONTROL_WEIGHT="${VALIDATION_CONTROL_WEIGHT:-5.0}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-0}"
DGM_GATE_ACTIVATION="${DGM_GATE_ACTIVATION:-tanh}"
DGM_CANDIDATE_ACTIVATION="${DGM_CANDIDATE_ACTIVATION:-tanh}"
DGM_INITIAL_ACTIVATION="${DGM_INITIAL_ACTIVATION:-tanh}"
DGM_SEED="${DGM_SEED:-0}"
DGM_TEACHER_FOLDER="${DGM_TEACHER_FOLDER:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000}"
DGM_DISTILL_STEPS="${DGM_DISTILL_STEPS:-20000}"
DGM_DISTILL_BATCH_SIZE="${DGM_DISTILL_BATCH_SIZE:-512}"
DGM_DISTILL_LR="${DGM_DISTILL_LR:-1e-3}"
DGM_DISTILL_LOG_FREQUENCY="${DGM_DISTILL_LOG_FREQUENCY:-1000}"

export LOGXI_MIN LOGXI_MAX
export VALIDATION_BATCH_SIZE VALIDATION_BATCHES VALIDATION_CONTROL_WEIGHT
export GRADIENT_CLIP_NORM ENABLE_TENSORBOARD
export DGM_GATE_ACTIVATION DGM_CANDIDATE_ACTIVATION DGM_INITIAL_ACTIVATION DGM_SEED
export DGM_TEACHER_FOLDER DGM_DISTILL_STEPS DGM_DISTILL_BATCH_SIZE
export DGM_DISTILL_LR DGM_DISTILL_LOG_FREQUENCY

if [ -z "${STAGE:-}" ]; then
    echo "STAGE must be set via sbatch --export=ALL,STAGE=<stage>" >&2
    exit 2
fi

if [[ "$STAGE" == "PostDamageIntermTech" || "$STAGE" == "PreDamageIntermTech" ]] \
    && awk -v pi="$TECH_JUMP_PROBABILITY" 'BEGIN { exit !(pi >= 1.0 - 1e-12) }'; then
    echo "Intermediate technology stage ${STAGE} is invalid when pi=${TECH_JUMP_PROBABILITY}." >&2
    exit 2
fi

case "$STAGE" in
    PostDamagePostTech|PostDamageIntermTech|PostDamagePreTech|PreDamagePostTech|PreDamageIntermTech|PreDamagePreTech)
        MODEL="${MODEL_DIR}/${STAGE}.py"
        ;;
    *)
        echo "Unknown DGM stage: ${STAGE}" >&2
        exit 2
        ;;
esac

mkdir -p "${JOB_NAME}/${STAGE}" "./job-outs-dgm/${FOLDERNAME}"

echo "Stage: ${STAGE}"
echo "Model: ${MODEL}"
echo "Output: ${JOB_NAME}"
echo "DGM width/layers: ${NUM_NEURONS}/${NUM_HIDDEN_LAYERS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rates: ${LEARNING_RATES}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Validation: ${VALIDATION_BATCHES} x ${VALIDATION_BATCH_SIZE}"
echo "Teacher folder: ${DGM_TEACHER_FOLDER}"
echo "Distillation: ${DGM_DISTILL_STEPS} steps at ${DGM_DISTILL_LR}"
echo "Program starts $(date)"

python "$MODEL" \
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
    "$FOLDERNAME" \
    "$TECH_JUMP_INTENSITY_SCALE" \
    "$TECH_JUMP_PROBABILITY"

echo "Program ends $(date)"
