#!/bin/bash

#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=21G
#SBATCH --job-name=oneTech_stage

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module load parallel
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

PREFIX="${PREFIX:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001}"
PRETRAINED_FOLDER="${PRETRAINED_FOLDER:-${PREFIX}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000}"
MODEL_DIR="${MODEL_DIR:-models}"
FOLDERNAME="${FOLDERNAME:?FOLDERNAME must be set}"
JOB_NAME="${JOB_NAME:-${PREFIX}/${FOLDERNAME}}"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_ITERATIONS_POSTTECH="${NUM_ITERATIONS_POSTTECH:-1000000}"
NUM_ITERATIONS_INTERMTECH="${NUM_ITERATIONS_INTERMTECH:-1000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
NUM_NEURONS="${NUM_NEURONS:-32}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}"
HIDDEN_LAYER_ACTIVATIONS="${HIDDEN_LAYER_ACTIVATIONS:-swish,tanh,tanh,softplus}"
OUTPUT_LAYER_ACTIVATIONS="${OUTPUT_LAYER_ACTIVATIONS:-softplus,custom,custom,softplus}"
LEARNING_RATE_SCHEDULE_TYPE="${LEARNING_RATE_SCHEDULE_TYPE:-warmup_cosine}"
LEARNING_RATES_POSTTECH="${LEARNING_RATES_POSTTECH:-10e-6,40e-4}"
LEARNING_RATES_INTERMTECH="${LEARNING_RATES_INTERMTECH:-10e-6,10e-4}"
TECH_JUMP_INTENSITY_SCALE="${TECH_JUMP_INTENSITY_SCALE:-1.0}"
TECH_JUMP_PROBABILITY="${TECH_JUMP_PROBABILITY:-0.0}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-1024}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-4}"
VALIDATION_CONTROL_WEIGHT="${VALIDATION_CONTROL_WEIGHT:-5.0}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-0}"
export VALIDATION_BATCH_SIZE VALIDATION_BATCHES VALIDATION_CONTROL_WEIGHT
export GRADIENT_CLIP_NORM ENABLE_TENSORBOARD

if [ -z "${STAGE:-}" ]; then
    echo "STAGE must be set via sbatch --export=ALL,STAGE=<stage>" >&2
    exit 2
fi

case "$STAGE" in
    PostDamagePostTech)
        MODEL="${MODEL_DIR}/PostDamagePostTech.py"
        LEARNING_RATES="${LEARNING_RATES_POSTTECH}"
        NUM_ITERATIONS="${NUM_ITERATIONS_POSTTECH}"
        ;;
    PostDamageIntermTech)
        MODEL="${MODEL_DIR}/PostDamageIntermTech.py"
        LEARNING_RATES="${LEARNING_RATES_INTERMTECH}"
        NUM_ITERATIONS="${NUM_ITERATIONS_INTERMTECH}"
        ;;
    PostDamagePreTech)
        MODEL="${MODEL_DIR}/PostDamagePreTech.py"
        LEARNING_RATES="${LEARNING_RATES_INTERMTECH}"
        NUM_ITERATIONS="${NUM_ITERATIONS_INTERMTECH}"
        ;;
    PreDamagePostTech)
        MODEL="${MODEL_DIR}/PreDamagePostTech.py"
        LEARNING_RATES="${LEARNING_RATES_POSTTECH}"
        NUM_ITERATIONS="${NUM_ITERATIONS_POSTTECH}"
        ;;
    PreDamageIntermTech)
        MODEL="${MODEL_DIR}/PreDamageIntermTech.py"
        LEARNING_RATES="${LEARNING_RATES_INTERMTECH}"
        NUM_ITERATIONS="${NUM_ITERATIONS_INTERMTECH}"
        ;;
    PreDamagePreTech)
        MODEL="${MODEL_DIR}/PreDamagePreTech.py"
        LEARNING_RATES="${LEARNING_RATES_INTERMTECH}"
        NUM_ITERATIONS="${NUM_ITERATIONS_INTERMTECH}"
        ;;
    *)
        echo "Unknown one-technology-jump stage: $STAGE" >&2
        exit 2
        ;;
esac

mkdir -p "${JOB_NAME}/${STAGE}" "./job-outs/${FOLDERNAME}"

echo "$SLURM_JOB_NAME"
echo "Stage: ${STAGE}"
echo "Model dir: ${MODEL_DIR}"
echo "Program starts $(date)"
echo "Output folder: ${JOB_NAME}"
echo "Pretrained folder: ${PRETRAINED_FOLDER}"
echo "Learning rates: ${LEARNING_RATES}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Tech jump intensity scale: ${TECH_JUMP_INTENSITY_SCALE}"
echo "Tech jump probability pi: ${TECH_JUMP_PROBABILITY}"
echo "Training batch size: ${BATCH_SIZE}"
echo "Validation: ${VALIDATION_BATCHES} x ${VALIDATION_BATCH_SIZE}"
echo "Gradient clip norm: ${GRADIENT_CLIP_NORM}"
start_time=$(date +%s)

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
end_time=$(date +%s)
elapsed=$((end_time - start_time))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
