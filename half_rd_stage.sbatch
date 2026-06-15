#!/bin/bash

#SBATCH --time=0-24:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=21G
#SBATCH --job-name=halfRD_stage

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module load parallel
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

PREFIX="${PREFIX:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001}"
PRETRAINED_FOLDER="${PRETRAINED_FOLDER:-${PREFIX}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000}"
FOLDERNAME="${FOLDERNAME:-TwoStageTech_RDIntensityScale_0p5_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000}"
JOB_NAME="${JOB_NAME:-${PREFIX}/${FOLDERNAME}}"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
NUM_NEURONS="${NUM_NEURONS:-32}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}"
HIDDEN_LAYER_ACTIVATIONS="${HIDDEN_LAYER_ACTIVATIONS:-swish,tanh,tanh,softplus}"
OUTPUT_LAYER_ACTIVATIONS="${OUTPUT_LAYER_ACTIVATIONS:-softplus,custom,custom,softplus}"
LEARNING_RATE_SCHEDULE_TYPE="${LEARNING_RATE_SCHEDULE_TYPE:-warmup_cosine}"
LEARNING_RATES_ACTIVE="${LEARNING_RATES_ACTIVE:-10e-6,10e-4}"
TECH_JUMP_INTENSITY_SCALE="${TECH_JUMP_INTENSITY_SCALE:-0.5}"
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
    PostDamageIntermTech)
        MODEL="models/PostDamageIntermTech.py"
        ;;
    PostDamagePreTech)
        MODEL="models/PostDamagePreTech.py"
        ;;
    PreDamageIntermTech)
        MODEL="models/PreDamageIntermTech.py"
        ;;
    PreDamagePreTech)
        MODEL="models/PreDamagePreTech.py"
        ;;
    *)
        echo "Unknown STAGE: $STAGE" >&2
        exit 2
        ;;
esac

mkdir -p "${JOB_NAME}/${STAGE}" "./job-outs/${FOLDERNAME}"

echo "$SLURM_JOB_NAME"
echo "Stage: ${STAGE}"
echo "Program starts $(date)"
echo "Output folder: ${JOB_NAME}"
echo "Pretrained folder: ${PRETRAINED_FOLDER}"
echo "Tech jump intensity scale: ${TECH_JUMP_INTENSITY_SCALE}"
start_time=$(date +%s)

python "$MODEL" \
    "$JOB_NAME" \
    "$BATCH_SIZE" \
    "$NUM_ITERATIONS" \
    "$PRETRAINED_FOLDER" \
    "$LOGGING_FREQUENCY" \
    "$LEARNING_RATES_ACTIVE" \
    "$HIDDEN_LAYER_ACTIVATIONS" \
    "$OUTPUT_LAYER_ACTIVATIONS" \
    "$NUM_HIDDEN_LAYERS" \
    "$NUM_NEURONS" \
    "$LEARNING_RATE_SCHEDULE_TYPE" \
    "$FOLDERNAME" \
    "$TECH_JUMP_INTENSITY_SCALE"

echo "Program ends $(date)"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
