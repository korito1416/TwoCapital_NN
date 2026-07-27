#!/bin/bash

#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=21G
#SBATCH --job-name=sensitivity_stage

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module load parallel
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

# --- A/B delta override (NON-INVASIVE): prepend sitecustomize dir + set MODEL_DELTA ---
# The sitecustomize.py in ab_delta_override/ patches params.PARAMS["δ"] on import when
# MODEL_DELTA is set. Production params.py / model code are untouched. Set both here so
# the job is self-contained even if --export=ALL propagation is imperfect.
export MODEL_DELTA="${MODEL_DELTA:?MODEL_DELTA must be set for the delta A/B stage}"
export PYTHONPATH="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/ab_delta_override:${PYTHONPATH:-}"
echo "[delta-AB] MODEL_DELTA=${MODEL_DELTA}"
echo "[delta-AB] PYTHONPATH head=${PYTHONPATH%%:*}"

PREFIX="${PREFIX:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_sensitivity_001}"
PRETRAINED_FOLDER="${PRETRAINED_FOLDER:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000}"
MODEL_DIR="${MODEL_DIR:-models}"
FOLDERNAME="${FOLDERNAME:?FOLDERNAME must be provided}"
JOB_NAME="${JOB_NAME:-${PREFIX}/${FOLDERNAME}}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
NUM_NEURONS="${NUM_NEURONS:-32}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}"
HIDDEN_LAYER_ACTIVATIONS="${HIDDEN_LAYER_ACTIVATIONS:-swish,tanh,tanh,softplus}"
OUTPUT_LAYER_ACTIVATIONS="${OUTPUT_LAYER_ACTIVATIONS:-softplus,custom,custom,softplus}"
LEARNING_RATE_SCHEDULE_TYPE="${LEARNING_RATE_SCHEDULE_TYPE:-warmup_cosine}"
LEARNING_RATES="${LEARNING_RATES:-10e-6,10e-4}"
TECH_JUMP_INTENSITY_SCALE="${TECH_JUMP_INTENSITY_SCALE:-1.0}"
TECH_JUMP_PROBABILITY="${TECH_JUMP_PROBABILITY:-0.04}"
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
        ;;
    PostDamageIntermTech)
        MODEL="${MODEL_DIR}/PostDamageIntermTech.py"
        ;;
    PostDamagePreTech)
        MODEL="${MODEL_DIR}/PostDamagePreTech.py"
        ;;
    PreDamagePostTech)
        MODEL="${MODEL_DIR}/PreDamagePostTech.py"
        ;;
    PreDamageIntermTech)
        MODEL="${MODEL_DIR}/PreDamageIntermTech.py"
        ;;
    PreDamagePreTech)
        MODEL="${MODEL_DIR}/PreDamagePreTech.py"
        ;;
    *)
        echo "Unknown STAGE: $STAGE" >&2
        exit 2
        ;;
esac

mkdir -p "${JOB_NAME}/${STAGE}" "./job-outs/sensitivity/${FOLDERNAME}"

echo "$SLURM_JOB_NAME"
echo "Stage: ${STAGE}"
echo "Program starts $(date)"
echo "Output folder: ${JOB_NAME}"
echo "Pretrained folder: ${PRETRAINED_FOLDER}"
echo "Learning rates: ${LEARNING_RATES}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "MODEL_SIGMA_D=${MODEL_SIGMA_D:-baseline}"
echo "MODEL_SIGMA_G=${MODEL_SIGMA_G:-baseline}"
echo "MODEL_GAMMA_D=${MODEL_GAMMA_D:-baseline}"
echo "MODEL_GAMMA_G=${MODEL_GAMMA_G:-baseline}"
echo "MODEL_THETA_D=${MODEL_THETA_D:-baseline}"
echo "MODEL_THETA_G=${MODEL_THETA_G:-baseline}"
echo "MODEL_PSI0=${MODEL_PSI0:-baseline}"
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
