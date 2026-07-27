#!/bin/bash
#
# A/B EXPERIMENT: is the damage-curvature (lambda3) worst-case distortion shift
# (near-uniform -> skewed at xi=0.1) driven by the delta CALIBRATION change
# (delta: 0.025 -> 0.01), or by the FOCIr code refactor?
#
# Method: retrain the CURRENT FOCIr code with delta forced back to 0.025 (everything
# else at CURRENT values), warm-started from the delta=0.01 two-stage base, then
# simulate and read lambda3_weights_distorted at xi=0.1.
#
#   delta override  : NON-INVASIVE via ab_delta_override/sitecustomize.py (MODEL_DELTA env).
#                     models/params.py and model code are UNTOUCHED.
#   model family    : TWO-STAGE tech, pi=0.04, TechIntensityScale=1.0 (matches historical run).
#   warm-start      : all 6 regimes from the canonical delta=0.01 two-stage base.
#   iterations      : 300000 per regime (warm-started; value LEVEL must re-converge since
#                     values scale ~1/delta, but the SHAPE transfers -> moderate iters suffice
#                     for the qualitative xi=0.1 lambda3 weight readout).
#   output          : NEW dir output_delta025_ab/  (never touches existing output*/).
#
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
ROOT="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"

OVR="${ROOT}/ab_delta_override"
PREFIX="${ROOT}/output_delta025_ab"
STAGE_SCRIPT="${OVR}/sbatch/sensitivity_stage_delta.sbatch"
DIAG_SCRIPT="${OVR}/sbatch/deterministic_stage_delta.sbatch"
MODEL_DIR="models"

# Canonical delta=0.01 two-stage base (all 6 regimes present, loss_v ~1.4e-3).
PRETRAINED_FOLDER="${ROOT}/output_001/TwoStageTech_LR_warmup_cosine_40e-5,40e-5,40e-5,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations1000000"

SCHEDULE_TYPE="warmup_cosine"
BATCH_SIZE="128"; NUM_NEURONS="32"; NUM_HIDDEN_LAYERS="4"
NUM_ITERATIONS="${NUM_ITERATIONS:-300000}"
LEARNING_RATES="${LEARNING_RATES:-40e-5,40e-5,40e-5,40e-5}"
TECH_JUMP_INTENSITY_SCALE="1.0"
TECH_JUMP_PROBABILITY="0.04"        # two-stage (matches pre-FOCIr comparison)

# delta A/B: force delta = 0.025 (the OLD value). Everything else CURRENT.
export MODEL_DELTA="${MODEL_DELTA:-0.025}"
# Reproducibility of collocation sampling / init across the arm.
export MODEL_SEED="${MODEL_SEED:-1}"

SIMULATION_XIS="${SIMULATION_XIS:-0.1,0.3,5.0,148.6}"
SIMULATION_Y0="${SIMULATION_Y0:-1.2}"

FOLDERNAME="TwoStageTech_DELTA025_AB_Pi0p04_TechIntensityScale_1p0_LR_${SCHEDULE_TYPE}_${LEARNING_RATES}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
EXPORT_FOLDER="${PREFIX}/${FOLDERNAME}"

# Guard: base must have all 6 regime checkpoints.
for stage in PostDamagePostTech PostDamageIntermTech PostDamagePreTech PreDamagePostTech PreDamageIntermTech PreDamagePreTech; do
    [ -f "${PRETRAINED_FOLDER}/${stage}/checkpoint" ] || { echo "MISSING base ckpt: ${PRETRAINED_FOLDER}/${stage}" >&2; exit 2; }
done

if [ -e "${EXPORT_FOLDER}/run_manifest.txt" ] && grep -Eq ' job: [0-9]+' "${EXPORT_FOLDER}/run_manifest.txt"; then
    echo "Refusing to duplicate an already-submitted run: ${FOLDERNAME}" >&2; exit 0
fi

mkdir -p "$EXPORT_FOLDER" "./job-outs/delta025_ab/${FOLDERNAME}"

submit_stage() {
    local stage="$1" dep="${2:-}"
    local dep_args=(); [ -n "$dep" ] && dep_args=(--dependency="$dep")
    PREFIX="$PREFIX" PRETRAINED_FOLDER="$PRETRAINED_FOLDER" MODEL_DIR="$MODEL_DIR" \
    FOLDERNAME="$FOLDERNAME" JOB_NAME="$EXPORT_FOLDER" \
    BATCH_SIZE="$BATCH_SIZE" NUM_NEURONS="$NUM_NEURONS" NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
    NUM_ITERATIONS="$NUM_ITERATIONS" LEARNING_RATES="$LEARNING_RATES" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" TECH_JUMP_PROBABILITY="$TECH_JUMP_PROBABILITY" \
    MODEL_DELTA="$MODEL_DELTA" MODEL_SEED="$MODEL_SEED" \
    sbatch --parsable "${dep_args[@]}" \
        --job-name="d025_${stage}" \
        --output="./job-outs/delta025_ab/${FOLDERNAME}/${stage}-%j.out" \
        --error="./job-outs/delta025_ab/${FOLDERNAME}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage",MODEL_DELTA="$MODEL_DELTA" "$STAGE_SCRIPT"
}

# Full 6-regime backward DAG (post -> pre). ALL regimes retrained: delta shifts EVERY
# regime's value level, so no post-tech stage can be copied.
jpp=$(submit_stage  PostDamagePostTech)
jpi=$(submit_stage  PostDamageIntermTech "afterok:${jpp}")
jppr=$(submit_stage PostDamagePreTech    "afterok:${jpi}")
jprp=$(submit_stage PreDamagePostTech    "afterok:${jpp}")
jpri=$(submit_stage PreDamageIntermTech  "afterok:${jpi}:${jprp}")
jpre=$(submit_stage PreDamagePreTech     "afterok:${jppr}:${jpri}")

sim=$(EXPORT_FOLDER="$EXPORT_FOLDER" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
    MODEL_DELTA="$MODEL_DELTA" MODEL_DIR="$MODEL_DIR" \
    sbatch --parsable --dependency="afterok:${jpre}" --job-name="d025_sim" \
    --output="./job-outs/delta025_ab/${FOLDERNAME}/Sim-%j.out" \
    --error="./job-outs/delta025_ab/${FOLDERNAME}/Sim-%j.err" \
    --export=ALL,MODE=simulate,MODEL_DELTA="$MODEL_DELTA" "$DIAG_SCRIPT")

plot=$(EXPORT_FOLDER="$EXPORT_FOLDER" MODEL_DELTA="$MODEL_DELTA" MODEL_DIR="$MODEL_DIR" \
    sbatch --parsable --dependency="afterok:${sim}" --job-name="d025_plot" \
    --output="./job-outs/delta025_ab/${FOLDERNAME}/Plot-%j.out" \
    --error="./job-outs/delta025_ab/${FOLDERNAME}/Plot-%j.err" \
    --export=ALL,MODE=plot,MODEL_DELTA="$MODEL_DELTA" "$DIAG_SCRIPT")

{
  echo "Run: delta=0.025 A/B (isolate delta as the driver of the lambda3 xi=0.1 distortion shift)"
  echo "Created: $(date)"
  echo "MODEL_DELTA (override): ${MODEL_DELTA}   MODEL_SEED: ${MODEL_SEED}"
  echo "Model family: TWO-STAGE, pi=${TECH_JUMP_PROBABILITY}, TechIntensityScale=${TECH_JUMP_INTENSITY_SCALE}"
  echo "Warm-start base (delta=0.01): ${PRETRAINED_FOLDER}"
  echo "LR: ${LEARNING_RATES}   iters/regime: ${NUM_ITERATIONS}"
  echo "Sim xi: ${SIMULATION_XIS}   Sim Y0: ${SIMULATION_Y0}"
  echo "PostDamagePostTech job: ${jpp}"
  echo "PostDamageIntermTech job: ${jpi} (afterok:${jpp})"
  echo "PostDamagePreTech job: ${jppr} (afterok:${jpi})"
  echo "PreDamagePostTech job: ${jprp} (afterok:${jpp})"
  echo "PreDamageIntermTech job: ${jpri} (afterok:${jpi}:${jprp})"
  echo "PreDamagePreTech job: ${jpre} (afterok:${jppr}:${jpri})"
  echo "Simulation job: ${sim} (afterok:${jpre})"
  echo "Plot job: ${plot} (afterok:${sim})"
} | tee "${EXPORT_FOLDER}/run_manifest.txt"

echo "=== delta=0.025 A/B submitted. Track: squeue -u \$USER ==="
echo "EXPORT_FOLDER=${EXPORT_FOLDER}"
