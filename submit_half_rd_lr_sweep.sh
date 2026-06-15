#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
PRETRAINED_FOLDER="${PREFIX}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
STAGE_SCRIPT="half_rd_full_stage.sbatch"
MODEL_DIR="models"
SCHEDULE_TYPE="warmup_cosine"
BATCH_SIZE="128"
NUM_NEURONS="32"
NUM_HIDDEN_LAYERS="4"
NUM_ITERATIONS1="1000000"
NUM_ITERATIONS2="1000000"
LEARNING_RATES_POSTTECH="10e-6,40e-4"
TECH_JUMP_INTENSITY_SCALE="0.5"

LEARNING_RATE_GRID=(
    "10e-4,10e-4,10e-4,10e-4"
    "20e-4,20e-4,20e-4,20e-4"
    "10e-5,10e-5,10e-5,10e-5"
    "40e-5,40e-5,40e-5,40e-5"
    "10e-6,10e-6,10e-6,10e-6"
    "40e-6,40e-6,40e-6,40e-6"
    "10e-7,10e-7,10e-7,10e-7"
    "40e-7,40e-7,40e-7,40e-7"
    "20e-7,20e-7,20e-7,20e-7"
    "10e-6,10e-4"
    "10e-6,40e-4"
    "10e-6,10e-5"
    "10e-6,40e-5"
    "10e-6,40e-6"
)

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local dependency="${3:-}"
    local dependency_args=()

    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    PREFIX="$PREFIX" \
    PRETRAINED_FOLDER="$PRETRAINED_FOLDER" \
    MODEL_DIR="$MODEL_DIR" \
    FOLDERNAME="$foldername" \
    JOB_NAME="${PREFIX}/${foldername}" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_NEURONS="$NUM_NEURONS" \
    NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
    NUM_ITERATIONS1="$NUM_ITERATIONS1" \
    NUM_ITERATIONS2="$NUM_ITERATIONS2" \
    LEARNING_RATES_POSTTECH="$LEARNING_RATES_POSTTECH" \
    LEARNING_RATES_ACTIVE="$learning_rates" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="halfRD_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
    foldername="TwoStageTech_RDIntensityScale_0p5_fullsweep_LR_${SCHEDULE_TYPE}_${learning_rates}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS2}"
    job_name="${PREFIX}/${foldername}"
    mkdir -p "$job_name" "./job-outs/${foldername}"

    cat > "${job_name}/run_manifest.txt" <<EOF
Run: half technology jump intensity full LR sweep
Created: $(date)
Tech jump intensity scale: ${TECH_JUMP_INTENSITY_SCALE}
Model directory: ${MODEL_DIR}
Pretrained source: ${PRETRAINED_FOLDER}
Post-tech learning rates: ${LEARNING_RATES_POSTTECH}
Active-regime learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Post-tech iterations: ${NUM_ITERATIONS1}
Active-regime iterations: ${NUM_ITERATIONS2}
EOF

    jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername")
    jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "afterok:${jid_post_post}")
    jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "afterok:${jid_post_interm}")
    jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "afterok:${jid_post_post}")
    jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "afterok:${jid_post_interm}:${jid_pre_post}")
    jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "afterok:${jid_post_pre}:${jid_pre_interm}")

    printf 'Submitted %s\n' "$foldername"
    printf '  PostDamagePostTech:  %s\n' "$jid_post_post"
    printf '  PostDamageIntermTech:%s\n' "$jid_post_interm"
    printf '  PostDamagePreTech:   %s\n' "$jid_post_pre"
    printf '  PreDamagePostTech:   %s\n' "$jid_pre_post"
    printf '  PreDamageIntermTech: %s\n' "$jid_pre_interm"
    printf '  PreDamagePreTech:    %s\n' "$jid_pre_pre"
done
