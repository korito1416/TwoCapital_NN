#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
PRETRAINED_FOLDER="${PREFIX}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
STAGE_SCRIPT="half_rd_stage.sbatch"
SCHEDULE_TYPE="warmup_cosine"
BATCH_SIZE="128"
NUM_NEURONS="32"
NUM_HIDDEN_LAYERS="4"
NUM_ITERATIONS="1000000"
LEARNING_RATES_POSTTECH="10e-6,40e-4"
TECH_JUMP_INTENSITY_SCALE="2.0"

# These are the uncommented active-regime learning rates in the requested setup.
LEARNING_RATE_GRID=(
    "10e-4,10e-4,10e-4,10e-4"
    "20e-4,20e-4,20e-4,20e-4"
    "10e-5,10e-5,10e-5,10e-5"
    "10e-6,10e-4"
)

for stage in PostDamagePostTech PreDamagePostTech; do
    if [ ! -f "${PRETRAINED_FOLDER}/${stage}/checkpoint" ]; then
        echo "Missing original-model post-tech checkpoint: ${PRETRAINED_FOLDER}/${stage}" >&2
        exit 2
    fi
done

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local learning_rates="$3"
    local dependency="${4:-}"
    local dependency_args=()

    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    PRETRAINED_FOLDER="$PRETRAINED_FOLDER" \
    FOLDERNAME="$foldername" \
    JOB_NAME="${PREFIX}/${foldername}" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_NEURONS="$NUM_NEURONS" \
    NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
    NUM_ITERATIONS="$NUM_ITERATIONS" \
    LEARNING_RATES_ACTIVE="$learning_rates" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="doubleTech_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
    foldername="TwoStageTech_TechIntensityScale_2p0_LR_${SCHEDULE_TYPE}_${learning_rates}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    job_name="${PREFIX}/${foldername}"
    mkdir -p "$job_name" "./job-outs/${foldername}"

    # Technology intensity does not enter either post-tech HJB.
    for stage in PostDamagePostTech PreDamagePostTech; do
        if [ ! -d "${job_name}/${stage}" ]; then
            cp -a "${PRETRAINED_FOLDER}/${stage}" "${job_name}/${stage}"
        fi
    done

    cat > "${job_name}/run_manifest.txt" <<EOF
Run: double technology jump intensity learning-rate sweep
Created: $(date)
Tech jump intensity scale: ${TECH_JUMP_INTENSITY_SCALE}
Interpretation: J_g = 2.0 * exp(logR) / varrho, with branch probabilities applied afterward.
Model directory: models
Pretrained source: ${PRETRAINED_FOLDER}
Post-tech learning rates: ${LEARNING_RATES_POSTTECH}
Active-regime learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Iterations per active regime: ${NUM_ITERATIONS}
Copied unchanged regimes: PostDamagePostTech, PreDamagePostTech
Submitted stages: PostDamageIntermTech, PostDamagePreTech, PreDamageIntermTech, PreDamagePreTech
Simulation submitted: no
EOF

    jid_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "$learning_rates")
    jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "$learning_rates" "afterok:${jid_interm}")
    jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "$learning_rates" "afterok:${jid_interm}")
    jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "$learning_rates" "afterok:${jid_post_pre}:${jid_pre_interm}")

    cat >> "${job_name}/run_manifest.txt" <<EOF
PostDamageIntermTech job: ${jid_interm}
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_interm})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_interm})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre}:${jid_pre_interm})
EOF

    printf 'Submitted %s\n' "$foldername"
    printf '  PostDamageIntermTech: %s\n' "$jid_interm"
    printf '  PostDamagePreTech:    %s (afterok:%s)\n' "$jid_post_pre" "$jid_interm"
    printf '  PreDamageIntermTech:  %s (afterok:%s)\n' "$jid_pre_interm" "$jid_interm"
    printf '  PreDamagePreTech:     %s (afterok:%s:%s)\n' "$jid_pre_pre" "$jid_post_pre" "$jid_pre_interm"
done
