#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
PRETRAINED_FOLDER="${PREFIX}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
STAGE_SCRIPT="one_tech_jump_stage.sbatch"
MODEL_DIR="models"
SCHEDULE_TYPE="warmup_cosine"
BATCH_SIZE="128"
NUM_NEURONS="32"
NUM_HIDDEN_LAYERS="4"
NUM_ITERATIONS_POSTTECH="1000000"
NUM_ITERATIONS_INTERMTECH="1000000"
LEARNING_RATES_POSTTECH="10e-6,40e-4"
TECH_JUMP_PROBABILITY="0.0"

TECH_JUMP_INTENSITY_SCALES=(
    "1.0"
    "2.0"
)

LEARNING_RATE_GRID=(
    "10e-4,10e-4,10e-4,10e-4"
    "20e-4,20e-4,20e-4,20e-4"
    "10e-5,10e-5,10e-5,10e-5"
    "10e-6,10e-4"
)

for stage in PostDamagePostTech PostDamageIntermTech PreDamagePostTech PreDamageIntermTech; do
    if [ ! -f "${PRETRAINED_FOLDER}/${stage}/checkpoint" ]; then
        echo "Missing original-model pretrained checkpoint: ${PRETRAINED_FOLDER}/${stage}" >&2
        exit 2
    fi
done

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local intensity_scale="$3"
    local learning_rates="$4"
    local dependency="${5:-}"
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
    NUM_ITERATIONS_POSTTECH="$NUM_ITERATIONS_POSTTECH" \
    NUM_ITERATIONS_INTERMTECH="$NUM_ITERATIONS_INTERMTECH" \
    LEARNING_RATES_POSTTECH="$LEARNING_RATES_POSTTECH" \
    LEARNING_RATES_INTERMTECH="$learning_rates" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$intensity_scale" \
    TECH_JUMP_PROBABILITY="$TECH_JUMP_PROBABILITY" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="oneTech_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

for intensity_scale in "${TECH_JUMP_INTENSITY_SCALES[@]}"; do
    intensity_tag="${intensity_scale/./p}"

    for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
        foldername="OneTechJump_Pi_0p0_TechIntensityScale_${intensity_tag}_LR_${SCHEDULE_TYPE}_${learning_rates}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS_INTERMTECH}"
        job_name="${PREFIX}/${foldername}"
        mkdir -p "$job_name" "./job-outs/${foldername}"

        cat > "${job_name}/run_manifest.txt" <<EOF
Run: one technology jump experiment
Created: $(date)
Tech jump probability pi: ${TECH_JUMP_PROBABILITY}
Tech jump intensity scale: ${intensity_scale}
Interpretation: pi = 0 eliminates the intermediate-to-post technology jump term.
Model directory: ${MODEL_DIR}
Pretrained source: ${PRETRAINED_FOLDER}
Post-tech learning rates: ${LEARNING_RATES_POSTTECH}
Intermediate-tech learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Post-tech iterations: ${NUM_ITERATIONS_POSTTECH}
Intermediate-tech iterations: ${NUM_ITERATIONS_INTERMTECH}
Submitted stages: PostDamagePostTech, PostDamageIntermTech, PreDamagePostTech, PreDamageIntermTech
Excluded stages: PostDamagePreTech, PreDamagePreTech
Simulation submitted: no
EOF

        jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername" "$intensity_scale" "$learning_rates")
        jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_post}")
        jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_post}")
        jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_interm}:${jid_pre_post}")

        cat >> "${job_name}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PostDamageIntermTech job: ${jid_post_interm} (afterok:${jid_post_post})
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm}:${jid_pre_post})
EOF

        printf 'Submitted %s\n' "$foldername"
        printf '  PostDamagePostTech:   %s\n' "$jid_post_post"
        printf '  PostDamageIntermTech: %s (afterok:%s)\n' "$jid_post_interm" "$jid_post_post"
        printf '  PreDamagePostTech:    %s (afterok:%s)\n' "$jid_pre_post" "$jid_post_post"
        printf '  PreDamageIntermTech:  %s (afterok:%s:%s)\n' "$jid_pre_interm" "$jid_post_interm" "$jid_pre_post"
    done
done
