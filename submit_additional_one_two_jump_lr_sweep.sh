#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
PRETRAINED_FOLDER="${PREFIX}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
ONE_JUMP_STAGE_SCRIPT="one_tech_jump_stage.sbatch"
TWO_JUMP_STAGE_SCRIPT="half_rd_stage.sbatch"
DIAGNOSTIC_STAGE_SCRIPT="deterministic_stage.sbatch"
MODEL_DIR="models"
SCHEDULE_TYPE="warmup_cosine"
BATCH_SIZE="128"
NUM_NEURONS="32"
NUM_HIDDEN_LAYERS="4"
NUM_ITERATIONS="1000000"
LEARNING_RATES_POSTTECH="10e-6,40e-4"
TECH_JUMP_PROBABILITY_ONE_JUMP="0.0"
SIMULATION_XIS="0.01,0.05,0.1,148.6"
SIMULATION_Y0="1.2"

TECH_JUMP_INTENSITY_SCALES=(
    "1.0"
    "2.0"
)

LEARNING_RATE_GRID=(
    "40e-5,40e-5,40e-5,40e-5"
    "10e-6,10e-6,10e-6,10e-6"
    "40e-6,40e-6,40e-6,40e-6"
    "10e-7,10e-7,10e-7,10e-7"
    "40e-7,40e-7,40e-7,40e-7"
    "20e-7,20e-7,20e-7,20e-7"
    "10e-6,40e-4"
    "10e-6,10e-5"
    "10e-6,40e-5"
    "10e-6,40e-6"
)

for stage in PostDamagePostTech PostDamageIntermTech PostDamagePreTech PreDamagePostTech PreDamageIntermTech PreDamagePreTech; do
    if [ ! -f "${PRETRAINED_FOLDER}/${stage}/checkpoint" ]; then
        echo "Missing original-model pretrained checkpoint: ${PRETRAINED_FOLDER}/${stage}" >&2
        exit 2
    fi
done

prepare_new_variant() {
    local foldername="$1"
    local job_name="${PREFIX}/${foldername}"

    if [ -e "${job_name}/run_manifest.txt" ]; then
        echo "Refusing to resubmit existing variant: ${foldername}" >&2
        return 1
    fi

    mkdir -p "$job_name" "./job-outs/${foldername}"
}

submit_one_jump_stage() {
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
    NUM_ITERATIONS_POSTTECH="$NUM_ITERATIONS" \
    NUM_ITERATIONS_INTERMTECH="$NUM_ITERATIONS" \
    LEARNING_RATES_POSTTECH="$LEARNING_RATES_POSTTECH" \
    LEARNING_RATES_INTERMTECH="$learning_rates" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$intensity_scale" \
    TECH_JUMP_PROBABILITY="$TECH_JUMP_PROBABILITY_ONE_JUMP" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="oneAdd_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$ONE_JUMP_STAGE_SCRIPT"
}

submit_two_jump_stage() {
    local stage="$1"
    local foldername="$2"
    local intensity_scale="$3"
    local learning_rates="$4"
    local dependency="${5:-}"
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
    TECH_JUMP_INTENSITY_SCALE="$intensity_scale" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="twoAdd_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$TWO_JUMP_STAGE_SCRIPT"
}

submit_density_diagnostics() {
    local foldername="$1"
    local training_jid="$2"
    local export_folder="${PREFIX}/${foldername}"

    DENSITY_SIM_JID=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${training_jid}" \
        --job-name=firstJumpDensitySim \
        --output="./job-outs/${foldername}/FirstJumpDensitySimulation-%j.out" \
        --error="./job-outs/${foldername}/FirstJumpDensitySimulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    DENSITY_PLOT_JID=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${DENSITY_SIM_JID}" \
        --job-name=firstJumpDensityPlot \
        --output="./job-outs/${foldername}/FirstJumpDensityPlot-%j.out" \
        --error="./job-outs/${foldername}/FirstJumpDensityPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    cat > "${export_folder}/first_jump_density_jobs.txt" <<EOF
Submitted: $(date)
Training dependency: ${training_jid}
Simulation scenarios xi: ${SIMULATION_XIS}
Simulation Y0: ${SIMULATION_Y0}
Simulation job: ${DENSITY_SIM_JID}
Plot job: ${DENSITY_PLOT_JID} (afterok:${DENSITY_SIM_JID})
EOF
}

submit_one_jump_variant() {
    local intensity_scale="$1"
    local learning_rates="$2"
    local intensity_tag="${intensity_scale/./p}"
    local foldername="OneTechJump_Pi_0p0_TechIntensityScale_${intensity_tag}_LR_${SCHEDULE_TYPE}_${learning_rates}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    local job_name="${PREFIX}/${foldername}"

    prepare_new_variant "$foldername"

    cat > "${job_name}/run_manifest.txt" <<EOF
Run: one technology jump additional learning-rate sweep
Created: $(date)
Tech jump probability pi: ${TECH_JUMP_PROBABILITY_ONE_JUMP}
Tech jump intensity scale: ${intensity_scale}
Model directory: ${MODEL_DIR}
Pretrained source: ${PRETRAINED_FOLDER}
Post-tech learning rates: ${LEARNING_RATES_POSTTECH}
Intermediate-tech learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Iterations per stage: ${NUM_ITERATIONS}
Submitted stages: PostDamagePostTech, PostDamageIntermTech, PreDamagePostTech, PreDamageIntermTech
Excluded stages: PostDamagePreTech, PreDamagePreTech
Deterministic simulation xi: ${SIMULATION_XIS}
Conditional first-jump density plots: yes
EOF

    jid_post_post=$(submit_one_jump_stage "PostDamagePostTech" "$foldername" "$intensity_scale" "$learning_rates")
    jid_post_interm=$(submit_one_jump_stage "PostDamageIntermTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_post=$(submit_one_jump_stage "PreDamagePostTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_interm=$(submit_one_jump_stage "PreDamageIntermTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_interm}:${jid_pre_post}")

    submit_density_diagnostics "$foldername" "$jid_pre_interm"

    cat >> "${job_name}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PostDamageIntermTech job: ${jid_post_interm} (afterok:${jid_post_post})
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm}:${jid_pre_post})
First-jump density simulation job: ${DENSITY_SIM_JID} (afterok:${jid_pre_interm})
First-jump density plot job: ${DENSITY_PLOT_JID} (afterok:${DENSITY_SIM_JID})
EOF

    printf 'Submitted one-jump scale=%s LR=%s: training final=%s, density=%s/%s\n' \
        "$intensity_scale" "$learning_rates" "$jid_pre_interm" "$DENSITY_SIM_JID" "$DENSITY_PLOT_JID"
}

submit_two_jump_variant() {
    local intensity_scale="$1"
    local learning_rates="$2"
    local intensity_tag="${intensity_scale/./p}"
    local foldername="TwoStageTech_TechIntensityScale_${intensity_tag}_LR_${SCHEDULE_TYPE}_${learning_rates}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    local job_name="${PREFIX}/${foldername}"

    prepare_new_variant "$foldername"

    for stage in PostDamagePostTech PreDamagePostTech; do
        cp -a "${PRETRAINED_FOLDER}/${stage}" "${job_name}/${stage}"
    done

    cat > "${job_name}/run_manifest.txt" <<EOF
Run: two technology jump additional learning-rate sweep
Created: $(date)
Tech jump probability pi: 0.04
Tech jump intensity scale: ${intensity_scale}
Model directory: ${MODEL_DIR}
Pretrained source: ${PRETRAINED_FOLDER}
Post-tech learning rates: ${LEARNING_RATES_POSTTECH}
Active-regime learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Iterations per active stage: ${NUM_ITERATIONS}
Copied unchanged stages: PostDamagePostTech, PreDamagePostTech
Submitted stages: PostDamageIntermTech, PostDamagePreTech, PreDamageIntermTech, PreDamagePreTech
Deterministic simulation xi: ${SIMULATION_XIS}
Conditional first-jump density plots: yes
EOF

    jid_post_interm=$(submit_two_jump_stage "PostDamageIntermTech" "$foldername" "$intensity_scale" "$learning_rates")
    jid_post_pre=$(submit_two_jump_stage "PostDamagePreTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_interm}")
    jid_pre_interm=$(submit_two_jump_stage "PreDamageIntermTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_interm}")
    jid_pre_pre=$(submit_two_jump_stage "PreDamagePreTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_pre}:${jid_pre_interm}")

    submit_density_diagnostics "$foldername" "$jid_pre_pre"

    cat >> "${job_name}/run_manifest.txt" <<EOF
PostDamageIntermTech job: ${jid_post_interm}
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_interm})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre}:${jid_pre_interm})
First-jump density simulation job: ${DENSITY_SIM_JID} (afterok:${jid_pre_pre})
First-jump density plot job: ${DENSITY_PLOT_JID} (afterok:${DENSITY_SIM_JID})
EOF

    printf 'Submitted two-jump scale=%s LR=%s: training final=%s, density=%s/%s\n' \
        "$intensity_scale" "$learning_rates" "$jid_pre_pre" "$DENSITY_SIM_JID" "$DENSITY_PLOT_JID"
}

for intensity_scale in "${TECH_JUMP_INTENSITY_SCALES[@]}"; do
    for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
        submit_one_jump_variant "$intensity_scale" "$learning_rates"
        submit_two_jump_variant "$intensity_scale" "$learning_rates"
    done
done
