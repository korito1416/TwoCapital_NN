#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
PRETRAINED_FOLDER="${PREFIX}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
ONE_JUMP_STAGE_SCRIPT="one_tech_jump_stage.sbatch"
DIAGNOSTIC_STAGE_SCRIPT="deterministic_stage.sbatch"
MODEL_DIR="models"
SCHEDULE_TYPE="warmup_cosine"
BATCH_SIZE="128"
NUM_NEURONS="32"
NUM_HIDDEN_LAYERS="4"
NUM_ITERATIONS="1000000"
LEARNING_RATES_POSTTECH="10e-6,40e-4"
TECH_JUMP_PROBABILITY_ONE_JUMP="1.0"
SIMULATION_XIS="0.01,0.05,0.1,148.6"
SIMULATION_Y0="1.2"

TECH_JUMP_INTENSITY_SCALES=(
    "1.0"
    "2.0"
)

LEARNING_RATE_GRID=(
    "10e-6,10e-4"
    "10e-6,40e-4"
    "10e-6,10e-5"
    "10e-6,40e-5"
    "10e-5,10e-5,10e-5,10e-5"
    "40e-5,40e-5,40e-5,40e-5"
    "10e-6,10e-6,10e-6,10e-6"
    "40e-6,40e-6,40e-6,40e-6"
)

EXISTING_DIAGNOSTIC_FOLDERS=(
    "${PREFIX}/OneTechJump_Pi_0p0_TechIntensityScale_1p0_LR_${SCHEDULE_TYPE}_10e-6,40e-4_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    "${PREFIX}/OneTechJump_Pi_0p0_TechIntensityScale_2p0_LR_${SCHEDULE_TYPE}_10e-6,40e-4_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    "${PREFIX}/TwoStageTech_LR_${SCHEDULE_TYPE}_10e-6,40e-4_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    "${PREFIX}/TwoStageTech_TechIntensityScale_1p0_LR_${SCHEDULE_TYPE}_10e-6,40e-4_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    "${PREFIX}/TwoStageTech_TechIntensityScale_2p0_LR_${SCHEDULE_TYPE}_10e-6,40e-4_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    "${PREFIX}/TwoStageTech_RDIntensityScale_0p5_fullsweep_LR_${SCHEDULE_TYPE}_10e-6,40e-4_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
)

for stage in PostDamagePostTech PreDamagePostTech PostDamagePreTech PreDamagePreTech; do
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
        --job-name="onePi1_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$ONE_JUMP_STAGE_SCRIPT"
}

submit_density_diagnostics() {
    local export_folder="$1"
    local dependency="${2:-}"
    local foldername
    local dependency_args=()

    foldername="$(basename "$export_folder")"
    mkdir -p "./job-outs/${foldername}"
    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    DENSITY_SIM_JID=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name=firstJumpY12Sim \
        --output="./job-outs/${foldername}/FirstJumpDensityY12Simulation-%j.out" \
        --error="./job-outs/${foldername}/FirstJumpDensityY12Simulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    DENSITY_PLOT_JID=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${DENSITY_SIM_JID}" \
        --job-name=firstJumpY12Plot \
        --output="./job-outs/${foldername}/FirstJumpDensityY12Plot-%j.out" \
        --error="./job-outs/${foldername}/FirstJumpDensityY12Plot-%j.err" \
        --export=ALL,MODE=plot \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    cat > "${export_folder}/first_jump_density_y12_jobs.txt" <<EOF
Submitted: $(date)
Training dependency: ${dependency:-none}
Simulation Y0: ${SIMULATION_Y0}
Simulation scenarios xi: ${SIMULATION_XIS}
Simulation job: ${DENSITY_SIM_JID}
Plot job: ${DENSITY_PLOT_JID} (afterok:${DENSITY_SIM_JID})
EOF
}

submit_direct_one_jump_variant() {
    local intensity_scale="$1"
    local learning_rates="$2"
    local intensity_tag="${intensity_scale/./p}"
    local foldername="OneTechJump_Pi_1p0_TechIntensityScale_${intensity_tag}_LR_${SCHEDULE_TYPE}_${learning_rates}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    local job_name="${PREFIX}/${foldername}"

    prepare_new_variant "$foldername"

    cat > "${job_name}/run_manifest.txt" <<EOF
Run: one technology jump direct-to-breakthrough learning-rate sweep
Created: $(date)
Tech jump probability pi: ${TECH_JUMP_PROBABILITY_ONE_JUMP}
Tech jump intensity scale: ${intensity_scale}
Model directory: ${MODEL_DIR}
Pretrained source: ${PRETRAINED_FOLDER}
Post-tech learning rates: ${LEARNING_RATES_POSTTECH}
Pre-tech learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Iterations per stage: ${NUM_ITERATIONS}
Submitted stages: PostDamagePostTech, PreDamagePostTech, PostDamagePreTech, PreDamagePreTech
Excluded stages: PostDamageIntermTech, PreDamageIntermTech
Deterministic simulation Y0: ${SIMULATION_Y0}
Deterministic simulation xi: ${SIMULATION_XIS}
Conditional first-jump density accounting: yes
EOF

    jid_post_post=$(submit_one_jump_stage "PostDamagePostTech" "$foldername" "$intensity_scale" "$learning_rates")
    jid_pre_post=$(submit_one_jump_stage "PreDamagePostTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_post}")
    jid_post_pre=$(submit_one_jump_stage "PostDamagePreTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_pre=$(submit_one_jump_stage "PreDamagePreTech" "$foldername" "$intensity_scale" "$learning_rates" "afterok:${jid_pre_post}:${jid_post_pre}")

    cat >> "${job_name}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_post})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_pre_post}:${jid_post_pre})
EOF

    if [ "$learning_rates" = "10e-6,40e-4" ]; then
        submit_density_diagnostics "$job_name" "afterok:${jid_pre_pre}"
        cat >> "${job_name}/run_manifest.txt" <<EOF
First-jump Y0=1.2 simulation job: ${DENSITY_SIM_JID} (afterok:${jid_pre_pre})
First-jump Y0=1.2 plot job: ${DENSITY_PLOT_JID} (afterok:${DENSITY_SIM_JID})
EOF
    fi

    printf 'Submitted pi=1 direct one-jump scale=%s LR=%s: final=%s\n' \
        "$intensity_scale" "$learning_rates" "$jid_pre_pre"
}

for export_folder in "${EXISTING_DIAGNOSTIC_FOLDERS[@]}"; do
    if [ ! -d "$export_folder" ]; then
        echo "Skipping missing diagnostic folder: $export_folder" >&2
        continue
    fi
    submit_density_diagnostics "$export_folder"
    printf 'Submitted Y0=1.2 diagnostics for existing folder: %s (%s/%s)\n' \
        "$(basename "$export_folder")" "$DENSITY_SIM_JID" "$DENSITY_PLOT_JID"
done

for intensity_scale in "${TECH_JUMP_INTENSITY_SCALES[@]}"; do
    for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
        submit_direct_one_jump_variant "$intensity_scale" "$learning_rates"
    done
done
