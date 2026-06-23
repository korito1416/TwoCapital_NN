#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="${PREFIX:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001}"
STAGE_SCRIPT="${STAGE_SCRIPT:-one_tech_jump_stage.sbatch}"
DETERMINISTIC_SCRIPT="${DETERMINISTIC_SCRIPT:-deterministic_stage.sbatch}"
MODEL_DIR="${MODEL_DIR:-models}"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_NEURONS="${NUM_NEURONS:-32}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
SCHEDULE_TYPE="${SCHEDULE_TYPE:-warmup_cosine}"
LEARNING_RATES="${LEARNING_RATES:-10e-6,40e-4}"
SIMULATION_XIS="${SIMULATION_XIS:-0.05,0.1,0.3,148.6}"
SIMULATION_Y0="${SIMULATION_Y0:-1.2}"

# tag|technology intensity|corresponding two-stage source folder
VARIANTS=(
    "1p0|1.0|TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
    "2p0|2.0|TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
)

copy_inherited_stage() {
    local source_folder="$1"
    local target_folder="$2"
    local stage="$3"

    if [ ! -f "${source_folder}/${stage}/checkpoint" ]; then
        echo "Missing inherited checkpoint: ${source_folder}/${stage}" >&2
        exit 2
    fi
    cp -a "${source_folder}/${stage}" "${target_folder}/${stage}"
}

submit_training_stage() {
    local stage="$1"
    local foldername="$2"
    local source_folder="$3"
    local intensity_scale="$4"
    local dependency="${5:-}"
    local dependency_args=()

    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    PREFIX="$PREFIX" \
    PRETRAINED_FOLDER="$source_folder" \
    MODEL_DIR="$MODEL_DIR" \
    FOLDERNAME="$foldername" \
    JOB_NAME="${PREFIX}/${foldername}" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_NEURONS="$NUM_NEURONS" \
    NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
    NUM_ITERATIONS_POSTTECH="$NUM_ITERATIONS" \
    NUM_ITERATIONS_INTERMTECH="$NUM_ITERATIONS" \
    LOGGING_FREQUENCY="$LOGGING_FREQUENCY" \
    LEARNING_RATES_POSTTECH="$LEARNING_RATES" \
    LEARNING_RATES_INTERMTECH="$LEARNING_RATES" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$intensity_scale" \
    TECH_JUMP_PROBABILITY=1.0 \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="pi1Fast_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r intensity_tag intensity_scale source_name <<< "$variant"
    source_folder="${PREFIX}/${source_name}"
    foldername="OneTechJump_Pi_1p0_InheritedPostTech_TechIntensityScale_${intensity_tag}_LR_${SCHEDULE_TYPE}_${LEARNING_RATES}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    export_folder="${PREFIX}/${foldername}"

    if [ -e "${export_folder}/run_manifest.txt" ]; then
        echo "Refusing to overwrite existing fast one-jump variant: ${foldername}" >&2
        exit 2
    fi

    mkdir -p "$export_folder" "./job-outs/${foldername}"
    copy_inherited_stage "$source_folder" "$export_folder" PostDamagePostTech
    copy_inherited_stage "$source_folder" "$export_folder" PreDamagePostTech

    cat > "${export_folder}/run_manifest.txt" <<EOF
Run: one technology jump direct-to-breakthrough with inherited post-tech regimes
Created: $(date)
Tech jump probability pi: 1.0
Tech jump intensity scale: ${intensity_scale}
Corresponding two-stage source: ${source_folder}
Inherited without retraining: PostDamagePostTech, PreDamagePostTech
Newly trained: PostDamagePreTech, PreDamagePreTech
Excluded: PostDamageIntermTech, PreDamageIntermTech
Model directory: ${MODEL_DIR}
Training batch size: ${BATCH_SIZE}
Learning rates: ${LEARNING_RATES}
Learning-rate schedule: ${SCHEDULE_TYPE}
Iterations per trained stage: ${NUM_ITERATIONS}
Simulation Y0: ${SIMULATION_Y0}
Simulation xi: ${SIMULATION_XIS}
EOF

    jid_post_pre=$(submit_training_stage \
        PostDamagePreTech "$foldername" "$source_folder" "$intensity_scale")
    jid_pre_pre=$(submit_training_stage \
        PreDamagePreTech "$foldername" "$source_folder" "$intensity_scale" \
        "afterok:${jid_post_pre}")

    sim_jid=$(EXPORT_FOLDER="$export_folder" \
        SIMULATION_XIS="$SIMULATION_XIS" \
        SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${jid_pre_pre}" \
        --job-name=pi1FastDetSim \
        --output="./job-outs/${foldername}/DeterministicSimulation-%j.out" \
        --error="./job-outs/${foldername}/DeterministicSimulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$DETERMINISTIC_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="$export_folder" \
        SIMULATION_XIS="$SIMULATION_XIS" \
        SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${sim_jid}" \
        --job-name=pi1FastDetPlot \
        --output="./job-outs/${foldername}/DeterministicPlot-%j.out" \
        --error="./job-outs/${foldername}/DeterministicPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$DETERMINISTIC_SCRIPT")

    cat >> "${export_folder}/run_manifest.txt" <<EOF
PostDamagePreTech job: ${jid_post_pre}
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre})
Deterministic simulation job: ${sim_jid} (afterok:${jid_pre_pre})
Deterministic plot job: ${plot_jid} (afterok:${sim_jid})
EOF

    printf '%s: train=%s/%s simulation=%s plot=%s\n' \
        "$foldername" "$jid_post_pre" "$jid_pre_pre" "$sim_jid" "$plot_jid"
done
