#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="one_tech_jump_stage.sbatch"
DIAGNOSTIC_STAGE_SCRIPT="deterministic_stage.sbatch"
MODEL_DIR="models"
SCHEDULE_TYPE="warmup_cosine"
BATCH_SIZE="128"
NUM_NEURONS="32"
NUM_HIDDEN_LAYERS="4"
NUM_ITERATIONS="500000"
TRAINING_LENGTH_LABEL="500000"
LOGGING_FREQUENCY="1000"
LOGXI_MIN="-4.0"
LOGXI_MAX="5.0"
TECH_JUMP_INTENSITY_SCALE="0.5"
TECH_JUMP_PROBABILITY="0.04"
SIMULATION_Y0="${SIMULATION_Y0:-1.2}"
# 148.6 is the existing deterministic-simulation proxy for xi=infinity.
SIMULATION_XIS="${SIMULATION_XIS:-0.02,0.05,0.1,0.3,148.6}"

LEARNING_RATE_GRID=(
    "10e-7,10e-7,10e-7,10e-7"
    "20e-8,20e-8,20e-8,20e-8"
    "10e-8,10e-8,10e-8,10e-8"
)

STAGES=(
    PostDamagePostTech
    PostDamageIntermTech
    PostDamagePreTech
    PreDamagePostTech
    PreDamageIntermTech
    PreDamagePreTech
)

logxi_tag="m${LOGXI_MIN#-}"
logxi_tag="${logxi_tag/./p}"

check_stages() {
    local pretrained_folder="$1"

    for stage in "${STAGES[@]}"; do
        if [ ! -f "${pretrained_folder}/${stage}/checkpoint" ]; then
            return 1
        fi
    done
}

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local pretrained_folder="$3"
    local learning_rates="$4"
    local dependency="${5:-}"
    local dependency_args=()

    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    PREFIX="$PREFIX" \
    PRETRAINED_FOLDER="$pretrained_folder" \
    MODEL_DIR="$MODEL_DIR" \
    FOLDERNAME="$foldername" \
    JOB_NAME="${PREFIX}/${foldername}" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_NEURONS="$NUM_NEURONS" \
    NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
    NUM_ITERATIONS_POSTTECH="$NUM_ITERATIONS" \
    NUM_ITERATIONS_INTERMTECH="$NUM_ITERATIONS" \
    LOGGING_FREQUENCY="$LOGGING_FREQUENCY" \
    LEARNING_RATES_POSTTECH="$learning_rates" \
    LEARNING_RATES_INTERMTECH="$learning_rates" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" \
    TECH_JUMP_PROBABILITY="$TECH_JUMP_PROBABILITY" \
    LOGXI_MIN="$LOGXI_MIN" \
    LOGXI_MAX="$LOGXI_MAX" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="logxiM4_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="${stage}",LOGXI_MIN="${LOGXI_MIN}",LOGXI_MAX="${LOGXI_MAX}" \
        "$STAGE_SCRIPT"
}

submit_density_diagnostics() {
    local export_folder="$1"
    local training_jid="$2"
    local foldername

    foldername="$(basename "$export_folder")"
    mkdir -p "./job-outs/${foldername}"

    sim_jid=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${training_jid}" \
        --job-name=logxiM4Sim \
        --output="./job-outs/${foldername}/LogXiM4Simulation-%j.out" \
        --error="./job-outs/${foldername}/LogXiM4Simulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${sim_jid}" \
        --job-name=logxiM4Plot \
        --output="./job-outs/${foldername}/LogXiM4Plot-%j.out" \
        --error="./job-outs/${foldername}/LogXiM4Plot-%j.err" \
        --export=ALL,MODE=plot \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    cat > "${export_folder}/logxi_m4_deterministic_jobs.txt" <<EOF
Submitted: $(date)
Training dependency: ${training_jid}
Simulation Y0: ${SIMULATION_Y0}
Simulation xi: ${SIMULATION_XIS}
Simulation job: ${sim_jid}
Plot job: ${plot_jid} (afterok:${sim_jid})
EOF
}

submit_two_variant() {
    local pretrained_folder="$1"
    local learning_rates="$2"
    local base_folder foldername export_folder

    base_folder="$(basename "$pretrained_folder")"
    foldername="${base_folder}_logximin_${logxi_tag}_rd05_${learning_rates}_iters${NUM_ITERATIONS}"
    export_folder="${PREFIX}/${foldername}"

    if [ -e "${export_folder}/run_manifest.txt" ]; then
        echo "Skipping existing variant: ${foldername}" >&2
        return 0
    fi

    mkdir -p "$export_folder" "./job-outs/${foldername}"

    cat > "${export_folder}/run_manifest.txt" <<EOF
Run: log-xi range extension m4 RD-0.5 continuation
Created: $(date)
Mode: two technology jumps, RD intensity scale 0.5
Pretrained source: ${pretrained_folder}
Extended logxi range: [${LOGXI_MIN}, ${LOGXI_MAX}]
Fine-tune learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Training length requested: ${TRAINING_LENGTH_LABEL}
Iterations per stage: ${NUM_ITERATIONS}
Tech jump intensity scale: ${TECH_JUMP_INTENSITY_SCALE}
Tech jump probability pi: ${TECH_JUMP_PROBABILITY}
Submitted stages: ${STAGES[*]}
Deterministic simulation Y0: ${SIMULATION_Y0}
Deterministic simulation xi: ${SIMULATION_XIS}
EOF

    jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername" "$pretrained_folder" "$learning_rates")
    jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "$pretrained_folder" "$learning_rates" "afterok:${jid_post_post}")
    jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "$pretrained_folder" "$learning_rates" "afterok:${jid_post_post}:${jid_post_interm}")
    jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "$pretrained_folder" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "$pretrained_folder" "$learning_rates" "afterok:${jid_pre_post}:${jid_post_interm}")
    jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "$pretrained_folder" "$learning_rates" "afterok:${jid_pre_post}:${jid_post_pre}:${jid_pre_interm}")
    submit_density_diagnostics "$export_folder" "$jid_pre_pre"

    cat >> "${export_folder}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PostDamageIntermTech job: ${jid_post_interm} (afterok:${jid_post_post})
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_post}:${jid_post_interm})
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_pre_post}:${jid_post_interm})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_pre_post}:${jid_post_pre}:${jid_pre_interm})
Deterministic simulation job: ${sim_jid} (afterok:${jid_pre_pre})
Deterministic plot job: ${plot_jid} (afterok:${sim_jid})
EOF

    printf 'Submitted logxi m4 RD-0.5 base=%s LR=%s final=%s sim/plot=%s/%s\n' "$base_folder" "$learning_rates" "$jid_pre_pre" "$sim_jid" "$plot_jid"
}

mapfile -t PRETRAINED_SOURCES < <(
    find "$PREFIX" -maxdepth 1 -type d -name 'TwoStageTech_RDIntensityScale_0p5*' \
        ! -name '*_logximin_m4p0_*' \
        | sort
)

complete_count=0
for pretrained_folder in "${PRETRAINED_SOURCES[@]}"; do
    if ! check_stages "$pretrained_folder"; then
        echo "Skipping incomplete pretrained source: ${pretrained_folder}" >&2
        continue
    fi

    complete_count=$((complete_count + 1))
    for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
        submit_two_variant "$pretrained_folder" "$learning_rates"
    done
done

if [ "$complete_count" -eq 0 ]; then
    echo "No complete RD-0.5 pretrained sources found." >&2
    exit 1
fi
