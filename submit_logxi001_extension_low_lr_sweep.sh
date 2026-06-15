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
NUM_ITERATIONS="1000000"
TRAINING_LENGTH_LABEL="2x500000"
LOGGING_FREQUENCY="1000"
LOGXI_MIN="-5.0"
LOGXI_MAX="5.0"
SIMULATION_XIS="0.01,0.05,0.1,148.6"
SIMULATION_Y0="1.2"

LEARNING_RATE_GRID=(
    "10e-7,10e-7,10e-7,10e-7"
    "40e-7,40e-7,40e-7,40e-7"
    "20e-7,20e-7,20e-7,20e-7"
)

# Fields: mode|folder|tech_intensity_scale|pi
# mode=one_interm uses PostDamageIntermTech/PreDamageIntermTech.
# mode=one_direct uses PostDamagePreTech/PreDamagePreTech with pi=1.
# mode=two uses the full two-stage technology jump tree.
BASE_MODELS=(
    "one_interm|OneTechJump_Pi_0p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|1.0|0.0"
    "one_interm|OneTechJump_Pi_0p0_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|2.0|0.0"
    "one_direct|OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|1.0|1.0"
    "one_direct|OneTechJump_Pi_1p0_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|2.0|1.0"
    "two|TwoStageTech_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|1.0|0.04"
    "two|TwoStageTech_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|1.0|0.04"
    "two|TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|2.0|0.04"
    "two|TwoStageTech_RDIntensityScale_0p5_fullsweep_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|0.5|0.04"
)

logxi_tag="m${LOGXI_MIN#-}"
logxi_tag="${logxi_tag/./p}"

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local pretrained_folder="$3"
    local intensity_scale="$4"
    local tech_probability="$5"
    local learning_rates="$6"
    local dependency="${7:-}"
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
    TECH_JUMP_INTENSITY_SCALE="$intensity_scale" \
    TECH_JUMP_PROBABILITY="$tech_probability" \
    LOGXI_MIN="$LOGXI_MIN" \
    LOGXI_MAX="$LOGXI_MAX" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="logxiLow_${stage}" \
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
        --job-name=logxiLowSim \
        --output="./job-outs/${foldername}/LogXi001LowLRSimulation-%j.out" \
        --error="./job-outs/${foldername}/LogXi001LowLRSimulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${sim_jid}" \
        --job-name=logxiLowPlot \
        --output="./job-outs/${foldername}/LogXi001LowLRPlot-%j.out" \
        --error="./job-outs/${foldername}/LogXi001LowLRPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    cat > "${export_folder}/logxi001_low_lr_deterministic_jobs.txt" <<EOF
Submitted: $(date)
Training dependency: ${training_jid}
Simulation Y0: ${SIMULATION_Y0}
Simulation xi: ${SIMULATION_XIS}
Simulation job: ${sim_jid}
Plot job: ${plot_jid} (afterok:${sim_jid})
EOF
}

prepare_variant() {
    local foldername="$1"
    local export_folder="${PREFIX}/${foldername}"

    if [ -e "${export_folder}/run_manifest.txt" ]; then
        echo "Skipping existing variant: ${foldername}" >&2
        return 1
    fi

    mkdir -p "$export_folder" "./job-outs/${foldername}"
    return 0
}

check_stages() {
    local pretrained_folder="$1"
    shift

    for stage in "$@"; do
        if [ ! -f "${pretrained_folder}/${stage}/checkpoint" ]; then
            echo "Missing pretrained checkpoint for ${stage}: ${pretrained_folder}" >&2
            return 1
        fi
    done
}

write_manifest_header() {
    local export_folder="$1"
    local mode_label="$2"
    local pretrained_folder="$3"
    local intensity_scale="$4"
    local tech_probability="$5"
    local learning_rates="$6"
    local submitted_stages="$7"

    cat > "${export_folder}/run_manifest.txt" <<EOF
Run: log-xi range extension low-learning-rate continuation
Created: $(date)
Mode: ${mode_label}
Pretrained source: ${pretrained_folder}
Extended logxi range: [${LOGXI_MIN}, ${LOGXI_MAX}]
Fine-tune learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Training length requested: ${TRAINING_LENGTH_LABEL}
Iterations per stage: ${NUM_ITERATIONS}
Tech jump intensity scale: ${intensity_scale}
Tech jump probability pi: ${tech_probability}
Submitted stages: ${submitted_stages}
Deterministic simulation Y0: ${SIMULATION_Y0}
Deterministic simulation xi: ${SIMULATION_XIS}
EOF
}

submit_one_interm_variant() {
    local base_folder="$1"
    local intensity_scale="$2"
    local tech_probability="$3"
    local learning_rates="$4"
    local pretrained_folder="${PREFIX}/${base_folder}"
    local foldername="${base_folder}_logximin_${logxi_tag}_lowLR_${learning_rates}_iters${NUM_ITERATIONS}"
    local export_folder="${PREFIX}/${foldername}"

    check_stages "$pretrained_folder" PostDamagePostTech PostDamageIntermTech PreDamagePostTech PreDamageIntermTech
    prepare_variant "$foldername" || return 0
    write_manifest_header "$export_folder" "one technology jump, intermediate-state target" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "PostDamagePostTech, PostDamageIntermTech, PreDamagePostTech, PreDamageIntermTech"

    jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates")
    jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_post_interm}:${jid_pre_post}")
    submit_density_diagnostics "$export_folder" "$jid_pre_interm"

    cat >> "${export_folder}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PostDamageIntermTech job: ${jid_post_interm} (afterok:${jid_post_post})
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm}:${jid_pre_post})
Deterministic simulation job: ${sim_jid} (afterok:${jid_pre_interm})
Deterministic plot job: ${plot_jid} (afterok:${sim_jid})
EOF

    printf 'Submitted low-LR logxi one-interm base=%s LR=%s final=%s sim/plot=%s/%s\n' "$base_folder" "$learning_rates" "$jid_pre_interm" "$sim_jid" "$plot_jid"
}

submit_one_direct_variant() {
    local base_folder="$1"
    local intensity_scale="$2"
    local tech_probability="$3"
    local learning_rates="$4"
    local pretrained_folder="${PREFIX}/${base_folder}"
    local foldername="${base_folder}_logximin_${logxi_tag}_lowLR_${learning_rates}_iters${NUM_ITERATIONS}"
    local export_folder="${PREFIX}/${foldername}"

    check_stages "$pretrained_folder" PostDamagePostTech PostDamagePreTech PreDamagePostTech PreDamagePreTech
    prepare_variant "$foldername" || return 0
    write_manifest_header "$export_folder" "one technology jump, direct breakthrough target" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "PostDamagePostTech, PostDamagePreTech, PreDamagePostTech, PreDamagePreTech"

    jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates")
    jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_post_post}")
    jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_pre_post}:${jid_post_pre}")
    submit_density_diagnostics "$export_folder" "$jid_pre_pre"

    cat >> "${export_folder}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_post})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_pre_post}:${jid_post_pre})
Deterministic simulation job: ${sim_jid} (afterok:${jid_pre_pre})
Deterministic plot job: ${plot_jid} (afterok:${sim_jid})
EOF

    printf 'Submitted low-LR logxi one-direct base=%s LR=%s final=%s sim/plot=%s/%s\n' "$base_folder" "$learning_rates" "$jid_pre_pre" "$sim_jid" "$plot_jid"
}

submit_two_variant() {
    local base_folder="$1"
    local intensity_scale="$2"
    local tech_probability="$3"
    local learning_rates="$4"
    local pretrained_folder="${PREFIX}/${base_folder}"
    local foldername="${base_folder}_logximin_${logxi_tag}_lowLR_${learning_rates}_iters${NUM_ITERATIONS}"
    local export_folder="${PREFIX}/${foldername}"

    check_stages "$pretrained_folder" PostDamagePostTech PostDamageIntermTech PostDamagePreTech PreDamagePostTech PreDamageIntermTech PreDamagePreTech
    prepare_variant "$foldername" || return 0
    write_manifest_header "$export_folder" "two technology jumps" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "PostDamagePostTech, PostDamageIntermTech, PostDamagePreTech, PreDamagePostTech, PreDamageIntermTech, PreDamagePreTech"

    jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates")
    jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_post_post}")
    jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_post_post}:${jid_post_interm}")
    jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_pre_post}:${jid_post_interm}")
    jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "$pretrained_folder" "$intensity_scale" "$tech_probability" "$learning_rates" "afterok:${jid_pre_post}:${jid_post_pre}:${jid_pre_interm}")
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

    printf 'Submitted low-LR logxi two-jump base=%s LR=%s final=%s sim/plot=%s/%s\n' "$base_folder" "$learning_rates" "$jid_pre_pre" "$sim_jid" "$plot_jid"
}

for spec in "${BASE_MODELS[@]}"; do
    IFS='|' read -r mode base_folder intensity_scale tech_probability <<< "$spec"
    for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
        case "$mode" in
            one_interm)
                submit_one_interm_variant "$base_folder" "$intensity_scale" "$tech_probability" "$learning_rates"
                ;;
            one_direct)
                submit_one_direct_variant "$base_folder" "$intensity_scale" "$tech_probability" "$learning_rates"
                ;;
            two)
                submit_two_variant "$base_folder" "$intensity_scale" "$tech_probability" "$learning_rates"
                ;;
            *)
                echo "Unknown mode in BASE_MODELS: ${mode}" >&2
                exit 2
                ;;
        esac
    done
done
