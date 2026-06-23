#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

ROOT="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
PREFIX="${ROOT}/output_sensitivity_001"
PRETRAINED_FOLDER="${ROOT}/output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
STAGE_SCRIPT="sensitivity_stage.sbatch"
DIAGNOSTIC_STAGE_SCRIPT="deterministic_stage.sbatch"
MODEL_DIR="models"
SCHEDULE_TYPE="warmup_cosine"
BATCH_SIZE="128"
NUM_NEURONS="32"
NUM_HIDDEN_LAYERS="4"
NUM_ITERATIONS="1000000"
TECH_JUMP_INTENSITY_SCALE="1.0"
TECH_JUMP_PROBABILITY="0.04"
SIMULATION_XIS="0.05,0.1,0.3,148.6"
SIMULATION_Y0="1.2"

LEARNING_RATE_GRID=(
    "10e-6,10e-4"
    "10e-6,40e-4"
    "10e-6,40e-5"
    "10e-6,10e-5"
)

for stage in PostDamagePostTech PostDamageIntermTech PostDamagePreTech PreDamagePostTech PreDamageIntermTech PreDamagePreTech; do
    if [ ! -f "${PRETRAINED_FOLDER}/${stage}/checkpoint" ]; then
        echo "Missing baseline checkpoint: ${PRETRAINED_FOLDER}/${stage}" >&2
        exit 2
    fi
done

mkdir -p "$PREFIX" "./job-outs/sensitivity"

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local learning_rates="$3"
    local dependency="${4:-}"
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
    NUM_ITERATIONS="$NUM_ITERATIONS" \
    LEARNING_RATES="$learning_rates" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" \
    TECH_JUMP_PROBABILITY="$TECH_JUMP_PROBABILITY" \
    MODEL_SIGMA_D="${MODEL_SIGMA_D:-}" \
    MODEL_SIGMA_G="${MODEL_SIGMA_G:-}" \
    MODEL_GAMMA_D="${MODEL_GAMMA_D:-}" \
    MODEL_GAMMA_G="${MODEL_GAMMA_G:-}" \
    MODEL_THETA_D="${MODEL_THETA_D:-}" \
    MODEL_THETA_G="${MODEL_THETA_G:-}" \
    MODEL_PSI0="${MODEL_PSI0:-}" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="sens_${EXPERIMENT_TAG}_${stage}" \
        --output="./job-outs/sensitivity/${foldername}/${stage}-%j.out" \
        --error="./job-outs/sensitivity/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

submit_deterministic_jobs() {
    local foldername="$1"
    local training_jid="$2"
    local export_folder="${PREFIX}/${foldername}"

    SIMULATION_JID=$(EXPORT_FOLDER="$export_folder" \
        SIMULATION_XIS="$SIMULATION_XIS" \
        SIMULATION_Y0="$SIMULATION_Y0" \
        MODEL_SIGMA_D="${MODEL_SIGMA_D:-}" \
        MODEL_SIGMA_G="${MODEL_SIGMA_G:-}" \
        MODEL_GAMMA_D="${MODEL_GAMMA_D:-}" \
        MODEL_GAMMA_G="${MODEL_GAMMA_G:-}" \
        MODEL_THETA_D="${MODEL_THETA_D:-}" \
        MODEL_THETA_G="${MODEL_THETA_G:-}" \
        MODEL_PSI0="${MODEL_PSI0:-}" \
        sbatch --parsable \
        --dependency="afterok:${training_jid}" \
        --job-name="sens_${EXPERIMENT_TAG}_sim" \
        --output="./job-outs/sensitivity/${foldername}/DeterministicSimulation-%j.out" \
        --error="./job-outs/sensitivity/${foldername}/DeterministicSimulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$DIAGNOSTIC_STAGE_SCRIPT")

    PLOT_JID=$(EXPORT_FOLDER="$export_folder" \
        sbatch --parsable \
        --dependency="afterok:${SIMULATION_JID}" \
        --job-name="sens_${EXPERIMENT_TAG}_plot" \
        --output="./job-outs/sensitivity/${foldername}/DeterministicPlot-%j.out" \
        --error="./job-outs/sensitivity/${foldername}/DeterministicPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$DIAGNOSTIC_STAGE_SCRIPT")
}

submit_experiment_variant() {
    local learning_rates="$1"
    local foldername="TwoStageTech_TechIntensityScale_1p0_Sensitivity_${EXPERIMENT_FOLDER_TAG}_LR_${SCHEDULE_TYPE}_${learning_rates}_${BATCH_SIZE}_neurons_${NUM_NEURONS}_#HiddenLayer_${NUM_HIDDEN_LAYERS}_num_iterations${NUM_ITERATIONS}"
    local export_folder="${PREFIX}/${foldername}"

    if [ -e "${export_folder}/run_manifest.txt" ]; then
        if grep -Eq ' job: [0-9]+' "${export_folder}/run_manifest.txt"; then
            echo "Refusing to duplicate a variant with submitted jobs: ${foldername}" >&2
            return 1
        fi
        echo "Restarting incomplete pre-submission variant: ${foldername}" >&2
    fi

    mkdir -p "$export_folder" "./job-outs/sensitivity/${foldername}"

    cat > "${export_folder}/run_manifest.txt" <<EOF
Run: two-stage, unit-technology-intensity parameter sensitivity
Created: $(date)
Experiment: ${EXPERIMENT_DESCRIPTION}
Exactly one sensitivity exercise is active in this folder.
Baseline source: ${PRETRAINED_FOLDER}
Model directory: ${MODEL_DIR}
Technology jump probability pi: ${TECH_JUMP_PROBABILITY}
Tech jump intensity scale: ${TECH_JUMP_INTENSITY_SCALE}
Learning rates: ${learning_rates}
Learning-rate schedule: ${SCHEDULE_TYPE}
Batch size: ${BATCH_SIZE}
Iterations per trained stage: ${NUM_ITERATIONS}
sigma_d: ${MODEL_SIGMA_D:-0.01}
sigma_g: ${MODEL_SIGMA_G:-0.01}
Gamma_d: ${MODEL_GAMMA_D:-0.060}
Gamma_g: ${MODEL_GAMMA_G:-0.060}
theta_d: ${MODEL_THETA_D:-16.7}
theta_g: ${MODEL_THETA_G:-16.7}
psi0: ${MODEL_PSI0:-0.10583}
Simulation Y0: ${SIMULATION_Y0}
Simulation xi: ${SIMULATION_XIS}
EOF

    local jid_post_post
    local jid_post_interm
    local jid_post_pre
    local jid_pre_post
    local jid_pre_interm
    local jid_pre_pre

    if [ "$COPY_POSTTECH" = "1" ]; then
        cp -a "${PRETRAINED_FOLDER}/PostDamagePostTech" "${export_folder}/PostDamagePostTech"
        cp -a "${PRETRAINED_FOLDER}/PreDamagePostTech" "${export_folder}/PreDamagePostTech"

        jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "$learning_rates")
        jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "$learning_rates" "afterok:${jid_post_interm}")
        jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "$learning_rates" "afterok:${jid_post_interm}")
        jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "$learning_rates" "afterok:${jid_post_pre}:${jid_pre_interm}")

        cat >> "${export_folder}/run_manifest.txt" <<EOF
Copied unchanged stages: PostDamagePostTech, PreDamagePostTech
PostDamageIntermTech job: ${jid_post_interm}
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_interm})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre}:${jid_pre_interm})
EOF
    else
        jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername" "$learning_rates")
        jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "$learning_rates" "afterok:${jid_post_post}")
        jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "$learning_rates" "afterok:${jid_post_interm}")
        jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "$learning_rates" "afterok:${jid_post_post}")
        jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "$learning_rates" "afterok:${jid_post_interm}:${jid_pre_post}")
        jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "$learning_rates" "afterok:${jid_post_pre}:${jid_pre_interm}")

        cat >> "${export_folder}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PostDamageIntermTech job: ${jid_post_interm} (afterok:${jid_post_post})
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_interm})
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm}:${jid_pre_post})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre}:${jid_pre_interm})
EOF
    fi

    submit_deterministic_jobs "$foldername" "$jid_pre_pre"

    cat >> "${export_folder}/run_manifest.txt" <<EOF
Deterministic simulation job: ${SIMULATION_JID} (afterok:${jid_pre_pre})
Deterministic plot job: ${PLOT_JID} (afterok:${SIMULATION_JID})
EOF

    printf '%s|%s|%s|%s|%s\n' \
        "$EXPERIMENT_TAG" "$learning_rates" "$foldername" "$SIMULATION_JID" "$PLOT_JID" \
        >> "${PREFIX}/submitted_jobs.tsv"
    printf 'Submitted experiment=%s LR=%s final=%s sim=%s plot=%s\n' \
        "$EXPERIMENT_TAG" "$learning_rates" "$jid_pre_pre" "$SIMULATION_JID" "$PLOT_JID"
}

if [ ! -f "${PREFIX}/submitted_jobs.tsv" ]; then
    printf 'experiment|learning_rates|folder|simulation_job|plot_job\n' > "${PREFIX}/submitted_jobs.tsv"
fi

# Exercise 1: halve both capital volatilities.
EXPERIMENT_TAG="capvol"
EXPERIMENT_FOLDER_TAG="CapitalVolatility0p005"
EXPERIMENT_DESCRIPTION="sigma_d=sigma_g=0.005; all other parameters at baseline"
MODEL_SIGMA_D="0.005"
MODEL_SIGMA_G="0.005"
MODEL_GAMMA_D=""
MODEL_GAMMA_G=""
MODEL_THETA_D=""
MODEL_THETA_G=""
MODEL_PSI0=""
COPY_POSTTECH="0"
for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
    submit_experiment_variant "$learning_rates"
done

# Exercise 2: halve adjustment costs using Gamma*2 and theta/2.
EXPERIMENT_TAG="adjcost"
EXPERIMENT_FOLDER_TAG="AdjustmentCostHalf_Gamma0p12_Theta8p35"
EXPERIMENT_DESCRIPTION="Gamma_d=Gamma_g=0.12 and theta_d=theta_g=8.35; all other parameters at baseline"
MODEL_SIGMA_D=""
MODEL_SIGMA_G=""
MODEL_GAMMA_D="0.12"
MODEL_GAMMA_G="0.12"
MODEL_THETA_D="8.35"
MODEL_THETA_G="8.35"
MODEL_PSI0=""
COPY_POSTTECH="0"
for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
    submit_experiment_variant "$learning_rates"
done

# Exercise 3: reduce R&D productivity/scaling to psi0=0.05.
EXPERIMENT_TAG="psi0"
EXPERIMENT_FOLDER_TAG="RDScalingPsi0_0p05"
EXPERIMENT_DESCRIPTION="psi0=0.05; all other parameters at baseline"
MODEL_SIGMA_D=""
MODEL_SIGMA_G=""
MODEL_GAMMA_D=""
MODEL_GAMMA_G=""
MODEL_THETA_D=""
MODEL_THETA_G=""
MODEL_PSI0="0.05"
COPY_POSTTECH="1"
for learning_rates in "${LEARNING_RATE_GRID[@]}"; do
    submit_experiment_variant "$learning_rates"
done
