#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="stochastic_jump_stage.sbatch"

XIS=(${STOCHASTIC_XIS:-0.05 0.1 0.3 148.6})
ARRAY_RANGE="${ARRAY_RANGE:-1-100}"
N_PATHS="${N_PATHS:-100}"
YEARS="${YEARS:-60}"
DT="${DT:-0.08333333333333333}"
Y0="${Y0:-1.2}"

# Fields: short_label|folder
MODELS=(
    "OneTechPi1S1|OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
    "TwoStageBase|TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
    "TwoStageS2|TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
)

required_stages_for_folder() {
    local folder="$1"
    if [[ "$folder" == OneTechJump_* ]]; then
        printf '%s\n' PostDamagePostTech PostDamagePreTech PreDamagePostTech PreDamagePreTech
    else
        printf '%s\n' PostDamagePostTech PostDamageIntermTech PostDamagePreTech PreDamagePostTech PreDamageIntermTech PreDamagePreTech
    fi
}

check_model_ready() {
    local export_folder="$1"
    local stage
    while read -r stage; do
        if [ ! -f "${export_folder}/${stage}/checkpoint" ]; then
            echo "Missing checkpoint for ${stage}: ${export_folder}" >&2
            return 1
        fi
    done < <(required_stages_for_folder "$(basename "$export_folder")")
}

for spec in "${MODELS[@]}"; do
    IFS='|' read -r label folder <<< "$spec"
    export_folder="${PREFIX}/${folder}"
    check_model_ready "$export_folder"

    for xi in "${XIS[@]}"; do
        log_dir="./job-outs/StochasticJump/${label}/xi_${xi}"
        mkdir -p "$log_dir"
        mkdir -p "${export_folder}/SimulationResults/paths_ξ_${xi}"

        jid=$(EXPORT_FOLDER="$export_folder" \
            OUTPUT_ROOT="${export_folder}/SimulationResults" \
            XI="$xi" \
            N_PATHS="$N_PATHS" \
            YEARS="$YEARS" \
            DT="$DT" \
            Y0="$Y0" \
            sbatch --parsable \
                --array="${ARRAY_RANGE}" \
                --job-name="stoch_${label}_xi_${xi}" \
                --output="${log_dir}/%A_%a.out" \
                --error="${log_dir}/%A_%a.err" \
                --export=ALL \
                "$STAGE_SCRIPT")

        cat > "${export_folder}/SimulationResults/stochastic_jobs_xi_${xi}.txt" <<EOF
Submitted: $(date)
Job id: ${jid}
Array range: ${ARRAY_RANGE}
Model label: ${label}
Export folder: ${export_folder}
xi: ${xi}
N paths per array task: ${N_PATHS}
Years: ${YEARS}
dt: ${DT}
Y0: ${Y0}
Output folder: ${export_folder}/SimulationResults/paths_ξ_${xi}
Script: ${STAGE_SCRIPT}
EOF
        printf 'Submitted stochastic simulation %s xi=%s job=%s array=%s\n' "$label" "$xi" "$jid" "$ARRAY_RANGE"
    done
done
