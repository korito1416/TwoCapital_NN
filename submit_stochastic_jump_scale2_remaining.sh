#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

EXPORT_FOLDER="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
LABEL="TwoStageS2"
STAGE_SCRIPT="stochastic_jump_stage.sbatch"

XIS=(${STOCHASTIC_XIS:-0.1 0.3 148.6})
ARRAY_RANGE="${ARRAY_RANGE:-9-100}"
N_PATHS="${N_PATHS:-100}"
YEARS="${YEARS:-60}"
DT="${DT:-0.08333333333333333}"
Y0="${Y0:-1.2}"

for xi in "${XIS[@]}"; do
    log_dir="./job-outs/StochasticJump/${LABEL}/xi_${xi}"
    mkdir -p "$log_dir" "${EXPORT_FOLDER}/SimulationResults/paths_ξ_${xi}"

    jid=$(EXPORT_FOLDER="$EXPORT_FOLDER" \
        OUTPUT_ROOT="${EXPORT_FOLDER}/SimulationResults" \
        XI="$xi" \
        N_PATHS="$N_PATHS" \
        YEARS="$YEARS" \
        DT="$DT" \
        Y0="$Y0" \
        sbatch --parsable \
            --array="${ARRAY_RANGE}" \
            --job-name="stoch_${LABEL}_xi_${xi}" \
            --output="${log_dir}/%A_%a.out" \
            --error="${log_dir}/%A_%a.err" \
            --export=ALL \
            "$STAGE_SCRIPT")

    cat > "${EXPORT_FOLDER}/SimulationResults/stochastic_jobs_xi_${xi}_remaining.txt" <<EOF
Submitted: $(date)
Job id: ${jid}
Array range: ${ARRAY_RANGE}
Model label: ${LABEL}
Export folder: ${EXPORT_FOLDER}
xi: ${xi}
N paths per array task: ${N_PATHS}
Years: ${YEARS}
dt: ${DT}
Y0: ${Y0}
Output folder: ${EXPORT_FOLDER}/SimulationResults/paths_ξ_${xi}
Script: ${STAGE_SCRIPT}
EOF
    printf 'Submitted remaining stochastic simulation %s xi=%s job=%s array=%s\n' "$LABEL" "$xi" "$jid" "$ARRAY_RANGE"
done
