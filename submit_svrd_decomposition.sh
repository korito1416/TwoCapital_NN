#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="svrd_decomposition_stage.sbatch"

MODEL_FOLDER="${MODEL_FOLDER:-TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000}"
MODEL_LABEL="${MODEL_LABEL:-TwoStageBase}"
XIS=(${SVRD_XIS:-0.05 0.1 148.6})
N_PATHS="${N_PATHS:-512}"
YEARS="${YEARS:-80}"
DT="${DT:-0.08333333333333333}"
SEED="${SEED:-1}"
EPS_LOGR="${EPS_LOGR:-0.0001}"
Y0="${Y0:-1.2}"
PROGRESS_EVERY="${PROGRESS_EVERY:-120}"

EXPORT_FOLDER="${PREFIX}/${MODEL_FOLDER}"

required_stages=(
    PostDamagePreTech
    PreDamagePostTech
    PreDamagePreTech
)
if [[ "$(basename "$EXPORT_FOLDER")" != OneTechJump_* ]]; then
    required_stages+=(PreDamageIntermTech)
fi

for stage in "${required_stages[@]}"; do
    if [ ! -f "${EXPORT_FOLDER}/${stage}/checkpoint" ]; then
        echo "Missing checkpoint for ${stage}: ${EXPORT_FOLDER}" >&2
        exit 1
    fi
done

for xi in "${XIS[@]}"; do
    log_dir="./job-outs/SVRD/${MODEL_LABEL}/xi_${xi}"
    mkdir -p "$log_dir"
    mkdir -p "${EXPORT_FOLDER}/SVRDDecomposition/xi_${xi}"

    jid=$(EXPORT_FOLDER="$EXPORT_FOLDER" \
        OUTPUT_ROOT="${EXPORT_FOLDER}/SVRDDecomposition" \
        XI="$xi" \
        N_PATHS="$N_PATHS" \
        YEARS="$YEARS" \
        DT="$DT" \
        SEED="$SEED" \
        EPS_LOGR="$EPS_LOGR" \
        Y0="$Y0" \
        BATCH_SIZE="$N_PATHS" \
        PROGRESS_EVERY="$PROGRESS_EVERY" \
        sbatch --parsable \
            --job-name="svrd_${MODEL_LABEL}_xi_${xi}" \
            --output="${log_dir}/%j.out" \
            --error="${log_dir}/%j.err" \
            --export=ALL \
            "$STAGE_SCRIPT")

    cat > "${EXPORT_FOLDER}/SVRDDecomposition/svrd_job_xi_${xi}.txt" <<EOF
Submitted: $(date)
Job id: ${jid}
Model label: ${MODEL_LABEL}
Export folder: ${EXPORT_FOLDER}
xi: ${xi}
N paths: ${N_PATHS}
Years: ${YEARS}
dt: ${DT}
seed: ${SEED}
eps_logR: ${EPS_LOGR}
Y0: ${Y0}
Output folder: ${EXPORT_FOLDER}/SVRDDecomposition/xi_${xi}
Script: ${STAGE_SCRIPT}
EOF
    printf 'Submitted SVRD decomposition %s xi=%s job=%s\n' "$MODEL_LABEL" "$xi" "$jid"
done
