#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="svrd_decomposition_stage.sbatch"
XIS=(${SVRD_XIS:-0.05 0.1 0.3 148.6})
N_PATHS="${N_PATHS:-512}"
YEARS="${YEARS:-80}"
DT="${DT:-0.08333333333333333}"
SEED="${SEED:-1}"
EPS_LOGR="${EPS_LOGR:-0.0001}"
Y0="${Y0:-1.2}"
PROGRESS_EVERY="${PROGRESS_EVERY:-120}"
FORCE="${FORCE:-0}"

# label|folder
MODELS=(
    "OneTechPi1S1|OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
    "TwoStageBase|TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
    "TwoStageS2|TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
)

summary_path="${PREFIX}/svrd_control_density_models_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "label,xi,status,job_id,export_folder,output_folder" > "$summary_path"

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

svrd_done_with_y0() {
    local output_folder="$1"
    [ -f "${output_folder}/metadata.txt" ] || return 1
    [ -f "${output_folder}/svrd_decomposition.csv" ] || return 1
    rg -q "^Y0 = ${Y0}$" "${output_folder}/metadata.txt"
}

for spec in "${MODELS[@]}"; do
    IFS='|' read -r label folder <<< "$spec"
    export_folder="${PREFIX}/${folder}"
    check_model_ready "$export_folder"

    for xi in "${XIS[@]}"; do
        output_folder="${export_folder}/SVRDDecomposition/xi_${xi}"
        if [ "$FORCE" != "1" ] && svrd_done_with_y0 "$output_folder"; then
            echo "\"${label}\",\"${xi}\",\"skipped_existing_y0_${Y0}\",\"\",\"${export_folder}\",\"${output_folder}\"" >> "$summary_path"
            printf 'Skipping existing SVRD %s xi=%s Y0=%s\n' "$label" "$xi" "$Y0"
            continue
        fi

        log_dir="./job-outs/SVRD/${label}/xi_${xi}"
        mkdir -p "$log_dir" "$output_folder"

        jid=$(EXPORT_FOLDER="$export_folder" \
            OUTPUT_ROOT="${export_folder}/SVRDDecomposition" \
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
                --job-name="svrd_${label}_xi_${xi}" \
                --output="${log_dir}/%j.out" \
                --error="${log_dir}/%j.err" \
                --export=ALL \
                "$STAGE_SCRIPT")

        cat > "${export_folder}/SVRDDecomposition/svrd_job_xi_${xi}_Y0_1p2.txt" <<EOF
Submitted: $(date)
Job id: ${jid}
Model label: ${label}
Export folder: ${export_folder}
xi: ${xi}
N paths: ${N_PATHS}
Years: ${YEARS}
dt: ${DT}
seed: ${SEED}
eps_logR: ${EPS_LOGR}
Y0: ${Y0}
Output folder: ${output_folder}
Script: ${STAGE_SCRIPT}
EOF
        echo "\"${label}\",\"${xi}\",\"submitted\",\"${jid}\",\"${export_folder}\",\"${output_folder}\"" >> "$summary_path"
        printf 'Submitted SVRD %s xi=%s job=%s Y0=%s\n' "$label" "$xi" "$jid" "$Y0"
    done
done

printf 'Wrote SVRD job summary: %s\n' "$summary_path"
