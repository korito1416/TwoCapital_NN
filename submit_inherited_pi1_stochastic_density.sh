#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

EXPORT_FOLDER="${EXPORT_FOLDER:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/OneTechJump_Pi_1p0_InheritedPostTech_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_512_neurons_32_#HiddenLayer_4_num_iterations1000000}"
STOCHASTIC_STAGE="${STOCHASTIC_STAGE:-stochastic_jump_stage.sbatch}"
PLOT_STAGE="${PLOT_STAGE:-stochastic_density_plot_stage.sbatch}"

XIS=(${STOCHASTIC_XIS:-0.05 0.1 0.3 148.6})
ARRAY_RANGE="${ARRAY_RANGE:-1-100}"
N_PATHS="${N_PATHS:-100}"
YEARS="${YEARS:-60}"
DT="${DT:-0.08333333333333333}"
Y0="${Y0:-1.2}"
SEED_MIN="${SEED_MIN:-1}"
SEED_MAX="${SEED_MAX:-100}"

required_stages=(
    PostDamagePostTech
    PreDamagePostTech
    PostDamagePreTech
    PreDamagePreTech
)

for stage in "${required_stages[@]}"; do
    if [ ! -f "${EXPORT_FOLDER}/${stage}/checkpoint" ]; then
        echo "Missing checkpoint for ${stage}: ${EXPORT_FOLDER}" >&2
        exit 2
    fi
done

for stage in PostDamageIntermTech PreDamageIntermTech; do
    if [ -d "${EXPORT_FOLDER}/${stage}" ]; then
        echo "Unexpected intermediate stage folder for pi=1 one-jump model: ${EXPORT_FOLDER}/${stage}" >&2
        exit 2
    fi
done

foldername="$(basename "$EXPORT_FOLDER")"
log_root="./job-outs/StochasticJump/${foldername}"
mkdir -p "$log_root" "${EXPORT_FOLDER}/SimulationResults" "${EXPORT_FOLDER}/SimulationResultsPlot"

summary="${EXPORT_FOLDER}/SimulationResults/stochastic_density_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "xi,job_id,array_range,n_paths,years,dt,Y0,output_folder" > "$summary"

dependencies=()
for xi in "${XIS[@]}"; do
    log_dir="${log_root}/xi_${xi}"
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
            --job-name="stoch_inhPi1_xi_${xi}" \
            --output="${log_dir}/%A_%a.out" \
            --error="${log_dir}/%A_%a.err" \
            --export=ALL \
            "$STOCHASTIC_STAGE")

    dependencies+=("$jid")
    output_folder="${EXPORT_FOLDER}/SimulationResults/paths_ξ_${xi}"
    echo "\"${xi}\",\"${jid}\",\"${ARRAY_RANGE}\",\"${N_PATHS}\",\"${YEARS}\",\"${DT}\",\"${Y0}\",\"${output_folder}\"" >> "$summary"
    cat > "${EXPORT_FOLDER}/SimulationResults/stochastic_jobs_xi_${xi}_inherited_pi1.txt" <<EOF
Submitted: $(date)
Job id: ${jid}
Array range: ${ARRAY_RANGE}
Export folder: ${EXPORT_FOLDER}
xi: ${xi}
N paths per array task: ${N_PATHS}
Years: ${YEARS}
dt: ${DT}
Y0: ${Y0}
Output folder: ${output_folder}
Script: ${STOCHASTIC_STAGE}
EOF
    printf 'Submitted inherited pi=1 stochastic simulation xi=%s job=%s array=%s\n' "$xi" "$jid" "$ARRAY_RANGE"
done

dependency_string=$(IFS=:; echo "${dependencies[*]}")
plot_jid=$(EXPORT_FOLDER="$EXPORT_FOLDER" \
    Y0="$Y0" \
    SEED_MIN="$SEED_MIN" \
    SEED_MAX="$SEED_MAX" \
    sbatch --parsable \
        --dependency="afterok:${dependency_string}" \
        --job-name="plot_inhPi1_density" \
        --output="${log_root}/DensityPlot-%j.out" \
        --error="${log_root}/DensityPlot-%j.err" \
        --export=ALL \
        "$PLOT_STAGE")

cat > "${EXPORT_FOLDER}/SimulationResults/stochastic_density_plot_job.txt" <<EOF
Submitted: $(date)
Plot job id: ${plot_jid}
Dependency: afterok:${dependency_string}
Export folder: ${EXPORT_FOLDER}
Y0: ${Y0}
Seed range: ${SEED_MIN}-${SEED_MAX}
Plot script: ${PLOT_STAGE}
Output folder: ${EXPORT_FOLDER}/SimulationResultsPlot
EOF

echo "\"plot\",\"${plot_jid}\",\"afterok:${dependency_string}\",\"\",\"\",\"\",\"${Y0}\",\"${EXPORT_FOLDER}/SimulationResultsPlot\"" >> "$summary"
printf 'Submitted inherited pi=1 density plot job=%s afterok:%s\n' "$plot_jid" "$dependency_string"
printf 'Wrote job summary: %s\n' "$summary"
