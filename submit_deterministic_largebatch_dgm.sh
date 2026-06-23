#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

LARGEBATCH_ROOT="${LARGEBATCH_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_largebatch_001}"
DGM_ROOT="${DGM_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_dgm_001}"
STANDARD_STAGE="${STANDARD_STAGE:-deterministic_stage.sbatch}"
DGM_STAGE="${DGM_STAGE:-dgm_deterministic_stage.sbatch}"
SIMULATION_XIS="${SIMULATION_XIS:-0.05,0.1,0.3,148.6}"
SIMULATION_Y0="${SIMULATION_Y0:-1.2}"
SIMULATION_T="${SIMULATION_T:-60.0}"
SIMULATION_DT="${SIMULATION_DT:-0.08333333333333333}"

required_stages_for_folder() {
    local folder="$1"
    if [[ "$(basename "$folder")" == OneTechJump_* ]]; then
        printf '%s\n' PostDamagePostTech PreDamagePostTech PostDamagePreTech PreDamagePreTech
    else
        printf '%s\n' PostDamagePostTech PostDamageIntermTech PreDamagePostTech PostDamagePreTech PreDamageIntermTech PreDamagePreTech
    fi
}

check_ready() {
    local folder="$1"
    local stage
    while read -r stage; do
        if [ ! -f "${folder}/${stage}/checkpoint" ]; then
            echo "Missing checkpoint for ${stage}: ${folder}" >&2
            return 1
        fi
    done < <(required_stages_for_folder "$folder")
}

submit_pair() {
    local family="$1"
    local folder="$2"
    local stage_script="$3"
    local index="$4"
    local foldername
    local log_dir
    local sim_jid
    local plot_jid

    foldername="$(basename "$folder")"
    log_dir="./job-outs/Deterministic/${family}/${foldername}"
    mkdir -p "$log_dir"

    sim_jid=$(EXPORT_FOLDER="$folder" \
        SIMULATION_XIS="$SIMULATION_XIS" \
        SIMULATION_Y0="$SIMULATION_Y0" \
        SIMULATION_T="$SIMULATION_T" \
        SIMULATION_DT="$SIMULATION_DT" \
        sbatch --parsable \
            --job-name="det_${family}_${index}" \
            --output="${log_dir}/Simulation-%j.out" \
            --error="${log_dir}/Simulation-%j.err" \
            --export=ALL,MODE=simulate \
            "$stage_script")

    plot_jid=$(EXPORT_FOLDER="$folder" \
        SIMULATION_XIS="$SIMULATION_XIS" \
        SIMULATION_Y0="$SIMULATION_Y0" \
        SIMULATION_T="$SIMULATION_T" \
        SIMULATION_DT="$SIMULATION_DT" \
        sbatch --parsable \
            --dependency="afterok:${sim_jid}" \
            --job-name="plot_${family}_${index}" \
            --output="${log_dir}/Plot-%j.out" \
            --error="${log_dir}/Plot-%j.err" \
            --export=ALL,MODE=plot \
            "$stage_script")

    mkdir -p "${folder}/SimulationDeterministic"
    cat > "${folder}/SimulationDeterministic/deterministic_jobs_y0_1p2.txt" <<EOF
Submitted: $(date)
Family: ${family}
Export folder: ${folder}
Simulation job: ${sim_jid}
Plot job: ${plot_jid} (afterok:${sim_jid})
Simulation xis: ${SIMULATION_XIS}
Simulation Y0: ${SIMULATION_Y0}
Simulation T: ${SIMULATION_T}
Simulation dt: ${SIMULATION_DT}
Stage script: ${stage_script}
Output plot folder: ${folder}/SimulationDeterministicPlot
EOF

    echo "\"${family}\",\"${foldername}\",\"${sim_jid}\",\"${plot_jid}\",\"${folder}\"" >> "$summary"
    printf 'Submitted deterministic %s %s: sim=%s plot=%s\n' "$family" "$foldername" "$sim_jid" "$plot_jid"
}

summary="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/deterministic_largebatch_dgm_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "family,folder,simulation_job,plot_job,export_folder" > "$summary"

mapfile -t largebatch_folders < <(find "$LARGEBATCH_ROOT" -maxdepth 1 -mindepth 1 -type d -name '*LargeBatch*' | sort)
mapfile -t dgm_folders < <(find "$DGM_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'DGM_TwoStage_*' | sort)

index=0
for folder in "${largebatch_folders[@]}"; do
    check_ready "$folder"
    index=$((index + 1))
    submit_pair "largebatch" "$folder" "$STANDARD_STAGE" "$index"
done

index=0
for folder in "${dgm_folders[@]}"; do
    check_ready "$folder"
    index=$((index + 1))
    submit_pair "dgm" "$folder" "$DGM_STAGE" "$index"
done

printf 'Wrote deterministic job summary: %s\n' "$summary"
