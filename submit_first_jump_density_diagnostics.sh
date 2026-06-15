#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="deterministic_stage.sbatch"
BASELINE_FOLDER="TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
SIMULATION_XIS="0.01,0.05,0.1,148.6"
SIMULATION_Y0="1.2"

final_training_job() {
    local export_folder="$1"
    local manifest="${export_folder}/run_manifest.txt"

    if [ -f "${export_folder}/PreDamagePreTech/v_nn_checkpoint_PreDamagePreTech.index" ]; then
        printf ''
        return
    fi
    if [ -f "${export_folder}/PreDamageIntermTech/v_nn_checkpoint_PreDamageIntermTech.index" ]; then
        printf ''
        return
    fi
    if [ ! -f "$manifest" ]; then
        echo "Cannot infer training dependency for incomplete folder: ${export_folder}" >&2
        return 2
    fi

    if grep -q '^PreDamagePreTech job:' "$manifest"; then
        awk '/^PreDamagePreTech job:/ {print $3; exit}' "$manifest"
    elif grep -q '^PreDamageIntermTech job:' "$manifest"; then
        awk '/^PreDamageIntermTech job:/ {print $3; exit}' "$manifest"
    else
        echo "No final training job found in ${manifest}" >&2
        return 2
    fi
}

submit_folder() {
    local export_folder="$1"
    local foldername
    local training_jid
    local dependency_args=()

    foldername="$(basename "$export_folder")"
    training_jid="$(final_training_job "$export_folder")"
    if [ -n "$training_jid" ]; then
        dependency_args=(--dependency="afterok:${training_jid}")
    fi

    mkdir -p "./job-outs/${foldername}"

    sim_jid=$(EXPORT_FOLDER="${export_folder}" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name=firstJumpDensitySim \
        --output="./job-outs/${foldername}/FirstJumpDensitySimulation-%j.out" \
        --error="./job-outs/${foldername}/FirstJumpDensitySimulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$STAGE_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="${export_folder}" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${sim_jid}" \
        --job-name=firstJumpDensityPlot \
        --output="./job-outs/${foldername}/FirstJumpDensityPlot-%j.out" \
        --error="./job-outs/${foldername}/FirstJumpDensityPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$STAGE_SCRIPT")

    cat > "${export_folder}/first_jump_density_jobs.txt" <<EOF
Submitted: $(date)
Training dependency: ${training_jid:-none}
Simulation scenarios xi: ${SIMULATION_XIS}
Simulation Y0: ${SIMULATION_Y0}
Simulation job: ${sim_jid}
Plot job: ${plot_jid} (afterok:${sim_jid})
EOF

    printf 'Submitted first-jump density diagnostics for %s\n' "$foldername"
    printf '  training dependency: %s\n' "${training_jid:-none}"
    printf '  simulation:          %s\n' "$sim_jid"
    printf '  comparison plots:    %s (afterok:%s)\n' "$plot_jid" "$sim_jid"
}

submit_folder "${PREFIX}/${BASELINE_FOLDER}"

while IFS= read -r export_folder; do
    submit_folder "$export_folder"
done < <(
    find "$PREFIX" -maxdepth 1 -type d \
        \( -name 'OneTechJump_Pi_0p0_*' -o -name 'TwoStageTech_TechIntensityScale_2p0_*' \) \
        | sort
)
