#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
DETERMINISTIC_STAGE_SCRIPT="deterministic_stage.sbatch"
BROWN_STAGE_SCRIPT="brown_capital_marginal_stage.sbatch"

stages=(
    PostDamagePostTech
    PostDamageIntermTech
    PostDamagePreTech
    PreDamagePostTech
    PreDamageIntermTech
    PreDamagePreTech
)

is_complete_folder() {
    local folder="$1"
    local stage
    for stage in "${stages[@]}"; do
        if [ ! -f "${folder}/${stage}/training_history.csv" ]; then
            return 1
        fi
    done
    return 0
}

submitted=0
while IFS= read -r export_folder; do
    if ! is_complete_folder "$export_folder"; then
        printf 'Skipping incomplete folder: %s\n' "$export_folder"
        continue
    fi

    foldername="$(basename "$export_folder")"
    mkdir -p "./job-outs/${foldername}"

    sim_jid=$(EXPORT_FOLDER="${export_folder}" \
        sbatch --parsable \
        --job-name=dtmSim_halfRD \
        --output="./job-outs/${foldername}/SimulationDeterministic-%j.out" \
        --error="./job-outs/${foldername}/SimulationDeterministic-%j.err" \
        --export=ALL,MODE=simulate \
        "$DETERMINISTIC_STAGE_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="${export_folder}" \
        sbatch --parsable \
        --dependency=afterok:${sim_jid} \
        --job-name=dtmPlot_halfRD \
        --output="./job-outs/${foldername}/SimulationDeterministicPlot-%j.out" \
        --error="./job-outs/${foldername}/SimulationDeterministicPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$DETERMINISTIC_STAGE_SCRIPT")

    brown_jid=$(EXPORT_FOLDER="${export_folder}" \
        sbatch --parsable \
        --job-name=brownMU_halfRD \
        --output="./job-outs/${foldername}/BrownCapitalMarginal-%j.out" \
        --error="./job-outs/${foldername}/BrownCapitalMarginal-%j.err" \
        --export=ALL \
        "$BROWN_STAGE_SCRIPT")

    submitted=$((submitted + 1))
    printf 'Submitted diagnostics for %s\n' "$foldername"
    printf '  deterministic simulate: %s\n' "$sim_jid"
    printf '  deterministic plot:     %s (afterok:%s)\n' "$plot_jid" "$sim_jid"
    printf '  brown capital marginal: %s\n' "$brown_jid"
done < <(find "$PREFIX" -maxdepth 1 -type d -name 'TwoStageTech_RDIntensityScale_0p5_*' | sort)

printf 'Submitted diagnostics for %d folders.\n' "$submitted"
