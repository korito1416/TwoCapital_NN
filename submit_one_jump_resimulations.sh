#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="deterministic_stage.sbatch"
SIMULATION_XIS="${SIMULATION_XIS:-148.6,0.3,0.1}"

while IFS= read -r export_folder; do
    foldername="$(basename "$export_folder")"

    if [ ! -f "${export_folder}/PreDamageIntermTech/v_nn_checkpoint_PreDamageIntermTech.index" ]; then
        echo "Skipping incomplete one-jump folder: ${foldername}" >&2
        continue
    fi

    mkdir -p "./job-outs/${foldername}"

    sim_jid=$(EXPORT_FOLDER="${export_folder}" SIMULATION_XIS="${SIMULATION_XIS}" \
        sbatch --parsable \
        --job-name=oneJumpResim \
        --output="./job-outs/${foldername}/OneJumpResimulation-%j.out" \
        --error="./job-outs/${foldername}/OneJumpResimulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$STAGE_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="${export_folder}" SIMULATION_XIS="${SIMULATION_XIS}" \
        sbatch --parsable \
        --dependency="afterok:${sim_jid}" \
        --job-name=oneJumpReplot \
        --output="./job-outs/${foldername}/OneJumpReplot-%j.out" \
        --error="./job-outs/${foldername}/OneJumpReplot-%j.err" \
        --export=ALL,MODE=plot \
        "$STAGE_SCRIPT")

    cat > "${export_folder}/one_jump_resimulation_jobs.txt" <<EOF
Submitted: $(date)
Simulation scenarios xi: ${SIMULATION_XIS}
Simulation job: ${sim_jid}
Plot job: ${plot_jid} (afterok:${sim_jid})
Simulation note: uses patched one-tech-jump deterministic logic with PreDamageIntermTech policy and lambda_technology = tech_jump_intensity_scale * R / varrho.
EOF

    printf 'Submitted one-jump resimulation for %s\n' "$foldername"
    printf '  simulation: %s\n' "$sim_jid"
    printf '  plot:       %s (afterok:%s)\n' "$plot_jid" "$sim_jid"
done < <(find "$PREFIX" -maxdepth 1 -type d -name 'OneTechJump_Pi_0p0_*' | sort)
