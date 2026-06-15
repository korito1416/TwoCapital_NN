#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="deterministic_stage.sbatch"
SIMULATION_XIS="0.05,0.1,0.3,148.6"
SIMULATION_Y0="1.2"

while IFS= read -r export_folder; do
    foldername="$(basename "$export_folder")"

    for stage in PostDamagePostTech PostDamagePreTech PreDamagePostTech PreDamagePreTech; do
        if [ ! -f "${export_folder}/${stage}/checkpoint" ]; then
            echo "Skipping incomplete folder, missing ${stage}: ${foldername}" >&2
            continue 2
        fi
    done

    mkdir -p "./job-outs/${foldername}"

    sim_jid=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --job-name=pi1S1DetSim \
        --output="./job-outs/${foldername}/Pi1Scale1DeterministicSimulation-%j.out" \
        --error="./job-outs/${foldername}/Pi1Scale1DeterministicSimulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$STAGE_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${sim_jid}" \
        --job-name=pi1S1DetPlot \
        --output="./job-outs/${foldername}/Pi1Scale1DeterministicPlot-%j.out" \
        --error="./job-outs/${foldername}/Pi1Scale1DeterministicPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$STAGE_SCRIPT")

    cat > "${export_folder}/pi1_scale1_all_deterministic_jobs.txt" <<EOF
Submitted: $(date)
Simulation xi: ${SIMULATION_XIS}
Simulation Y0: ${SIMULATION_Y0}
Simulation job: ${sim_jid}
Plot job: ${plot_jid} (afterok:${sim_jid})
Conditional density area annotation: yes
Conditional density accounting file: SimulationDeterministicPlot/first_jump_density_accounting.txt
EOF

    printf 'Submitted Pi1 scale-1 deterministic density for %s: sim=%s plot=%s\n' \
        "$foldername" "$sim_jid" "$plot_jid"
done < <(
    find "$PREFIX" -maxdepth 1 -type d \
        -name 'OneTechJump_Pi_1p0_TechIntensityScale_1p0*' \
        | sort
)
