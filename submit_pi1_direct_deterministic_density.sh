#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="deterministic_stage.sbatch"
SIMULATION_XIS="0.05,0.1,0.3,148.6"
SIMULATION_Y0="1.2"

FOLDERS=(
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
    "OneTechJump_Pi_1p0_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
)

for foldername in "${FOLDERS[@]}"; do
    export_folder="${PREFIX}/${foldername}"
    for stage in PostDamagePostTech PostDamagePreTech PreDamagePostTech PreDamagePreTech; do
        if [ ! -f "${export_folder}/${stage}/checkpoint" ]; then
            echo "Missing checkpoint: ${export_folder}/${stage}" >&2
            exit 2
        fi
    done

    mkdir -p "./job-outs/${foldername}"

    sim_jid=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --job-name=pi1DetSim \
        --output="./job-outs/${foldername}/Pi1DeterministicSimulation-%j.out" \
        --error="./job-outs/${foldername}/Pi1DeterministicSimulation-%j.err" \
        --export=ALL,MODE=simulate \
        "$STAGE_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="$export_folder" SIMULATION_XIS="$SIMULATION_XIS" SIMULATION_Y0="$SIMULATION_Y0" \
        sbatch --parsable \
        --dependency="afterok:${sim_jid}" \
        --job-name=pi1DetPlot \
        --output="./job-outs/${foldername}/Pi1DeterministicPlot-%j.out" \
        --error="./job-outs/${foldername}/Pi1DeterministicPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$STAGE_SCRIPT")

    cat > "${export_folder}/pi1_direct_deterministic_density_jobs.txt" <<EOF
Submitted: $(date)
Simulation xi: ${SIMULATION_XIS}
Simulation Y0: ${SIMULATION_Y0}
Simulation job: ${sim_jid}
Plot job: ${plot_jid} (afterok:${sim_jid})
Conditional density area annotation: yes
Conditional density accounting file: SimulationDeterministicPlot/first_jump_density_accounting.txt
EOF

    printf 'Submitted Pi1 direct deterministic density for %s: sim=%s plot=%s\n' \
        "$foldername" "$sim_jid" "$plot_jid"
done
