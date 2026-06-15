#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
STAGE_SCRIPT="deterministic_stage.sbatch"

folders=(
    "TwoStageTech_RDIntensityScale_0p5_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
    "TwoStageTech_RDIntensityScale_0p5_finetune_LR_warmup_cosine_5e-6,2e-4_128_neurons_32_#HiddenLayer_4_num_iterations500000"
    "TwoStageTech_RDIntensityScale_0p5_finetune_LR_None_2e-6,5e-5_128_neurons_32_#HiddenLayer_4_num_iterations500000"
)

for foldername in "${folders[@]}"; do
    export_folder="${PREFIX}/${foldername}"
    mkdir -p "./job-outs/${foldername}" "./bash/${foldername}"

    sim_jid=$(EXPORT_FOLDER="${export_folder}" \
        sbatch --parsable \
        --job-name=dtmSim_halfRD \
        --output="./job-outs/${foldername}/SimulationDeterministic-%j.out" \
        --error="./job-outs/${foldername}/SimulationDeterministic-%j.err" \
        --export=ALL,MODE=simulate \
        "$STAGE_SCRIPT")

    plot_jid=$(EXPORT_FOLDER="${export_folder}" \
        sbatch --parsable \
        --dependency=afterok:${sim_jid} \
        --job-name=dtmPlot_halfRD \
        --output="./job-outs/${foldername}/SimulationDeterministicPlot-%j.out" \
        --error="./job-outs/${foldername}/SimulationDeterministicPlot-%j.err" \
        --export=ALL,MODE=plot \
        "$STAGE_SCRIPT")

    printf 'Submitted deterministic paths for %s\n' "$foldername"
    printf '  simulate: %s\n' "$sim_jid"
    printf '  plot:     %s (afterok:%s)\n' "$plot_jid" "$sim_jid"
done
