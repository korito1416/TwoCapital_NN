#!/bin/bash

#SBATCH --time=0-1:30:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=dgm_deterministic_stage

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

if [ -z "${EXPORT_FOLDER:-}" ]; then
    echo "EXPORT_FOLDER must be provided." >&2
    exit 2
fi

case "${MODE:-simulate}" in
    simulate)
        SIMULATION_XIS="${SIMULATION_XIS:-0.05,0.1,0.3,148.6}" \
        SIMULATION_Y0="${SIMULATION_Y0:-1.2}" \
            python3 models_dgm/SimulationDeterministic.py "$EXPORT_FOLDER"
        ;;
    plot)
        SIMULATION_XIS="${SIMULATION_XIS:-0.05,0.1,0.3,148.6}" \
            python3 models/SimulationDeterministicPlot.py "$EXPORT_FOLDER"
        ;;
    *)
        echo "Unknown MODE: ${MODE}" >&2
        exit 2
        ;;
esac
