#!/bin/bash

#SBATCH --time=0-1:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --job-name=deterministic_stage

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

# --- A/B delta override (NON-INVASIVE). NOTE: the lambda3 distorted-weight readout depends
# only on xi and the TRAINED value nets, not on live delta at sim time; we still set the
# override so `import params` reports the run's delta consistently. ---
export MODEL_DELTA="${MODEL_DELTA:-0.025}"
export PYTHONPATH="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/ab_delta_override:${PYTHONPATH:-}"
echo "[delta-AB] MODEL_DELTA=${MODEL_DELTA}"

if [ -z "${EXPORT_FOLDER:-}" ]; then
    echo "EXPORT_FOLDER must be provided." >&2
    exit 2
fi

case "${MODE:-simulate}" in
    simulate)
        SIMULATION_XIS="${SIMULATION_XIS:-0.01,0.05,0.1,148.6}" \
        SIMULATION_Y0="${SIMULATION_Y0:-1.2}" \
            python3 "${MODEL_DIR:-models}/SimulationDeterministic.py" "$EXPORT_FOLDER"
        ;;
    plot)
        python3 "${MODEL_DIR:-models}/SimulationDeterministicPlot.py" "$EXPORT_FOLDER"
        ;;
    *)
        echo "Unknown MODE: ${MODE}" >&2
        exit 2
        ;;
esac
