#!/bin/bash
#SBATCH --time=0-1:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --job-name=lowxi_thr

set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
module unload cuda 2>/dev/null || true
module unload python 2>/dev/null || true
module load cuda/11.2
module load python/anaconda-2021.05

# CALIB is passed as $1 (full or half); paths hardcoded+quoted to survive commas
CALIB="$1"
if [ "$CALIB" = "full" ]; then
  EXPORT_FOLDER="output_lowxi_float64/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000"
else
  EXPORT_FOLDER="output_lowxi_float64/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m5p30_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000"
fi

export MODEL_DIR=models_float64
export SIMULATION_XIS="0.02,0.015,0.0125"
export SIMULATION_Y0=1.2

echo "EXPORT_FOLDER=[$EXPORT_FOLDER]"
echo "SIMULATION_XIS=[$SIMULATION_XIS]"
python3 "${MODEL_DIR}/SimulationDeterministic.py" "$EXPORT_FOLDER"
