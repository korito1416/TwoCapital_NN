#!/bin/bash
#SBATCH --time=0-00:40:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=rich_s7_12
#SBATCH --output=logging/richardson_seeds7_12-%j.out
#SBATCH --error=logging/richardson_seeds7_12-%j.err

set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
mkdir -p logging
module unload python 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=4
python -u richardson_step2_seeds7_12.py
echo "EXIT_CODE=$?"
