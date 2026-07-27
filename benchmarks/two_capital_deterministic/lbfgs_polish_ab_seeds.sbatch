#!/bin/bash
#SBATCH --time=0-00:40:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=lbfgs_polish_seeds
#SBATCH --output=slurm-lbfgs_polish_seeds-%j.out
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
module unload python 2>/dev/null || true
module load python/anaconda-2021.05
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=4
python -u lbfgs_polish_ab_seeds.py 1 2 3
