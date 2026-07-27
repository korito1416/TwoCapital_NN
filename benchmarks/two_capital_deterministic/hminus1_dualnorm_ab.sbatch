#!/bin/bash
#SBATCH --job-name=hm1_ab
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --time=0:35:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-hm1_ab-%j.out
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
export TF_CPP_MIN_LOG_LEVEL=3
python hminus1_dualnorm_ab.py
