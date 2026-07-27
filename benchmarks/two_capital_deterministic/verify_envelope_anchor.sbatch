#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=verify_env_anchor
#SBATCH --output=slurm-verify_env_anchor-%j.out
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
python verify_envelope_anchor.py
