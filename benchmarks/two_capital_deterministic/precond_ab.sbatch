#!/bin/bash
#SBATCH --time=0-00:25:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=precond_ab
#SBATCH --output=slurm-precond_ab-%j.out
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
module unload python 2>/dev/null || true
module load python/anaconda-2021.05
python -u precond_ab_multiseed.py
