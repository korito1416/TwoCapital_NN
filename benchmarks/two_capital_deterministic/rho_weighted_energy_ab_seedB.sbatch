#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --job-name=rho_E_seedB
#SBATCH --time=0:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-rho_E_seedB-%j.out
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
python rho_weighted_energy_ab_seedB.py
