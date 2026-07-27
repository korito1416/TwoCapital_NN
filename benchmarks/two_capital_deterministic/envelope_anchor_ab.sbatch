#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=env_anchor_ab
#SBATCH --output=slurm-env_anchor_ab-%j.out
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
python envelope_anchor_ab.py
