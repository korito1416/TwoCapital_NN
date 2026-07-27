#!/bin/bash
#SBATCH --time=0-00:25:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=policy_eval
#SBATCH --output=logging/poleval-%j.out
#SBATCH --error=logging/poleval-%j.err
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
module unload python 2>/dev/null||true; module unload cuda 2>/dev/null||true
module load cuda/11.2; module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0
python -u policy_eval.py
