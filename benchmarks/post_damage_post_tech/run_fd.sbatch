#!/bin/bash
#SBATCH --time=0-01:30:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=pdpt_fd_vs_nn
#SBATCH --output=logging/fd_vs_nn-%j.out
#SBATCH --error=logging/fd_vs_nn-%j.err
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
mkdir -p logging
module unload python 2>/dev/null||true; module unload cuda 2>/dev/null||true
module load cuda/11.2; module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "start $(date)"
python -u fd_vs_nn.py
echo "end $(date)"
