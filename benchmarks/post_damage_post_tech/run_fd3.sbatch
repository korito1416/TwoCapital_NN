#!/bin/bash
#SBATCH --time=0-00:40:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=pdpt_fd_v3
#SBATCH --output=logging/fd_v3-%j.out
#SBATCH --error=logging/fd_v3-%j.err
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
mkdir -p logging
module unload python 2>/dev/null||true; module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "start $(date)"; python -u fd_pdpt_v3.py; echo "end $(date)"
