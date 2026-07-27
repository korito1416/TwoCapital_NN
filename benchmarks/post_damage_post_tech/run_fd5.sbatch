#!/bin/bash
#SBATCH --time=0-01:40:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=fd5_pibys
#SBATCH --output=logging/fd5-%j.out
#SBATCH --error=logging/fd5-%j.err
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
module unload python 2>/dev/null||true
module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
echo "=== smoke test first ==="
python -u fd_pdpt_v5.py
echo
echo "=== grid-convergence study ==="
python -u fd_v5_converge.py
