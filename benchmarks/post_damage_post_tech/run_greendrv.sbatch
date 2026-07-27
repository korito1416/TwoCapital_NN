#!/bin/bash
#SBATCH --time=0-00:40:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --job-name=green_drv
#SBATCH --output=logging/greendrv-%j.out
#SBATCH --error=logging/greendrv-%j.err
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
module unload python 2>/dev/null||true
module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
python -u fd_v5_green_drivers.py
