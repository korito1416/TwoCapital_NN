#!/bin/bash
#SBATCH --time=0-01:30:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=egm_fine
#SBATCH --output=logging/egmfine-%j.out
#SBATCH --error=logging/egmfine-%j.err
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
module load python/anaconda-2021.05
export OMP_NUM_THREADS=4
# FINE grid = EXACTLY the FD-truth grid (31x61x31) -> removes the coarse-grid oracle truncation,
# isolating the autodiff-slope floor (Q1). If controls error drops toward the fine FD floor, the
# coarse-grid 3.8e-2 i_d error was grid truncation, NOT the autodiff slope.
python -u egm_howard_ctrlfit_ab.py --seeds 1 --nK 31 --nZ 61 --nY 31 \
    --howard 14 --fit 800 --warm 1200 --dt 2.5 --only ctrlfit
