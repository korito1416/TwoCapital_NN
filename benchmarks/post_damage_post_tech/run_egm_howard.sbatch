#!/bin/bash
#SBATCH --time=0-02:30:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=egm_howard
#SBATCH --output=logging/egmhoward-%j.out
#SBATCH --error=logging/egmhoward-%j.err
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
module load python/anaconda-2021.05
export OMP_NUM_THREADS=4
python -u egm_howard_ctrlfit_ab.py --seeds 1 2 3 --nK 21 --nZ 31 --nY 21 \
    --howard 14 --fit 800 --warm 1200 --dt 2.5
