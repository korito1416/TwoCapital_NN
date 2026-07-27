#!/bin/bash
#SBATCH --job-name=fd_anchor
#SBATCH --partition=caslake
#SBATCH --account=pi-lhansen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
# Runs an arbitrary FD-anchor python command passed via env RUNCMD.
set -euo pipefail
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_selection_criterion
echo "RUNCMD: ${RUNCMD:?}"
eval "python -u ${RUNCMD}"
echo "FD-JOB-DONE rc=$?"
