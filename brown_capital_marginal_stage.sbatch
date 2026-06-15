#!/bin/bash

#SBATCH --time=0-1:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --job-name=brown_capital_marginal

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

if [ -z "${EXPORT_FOLDER:-}" ]; then
    echo "EXPORT_FOLDER must be provided." >&2
    exit 2
fi

python3 plot_brown_capital_marginal_tech_jump.py "$EXPORT_FOLDER" --regime-set all
