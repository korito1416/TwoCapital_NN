#!/bin/bash

#SBATCH --time=0-03:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --job-name=stochDensityPlot

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload python
module unload cuda
module load cuda/11.2
module load python/anaconda-2021.05

export PYTHONNOUSERSITE=1
export TF_ENABLE_ONEDNN_OPTS=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

EXPORT_FOLDER="${EXPORT_FOLDER:?EXPORT_FOLDER must be set}"
Y0="${Y0:-1.2}"
SEED_MIN="${SEED_MIN:-1}"
SEED_MAX="${SEED_MAX:-100}"
COMPARISON="${COMPARISON:-pre-post}"

echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"
echo "Export folder: ${EXPORT_FOLDER}"
echo "Y0 filter: ${Y0}"
echo "Seed range: ${SEED_MIN}-${SEED_MAX}"
start_time=$(date +%s)

python -u plot_stochastic_control_densities.py \
    --export-folder "$EXPORT_FOLDER" \
    --control-group all \
    --comparison "$COMPARISON" \
    --y0 "$Y0" \
    --seed-min "$SEED_MIN" \
    --seed-max "$SEED_MAX"

python -u plot_stochastic_marginal_value_densities.py \
    --export-folder "$EXPORT_FOLDER" \
    --comparison "$COMPARISON" \
    --y0 "$Y0" \
    --seed-min "$SEED_MIN" \
    --seed-max "$SEED_MAX"

python - <<'PY'
import os
import shutil
from pathlib import Path

export_folder = Path(os.environ["EXPORT_FOLDER"])
comparison = os.environ.get("COMPARISON", "pre-post")
for paths_folder in sorted((export_folder / "SimulationResults").glob("paths_*")):
    source_root = paths_folder / "ControlDensities"
    if not source_root.exists():
        continue
    dest_root = export_folder / "SimulationResultsPlot" / paths_folder.name / "ControlDensities"
    for group in ("level", "rate", "value", "marginal"):
        source = source_root / f"{group}_{comparison}"
        if not source.exists():
            continue
        dest = dest_root / source.name
        dest.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
PY

echo "Program ends $(date)"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
