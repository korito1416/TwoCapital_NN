#!/bin/bash
# jones_ladder.sbatch -- one JONES ladder cell: (REGIME [, L3IDX]) x the full
# xi ladder {148.4, 0.1, 0.05} warm-chained inside the job (jones_callbacks.py
# `ladder`). Submitted (with per-stage dependencies and time overrides) by
# submit_jones_ladder.sh -- do not submit by hand unless you know the DAG.
#
# env in (via --export): REGIME (required), L3IDX (post-damage regimes only),
#                        NZ NY NS (default 61 31 41), HOWARD, HOWARD_WARM
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --job-name=jones_ladder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/economy_zoo/joblogs/%x-%j.out
#SBATCH --error=/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/economy_zoo/joblogs/%x-%j.err

set -euo pipefail
module load python/anaconda-2021.05
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

ZOO="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/economy_zoo"
OUTDIR="$ZOO/solvers/outputs/jones"
mkdir -p "$OUTDIR"

: "${REGIME:?REGIME must be exported (PostDamagePostTech|PreDamagePostTech|PostDamagePreTech|PreDamagePreTech)}"
L3ARG=""
if [[ -n "${L3IDX:-}" ]]; then L3ARG="--l3idx ${L3IDX}"; fi

echo "[jones_ladder] REGIME=$REGIME L3IDX=${L3IDX:-none} grid=(${NZ:-61},${NY:-31},${NS:-41}) host=$(hostname) job=${SLURM_JOB_ID:-local}"
python "$ZOO/solvers/jones_callbacks.py" ladder \
    --regime "$REGIME" $L3ARG \
    --nz "${NZ:-61}" --ny "${NY:-31}" --ns "${NS:-46}" \
    --outdir "$OUTDIR" \
    --howard "${HOWARD:-100}" --howard-warm "${HOWARD_WARM:-60}"
echo "[jones_ladder] DONE $REGIME ${L3IDX:-}"
