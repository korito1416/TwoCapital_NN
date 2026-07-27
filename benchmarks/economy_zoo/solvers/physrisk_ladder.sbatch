#!/bin/bash
# physrisk_ladder.sbatch -- ONE rung of the PHYSRISK full-grid FD ladder:
# the 12-solve backward regime chain (5x PostDamagePostTech(l) -> PreDamagePostTech
# -> 5x PostDamagePreTech(l) -> PreDamagePreTech) for ONE xi on the full grid
# (Z,Y,s) = (61,41,61), s in [-6,2].  Submit via submit_physrisk_ladder.sh, which
# chains xi = 148.4 -> 0.1 -> 0.05 with afterok (the small-xi rungs warm-start
# W_init/controls from the previous rung's npz -- required for the g-distortion lag).
#
# Env:
#   PHYSRISK_XI       (required)  148.4 | 0.1 | 0.05
#   PHYSRISK_WARM_XI  (optional)  xi of a finished rung to warm-start from
#   PHYSRISK_HOWARD   (optional)  Howard budget, default 60
#   PHYSRISK_OUTDIR   (optional)  default ../outputs/physrisk
#
#SBATCH --job-name=physrisk_ladder
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=36:00:00
#SBATCH --output=/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/economy_zoo/joblogs/physrisk_ladder_%j.out

set -euo pipefail

module unload python 2>/dev/null || true
module load python/anaconda-2021.05

cd "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/economy_zoo/solvers"

XI="${PHYSRISK_XI:?set PHYSRISK_XI (148.4 | 0.1 | 0.05)}"
OUTDIR="${PHYSRISK_OUTDIR:-../outputs/physrisk}"
# dt2d/dt3d deviation from the harness-reference dt=2.5 is REQUIRED for PHYSRISK
# (fast destruction dynamics; see run_chain docstring in physrisk_callbacks.py)
ARGS=(--chain --xi "$XI" --grid full --outdir "$OUTDIR"
      --howard "${PHYSRISK_HOWARD:-60}" --robust 1
      --dt2d 0.5 --dt3d 1.0 --relax 0.2 --tail 10)
if [ -n "${PHYSRISK_WARM_XI:-}" ]; then
    ARGS+=(--warm-xi "$PHYSRISK_WARM_XI")
fi

echo "[physrisk_ladder] xi=$XI warm=${PHYSRISK_WARM_XI:-none} outdir=$OUTDIR host=$(hostname) $(date)"
python3 physrisk_callbacks.py "${ARGS[@]}"
echo "[physrisk_ladder] done $(date)"
