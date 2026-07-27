#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=two_capital_shock_nn
#SBATCH --output=logging/shock_nn-%j.out
#SBATCH --error=logging/shock_nn-%j.err

set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_shock
mkdir -p logging

module unload python 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/11.2
module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0

# NB: use NN_ARCH (not ARCH) -- the cluster login env exports ARCH=x86_64,
# which --export=ALL would otherwise inherit and pass to --arch.
SIGMA="${SIGMA:-0.01}"
NN_ARCH="${NN_ARCH:-forwardnet}"
FD_SUPERVISE="${FD_SUPERVISE:-0}"
PRECOND="${PRECOND:-1}"
NN_ITERS="${NN_ITERS:-120000}"
LBFGS_ITERS="${LBFGS_ITERS:-0}"
LABEL="${LABEL:-}"

sup=""; [ "${FD_SUPERVISE}" = "1" ] && sup="--fd-supervise"
pc="";  [ "${PRECOND}" = "1" ] && pc="--precond"

echo "Program starts $(date); sigma=${SIGMA} arch=${NN_ARCH} fd_sup=${FD_SUPERVISE} precond=${PRECOND} iters=${NN_ITERS} lbfgs=${LBFGS_ITERS} label=${LABEL}"
python -u compare_shock.py --sigma "${SIGMA}" --arch "${NN_ARCH}" ${sup} ${pc} \
    --nn-iters "${NN_ITERS}" --lbfgs-iters "${LBFGS_ITERS}" --label "${LABEL}"
echo "Program ends $(date)"
