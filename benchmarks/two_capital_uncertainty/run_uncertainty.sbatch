#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=two_capital_unc_nn
#SBATCH --output=logging/unc_nn-%j.out
#SBATCH --error=logging/unc_nn-%j.err

set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_uncertainty
mkdir -p logging

module unload python 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/11.2
module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0

# NB: use NN_ARCH (not ARCH); the login env exports ARCH=x86_64.
SIGMA="${SIGMA:-0.2}"
NN_ARCH="${NN_ARCH:-forwardnet}"
FD_SUPERVISE="${FD_SUPERVISE:-1}"
PRECOND="${PRECOND:-1}"
NN_ITERS="${NN_ITERS:-150000}"
NUM_NEURONS="${NUM_NEURONS:-32}"
NUM_LAYERS="${NUM_LAYERS:-4}"
LABEL="${LABEL:-}"

sup=""; [ "${FD_SUPERVISE}" = "1" ] && sup="--fd-supervise"
pc="";  [ "${PRECOND}" = "1" ] && pc="--precond"

echo "Program starts $(date); sigma=${SIGMA} arch=${NN_ARCH} fd_sup=${FD_SUPERVISE} precond=${PRECOND} iters=${NN_ITERS} neurons=${NUM_NEURONS} layers=${NUM_LAYERS} label=${LABEL}"
python -u compare_uncertainty_nn.py --sigma "${SIGMA}" --arch "${NN_ARCH}" ${sup} ${pc} \
    --nn-iters "${NN_ITERS}" --num-neurons "${NUM_NEURONS}" --num-layers "${NUM_LAYERS}" --label "${LABEL}"
echo "Program ends $(date)"
