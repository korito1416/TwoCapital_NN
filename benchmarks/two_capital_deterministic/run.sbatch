#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=two_capital_fd_vs_nn
#SBATCH --output=logging/two_capital_fd_vs_nn-%j.out
#SBATCH --error=logging/two_capital_fd_vs_nn-%j.err

set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
mkdir -p logging

module unload python 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/11.2
module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

A_G="${A_G:-A_g_prime_prime}"
NN_ITERS="${NN_ITERS:-200000}"
FD_N="${FD_N:-600}"
ARCH="${ARCH:-forwardnet}"
FD_SUPERVISE="${FD_SUPERVISE:-0}"
PRECOND="${PRECOND:-0}"
PRECOND_EPS="${PRECOND_EPS:-1e-3}"
LBFGS_ITERS="${LBFGS_ITERS:-5000}"
LR_V="${LR_V:-1e-4}"
LR_C="${LR_C:-4e-3}"
LABEL="${LABEL:-}"

sup_flag=""; [ "${FD_SUPERVISE}" = "1" ] && sup_flag="--fd-supervise"
pc_flag="";  [ "${PRECOND}" = "1" ] && pc_flag="--precond"

echo "Program starts $(date); A_g=${A_G}, nn_iters=${NN_ITERS}, fd_n=${FD_N}, arch=${ARCH}, fd_supervise=${FD_SUPERVISE}, precond=${PRECOND}, precond_eps=${PRECOND_EPS}, lbfgs_iters=${LBFGS_ITERS}, lr_v=${LR_V}, lr_c=${LR_C}, label=${LABEL}"
python -u compare.py --a-g "${A_G}" --nn-iters "${NN_ITERS}" --fd-n "${FD_N}" --arch "${ARCH}" ${sup_flag} ${pc_flag} --precond-eps "${PRECOND_EPS}" --lbfgs-iters "${LBFGS_ITERS}" --lr-v "${LR_V}" --lr-c "${LR_C}" --label "${LABEL}"
echo "Program ends $(date)"
