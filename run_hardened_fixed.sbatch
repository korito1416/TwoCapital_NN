#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=hard_fixed
#SBATCH --output=hardened_fixed_%x_%j.out
#SBATCH --error=hardened_fixed_%x_%j.err

# Two-arm decisive run of the FIXED hardened PostDamagePostTech.
#   ARM env COSTATE_SUP_WEIGHT controls supervision (0 = hardening-bundle only; 0.5 = + fixed supervision)
# Warm-started from the incumbent into a SCRATCH folder (NEVER output_001).
# Periodic export (every 10k) makes it gradeable mid-flight.

set -euo pipefail
module load python/anaconda-2021.05

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

INC="output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"

# ARM is passed via env: ARM=A (sup off) or ARM=B (sup on). Default B.
ARM="${ARM:-B}"
if [ "$ARM" = "A" ]; then
  export COSTATE_SUP_WEIGHT=0.0
  OUT="output_hardened_scratch/fixed_armA_supoff"
else
  export COSTATE_SUP_WEIGHT=0.5
  OUT="output_hardened_scratch/fixed_armB_supon"
fi

export CHECKPOINT_EXPORT_EVERY=10000
export COSTATE_SIG_LOGXI=""        # logxi proximity DROPPED -> supervision active at small-xi
export XI_CURRICULUM_FRAC=0.3
export VALIDATION_BATCHES=4
export VALIDATION_BATCH_SIZE=2048
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$OUT"
NITERS="${NITERS:-150000}"

python models/PostDamagePostTech_hardened.py \
  "$OUT" 128 "$NITERS" "$INC" 1000 "10e-6,40e-4" \
  "swish,tanh,tanh" "softplus,custom,custom" 4 32 \
  "warmup_cosine" "$OUT/out" 1.0 1.0

echo "DONE arm=$ARM out=$OUT"
