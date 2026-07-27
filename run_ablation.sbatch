#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=hard_abl
#SBATCH --output=hardened_abl_%x_%j.out
#SBATCH --error=hardened_abl_%x_%j.err

# ABLATION of the hardening bundle: isolate which piece drives the ~3.45x HJB-residual
# win and which drives the de-invest/costate REGRESSION.  Supervision OFF in all arms.
# Matched to arm A: NITERS=150000, curriculum_frac=0.3 (anneal over 45k); grade at ~80k.
#   ABLCFG=rel   -> relative-residual ONLY (curriculum off)   : is rel the residual-win driver, controls clean?
#   ABLCFG=curr  -> xi-curriculum ONLY    (relative off)      : does curriculum alone help/regress?
#   ABLCFG=base  -> C>0 + Bug-A ONLY      (rel off, curr off) : the minimal-safe baseline (~incumbent?)

set -euo pipefail
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

INC="output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"

export COSTATE_SUP_WEIGHT=0.0     # supervision OFF in every ablation arm
export CHECKPOINT_EXPORT_EVERY=10000
export XI_CURRICULUM_FRAC=0.3
export VALIDATION_BATCHES=4
export VALIDATION_BATCH_SIZE=2048
export CUDA_VISIBLE_DEVICES=""

ABLCFG="${ABLCFG:-base}"
case "$ABLCFG" in
  rel)  export RELATIVE_RESIDUAL=1; export XI_CURRICULUM=0; OUT="output_hardened_scratch/abl_rel_only" ;;
  curr) export RELATIVE_RESIDUAL=0; export XI_CURRICULUM=1; OUT="output_hardened_scratch/abl_curr_only" ;;
  base) export RELATIVE_RESIDUAL=0; export XI_CURRICULUM=0; OUT="output_hardened_scratch/abl_base_c0" ;;
  *) echo "unknown ABLCFG=$ABLCFG"; exit 1 ;;
esac

mkdir -p "$OUT"
NITERS="${NITERS:-150000}"
echo "ABLATION arm=$ABLCFG  RELATIVE_RESIDUAL=$RELATIVE_RESIDUAL  XI_CURRICULUM=$XI_CURRICULUM  out=$OUT"

python models/PostDamagePostTech_hardened.py \
  "$OUT" 128 "$NITERS" "$INC" 1000 "10e-6,40e-4" \
  "swish,tanh,tanh" "softplus,custom,custom" 4 32 \
  "warmup_cosine" "$OUT/out" 1.0 1.0

echo "DONE ablation arm=$ABLCFG out=$OUT"
