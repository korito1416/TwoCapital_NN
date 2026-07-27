#!/bin/bash
#SBATCH --job-name=wsroll_jumps
#SBATCH --partition=caslake
#SBATCH --account=pi-lhansen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/job-outs/ws_eval/rolljumps-%j.out
#SBATCH --error=/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/job-outs/ws_eval/rolljumps-%j.err

module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/policy_rollout

ABS=/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
RUNA="$ABS/output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
P1M="$ABS/output_warmstart_1M_20260714"
WLX="$ABS/output_warmstart_1M_xi0p005_20260714"

# jumps ON (simulator default), same seed => same Brownian/uniform draws (CRN) as
# the no-jump run paths_stoch_xi0p05.npz
python -u rollout.py \
  --run "reference=$RUNA" \
  --run "nber_s1=$P1M/nber_s1" --run "nber_s2=$P1M/nber_s2" \
  --run "perturb_s1=$P1M/perturb_s1" --run "perturb_s2=$P1M/perturb_s2" \
  --run "wlx_s1=$WLX/nber_s1" --run "wlx_s2=$WLX/nber_s2" \
  --xi 0.05 --n-paths 2000 --years 60 --seed 0 \
  --record-stride 4 --out paths_stoch_jumps_xi0p05.npz
echo PATHS-JUMPS-DONE
