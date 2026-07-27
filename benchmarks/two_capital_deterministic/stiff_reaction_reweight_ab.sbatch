#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=stiff_reac_ab
#SBATCH --output=slurm-stiff_reac_ab-%j.out
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/two_capital_deterministic
for SPAN in 1.0 2.0 3.0; do
  for S in 0 1 2; do
    echo "########## DECADE_SPAN=$SPAN SEED=$S ##########"
    DECADE_SPAN=$SPAN SEED=$S python stiff_reaction_reweight_ab.py
  done
done
echo ALL_DONE
