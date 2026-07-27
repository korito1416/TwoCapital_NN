#!/bin/bash
#SBATCH --job-name=ablat_root
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/home/kunjianli/claude-1983122066/-project-lhansen-Cap-damage-TwoStageTechJump-FOCIr-orignal/b1cc181d-78a1-47f3-9df3-aca58828ef92/scratchpad/ablation_root_%j.out

module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
SC=/home/kunjianli/claude-1983122066/-project-lhansen-Cap-damage-TwoStageTechJump-FOCIr-orignal/b1cc181d-78a1-47f3-9df3-aca58828ef92/scratchpad
python -u "$SC/ablation_root.py"
echo "=== SBATCH DONE ==="
