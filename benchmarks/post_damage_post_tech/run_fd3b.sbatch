#!/bin/bash
#SBATCH --time=0-01:30:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --job-name=pdpt_fd_v3b
#SBATCH --output=logging/fd_v3b-%j.out
#SBATCH --error=logging/fd_v3b-%j.err
set -euo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
mkdir -p logging
module unload python 2>/dev/null||true; module load python/anaconda-2021.05
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "start $(date)"
python -u - <<'PY'
import numpy as np, fd_pdpt_v3 as F
for (nK,nZ,nY) in [(31,31,31),(41,41,41),(61,41,41)]:
    out=F.solve(nK=nK,nZ=nZ,nY=nY,howard_max=70,kappa=0.4,tol=1e-6,verbose=False)
    ik=np.argmin(np.abs(out['logK']-np.log(880))); jz=np.argmin(np.abs(out['Z']-0.7)); ky=np.argmin(np.abs(out['Y']-3.0))
    lNy=0.00017675+2*0.0022*3.0+(1/6.0)*(3.0-2.5)
    print(f"grid {nK}x{nZ}x{nY}: iters={out['iters']} {out['time']:.0f}s resid={out['max_abs_residual']:.2e} | "
          f"i_d={out['i_d'][ik,jz,ky]:+.4f} i_g={out['i_g'][ik,jz,ky]:+.4f} vlK={out['vlK'][ik,jz,ky]:.3f} "
          f"c={out['c'][ik,jz,ky]:.4f} V_Y={out['vY'][ik,jz,ky]-lNy:+.4f}",flush=True)
print("TARGET (NN): i_d~0.040 i_g~0.104 vlK~0.54 c~0.064 V_Y~-0.16")
PY
echo "end $(date)"
