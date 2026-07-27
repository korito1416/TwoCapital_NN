#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --mem=8G
# CPU throughput bench: loops (intra,inter) x batch configs; one python process each
# (threading must be set before TF runs any op). Submit with -c {1,2,4,8}.
set -uo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/cpu_throughput
module load python/anaconda-2021.05
C="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$C" OPENBLAS_NUM_THREADS="$C" TF_CPP_MIN_LOG_LEVEL=2

# threading configs scaled to allotted cpus
if   [ "$C" -eq 1 ]; then TCONFS="1:1"
elif [ "$C" -eq 2 ]; then TCONFS="1:1 2:1 1:2"
else                      TCONFS="1:1 $C:1 $((C/2)):2 2:$((C/2))"
fi

echo "cpus=$C  tconfs=$TCONFS"
for tc in $TCONFS; do
  intra="${tc%%:*}"; inter="${tc##*:}"
  for b in 128 512 2048; do
    BENCH_INTRA="$intra" BENCH_INTER="$inter" BENCH_BATCH="$b" \
      python -u bench.py 2>/dev/null | grep "^RESULT" || echo "RESULT,$C,$intra,$inter,$b,FAIL"
  done
done
echo "bench done (cpus=$C)"
