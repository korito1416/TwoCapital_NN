#!/bin/bash
# submit_physrisk_ladder.sh -- submit the PHYSRISK full-grid FD ladder:
# 3 afterok-chained rungs (one per xi), each running the 12-solve backward regime
# chain, i.e. the full 36-solve matrix (4 regimes x lambda3 x xi):
#     rung 1: xi = 148.4  (near-neutral family member; cold start)
#     rung 2: xi = 0.1    (warm-started from the 148.4 rung: W_init for the
#                          jump-distortion lag + controls)
#     rung 3: xi = 0.05   (warm-started from the 0.1 rung)
# Grid (61,41,61), s in [-6,2]; T=1200 dt=2.5 PIBYS, step_exact jump sinks,
# robust drift feedback ON. Outputs: benchmarks/economy_zoo/outputs/physrisk/
# (12 npz + 12 _PROVENANCE.json + a chain SUMMARY json per rung).
#
# PRE-REQ: the coarse validation gates in
#   benchmarks/economy_zoo/outputs/physrisk_coarse/physrisk_validation_PROVENANCE.json
# must be PASS before trusting/submitting this ladder.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../joblogs ../outputs/physrisk

j1=$(sbatch --parsable --export=ALL,PHYSRISK_XI=148.4 physrisk_ladder.sbatch)
echo "rung 1 (xi=148.4): job $j1"
j2=$(sbatch --parsable --dependency=afterok:"$j1" \
     --export=ALL,PHYSRISK_XI=0.1,PHYSRISK_WARM_XI=148.4 physrisk_ladder.sbatch)
echo "rung 2 (xi=0.1, warm from 148.4): job $j2 (afterok:$j1)"
j3=$(sbatch --parsable --dependency=afterok:"$j2" \
     --export=ALL,PHYSRISK_XI=0.05,PHYSRISK_WARM_XI=0.1 physrisk_ladder.sbatch)
echo "rung 3 (xi=0.05, warm from 0.1): job $j3 (afterok:$j2)"
echo "ladder submitted: $j1 -> $j2 -> $j3"
