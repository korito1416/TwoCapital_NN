#!/bin/bash
# submit_jones_ladder.sh -- submit the FULL JONES ladder as a dependency-ordered
# Slurm DAG (account pi-lhansen, partition caslake). Post regimes before pre;
# each job runs its own 3-xi warm chain {148.4 -> 0.1 -> 0.05} internally.
#
#   stage 1: PostDamagePostTech  x l3 in {0..4}   (2-D, cheap)
#   stage 2: PreDamagePostTech                    afterok: ALL of stage 1
#   stage 3: PostDamagePreTech   x l3 in {0..4}   afterok: stage-1 job of SAME l3
#   stage 4: PreDamagePreTech                     afterok: stage 2 + ALL stage 3
#
# 12 jobs, 36 solves total (4 regimes x lambda3 slices x xi in {148.4,0.1,0.05}),
# grid (61,31,41), s in [-6,2]. Measured: one full-grid evaluation sweep of the
# heaviest regime = ~200s (login node) -> stage1 ~1h/job, stage2 ~2h,
# stage3 ~8h/job, stage4 ~13h; ~230 core-hours at 4 cpus/job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="$HERE/jones_ladder.sbatch"
mkdir -p "$HERE/../joblogs"

NZ=${NZ:-61}; NY=${NY:-31}; NS=${NS:-46}
COMMON="ALL,NZ=$NZ,NY=$NY,NS=$NS"

# ---- stage 1: 5x PostDamagePostTech
S1=()
for K in 0 1 2 3 4; do
  jid=$(sbatch --parsable --job-name="jonesS1_pdpt_l3$K" --time=12:00:00 \
        --export="$COMMON,REGIME=PostDamagePostTech,L3IDX=$K" "$SB")
  echo "stage1 l3=$K -> $jid"
  S1+=("$jid")
done
S1DEP=$(IFS=:; echo "${S1[*]}")

# ---- stage 2: PreDamagePostTech (needs all 5 terminals at every xi)
S2=$(sbatch --parsable --job-name="jonesS2_prdpt" --time=12:00:00 \
     --dependency="afterok:$S1DEP" \
     --export="$COMMON,REGIME=PreDamagePostTech" "$SB")
echo "stage2       -> $S2 (afterok:$S1DEP)"

# ---- stage 3: 5x PostDamagePreTech (each needs its own-l3 terminal)
S3=()
for K in 0 1 2 3 4; do
  jid=$(sbatch --parsable --job-name="jonesS3_pdprt_l3$K" --time=34:00:00 \
        --dependency="afterok:${S1[$K]}" \
        --export="$COMMON,REGIME=PostDamagePreTech,L3IDX=$K" "$SB")
  echo "stage3 l3=$K -> $jid (afterok:${S1[$K]})"
  S3+=("$jid")
done
S3DEP=$(IFS=:; echo "${S3[*]}")

# ---- stage 4: PreDamagePreTech (needs stage 2 + all stage 3)
S4=$(sbatch --parsable --job-name="jonesS4_prdprt" --time=34:00:00 \
     --dependency="afterok:$S2:$S3DEP" \
     --export="$COMMON,REGIME=PreDamagePreTech" "$SB")
echo "stage4       -> $S4 (afterok:$S2:$S3DEP)"

echo "JONES ladder DAG submitted: S1=${S1[*]} S2=$S2 S3=${S3[*]} S4=$S4"
echo "outputs -> $HERE/outputs/jones/  logs -> $HERE/../joblogs/"
