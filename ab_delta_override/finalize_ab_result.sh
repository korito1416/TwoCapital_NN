#!/bin/bash
# Wait for the delta=0.025 A/B DAG to finish, then read the xi=0.1 lambda3 distorted
# weights and append the final verdict to ab_delta025_result.txt.
set -uo pipefail
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
ROOT="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
SIM_JOB="51346363"
PRE_JOB="51346362"
RES="${ROOT}/benchmarks/report_one_jump_lowxi_stable/mike_diagnostics/ab_delta025_result.txt"
EXPORT_FOLDER="${ROOT}/output_delta025_ab/TwoStageTech_DELTA025_AB_Pi0p04_TechIntensityScale_1p0_LR_warmup_cosine_40e-5,40e-5,40e-5,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations300000"
SIMDIR="${EXPORT_FOLDER}/PreDamagePreTech"

# Poll until the sim job leaves the queue (completed or failed).
while squeue -j "${SIM_JOB}" -h 2>/dev/null | grep -q .; do
    sleep 120
done

# Give the filesystem a moment.
sleep 20

module unload python 2>/dev/null; module load python/anaconda-2021.05 2>/dev/null

REF_OLD="[0.203, 0.201, 0.200, 0.199, 0.197]"
REF_NEW="[0.006, 0.023, 0.080, 0.244, 0.647]"

# Find the xi=0.1 output dir (sim uses f"_ξ_{xi:.3f}").
W01=""
for cand in "${SIMDIR}/SimulationOutputs_ξ_0.100" "${SIMDIR}/SimulationOutputs_ξ_0.1" "${SIMDIR}"/SimulationOutputs_ξ_0.100*; do
    [ -f "${cand}/lambda3_weights_distorted.txt" ] && { W01="${cand}/lambda3_weights_distorted.txt"; break; }
done

{
  echo ""
  echo "=== FINALIZED $(date) ==="
  echo "Sim job ${SIM_JOB} final state:"
  sacct -j "${SIM_JOB}" --format=JobID,JobName%22,State,ExitCode -n 2>/dev/null | sed 's/^/  /'
  echo "PreDamagePreTech job ${PRE_JOB} final state:"
  sacct -j "${PRE_JOB}" --format=JobID,JobName%22,State,ExitCode -n 2>/dev/null | sed 's/^/  /'
  echo ""
  echo "Available SimulationOutputs dirs:"
  ls -d "${SIMDIR}"/SimulationOutputs_* 2>/dev/null | sed 's/^/  /' || echo "  (none)"
  echo ""
  if [ -n "${W01}" ]; then
      echo "READ FILE: ${W01}"
      RAW="$(cat "${W01}")"
      echo "RAW xi=0.1 lambda3 distorted weights:"
      echo "  ${RAW}"
      # Normalize to a python list + compute verdict.
      python3 - "$W01" <<'PY'
import sys, numpy as np
w = np.atleast_1d(np.loadtxt(sys.argv[1])).astype(float)
w = w / w.sum() if w.sum() > 0 else w
fmt = "[" + ", ".join(f"{x:.3f}" for x in w) + "]"
worst = float(w[-1]); umax = float(np.max(np.abs(w - 0.2)))
print("NORMALIZED xi=0.1 weights (grid [0,1/12,1/6,1/4,1/3]):")
print("  delta=0.025 RETRAIN :", fmt)
print("  ref pre-FOCIr 0.025 : [0.203, 0.201, 0.200, 0.199, 0.197]  (near-uniform)")
print("  ref current   0.01  : [0.006, 0.023, 0.080, 0.244, 0.647]  (skewed)")
print(f"  weight-on-worst(lambda3=1/3) = {worst:.3f}   max|w-0.20| = {umax:.3f}")
NEAR = umax < 0.05 and worst < 0.30
SKEW = worst > 0.45
print("")
if NEAR:
    print("VERDICT: NEAR-UNIFORM reproduced. delta=0.025 recovers the pre-FOCIr near-uniform")
    print("         xi=0.1 lambda3 weights => delta IS the driver; the FOCIr code refactor is")
    print("         EXONERATED for this distortion shift.")
elif SKEW:
    print("VERDICT: STILL SKEWED at delta=0.025. delta alone does NOT recover near-uniform =>")
    print("         the code refactor (or another change) contributes; needs a deeper look.")
else:
    print("VERDICT: PARTIAL / INTERMEDIATE. Not cleanly near-uniform nor as skewed as current;")
    print("         inspect convergence (loss_v) before concluding.")
PY
  else
      echo "READOUT NOT FOUND: no lambda3_weights_distorted.txt for xi=0.1 under ${SIMDIR}"
      echo "  (check sim job ${SIM_JOB} logs; DAG may have failed upstream.)"
  fi
  echo ""
  echo "Convergence check (final loss_v per trained regime; ~1e-3 = paper-grade):"
  for R in PostDamagePostTech PostDamageIntermTech PostDamagePreTech PreDamagePostTech PreDamageIntermTech PreDamagePreTech; do
      LV="$(tail -1 "${EXPORT_FOLDER}/${R}/training_history.csv" 2>/dev/null | awk -F, '{print $2}')"
      printf "  %-22s loss_v=%s\n" "$R" "${LV:-<none>}"
  done
} >> "${RES}"

echo "finalize done -> ${RES}"
