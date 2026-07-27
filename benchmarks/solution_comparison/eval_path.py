"""Per-jump-state, per-xi evaluation of every objective term along the simulated
60-year path (RMS across years). Generalizes the CS1/CS2 pipeline to any set of runs.

Example (the two-trainings comparison):
  python eval_path.py \
    --run "training 1=<RUNA dir>" --run "training 2=<RUNB dir>" \
    --paths-from "<RUNB dir>" --out cross_state_path.npy
For the warm-start RCT: one --run per arm, --paths-from a reference run that has
SimulationDeterministic outputs (states vary negligibly across solutions).
"""
import argparse, os
import numpy as np
import solution_loader as SL

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True, help="label=/abs/run/root (repeatable)")
ap.add_argument("--paths-from", required=True, help="run root whose SimulationDeterministic supplies path states")
ap.add_argument("--xis", default="0.05,0.075,0.1,0.2,0.3,0.7,1.5,3,7,15,35,70,148.6")
ap.add_argument("--path-xis", default="0.050,148.600", help="available SimulationOutputs_ξ_<dir> tags")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--out", default="cross_state_path.npy")
args = ap.parse_args()

RUNS = SL.parse_runs(args.run)
XIS = [float(x) for x in args.xis.split(",")]
PTAGS = args.path_xis.split(",")
PATHS = {t: SL.path_states(os.path.abspath(args.paths_from), t, args.stride) for t in PTAGS}

def nearest(xi):
    return min(PTAGS, key=lambda t: abs(np.log(xi) - np.log(float(t))))

out = {}
for reg in SL.REGS:
    for lab, root in RUNS:
        m = SL.make_model(reg, root)
        for xi in XIS:
            lk, Z, Y, lr = PATHS[nearest(xi)]
            Yr = Y if reg.startswith("PreDamage") else np.full_like(Y, SL.Y_ENTRY)
            out[f"{reg}|{lab}|{xi}"] = SL.eval_terms(m, reg, lk, Z, Yr, lr, xi)
        del m
    a = out[f"{reg}|{RUNS[0][0]}|{XIS[0]}"]
    print(f"{reg:22s} {RUNS[0][0]} xi={XIS[0]}: res={a['res']:.2e}")

np.save(args.out, {"XIS": XIS, "REGS": SL.REGS, "HASR": SL.HASR,
                   "labels": [lab for lab, _ in RUNS], "data": out})
print("saved", args.out)
