"""Per-jump-state, per-xi evaluation of every objective term on ONE shared
Latin-hypercube sample of states (the 'same evaluation sample' comparison).
Post-damage jump states are sampled on their reachable region Y in [2.5, 4].

Also supports --checkpoint-glob to score periodic checkpoints of a single run
(convergence-of-the-solution curves for the warm-start RCT).
"""
import argparse
import numpy as np
import solution_loader as SL

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True, help="label=/abs/run/root (repeatable)")
ap.add_argument("--xis", default="0.05,0.1,0.3,1,10,148.6")
ap.add_argument("--n", type=int, default=8192)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="cross_state_lhs.npy")
args = ap.parse_args()

RUNS = SL.parse_runs(args.run)
XIS = [float(x) for x in args.xis.split(",")]

out = {}
for reg in SL.REGS:
    ylo = SL.Y_ENTRY if reg.startswith("PostDamage") else 0.0
    lk, Z, Y, lr, l3 = SL.lhs(args.n, [(4, 7), (0.01, 0.99), (ylo, 4), (1, 6), (0, 1 / 3)], seed=args.seed)
    for lab, root in RUNS:
        m = SL.make_model(reg, root)
        for xi in XIS:
            out[f"{reg}|{lab}|{xi}"] = SL.eval_terms(m, reg, lk, Z, Y, lr, xi)
        del m
    print(reg, "done")

np.save(args.out, {"XIS": XIS, "REGS": SL.REGS, "HASR": SL.HASR,
                   "labels": [lab for lab, _ in RUNS], "data": out})
print("saved", args.out)
