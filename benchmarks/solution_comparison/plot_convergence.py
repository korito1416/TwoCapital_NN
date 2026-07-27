"""Convergence-of-the-solution curves for the warm-start RCT: evaluate each run's
periodic checkpoints (ckpt_step<N>/ dirs written by models_warmstart) on ONE fixed
Latin-hypercube evaluation sample, per jump state, at a fixed xi.

This is the sampling-noise-free convergence comparison (unlike training_history,
whose per-step loss is a fresh-batch Monte Carlo draw).

Mid-training checkpoints of a pre-jump state are paired with the run's FINAL
downstream value functions — correct, because the chain trains post-jump states
to completion before any pre-jump state starts.
"""
import argparse, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solution_loader as SL

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True, help="label=/abs/run/root")
ap.add_argument("--xi", type=float, default=0.05)
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="convergence.png")
ap.add_argument("--npy", default="convergence.npy")
args = ap.parse_args()

RUNS = SL.parse_runs(args.run)
COLORS = ["#c1272d", "#1f4e79", "#2e7d32", "#e65100", "#6a1b9a", "#00838f", "#795548", "#546e7a",
          "#9e9d24", "#4527a0"]

def ckpt_steps(root, reg):
    steps = []
    for d in glob.glob(os.path.join(root, reg, "ckpt_step*")):
        m = re.search(r"ckpt_step(\d+)$", d)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)

def model_at(reg, root, step):
    """Regime model with the regime's OWN nets from ckpt_step<step>, downstream final."""
    m = SL.make_model(reg, root)             # final weights everywhere (incl. downstream)
    sub = os.path.join(root, reg, f"ckpt_step{step}")
    for nm in SL.NETS[reg]:
        w = SL.cast_w(os.path.join(sub, f"{nm}_nn_checkpoint_{reg}"), SL.CFG[nm], SL.DIMS[reg])
        getattr(m, f"{nm}_nn").set_weights(w)
    return m

out = {}
for reg in SL.REGS:
    ylo = SL.Y_ENTRY if reg.startswith("PostDamage") else 0.0
    lk, Z, Y, lr, _ = SL.lhs(args.n, [(4, 7), (0.01, 0.99), (ylo, 4), (1, 6), (0, 1 / 3)], seed=args.seed)
    for lab, root in RUNS:
        steps = ckpt_steps(root, reg)
        if not steps:
            continue
        curve = []
        for st in steps:
            m = model_at(reg, root, st)
            curve.append(SL.eval_terms(m, reg, lk, Z, Y, lr, args.xi)["res"])
            del m
        # final checkpoint as the last point (x = the run's total iterations)
        m = SL.make_model(reg, root)
        total = None
        ptxt = os.path.join(root, reg, "params.txt")
        if os.path.exists(ptxt):
            for line in open(ptxt, encoding="utf-8"):
                if line.startswith("num_iterations:"):
                    total = int(line.split(":", 1)[1].strip()); break
        steps_f = steps + [total if total else max(steps) + (steps[1] - steps[0] if len(steps) > 1 else 1)]
        curve.append(SL.eval_terms(m, reg, lk, Z, Y, lr, args.xi)["res"])
        del m
        out[f"{reg}|{lab}"] = (steps_f, curve)
        print(f"{reg:22s} {lab:14s} {len(steps)} ckpts, final res={curve[-1]:.2e}")

np.save(args.npy, {"xi": args.xi, "labels": [l for l, _ in RUNS], "data": out})

plt.rcParams.update({"font.size": 16, "axes.linewidth": 1.0, "xtick.labelsize": 13, "ytick.labelsize": 13})
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
for k, reg in enumerate(SL.REGS):
    a = axes.flat[k]
    for i, (lab, _) in enumerate(RUNS):
        key = f"{reg}|{lab}"
        if key not in out:
            continue
        st, cv = out[key]
        a.semilogy(np.array(st) / 1e3, cv, "o-", color=COLORS[i % 10], lw=2.0, ms=5, label=lab)
    a.grid(alpha=.25, lw=.6, which="major")
    a.set_title(SL.RLAB[reg], fontsize=16)
    a.set_xlabel("training step (thousands)")
    a.set_ylabel("error in the HJB equation")
axes.flat[0].legend(frameon=False, fontsize=11, loc="best")
fig.tight_layout()
fig.savefig(args.out, dpi=135, bbox_inches="tight")
print("wrote", args.out, "and", args.npy)
