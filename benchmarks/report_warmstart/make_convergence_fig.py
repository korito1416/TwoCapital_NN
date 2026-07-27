"""Merge the per-arm convergence npys (solution_comparison/plot_convergence.py output)
into one 4-panel figure: per jump state, error in the HJB equation on a fixed common
sample at xi=0.05, vs training iteration. Method colors match the policy fan.
"""
import os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [("nber", "#1f4e79"), ("perturb", "#2e7d32"), ("paperlr2M", "#00838f")]
DISPLAY = {"nber": "warm start A", "perturb": "warm start B", "paperlr2M": "cold start"}
REGS = ["PreDamagePreTech", "PreDamagePostTech", "PostDamagePreTech", "PostDamagePostTech"]
TITLES = {"PreDamagePreTech": "pre-damage, pre-tech",
          "PreDamagePostTech": "pre-damage, post-tech",
          "PostDamagePreTech": "post-damage, pre-tech",
          "PostDamagePostTech": "post-damage, post-tech"}

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
curves = {}   # (reg, arm) -> (steps, curve)
for f in glob.glob(os.path.join(HERE, DATA, "conv_*.npy")) + glob.glob(os.path.join(HERE, "data_ext", "conv_*.npy")):
    d = np.load(f, allow_pickle=True).item()
    for out in np.atleast_1d(d["data"]):
        for key, (steps, curve) in out.items():
            reg, lab = key.split("|")
            curves[(reg, lab)] = (np.asarray(steps), np.asarray(curve))

ref_file = os.path.join(HERE, "data", "runa_fixed_sample.npy")  # same frozen sample for both waves
REF = np.load(ref_file, allow_pickle=True).item() if os.path.exists(ref_file) else {}

plt.rcParams.update({"font.size": 17, "axes.linewidth": 1.1,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})
fig, axes = plt.subplots(1, 4, figsize=(24, 6.0), sharey=True)
import matplotlib.lines as mlines
for k, reg in enumerate(REGS):
    a = axes[k]
    if reg in REF:
        a.axhline(REF[reg], color="0.25", lw=3.2, ls=":")
    for meth, col in METHODS:
        for s, ls in [(1, "-"), (2, "--")]:
            key = (reg, f"{meth}_s{s}")
            if key not in curves:
                continue
            steps, c = curves[key]
            n = min(len(steps), len(c))
            a.plot(steps[:n] / 1e3, c[:n], color=col, lw=2.0, ls=ls, marker="o", ms=3.5)
    a.set_yscale("log")
    a.set_title(TITLES[reg], fontsize=17)
    a.set_xlabel("iteration (thousands)")
    a.grid(alpha=.25, lw=.6, which="both")
axes[0].set_ylabel("error in the HJB equation (fixed sample, $\\xi=0.05$)")
present = {m for (reg, lab) in curves for m, _ in METHODS if lab.startswith(m)}
handles = [mlines.Line2D([], [], color="0.25", lw=3.2, ls=":", label="reference (base run, final)")]
handles += [mlines.Line2D([], [], color=col, lw=2.0, label=DISPLAY[meth]) for meth, col in METHODS if meth in present]
fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "figures", f"convergence_by_state{SUFFIX}.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out, "|", len(curves), "curves")
