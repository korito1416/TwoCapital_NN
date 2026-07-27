"""Investment sensitivity to xi (Lars's request): year-60 investments vs xi,
deterministic paths, replicate means; reference + the two warm starts."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
XIS = ["0.050", "0.100", "0.300", "1.000", "3.000", "10.000", "35.000", "148.600"]
xv = [float(x) for x in XIS]

def yr60(run, q, xi):
    p = f"{ROOT}/{run}/SimulationDeterministic/SimulationOutputs_ξ_{xi}/{q}.txt"
    try:
        return float(np.loadtxt(p)[-1])
    except Exception:
        return np.nan

SETTINGS = [
    ("reference",    "0.1",     ["output_xiprofile_ref_20260714"]),
    ("warm start A", "#0072B2", ["output_warmstart_1M_20260714/nber_s1",
                                 "output_warmstart_1M_20260714/nber_s2"]),
    ("warm start B", "#009E73", ["output_warmstart_1M_20260714/perturb_s1",
                                 "output_warmstart_1M_20260714/perturb_s2"]),
]
QUANT = [("I_g", "green investment $I_g$, year 60"),
         ("I_d", "dirty investment $I_d$, year 60"),
         ("I_r", "R&D investment $I_r$, year 60")]

plt.rcParams.update({"font.size": 15, "axes.linewidth": 1.0})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
for k, (q, ylab) in enumerate(QUANT):
    a = axes[k]
    for lab, col, runs in SETTINGS:
        ys = []
        for xi in XIS:
            vals = [yr60(r, q, xi) for r in runs]
            vals = [v for v in vals if np.isfinite(v)]
            ys.append(np.mean(vals) if vals else np.nan)
        a.plot(xv, ys, "-o", color=col, lw=2.3, ms=6, label=(lab if k == 0 else None))
    a.set_xscale("log")
    a.set_xlabel(r"uncertainty aversion $\xi$ (log scale)")
    a.set_ylabel(ylab)
    a.grid(alpha=0.22, lw=0.6, which="both")
fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(HERE, "figures", "investments_vs_xi.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
