"""Policy fan for the warm-start RCT: investment paths at xi=0.05 across all arms
plus the RUNA reference. Same color = same warm-start method; solid/dashed = the
two seeds. Tight same-color pairs = the method reproduces; spread across colors =
the warm start selects the solution."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WS = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "output_warmstart_20260714")
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
RUNA = os.path.join(ROOT, "output_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_"
    "#HiddenLayer_4_num_iterations1000000")
XI = "0.050"

SETTINGS = [
    ("warm start A", "#0072B2", [os.path.join(ROOT, "output_warmstart_1M_20260714", "nber_s1"),
                                 os.path.join(ROOT, "output_warmstart_1M_20260714", "nber_s2")]),
    ("warm start B", "#009E73", [os.path.join(ROOT, "output_warmstart_1M_20260714", "perturb_s1"),
                                 os.path.join(ROOT, "output_warmstart_1M_20260714", "perturb_s2")]),
    ("warm start A, wider ξ", "#CC79A7",
                                [os.path.join(ROOT, "output_warmstart_1M_xi0p005_20260714", "nber_s1"),
                                 os.path.join(ROOT, "output_warmstart_1M_xi0p005_20260714", "nber_s2")]),
]
QUANT = [("I_g", "green investment $I_g$"), ("I_d", "dirty investment $I_d$"),
         ("I_r", "R&D investment $I_r$")]

def series(run, q):
    p = os.path.join(run, "SimulationDeterministic", f"SimulationOutputs_ξ_{XI}", f"{q}.txt")
    return np.loadtxt(p) if os.path.exists(p) else None

plt.rcParams.update({"font.size": 17, "axes.linewidth": 1.1,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0))
t = None
for k, (q, ylab) in enumerate(QUANT):
    a = axes[k]
    ref = series(RUNA, q)
    if ref is not None:
        if t is None:
            t = np.arange(len(ref)) / 12.0
        a.plot(t[:len(ref)], ref, color="0.1", lw=3.2, ls=":", label="reference (base run)")
    for lab, col, runs in SETTINGS:
        ys = [series(r, q) for r in runs]
        ys = [y for y in ys if y is not None]
        if not ys:
            continue
        n = min(len(y) for y in ys)
        ym = np.mean([y[:n] for y in ys], axis=0)
        a.plot(t[:n], ym, color=col, lw=2.4, label=lab)
    a.set_xlabel("year"); a.set_ylabel(ylab); a.grid(alpha=.25, lw=.6)
    a.set_xlim(0, 60)
axes[0].legend(frameon=False, fontsize=13, loc="best")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "figures", f"policy_fan_xi0p05{SUFFIX}.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
# numbers: year-60 seed means per setting
print("year-60 seed-mean values (I_g | I_d | I_r):")
for lab, _, runs in SETTINGS:
    vals = [[series(r, q) for q, _ in QUANT] for r in runs]
    vals = [v for v in vals if all(x is not None for x in v)]
    means = [np.mean([v[i][-1] for v in vals]) for i in range(3)]
    print(f"  {lab:14s} {means[0]:7.2f} | {means[1]:6.2f} | {means[2]:6.2f}")
ref = [series(RUNA, q) for q, _ in QUANT]
print(f"  {'reference':14s} {ref[0][-1]:7.2f} | {ref[1][-1]:6.2f} | {ref[2][-1]:6.2f}")
print("wrote", out)
