"""INTERNAL — the mechanism behind warm start A's rising dirty investment and
emissions, and whether widening the ξ range changes it. Four channels at ξ = 0.05:
total capital, consumption share, green share, emissions. warm start A + wider ξ is
warm start A's own initialization retrained with the ξ floor at 0.005; overlaying it
shows the wider range does not move the solution.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REF = os.path.join(ROOT, "output_xiprofile_ref_20260714")
P1M = os.path.join(ROOT, "output_warmstart_1M_20260714")
WLX = os.path.join(ROOT, "output_warmstart_1M_xi0p005_20260714")
OUT = os.path.join(os.path.dirname(__file__), "figures_internal")

SETTINGS = [
    ("reference",             "0.1",     "-",   [REF]),
    ("warm start A",          "#0072B2", "-",   [f"{P1M}/nber_s1", f"{P1M}/nber_s2"]),
    ("warm start B",          "#009E73", "-",   [f"{P1M}/perturb_s1", f"{P1M}/perturb_s2"]),
    ("warm start A + wider ξ","#CC79A7", "--",  [f"{WLX}/nber_s1", f"{WLX}/nber_s2"]),
]
XI = "0.050"

def load(root, name):
    d = os.path.join(root, "SimulationDeterministic", f"SimulationOutputs_ξ_{XI}")
    return np.loadtxt(os.path.join(d, f"{name}.txt"))

def path(roots, name, scale=1.0):
    curves, t0 = [], None
    for r in roots:
        t = load(r, "t"); v = load(r, name) * scale
        n = min(len(t), len(v)); m = t[:n] <= 60.0
        curves.append(v[:n][m]); t0 = t[:n][m]
    n = min(len(c) for c in curves)
    return t0[:n], np.mean([c[:n] for c in curves], axis=0)

PANELS = [("K", "total capital $K$", 1.0),
          ("ConsumptionOutputRatio", "consumption share $C/Y$ (%)", 100.0),
          ("Z", "green capital share $Z$", 1.0),
          ("E", "emissions $\\mathcal{E}$ (GtC)", 1.0)]
plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
for k, (name, ylab, sc) in enumerate(PANELS):
    a = axes[k]
    for lab, col, ls, roots in SETTINGS:
        t, v = path(roots, name, sc)
        a.plot(t, v, color=col, ls=ls, lw=2.6, label=(lab if k == 0 else None))
    a.set_xlabel("year"); a.set_ylabel(ylab); a.set_xlim(0, 60); a.grid(alpha=.25, lw=.6)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=13.5,
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Mechanism at ξ = 0.05: warm A over-accumulates; wider ξ inherits it (INTERNAL)",
             fontsize=14)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
out = os.path.join(OUT, "grid5_mechanism.png")
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
