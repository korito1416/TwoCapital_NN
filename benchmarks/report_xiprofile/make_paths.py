"""ξ-profile of the reference (smart-guess) solution, no-shock deterministic
simulation, in Haoyang's format. Reads the sandbox re-simulation of RUNA
(bit-identical to RUNA's own SimulationDeterministic; see the alignment table in
the report). One overlaid curve per ξ on the grid used before.

Outputs:
  figures/paths_grid.png        — I_g, I_d, RD, E, C/Y over 60 years, ξ overlaid
  figures/jump_densities.png    — technology and damage first-jump densities, ξ overlaid
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SB = os.path.join(ROOT, "output_xiprofile_ref_20260714", "SimulationDeterministic")

# same grid as before; warmer color = more uncertainty-averse (smaller ξ)
XIS = [("0.050", "ξ = 0.05", "#D55E00"),
       ("0.100", "ξ = 0.1",  "#E69F00"),
       ("0.300", "ξ = 0.3",  "#0072B2"),
       ("148.600", "ξ = 148.6 (neutral)", "#000000")]

def load(xi, name):
    d = os.path.join(SB, f"SimulationOutputs_ξ_{xi}")
    return np.loadtxt(os.path.join(d, f"{name}.txt"))

def clip60(t, v):
    n = min(len(t), len(v))
    m = t[:n] <= 60.0
    return t[:n][m], v[:n][m]

# ---- Figure 1: investment / emissions / consumption paths -------------------
PANELS = [
    ("I_g", "green investment $I_g$", 1.0),
    ("I_d", "dirty investment $I_d$", 1.0),
    ("RD",  "R&D investment as % of output ($I_r/Y$)", 100.0),
    ("E",   "emissions $\\mathcal{E}$ (GtC)", 1.0),
    ("ConsumptionOutputRatio", "consumption as % of output ($C/Y$)", 100.0),
]
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.0})
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
axf = axes.ravel()
for k, (name, ylab, scale) in enumerate(PANELS):
    a = axf[k]
    for xi, lab, col in XIS:
        t = load(xi, "t"); v = load(xi, name)
        t, v = clip60(t, v)
        a.plot(t, v * scale, color=col, lw=2.6, label=(lab if k == 0 else None))
    a.set_xlabel("year"); a.set_ylabel(ylab); a.set_xlim(0, 60); a.grid(alpha=.25, lw=.6)
axf[5].axis("off")
h, l = axf[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=15,
           bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(rect=(0, 0.05, 1, 1))
out = os.path.join(os.path.dirname(__file__), "figures", "paths_grid.png")
fig.savefig(out, dpi=155, bbox_inches="tight"); print("wrote", out)

# ---- Figure 2: first-jump densities -----------------------------------------
DENS = [("tech_jump_density", "technology first-jump density"),
        ("dmg_jump_density",  "damage first-jump density")]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
for k, (name, title) in enumerate(DENS):
    a = axes[k]
    for xi, lab, col in XIS:
        t = load(xi, "t"); v = load(xi, name)
        t, v = clip60(t, v)
        a.plot(t, v, color=col, lw=2.6, label=(lab if k == 0 else None))
    a.set_title(title); a.set_xlabel("year"); a.set_ylabel("density")
    a.set_xlim(0, 60); a.grid(alpha=.25, lw=.6)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = os.path.join(os.path.dirname(__file__), "figures", "jump_densities.png")
fig.savefig(out, dpi=155, bbox_inches="tight"); print("wrote", out)

# ---- print the year-60 alignment numbers for the report table ---------------
print("\nyear-60 values (reference re-simulation):")
print(f"{'xi':>10} {'I_g':>9} {'I_d':>9} {'RD%':>9} {'E':>9} {'C/Y%':>9}")
for xi, lab, _ in XIS:
    def v60(nm, sc=1.0):
        t = load(xi, "t"); v = load(xi, nm); t, v = clip60(t, v); return v[-1] * sc
    print(f"{lab:>22} {v60('I_g'):9.3f} {v60('I_d'):9.3f} {v60('RD',100):9.4f} "
          f"{v60('E'):9.3f} {v60('ConsumptionOutputRatio',100):9.3f}")
