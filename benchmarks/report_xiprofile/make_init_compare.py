"""INTERNAL comparison (not for PIs): how the no-shock ξ-profile depends on the
initialization. Reference vs warm start A (nber) vs warm start B (perturb), each
seed-averaged, over the ξ grid on which all three have full Haoyang-format
simulations. Cold/anchor deliver degenerate economies (I_g ~ 0) at these
checkpoints and are reported as a note, not plotted.

Outputs (figures_internal/):
  init_outcomes_vs_xi.png  — year-60 outcomes vs ξ, one curve per initialization
  init_paths_xi0p05.png    — full 60y paths at ξ=0.05, initializations overlaid
  init_densities.png       — distorted first-jump densities at ξ=0.05 and 0.3
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REF = os.path.join(ROOT, "output_xiprofile_ref_20260714")
P1M = os.path.join(ROOT, "output_warmstart_1M_20260714")
OUT = os.path.join(os.path.dirname(__file__), "figures_internal")
os.makedirs(OUT, exist_ok=True)

# each initialization = one or more run roots (seed-averaged)
INITS = [
    ("reference",     "0.1",     [REF]),
    ("warm start A",  "#0072B2", [f"{P1M}/nber_s1", f"{P1M}/nber_s2"]),
    ("warm start B",  "#009E73", [f"{P1M}/perturb_s1", f"{P1M}/perturb_s2"]),
]
XIS = ["0.050", "0.100", "0.300", "148.600"]
XIVAL = {"0.050": 0.05, "0.100": 0.1, "0.300": 0.3, "148.600": 148.6}

def load(root, xi, name):
    d = os.path.join(root, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}")
    return np.loadtxt(os.path.join(d, f"{name}.txt"))

def path60(root, xi, name, scale=1.0):
    t = load(root, xi, "t"); v = load(root, xi, name)
    n = min(len(t), len(v)); m = t[:n] <= 60.0
    return t[:n][m], v[:n][m] * scale

def y60(root, xi, name, scale=1.0):
    _, v = path60(root, xi, name, scale); return v[-1]

def avg_over_seeds(roots, fn):
    vals = [fn(r) for r in roots]
    return float(np.mean(vals))

# ---- Figure 1: year-60 outcomes vs ξ, one curve per initialization ----------
PANELS = [("I_g", "green investment $I_g$", 1.0),
          ("I_d", "dirty investment $I_d$", 1.0),
          ("RD",  "R&D as % of output", 100.0),
          ("E",   "emissions (GtC)", 1.0),
          ("ConsumptionOutputRatio", "consumption as % of output", 100.0)]
plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
axf = axes.ravel()
xs = [XIVAL[x] for x in XIS]
for k, (name, ylab, sc) in enumerate(PANELS):
    a = axf[k]
    for lab, col, roots in INITS:
        ys = [avg_over_seeds(roots, lambda r: y60(r, xi, name, sc)) for xi in XIS]
        a.plot(xs, ys, "o-", color=col, lw=2.4, ms=7, label=(lab if k == 0 else None))
    a.set_xscale("log"); a.set_xlabel("ξ (log scale)"); a.set_ylabel(ylab)
    a.grid(alpha=.25, lw=.6)
axf[5].axis("off")
h, l = axf[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=15,
           bbox_to_anchor=(0.5, 0.01))
fig.suptitle("Year-60 outcomes vs ξ by initialization (INTERNAL)", fontsize=15)
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fig.savefig(os.path.join(OUT, "init_outcomes_vs_xi.png"), dpi=150, bbox_inches="tight")
print("wrote init_outcomes_vs_xi.png")

# ---- Figure 2: full paths at xi=0.05, initializations overlaid --------------
PP = [("I_g", "green investment $I_g$", 1.0), ("I_d", "dirty investment $I_d$", 1.0),
      ("RD", "R&D as % of output", 100.0), ("E", "emissions (GtC)", 1.0)]
fig, axes = plt.subplots(1, 4, figsize=(19, 5))
for k, (name, ylab, sc) in enumerate(PP):
    a = axes[k]
    for lab, col, roots in INITS:
        # seed-average the path (align on the shorter length)
        curves = [path60(r, "0.050", name, sc)[1] for r in roots]
        n = min(len(c) for c in curves)
        t = path60(roots[0], "0.050", name, sc)[0][:n]
        y = np.mean([c[:n] for c in curves], axis=0)
        a.plot(t, y, color=col, lw=2.6, label=(lab if k == 0 else None))
    a.set_xlabel("year"); a.set_ylabel(ylab); a.set_xlim(0, 60); a.grid(alpha=.25, lw=.6)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("60-year paths at ξ = 0.05 by initialization (INTERNAL)", fontsize=15)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(os.path.join(OUT, "init_paths_xi0p05.png"), dpi=150, bbox_inches="tight")
print("wrote init_paths_xi0p05.png")

# ---- Figure 3: distorted jump densities by initialization, at two ξ ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for row, xi in enumerate(["0.050", "0.300"]):
    for col_i, (dens, title) in enumerate([("tech_jump_density", "technology"),
                                            ("dmg_jump_density", "damage")]):
        a = axes[row, col_i]
        for lab, col, roots in INITS:
            curves = [path60(r, xi, dens)[1] for r in roots]
            n = min(len(c) for c in curves)
            t = path60(roots[0], xi, dens)[0][:n]
            y = np.mean([c[:n] for c in curves], axis=0)
            a.plot(t, y, color=col, lw=2.6, label=(lab if (row == 0 and col_i == 0) else None))
        a.set_title(f"{title} first-jump density, ξ = {XIVAL[xi]}")
        a.set_xlabel("year"); a.set_ylabel("density"); a.set_xlim(0, 60); a.grid(alpha=.25, lw=.6)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Distorted first-jump densities by initialization (INTERNAL)", fontsize=15)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(os.path.join(OUT, "init_densities.png"), dpi=150, bbox_inches="tight")
print("wrote init_densities.png")

# ---- console table ----------------------------------------------------------
print("\nyear-60 I_g by initialization and ξ:")
print(f"{'init':>16} " + " ".join(f"{XIVAL[x]:>9}" for x in XIS))
for lab, _, roots in INITS:
    row = [avg_over_seeds(roots, lambda r: y60(r, xi, "I_g")) for xi in XIS]
    print(f"{lab:>16} " + " ".join(f"{v:9.2f}" for v in row))
