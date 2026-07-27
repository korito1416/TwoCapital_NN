"""INTERNAL — the four 07-20 report figures, each gaining one ROW per initialization
(row 1 reference, row 2 warm start A, row 3 warm start B), so each initialization's
own ξ-profile 'report' is visible and comparable down the columns. Same formats,
normalizations, and ξ colors as the 07-20 note. All data read from existing
SimulationDeterministic outputs; climate distortion recomputed from stored h_y (shift
= ϛ·h_y(T)), exactly as the pipeline does it.
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
VARSIGMA = 1.2 * 1.86 / 1000
import pandas as pd
THETA_ELL = (pd.read_csv(os.path.join(ROOT, "models", "model144.csv"),
                         header=None).to_numpy()[:, 0] / 1000.0).astype(np.float64)

INITS = [  # (label, [seed roots])
    ("reference",    [REF]),
    ("warm start A", [f"{P1M}/nber_s1", f"{P1M}/nber_s2"]),
    ("warm start B", [f"{P1M}/perturb_s1", f"{P1M}/perturb_s2"]),
]
XIS = [("0.050", "ξ = 0.05", "#D55E00"), ("0.100", "ξ = 0.1", "#E69F00"),
       ("0.300", "ξ = 0.3", "#0072B2"), ("148.600", "ξ = 148.6 (neutral)", "#000000")]
XIS_DIST = [("0.300", "ξ = 0.3"), ("0.100", "ξ = 0.1"), ("0.050", "ξ = 0.05")]

def sim(root, xi):
    return os.path.join(root, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}")

def load(root, xi, name):
    return np.loadtxt(os.path.join(sim(root, xi), f"{name}.txt"))

def path60(roots, xi, name, scale=1.0):
    curves, t0 = [], None
    for r in roots:
        t = load(r, xi, "t"); v = load(r, xi, name) * scale
        n = min(len(t), len(v)); m = t[:n] <= 60.0
        curves.append(v[:n][m]); t0 = t[:n][m]
    n = min(len(c) for c in curves)
    return t0[:n], np.mean([c[:n] for c in curves], axis=0)

def rowlabel(fig, ax, text):
    ax.annotate(text, xy=(-0.32, 0.5), xycoords="axes fraction",
                rotation=90, va="center", ha="center", fontsize=15, fontweight="bold")

# ---- Figure 1: paths, 3 inits (rows) x 5 quantities (cols), ξ overlaid --------
PANELS = [("I_g", "green $I_g$", 1.0), ("I_d", "dirty $I_d$", 1.0),
          ("RD", "R&D % output", 100.0), ("E", "emissions", 1.0),
          ("ConsumptionOutputRatio", "C/Y %", 100.0)]
plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(3, 5, figsize=(19, 10), sharex=True)
for ri, (ilab, roots) in enumerate(INITS):
    for ci, (name, ql, sc) in enumerate(PANELS):
        a = axes[ri, ci]
        for xi, xlab, col in XIS:
            t, v = path60(roots, xi, name, sc)
            a.plot(t, v, color=col, lw=2.2, label=(xlab if (ri == 0 and ci == 0) else None))
        a.grid(alpha=.22, lw=.6); a.set_xlim(0, 60)
        if ri == 0: a.set_title(ql, fontsize=13)
        if ri == 2: a.set_xlabel("year")
    rowlabel(fig, axes[ri, 0], ilab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0.03, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "grid1_paths.png"), dpi=140, bbox_inches="tight"); print("grid1_paths")

# ---- Figure 2: jump densities, 3 inits x 2 jumps, ξ overlaid ------------------
DENS = [("tech_jump_density", "technology first-jump density"),
        ("dmg_jump_density", "damage first-jump density")]
fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
for ri, (ilab, roots) in enumerate(INITS):
    for ci, (name, title) in enumerate(DENS):
        a = axes[ri, ci]
        for xi, xlab, col in XIS:
            t, v = path60(roots, xi, name)
            a.plot(t, v, color=col, lw=2.2, label=(xlab if (ri == 0 and ci == 0) else None))
        a.grid(alpha=.22, lw=.6); a.set_xlim(0, 60)
        if ri == 0: a.set_title(title, fontsize=13)
        if ri == 2: a.set_xlabel("year")
        if ci == 0: a.set_ylabel("density")
    rowlabel(fig, axes[ri, 0], ilab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0.04, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "grid2_densities.png"), dpi=140, bbox_inches="tight"); print("grid2_densities")

# ---- Figure 3: climate-sensitivity distortion, 3 inits x 3 ξ ------------------
theta_base = 1000.0 * THETA_ELL
w_unif = np.ones_like(THETA_ELL) / len(THETA_ELL)
cbins = np.linspace(0.8, 3.0, 16)
fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey=True)
for ri, (ilab, roots) in enumerate(INITS):
    for ci, (xi, xlab) in enumerate(XIS_DIST):
        a = axes[ri, ci]
        hy_last = np.mean([load(r, xi, "h_y")[-1] for r in roots])
        theta_dist = 1000.0 * (THETA_ELL + VARSIGMA * hy_last)
        a.hist(theta_base, weights=w_unif, bins=cbins, density=True, color="C3",
               alpha=0.5, ec="darkgrey", label=("baseline" if (ri == 0 and ci == 0) else None))
        a.hist(theta_dist, weights=w_unif, bins=cbins, density=True, color="C0",
               alpha=0.5, ec="darkgrey", label=("distorted" if (ri == 0 and ci == 0) else None))
        a.set_xlim(0.8, 3.0)
        if ri == 0: a.set_title(xlab, fontsize=13)
        if ri == 2: a.set_xlabel("climate sensitivity")
    rowlabel(fig, axes[ri, 0], ilab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0.04, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "grid3_climate.png"), dpi=140, bbox_inches="tight"); print("grid3_climate")

# ---- Figure 4: λ3 damage-curvature distortion, 3 inits x 3 ξ ------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey=True)
for ri, (ilab, roots) in enumerate(INITS):
    for ci, (xi, xlab) in enumerate(XIS_DIST):
        a = axes[ri, ci]
        grid = load(roots[0], xi, "lambda3_grid")
        w_dist = np.mean([load(r, xi, "lambda3_weights_distorted") for r in roots], axis=0)
        L = len(grid); base = np.ones(L) / L
        edges = np.linspace(float(grid.min()), float(grid.max()), L + 1)
        a.hist(grid, weights=base, bins=edges, color="C3", alpha=0.5, ec="darkgrey",
               label=("baseline" if (ri == 0 and ci == 0) else None))
        a.hist(grid, weights=w_dist, bins=edges, color="C0", alpha=0.5, ec="darkgrey",
               label=("distorted" if (ri == 0 and ci == 0) else None))
        if ri == 0: a.set_title(xlab, fontsize=13)
        if ri == 2: a.set_xlabel(r"$\lambda_3$")
    rowlabel(fig, axes[ri, 0], ilab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0.04, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "grid4_lambda3.png"), dpi=140, bbox_inches="tight"); print("grid4_lambda3")
