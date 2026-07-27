"""Two-row (Haoyang stored results / this note's re-simulation) versions of the
belief figures for the 07-20 report — the append-a-row convention: top row =
Haoyang's stored results, bottom row = our re-simulation, same quantities, same
xi. The rows are visually indistinguishable at every xi (max |diff| = 3.8e-6),
so every figure is itself a replication check.

Overwrites, in figures/:
  jump_densities.png    2 rows x 2 cols (tech, damage first-jump density), xi overlaid
  climate_dist_row.png  2 rows x 3 cols (xi=0.3,0.1,0.05), baseline vs worst-case climate models
  dmg_dist_row.png      2 rows x 3 cols (xi=0.3,0.1,0.05), baseline vs worst-case damage models over lambda3
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "figures")
VARSIGMA = 1.2 * 1.86 / 1000
THETA = (pd.read_csv(os.path.join(ROOT, "models", "model144.csv"), header=None)
         .to_numpy()[:, 0] / 1000.0).astype(np.float64)

ROWS = [
    ("reference — Haoyang (stored)", os.path.join(ROOT, "output_001",
        "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_"
        "#HiddenLayer_4_num_iterations1000000")),
    ("reference — re-simulation", os.path.join(ROOT, "output_xiprofile_ref_20260714")),
    ("warm start B", os.path.join(ROOT, "output_warmstart_1M_20260714", "perturb_s1")),
]
NROW = len(ROWS)
XIS = ["0.050", "0.100", "0.300", "148.600"]
XICOL = {"0.050": "#D55E00", "0.100": "#E69F00", "0.300": "#0072B2", "148.600": "#000000"}
XILAB = {"0.050": "ξ = 0.05", "0.100": "ξ = 0.1", "0.300": "ξ = 0.3", "148.600": "ξ = 148.6 (neutral)"}
# the belief histograms omit the neutral case (no distortion)
XIS_HIST = [("0.300", "ξ = 0.3"), ("0.100", "ξ = 0.1"), ("0.050", "ξ = 0.05")]


def load(root, xi, name):
    d = os.path.join(root, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}")
    return np.loadtxt(os.path.join(d, f"{name}.txt"))


def p60(root, xi, name):
    t = load(root, xi, "t"); v = load(root, xi, name)
    n = min(len(t), len(v)); m = t[:n] <= 60.0
    return t[:n][m], v[:n][m]


def rowlabel(ax, text):
    ax.annotate(text, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                va="center", ha="center", fontsize=13, fontweight="bold")


# ---- Figure: first-jump densities, 2 rows x 2 cols --------------------------
plt.rcParams.update({"font.size": 13})
DENS = [("tech_jump_density", "technology first-jump density"),
        ("dmg_jump_density",  "damage first-jump density")]
fig, axes = plt.subplots(NROW, 2, figsize=(14, 4.8 * NROW), sharex=True)
for ri, (rlab, root) in enumerate(ROWS):
    for ci, (name, title) in enumerate(DENS):
        a = axes[ri, ci]
        for xi in XIS:
            t, v = p60(root, xi, name)
            a.plot(t, v, color=XICOL[xi], lw=2.6,
                   label=(XILAB[xi] if (ri == 0 and ci == 0) else None))
        a.set_xlim(0, 60); a.grid(alpha=.25, lw=.6); a.set_ylabel("density")
        if ri == 0: a.set_title(title, fontsize=14)
        if ri == NROW - 1: a.set_xlabel("year")
    rowlabel(axes[ri, 0], rlab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0.03, 0.05, 1, 1))
out = os.path.join(FIG, "jump_densities.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


# ---- Figure: climate-sensitivity models, 2 rows x 3 cols --------------------
w_unif = np.ones_like(THETA) / len(THETA)
cbins = np.linspace(0.8, 3.0, 16)
fig, axes = plt.subplots(NROW, 3, figsize=(16.5, 4.3 * NROW), sharex=True, sharey=True)
for ri, (rlab, root) in enumerate(ROWS):
    for ci, (xi, xlab) in enumerate(XIS_HIST):
        a = axes[ri, ci]
        hy = load(root, xi, "h_y")[-1]
        a.hist(1000 * THETA, weights=w_unif, bins=cbins, density=True, color="C3",
               alpha=0.5, ec="darkgrey", label=("baseline" if (ri == 0 and ci == 0) else None))
        a.hist(1000 * (THETA + VARSIGMA * hy), weights=w_unif, bins=cbins, density=True,
               color="C0", alpha=0.5, ec="darkgrey",
               label=("worst case" if (ri == 0 and ci == 0) else None))
        a.set_xlim(0.8, 3.0)
        if ri == 0: a.set_title(xlab, fontsize=14)
        if ri == NROW - 1: a.set_xlabel("climate sensitivity")
    rowlabel(axes[ri, 0], rlab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0.03, 0.05, 1, 1))
out = os.path.join(FIG, "climate_dist_row.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


# ---- Figure: damage models over lambda3, 2 rows x 3 cols --------------------
fig, axes = plt.subplots(NROW, 3, figsize=(16.5, 4.3 * NROW), sharex=True, sharey=True)
for ri, (rlab, root) in enumerate(ROWS):
    for ci, (xi, xlab) in enumerate(XIS_HIST):
        a = axes[ri, ci]
        grid = load(root, xi, "lambda3_grid")
        wd = load(root, xi, "lambda3_weights_distorted")
        L = len(grid); edges = np.linspace(float(grid.min()), float(grid.max()), L + 1)
        a.hist(grid, weights=np.ones(L) / L, bins=edges, color="C3", alpha=0.5,
               ec="darkgrey", label=("baseline" if (ri == 0 and ci == 0) else None))
        a.hist(grid, weights=wd, bins=edges, color="C0", alpha=0.5, ec="darkgrey",
               label=("worst case" if (ri == 0 and ci == 0) else None))
        if ri == 0: a.set_title(xlab, fontsize=14)
        if ri == NROW - 1: a.set_xlabel(r"$\lambda_3$ (damage curvature)")
    rowlabel(axes[ri, 0], rlab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0.03, 0.05, 1, 1))
out = os.path.join(FIG, "dmg_dist_row.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
