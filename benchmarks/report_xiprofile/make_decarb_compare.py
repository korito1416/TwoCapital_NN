"""REPORT DISCIPLINE (2026-07-19): every decarbonization-successful solution is
compared on MIKE'S figure set, one ROW per solution, Haoyang formats:
  Fig 1  no-shock 60-year paths: I_g, I_d, R&D %, emissions, C/Y %  (xi overlaid)
  Fig 2  distorted tech & damage first-jump densities               (xi overlaid)
  Fig 3  distorted probability of the climate models  (baseline vs worst-case, xi=0.05)
  Fig 4  distorted probability of the damage models over lambda3    (xi=0.05)
(the four-channels chart is NOT part of Mike's set - internal diagnostic only.)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGD = os.path.join(HERE, "figures_internal")
VARSIGMA = 1.2 * 1.86 / 1000
THETA = (pd.read_csv(os.path.join(ROOT, "models", "model144.csv"), header=None)
         .to_numpy()[:, 0] / 1000.0).astype(np.float64)

ARMS = [  # every decarbonization-successful solution (E60 < 10 and falling)
    ("reference (inherited)",  f"{ROOT}/output_xiprofile_ref_20260714",                ["0.050","0.100","0.300","148.600"]),
    ("warm start B (perturb)", f"{ROOT}/output_warmstart_1M_20260714/perturb_s1",      ["0.050","0.100","0.300","148.600"]),
    ("FD anchor (base)",       f"{ROOT}/output_fdanchor_20260718/fdanchor_base",       ["0.050","0.100","0.300","148.600"]),
    ("FD anchor (delta 0.008)",f"{ROOT}/output_fdanchor_20260718/fdanchor_dlt0080",    ["0.050","0.100","0.300","148.600"]),
    ("FD anchor (delta 0.0125)",f"{ROOT}/output_fdanchor_20260718/fdanchor_dlt0125",   ["0.050","0.100","0.300","148.600"]),
    ("ABSORB (catch-up)",      f"{ROOT}/output_zoo_20260718/zoo_absorb",               ["0.050","0.100","0.300","148.600"]),
    ("RACE (patent race)",     f"{ROOT}/output_zoo_20260718/zoo_race",                 ["0.050","0.100","0.300","148.600"]),
    ("GHKM (knowledge-in-prod)",f"{ROOT}/output_zoo_20260718/zoo_ghkm",                ["0.050","0.100","0.300","148.600"]),
]
XICOL = {"0.050": "#D55E00", "0.100": "#E69F00", "0.300": "#0072B2", "148.600": "#000000"}
XILAB = {"0.050": "ξ = 0.05", "0.100": "ξ = 0.1", "0.300": "ξ = 0.3", "148.600": "ξ = 148.6 (neutral)"}

def load(root, xi, name):
    d = f"{root}/SimulationDeterministic/SimulationOutputs_ξ_{xi}"
    return np.loadtxt(os.path.join(d, f"{name}.txt"))

def p60(root, xi, name, sc=1.0):
    t = load(root, xi, "t"); v = load(root, xi, name)
    n = min(len(t), len(v)); m = t[:n] <= 60.0
    return t[:n][m], v[:n][m] * sc

def rowlabel(ax, text):
    ax.annotate(text, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90,
                va="center", ha="center", fontsize=11, fontweight="bold")

# ---- Fig 1: paths -----------------------------------------------------------
PANELS = [("I_g", "green $I_g$", 1.0), ("I_d", "dirty $I_d$", 1.0),
          ("RD", "R&D % output", 100.0), ("E", "emissions", 1.0),
          ("ConsumptionOutputRatio", "C/Y %", 100.0)]
plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(len(ARMS), 5, figsize=(17, 2.6 * len(ARMS)), sharex=True)
for ri, (lab, root, xis) in enumerate(ARMS):
    for ci, (nm, ql, sc) in enumerate(PANELS):
        a = axes[ri, ci]
        for xi in xis:
            try:
                t, v = p60(root, xi, nm, sc)
                a.plot(t, v, color=XICOL[xi], lw=2.0,
                       label=(XILAB[xi] if (ri == 0 and ci == 0) else None))
            except Exception: pass
        a.grid(alpha=.22, lw=.5); a.set_xlim(0, 60)
        if ri == 0: a.set_title(ql, fontsize=12)
        if ri == len(ARMS) - 1: a.set_xlabel("year")
    rowlabel(axes[ri, 0], lab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.004))
fig.tight_layout(rect=(0.035, 0.015, 1, 1))
fig.savefig(os.path.join(FIGD, "decarb_paths.png"), dpi=140, bbox_inches="tight"); print("fig1")

# ---- Fig 2: first-jump densities -------------------------------------------
fig, axes = plt.subplots(len(ARMS), 2, figsize=(12, 2.6 * len(ARMS)), sharex=True)
for ri, (lab, root, xis) in enumerate(ARMS):
    for ci, (nm, title) in enumerate([("tech_jump_density", "technology first-jump density"),
                                       ("dmg_jump_density", "damage first-jump density")]):
        a = axes[ri, ci]
        for xi in xis:
            try:
                t, v = p60(root, xi, nm)
                a.plot(t, v, color=XICOL[xi], lw=2.0,
                       label=(XILAB[xi] if (ri == 0 and ci == 0) else None))
            except Exception: pass
        a.grid(alpha=.22, lw=.5); a.set_xlim(0, 60)
        if ri == 0: a.set_title(title, fontsize=12)
        if ri == len(ARMS) - 1: a.set_xlabel("year")
    rowlabel(axes[ri, 0], lab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.004))
fig.tight_layout(rect=(0.045, 0.015, 1, 1))
fig.savefig(os.path.join(FIGD, "decarb_densities.png"), dpi=140, bbox_inches="tight"); print("fig2")

# ---- Fig 3: climate-model histograms (xi=0.05), 2-col panel grid ------------
w_unif = np.ones_like(THETA) / len(THETA)
cbins = np.linspace(0.8, 3.0, 16)
nr = (len(ARMS) + 1) // 2
fig, axes = plt.subplots(nr, 2, figsize=(13, 3.0 * nr), sharex=True, sharey=True)
axf = axes.ravel()
for k, (lab, root, xis) in enumerate(ARMS):
    a = axf[k]
    hy = load(root, "0.050", "h_y")[-1]
    a.hist(1000*THETA, weights=w_unif, bins=cbins, density=True, color="C3", alpha=0.5,
           ec="darkgrey", label=("baseline" if k == 0 else None))
    a.hist(1000*(THETA + VARSIGMA*hy), weights=w_unif, bins=cbins, density=True, color="C0",
           alpha=0.5, ec="darkgrey", label=("distorted (ξ=0.05)" if k == 0 else None))
    a.set_title(lab, fontsize=11.5); a.set_xlim(0.8, 3.0)
for k in range(len(ARMS), len(axf)): axf[k].axis("off")
for a in axes[-1]: a.set_xlabel("climate sensitivity")
h, l = axf[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.006))
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(os.path.join(FIGD, "decarb_climate.png"), dpi=140, bbox_inches="tight"); print("fig3")

# ---- Fig 4: lambda3 histograms (xi=0.05) ------------------------------------
fig, axes = plt.subplots(nr, 2, figsize=(13, 3.0 * nr), sharex=True, sharey=True)
axf = axes.ravel()
for k, (lab, root, xis) in enumerate(ARMS):
    a = axf[k]
    grid = load(root, "0.050", "lambda3_grid")
    wd = load(root, "0.050", "lambda3_weights_distorted")
    L = len(grid); edges = np.linspace(float(grid.min()), float(grid.max()), L + 1)
    a.hist(grid, weights=np.ones(L)/L, bins=edges, color="C3", alpha=0.5, ec="darkgrey",
           label=("baseline" if k == 0 else None))
    a.hist(grid, weights=wd, bins=edges, color="C0", alpha=0.5, ec="darkgrey",
           label=("distorted (ξ=0.05)" if k == 0 else None))
    mono = "ok" if np.all(np.diff(wd) > 0) else ("WRONG DIRECTION" if np.all(np.diff(wd) < 0) else "non-monotone")
    col = {"ok": "0.25", "WRONG DIRECTION": "#B00020", "non-monotone": "#B06000"}[mono]
    a.set_title(f"{lab}  [{mono}]", fontsize=11.5, color=col)
for k in range(len(ARMS), len(axf)): axf[k].axis("off")
for a in axes[-1]: a.set_xlabel(r"$\lambda_3$ (damage curvature)")
h, l = axf[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.006))
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(os.path.join(FIGD, "decarb_lambda3.png"), dpi=140, bbox_inches="tight"); print("fig4")
