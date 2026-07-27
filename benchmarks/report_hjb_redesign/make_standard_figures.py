"""Mike's REQUIRED figure set for the one-jump HJB redesign arms, in Haoyang's formats.

Follows the house reporting standard (benchmarks/report_xiprofile/ templates):
  1. no-shock 60-year deterministic paths — green I_g, dirty I_d as LEVELS; R&D and C/Y as % of
     output; emissions; xi overlaid inside each panel
  2. distorted technology and damage FIRST-JUMP densities — xi overlaid
  3. distorted probability of the climate-sensitivity models — baseline (equal weight) vs worst case,
     recomputed from stored h_y and models/model144.csv
  4. distorted probability of the damage models over lambda3 — lambda3_grid + lambda3_weights_distorted

APPEND-A-ROW convention: one ROW per arm, quantities as columns, bold rotated row label on the left.
NO text baked into the figures (titles/readings belong in the note). Histograms are REPLOTTED FROM
DATA, never imshow'd from the pipeline PNGs (those carry embedded titles).

QUANTITY DICTIONARY (the recurring trap): green/dirty are I_g.txt / I_d.txt LEVELS (NOT
GreenInvestment/DirtyInvestment, which are I/Y ratios); R&D is RD.txt (= I_r/Output, NOT i_r.txt
= I_r/K).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
FULL = f"{ROOT}/output_redesign_full_20260726"
OUT = f"{ROOT}/benchmarks/report_hjb_redesign/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "lines.linewidth": 2.6, "savefig.facecolor": "white",
})

# one ROW per arm (the append-a-row convention); reference row first
ROWS = [
    ("reference (production loss)", f"{FULL}/A0_baseline_seed1"),
    ("non-dimensionalised HJB",     f"{FULL}/A1_nondim_seed1"),
    ("level anchor",                f"{FULL}/A2_anchor_seed1"),
    (r"$1/\xi$ pseudo-state",       f"{FULL}/A4_theta_seed1"),
]
ROWS = [(lab, r) for lab, r in ROWS if os.path.isdir(os.path.join(r, "SimulationDeterministic"))]
NROW = max(len(ROWS), 1)

XIS = ["0.050", "0.100", "0.300", "148.600"]
XILAB = {"0.050": r"$\xi=0.05$", "0.100": r"$\xi=0.1$",
         "0.300": r"$\xi=0.3$", "148.600": "uncertainty neutral"}
XICOL = {"0.050": "#d62728", "0.100": "#ff7f0e", "0.300": "#2ca02c", "148.600": "#1f77b4"}
XIS_HIST = [("0.300", r"$\xi=0.3$"), ("0.100", r"$\xi=0.1$"), ("0.050", r"$\xi=0.05$")]

THETA = None
_m144 = os.path.join(ROOT, "models", "model144.csv")
if os.path.exists(_m144):
    THETA = (pd.read_csv(_m144, header=None).to_numpy().ravel().astype(float))
VARSIGMA = 1.2 * 1.86 / 1000.0


def sdir(root, xi):
    return os.path.join(root, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}")


def load(root, xi, name):
    p = os.path.join(sdir(root, xi), f"{name}.txt")
    return np.loadtxt(p) if os.path.exists(p) else None


def rowlabel(ax, text):
    ax.annotate(text, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                fontweight="bold", fontsize=15, va="center", ha="center")


# ---------------------------------------------------------------- 1. no-shock paths
def fig_paths():
    cols = [("I_g", "green investment"), ("I_d", "dirty investment"),
            ("RD", "R&D (% of output)"), ("ConsumptionOutputRatio", "consumption (% of output)"),
            ("E", "emissions")]
    fig, axes = plt.subplots(NROW, len(cols), figsize=(5.0 * len(cols), 4.3 * NROW), squeeze=False)
    for ri, (lab, root) in enumerate(ROWS):
        for ci, (q, ylab) in enumerate(cols):
            ax = axes[ri][ci]
            for xi in XIS:
                t, y = load(root, xi, "t"), load(root, xi, q)
                if t is None or y is None:
                    continue
                n = min(len(t), len(y))
                v = y[:n] * (100.0 if q in ("RD", "ConsumptionOutputRatio") else 1.0)
                ax.plot(t[:n], v, color=XICOL[xi], label=XILAB[xi] if (ri == 0 and ci == 0) else None)
            ax.set_xlim(0, 60); ax.grid(alpha=.25, lw=.6); ax.set_ylabel(ylab)
            if ri == NROW - 1:
                ax.set_xlabel("year")
            if ci == 0:
                rowlabel(ax, lab)
    if NROW and len(ROWS):
        axes[0][0].legend(frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(f"{OUT}/paths_rows.png"); plt.close(fig)
    print("wrote paths_rows.png")


# ------------------------------------------------------- 2. first-jump densities
def fig_densities():
    DENS = [("tech_jump_density", "technology first-jump density"),
            ("dmg_jump_density", "damage first-jump density")]
    fig, axes = plt.subplots(NROW, 2, figsize=(14, 4.8 * NROW), squeeze=False, sharex=True)
    for ri, (lab, root) in enumerate(ROWS):
        for ci, (q, ylab) in enumerate(DENS):
            ax = axes[ri][ci]
            for xi in XIS:
                t, d = load(root, xi, "t"), load(root, xi, q)
                if t is None or d is None:
                    continue
                n = min(len(t), len(d))
                ax.plot(t[:n], d[:n], color=XICOL[xi],
                        label=XILAB[xi] if (ri == 0 and ci == 0) else None)
            ax.set_xlim(0, 60); ax.grid(alpha=.25, lw=.6); ax.set_ylabel(ylab)
            if ri == NROW - 1:
                ax.set_xlabel("year")
            if ci == 0:
                rowlabel(ax, lab)
    if len(ROWS):
        axes[0][0].legend(frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(f"{OUT}/jump_densities_rows.png"); plt.close(fig)
    print("wrote jump_densities_rows.png")


# --------------------------------------- 3. distorted climate-sensitivity model weights
def fig_climate():
    if THETA is None:
        print("skip climate: models/model144.csv not found"); return
    fig, axes = plt.subplots(NROW, 3, figsize=(16.5, 4.3 * NROW), squeeze=False,
                             sharex=True, sharey=True)
    n144 = len(THETA)
    base = np.ones(n144) / n144
    for ri, (lab, root) in enumerate(ROWS):
        for ci, (xi, xlab) in enumerate(XIS_HIST):
            ax = axes[ri][ci]
            hy = load(root, xi, "h_y")
            if hy is not None:
                hy = float(np.atleast_1d(hy)[-1])
                # worst-case tilt of the equal-weighted climate models under the drift distortion
                w = base * np.exp(THETA * hy / VARSIGMA)
                w = w / w.sum()
                ax.bar(THETA * 1000, base, width=0.02, color="#999999", alpha=.65)
                ax.bar(THETA * 1000, w, width=0.02, color=XICOL[xi], alpha=.75)
            ax.grid(alpha=.25, lw=.6)
            if ci == 0:
                ax.set_ylabel("probability"); rowlabel(ax, lab)
            if ri == NROW - 1:
                ax.set_xlabel("climate sensitivity")
    fig.tight_layout(); fig.savefig(f"{OUT}/climate_dist_rows.png"); plt.close(fig)
    print("wrote climate_dist_rows.png")


# ------------------------------------------------ 4. distorted damage-model weights over lambda3
def fig_lambda3():
    fig, axes = plt.subplots(NROW, 3, figsize=(16.5, 4.3 * NROW), squeeze=False,
                             sharex=True, sharey=True)
    for ri, (lab, root) in enumerate(ROWS):
        for ci, (xi, xlab) in enumerate(XIS_HIST):
            ax = axes[ri][ci]
            grid = load(root, xi, "lambda3_grid")
            wd = load(root, xi, "lambda3_weights_distorted")
            if grid is not None and wd is not None:
                grid = np.atleast_1d(grid); wd = np.atleast_1d(wd)
                n = min(len(grid), len(wd))
                base = np.ones(n) / n
                width = (grid[1] - grid[0]) * 0.4 if n > 1 else 0.02
                ax.bar(grid[:n] - width / 2, base, width=width, color="#999999", alpha=.65)
                ax.bar(grid[:n] + width / 2, wd[:n], width=width, color=XICOL[xi], alpha=.75)
            ax.grid(alpha=.25, lw=.6)
            if ci == 0:
                ax.set_ylabel("probability"); rowlabel(ax, lab)
            if ri == NROW - 1:
                ax.set_xlabel(r"$\lambda_3$ (damage curvature)")
    fig.tight_layout(); fig.savefig(f"{OUT}/lambda3_dist_rows.png"); plt.close(fig)
    print("wrote lambda3_dist_rows.png")


if __name__ == "__main__":
    if not ROWS:
        raise SystemExit(f"no simulated arms found under {FULL} — the 4-regime chains must finish first")
    print(f"rows: {[l for l, _ in ROWS]}")
    fig_paths(); fig_densities(); fig_climate(); fig_lambda3()
