"""
Overlay deterministic-simulation trajectories from several trained runs.

Reads the per-xi .txt series the deterministic simulation writes under
  <run>/SimulationDeterministic/SimulationOutputs_<xi>/{DirtyInvestment,GreenInvestment,ConsumptionOutputRatio}.txt
and overlays them so two calibrations / two solution methods can be compared on
the same axes (I_d/Y, I_g/Y, C/Y as % of output, vs years).

Usage:
  python compare_deterministic_runs.py OUT.png XI  "label1=run_folder1" "label2=run_folder2" ...
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 14, "axes.titlesize": 16,
})

PANELS = [
    ("DirtyInvestment",        "Dirty investment  $I_d/Y$  (%)"),
    ("GreenInvestment",        "Green investment  $I_g/Y$  (%)"),
    ("ConsumptionOutputRatio", "Consumption  $C/Y$  (%)"),
]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
STYLES = ["-", "--", "-.", ":"]


def xi_dir(run, xi):
    base = os.path.join(run, "SimulationDeterministic")
    # match the SimulationOutputs_<xi> folder (xi printed with 3 decimals)
    want = f"SimulationOutputs_ξ_{float(xi):.3f}"
    cand = os.path.join(base, want)
    if os.path.isdir(cand):
        return cand
    # fall back: scan for closest xi
    for d in sorted(os.listdir(base)):
        if d.startswith("SimulationOutputs_ξ_"):
            try:
                v = float(d.rsplit("_", 1)[1])
            except ValueError:
                continue
            if abs(v - float(xi)) < 1e-6:
                return os.path.join(base, d)
    raise FileNotFoundError(f"no xi={xi} folder under {base}")


def load(run, xi, key):
    f = os.path.join(xi_dir(run, xi), key + ".txt")
    y = np.loadtxt(f, ndmin=1)
    if np.nanmax(np.abs(y)) < 1.5:
        y = 100.0 * y  # I_d/Y, I_g/Y, C/Y are stored as fractions -> percent
    return y


def main():
    out, xi = sys.argv[1], sys.argv[2]
    runs = [a.rsplit("=", 1) for a in sys.argv[3:]]  # label may contain '='; path has none

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for j, (key, ylab) in enumerate(PANELS):
        ax = axes[j]
        for i, (label, run) in enumerate(runs):
            y = load(run, xi, key)
            t = np.arange(len(y)) / 12.0  # 721 monthly steps -> 0..60 years
            ax.plot(t, y, STYLES[i % len(STYLES)], color=COLORS[i % len(COLORS)],
                    lw=2.2, label=label)
            if j == 0:
                print(f"  [{label}] {key}: start={y[0]:.3f} end={y[-1]:.3f}")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Deterministic paths  (ξ = {xi})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
