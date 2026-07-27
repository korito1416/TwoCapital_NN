"""
Section-3 "consequences" comparison: do the investment shifts from halving the
adjustment cost translate into faster decarbonization (emissions) and an earlier
green-capital transition (share Z)? Overlays baseline vs half along the xi path.

  python compare_consequences.py OUT.png XI "label1=run1" "label2=run2"
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

# (key, y-label, multiply-by-100?)
PANELS = [
    ("E", r"Emissions  $E$", False),
    ("Z", r"Green capital share  $Z$  (%)", True),
]
COLORS = ["#1f77b4", "#d62728"]
STYLES = ["-", "--"]


def xi_dir(run, xi):
    return os.path.join(run, "SimulationDeterministic",
                        f"SimulationOutputs_ξ_{float(xi):.3f}")


def load(run, xi, key, pct):
    y = np.loadtxt(os.path.join(xi_dir(run, xi), key + ".txt"), ndmin=1)
    return 100.0 * y if pct else y


def main():
    out, xi = sys.argv[1], sys.argv[2]
    runs = [a.rsplit("=", 1) for a in sys.argv[3:]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for j, (key, ylab, pct) in enumerate(PANELS):
        ax = axes[j]
        for i, (label, run) in enumerate(runs):
            y = load(run, xi, key, pct)
            t = np.arange(len(y)) / 12.0
            ax.plot(t, y, STYLES[i % len(STYLES)], color=COLORS[i % len(COLORS)],
                    lw=2.3, label=label)
            print(f"  [{label}] {key}: start={y[0]:.3f}  end={y[-1]:.3f}")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Consequences of halving the adjustment cost  (ξ = {xi})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
