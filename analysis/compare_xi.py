"""
Magnitude of the uncertainty-aversion (xi) effect on the half-cost model, as a 2x2:
top row = investment rates i_d, i_g (xi cuts dirty more than green -> a green tilt);
bottom row = the two distorted jump probabilities (xi distorts beliefs strongly).

  python compare_xi.py RUN_FOLDER OUT.png
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

XIS = [("0.050", r"$\xi=0.05$", "#ff7f0e"),
       ("0.100", r"$\xi=0.1$",  "#1f77b4"),
       ("0.300", r"$\xi=0.3$",  "#2ca02c"),
       ("148.600", r"$\xi=\infty$", "#d62728")]
PANELS = [
    ("i_d",            r"Dirty investment rate $i_d$"),
    ("i_g",            r"Green investment rate $i_g$"),
    ("tech_jump_prob", r"Tech-jump cumulative prob."),
    ("dmg_jump_prob",  r"Damage-jump cumulative prob."),
]


def main():
    run, out = sys.argv[1], sys.argv[2]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, ylab) in zip(axes.ravel(), PANELS):
        for xi, lab, col in XIS:
            f = os.path.join(run, "SimulationDeterministic",
                             f"SimulationOutputs_ξ_{xi}", key + ".txt")
            y = np.loadtxt(f, ndmin=1)
            t = np.arange(len(y)) / 12.0
            ax.plot(t, y, color=col, lw=2.6, label=lab)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(r"Uncertainty aversion $\xi$ (half-adjustment-cost model)",
                 fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
