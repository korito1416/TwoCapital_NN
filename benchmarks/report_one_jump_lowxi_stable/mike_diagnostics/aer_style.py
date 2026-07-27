"""Clean, LARGE, project-coloured figure style. Text lives in the caption, NOT on the figure.
Matches the repo's own plotting convention (baseline = C3 red, distorted = C0 blue, alpha 0.5,
edgecolor darkgrey; matplotlib default sans, default spines). `import aer_style; aer_style.apply()`."""
import numpy as np
import matplotlib.pyplot as plt

# project convention (models/SimulationDeterministic.py): baseline red, distorted blue
BASELINE = "C3"
DISTORTED = "C0"
# kept for backward-compat with existing scripts; mapped to the project palette
NAVY, MAROON, SLATE, GOLD, GREEN = "C0", "C3", "0.4", "C1", "C2"
FORBIDDEN = "0.90"
PALETTE = ["C0", "C3", "C2", "C1", "C4"]


def apply():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.6,
        "lines.markersize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def xi_colors(n):
    """Ordered xi curves: light (neutral) -> dark blue (very averse), matching 'blue = more averse'."""
    return plt.cm.viridis(np.linspace(0.08, 0.9, n))
