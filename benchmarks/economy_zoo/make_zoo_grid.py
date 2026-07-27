"""THE ZOO GRID (Mike format): one ROW per warm-start map, columns = delivered
quantities over the 60-year no-shock path, xi overlaid where simulated.
Rows: reference (inherited) / FD anchor / ABSORB / RACE / GHKM / dirty-attractor ref.
Appended below the existing figures per the delivery convention.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

ROWS = [
    ("reference (inherited)", os.path.join(ROOT, "output_xiprofile_ref_20260714"),
     ["0.050", "0.100", "0.300", "148.600"]),
    ("FD anchor (production model)", os.path.join(ROOT, "output_fdanchor_20260718", "fdanchor_base"),
     ["0.050", "148.600"]),
    ("ABSORB (catch-up absorption)", os.path.join(ROOT, "output_zoo_20260718", "zoo_absorb"),
     ["0.050", "148.600"]),
    ("RACE (patent race)", os.path.join(ROOT, "output_zoo_20260718", "zoo_race"),
     ["0.050", "148.600"]),
    ("GHKM (knowledge-in-production)", os.path.join(ROOT, "output_zoo_20260718", "zoo_ghkm"),
     ["0.050", "148.600"]),
    ("refit map (dirty attractor)", os.path.join(ROOT, "output_levelshift_gentle_20260718", "lvldng_s1"),
     ["0.050", "148.600"]),
]
XICOL = {"0.050": "#D55E00", "0.100": "#E69F00", "0.300": "#0072B2", "148.600": "#000000"}
XILAB = {"0.050": "ξ = 0.05", "0.100": "ξ = 0.1", "0.300": "ξ = 0.3", "148.600": "ξ = 148.6 (neutral)"}
PANELS = [("I_g", "green $I_g$", 1.0), ("I_d", "dirty $I_d$", 1.0),
          ("E", "emissions", 1.0), ("ConsumptionOutputRatio", "C/Y %", 100.0)]

def path60(root, xi, name, scale=1.0):
    d = os.path.join(root, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}")
    t = np.loadtxt(os.path.join(d, "t.txt")); v = np.loadtxt(os.path.join(d, f"{name}.txt"))
    n = min(len(t), len(v)); m = t[:n] <= 60.0
    return t[:n][m], v[:n][m] * scale

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(len(ROWS), len(PANELS), figsize=(17, 3.1 * len(ROWS)), sharex=True)
for ri, (rlab, root, xis) in enumerate(ROWS):
    for ci, (name, ql, sc) in enumerate(PANELS):
        a = axes[ri, ci]
        for xi in xis:
            try:
                t, v = path60(root, xi, name, sc)
                a.plot(t, v, color=XICOL[xi], lw=2.2,
                       label=(XILAB[xi] if (ri == 0 and ci == 0) else None))
            except Exception:
                pass
        a.grid(alpha=.22, lw=.6); a.set_xlim(0, 60)
        if ri == 0: a.set_title(ql, fontsize=13)
        if ri == len(ROWS) - 1: a.set_xlabel("year")
    axes[ri, 0].annotate(rlab, xy=(-0.34, 0.5), xycoords="axes fraction",
                         rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=13,
           bbox_to_anchor=(0.5, -0.006))
fig.tight_layout(rect=(0.035, 0.02, 1, 1))
out = os.path.join(HERE, "figures", "zoo_grid.png")
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
