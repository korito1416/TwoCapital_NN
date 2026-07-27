"""Two mechanism figures for the final report, identification-report style.

Fig A (basins): dumbbell/arrow chart -- each retraining experiment as an arrow from
its warm-start base's year-60 green investment to the delivered one, grouped by
training protocol. Shows: annealed and non-reheated runs stay at their base; the
reheated full cycle moves BOTH bases toward an intermediate band.

Fig B (xi): gap(xi) profile -- year-60 green investment vs xi for the three
converged families; the nber/reference ratio is flat (1.92-1.96), so the selection
is not amplified by uncertainty aversion.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

def yr60(run, q="I_g", xi="0.050"):
    p = f"{ROOT}/{run}/SimulationDeterministic/SimulationOutputs_ξ_{xi}/{q}.txt"
    try:
        return float(np.loadtxt(p)[-1])
    except Exception:
        return np.nan

plt.rcParams.update({"font.size": 15, "axes.linewidth": 1.0})

# ---------------- Fig A: basins ----------------
REF, NBER = 67.7, 133.1
rows = [
    ("perturbed reference,\n300k anneal",        REF,  [70.7, 65.6],               "#2e7d32"),
    ("perturbed reference,\nreheated 1M cycle",  REF,  [86.4, 77.3, 95.2, 88.2],   "#2e7d32"),
    ("continuation, no reheat\n(+700k constant 1e-5)", REF, [68.7, 63.7],          "#2e7d32"),
    ("perturbed warm-start-A solution,\nreheated 1M cycle", NBER, [115.0, 107.8],  "#1f4e79"),
]
fig, ax = plt.subplots(figsize=(11.5, 5.2))
for i, (lab, base, ends, col) in enumerate(rows):
    y = len(rows) - 1 - i
    for e in ends:
        ax.annotate("", xy=(e, y), xytext=(base, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, alpha=0.85))
        ax.plot([e], [y], "o", color=col, ms=8, zorder=3)
    ax.plot([base], [y], "s", color="0.15", ms=9, zorder=4)
    ax.text(28, y, lab, ha="right", va="center", fontsize=12.5)
ax.axvline(REF, color="0.15", ls=":", lw=2.2)
ax.axvline(NBER, color="#1f4e79", ls=":", lw=2.2)
ax.text(REF, len(rows) - 0.45, "reference", ha="center", fontsize=12, color="0.15")
ax.text(NBER, len(rows) - 0.45, "warm start A", ha="center", fontsize=12, color="#1f4e79")
ax.set_xlim(24, 142); ax.set_ylim(-0.6, len(rows) - 0.1)
ax.set_yticks([])
ax.set_xlabel("green investment $I_g$ at year 60 (ξ = 0.05)")
ax.grid(axis="x", alpha=0.22, lw=0.6)
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "basins_reheat.png"), dpi=160, bbox_inches="tight")
print("wrote basins_reheat.png")

# ---------------- Fig B: gap(xi) ----------------
XIS = ["0.050", "0.100", "0.300", "1.000", "3.000", "10.000", "35.000", "148.600"]
xv = [float(x) for x in XIS]
runs = {"reference": ("output_xiprofile_ref_20260714", "0.15"),
        "warm start A": ("output_warmstart_1M_20260714/nber_s1", "#1f4e79"),
        "warm start B": ("output_warmstart_1M_20260714/perturb_s1", "#2e7d32")}
fig, ax = plt.subplots(figsize=(9.5, 5.6))
series = {}
for lab, (r, col) in runs.items():
    ys = [yr60(r, "I_g", x) for x in XIS]
    series[lab] = np.array(ys)
    ax.plot(xv, ys, "-o", color=col, lw=2.2, ms=6, label=lab)
ax.set_xscale("log")
ax.set_xlabel(r"uncertainty aversion $\xi$ (log scale)")
ax.set_ylabel("green investment $I_g$ at year 60")
ratio = series["warm start A"] / series["reference"]
ax.text(0.35, 104, f"warm-A / reference ratio: {ratio.min():.2f}–{ratio.max():.2f}\nacross the whole ξ range",
        fontsize=12.5, color="0.25")
ax.legend(frameon=False, fontsize=13, loc="center right")
ax.grid(alpha=0.22, lw=0.6, which="both")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "gap_vs_xi.png"), dpi=160, bbox_inches="tight")
print("wrote gap_vs_xi.png")
