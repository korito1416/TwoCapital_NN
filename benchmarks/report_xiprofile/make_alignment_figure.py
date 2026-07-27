"""Alignment figure for the 07-20 report: Haoyang's stored results (row 1) vs this
note's re-simulation (row 2), same quantities, same xi overlay — the visual form of
the bit-identical alignment table (max |diff| = 3.8e-6 across everything)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ROWS = [
    ("reference — Haoyang (stored)", os.path.join(ROOT, "output_001",
        "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_"
        "#HiddenLayer_4_num_iterations1000000")),
    ("reference — re-simulation", os.path.join(ROOT, "output_xiprofile_ref_20260714")),
    ("warm start B", os.path.join(ROOT, "output_warmstart_1M_20260714", "perturb_s1")),
]
XIS = ["0.050", "0.100", "0.300", "148.600"]
XICOL = {"0.050": "#D55E00", "0.100": "#E69F00", "0.300": "#0072B2", "148.600": "#000000"}
XILAB = {"0.050": "ξ = 0.05", "0.100": "ξ = 0.1", "0.300": "ξ = 0.3", "148.600": "ξ = 148.6 (neutral)"}
PANELS = [("I_g", "green $I_g$", 1.0), ("I_d", "dirty $I_d$", 1.0),
          ("RD", "R&D % output", 100.0), ("E", "emissions", 1.0),
          ("ConsumptionOutputRatio", "C/Y %", 100.0)]

def p60(root, xi, name, sc=1.0):
    d = f"{root}/SimulationDeterministic/SimulationOutputs_ξ_{xi}"
    t = np.loadtxt(os.path.join(d, "t.txt")); v = np.loadtxt(os.path.join(d, f"{name}.txt"))
    n = min(len(t), len(v)); m = t[:n] <= 60.0
    return t[:n][m], v[:n][m] * sc

plt.rcParams.update({"font.size": 12})
nrow = len(ROWS)
fig, axes = plt.subplots(nrow, 5, figsize=(17, 3.0 * nrow), sharex=True)
for ri, (lab, root) in enumerate(ROWS):
    for ci, (nm, ql, sc) in enumerate(PANELS):
        a = axes[ri, ci]
        for xi in XIS:
            t, v = p60(root, xi, nm, sc)
            a.plot(t, v, color=XICOL[xi], lw=2.0,
                   label=(XILAB[xi] if (ri == 0 and ci == 0) else None))
        a.grid(alpha=.22, lw=.5); a.set_xlim(0, 60)
        if ri == 0: a.set_title(ql, fontsize=13)
        if ri == nrow - 1: a.set_xlabel("year")
    axes[ri, 0].annotate(lab, xy=(-0.40, 0.5), xycoords="axes fraction", rotation=90,
                         va="center", ha="center", fontsize=12, fontweight="bold")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=12.5,
           bbox_to_anchor=(0.5, -0.015))
fig.tight_layout(rect=(0.035, 0.03, 1, 1))
out = os.path.join(HERE, "figures", "alignment_paths.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
