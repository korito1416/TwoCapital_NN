"""Headline figure: the delivered-solution map. Every run in the study as one point
in (year-60 green investment, year-60 dirty investment) space; color = warm-start
family (continuing the report's fixed method colors), marker = training protocol.
Clusters direct-labeled; degenerate family shaded. One glance = the whole finding:
low-loss solutions form a manifold, and the warm start + schedule pick the point.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

def yr60(run, q, xi="0.050"):
    p = f"{ROOT}/{run}/SimulationDeterministic/SimulationOutputs_ξ_{xi}/{q}.txt"
    try:
        return float(np.loadtxt(p)[-1])
    except Exception:
        return np.nan

W1M, WLX = "output_warmstart_1M_20260714", "output_warmstart_1M_xi0p005_20260714"
WI2, WEX = "output_warmstart_init2_20260714", "output_warmstart_ext_20260714"
W300 = "output_warmstart_20260714"
RUNA = ("output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_"
        "10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000")

# (family, marker, list of runs) — color follows the warm-start family everywhere
GROUPS = [
    ("reference",              "0.15",    "*", [RUNA]),
    ("warm start (variants)",  "#1f4e79", "o", [f"{W1M}/nber_s1", f"{W1M}/nber_s2",
                                                f"{WEX}/nber_s3", f"{WEX}/nber_s4",
                                                f"{WLX}/nber_s1", f"{WLX}/nber_s2",
                                                f"{W1M}/perturb_s1", f"{W1M}/perturb_s2",
                                                f"{WEX}/perturb_s3", f"{WEX}/perturb_s4",
                                                f"{WEX}/nberperturb_s1", f"{WEX}/nberperturb_s2",
                                                f"{WEX}/noreheat_s1", f"{WEX}/noreheat_s2"]),
    ("cold start",             "#00838f", "D", [f"{WEX}/paperlr2M_s{i}" for i in (1, 2, 3)]),
]

plt.rcParams.update({"font.size": 15, "axes.linewidth": 1.0})
fig, ax = plt.subplots(figsize=(11.5, 8))
for label, col, mk, runs in GROUPS:
    xs = [yr60(r, "I_g") for r in runs]
    ys = [yr60(r, "I_d") for r in runs]
    xs, ys = zip(*[(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)])
    big = 340 if mk == "*" else 130
    ax.scatter(xs, ys, s=big, marker=mk, c=col, label=label,
               edgecolors="white", linewidths=1.2, zorder=3)

ax.set_xlabel("green investment $I_g$ at year 60 (ξ = 0.05)")
ax.set_ylabel("dirty investment $I_d$ at year 60")
ax.set_xlim(55, 205); ax.set_ylim(0, 62)
ax.grid(alpha=0.22, lw=0.6)

ann = [(67.7, 4.8, "reference and its\ncontinuations", (60, 13.5), "0.15"),
       (86, 9.2, "warm starts:\nperturbations of the reference", (78, 1.5), "#1f4e79"),
       (131.8, 19.0, "warm starts:\nearlier model vintage", (118, 27.5), "#1f4e79"),
       (176, 43, "cold starts\n(constant learning rate, 2M)", (150, 54.5), "#00838f")]
for x, y, txt, (tx, ty), col in ann:
    ax.annotate(txt, xy=(x, y), xytext=(tx, ty), fontsize=12.5, color=col,
                ha="left", arrowprops=dict(arrowstyle="-", color=col, lw=0.9, alpha=0.6))

ax.legend(frameon=False, fontsize=12.5, loc="upper left")
fig.tight_layout()
out = os.path.join(HERE, "figures", "solution_map_xi0p05.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
