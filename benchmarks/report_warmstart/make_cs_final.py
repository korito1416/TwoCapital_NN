"""CS1/CS2 figures in the identification-report style, for the warm-start families:
per jump state, error in the HJB equation (CS1) and the three investment-optimality
errors (CS2) across xi, re-evaluated along the simulated 60-year path (RMS over
years). Same color = same warm-start family; solid/dashed = seeds.
Input: data_1M/cs_final.npy from solution_comparison/eval_path.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "data_1M", "cs_final.npy"), allow_pickle=True).item()
XIS = np.array(D["XIS"], dtype=float); data = D["data"]
D2 = np.load(os.path.join(HERE, "data_lowxi", "cs_warmstart_lowxi.npy"), allow_pickle=True).item()
XIS2 = np.array(D2["XIS"], dtype=float); data2 = D2["data"]
_extra = os.path.join(HERE, "data_1M", "cs_final_coldextra.npy")
if os.path.exists(_extra):
    data.update(np.load(_extra, allow_pickle=True).item()["data"])
REGS = ["PreDamagePreTech", "PreDamagePostTech", "PostDamagePreTech", "PostDamagePostTech"]
TITLES = {"PreDamagePreTech": "pre-dmg / pre-tech", "PreDamagePostTech": "pre-dmg / post-tech",
          "PostDamagePreTech": "post-dmg / pre-tech", "PostDamagePostTech": "post-dmg / post-tech"}
RUNS = [("reference", "0.15", "-", "o"), ("nber_s1", "#1f4e79", "-", "o"),
        ("nber_s2", "#1f4e79", "--", "o"), ("perturb_s1", "#2e7d32", "-", "o"),
        ("perturb_s2", "#2e7d32", "--", "o"), ("paperlr2M_s1", "#00838f", "-", "D")]
DISPLAY = {"reference": "reference", "nber_s1": "warm start A", "perturb_s1": "warm start B",
           "paperlr2M_s1": "cold start"}

def series(lab, reg, key):
    return np.array([data.get(f"{reg}|{lab}|{xi}", {}).get(key, np.nan) or np.nan for xi in D["XIS"]],
                    dtype=float)

def series2(lab, reg, key):
    return np.array([data2.get(f"{reg}|{lab}|{xi}", {}).get(key, np.nan) or np.nan for xi in D2["XIS"]],
                    dtype=float)

plt.rcParams.update({"font.size": 16, "axes.linewidth": 1.1,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})

# ---------- CS1: HJB error ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), sharey=True)
SETTINGS = [
    ("warm start A", "#0072B2", "-", "o", ["nber_s1", "nber_s2"], series),
    ("warm start B", "#009E73", "-", "o", ["perturb_s1", "perturb_s2"], series),
]
for k, reg in enumerate(REGS):
    a = axes.flat[k]
    yref = series2("RUNA(base)", reg, "res")
    a.loglog(XIS2, yref, "-", marker="o", color="0.1", lw=2.2, ms=6,
             label=("reference" if k == 0 else None))
    for lab, col, ls, mk, seeds, fn in SETTINGS:
        ys = [fn(sd, reg, "res") for sd in seeds]
        ys = [y for y in ys if not np.all(np.isnan(y))]
        if not ys:
            continue
        ym = np.nanmean(ys, axis=0)
        a.loglog(XIS, ym, ls, marker=mk, color=col, lw=2.2, ms=6,
                 label=(lab if k == 0 else None))
    yw = [series2(sd, reg, "res") for sd in ["nber_s1", "nber_s2"]]
    yw = [y for y in yw if not np.all(np.isnan(y))]
    if yw:
        a.loglog(XIS2, np.nanmean(yw, axis=0), "-.", marker="^", color="#CC79A7", lw=2.2, ms=6,
                 label=("warm start A, wider ξ" if k == 0 else None))
    a.set_title(TITLES[reg], fontsize=16)
    a.set_xlabel(r"$\xi$"); a.grid(alpha=.22, lw=.6, which="both")
for a in (axes[0,0], axes[1,0]):
    a.set_ylabel("error in the HJB equation (path RMS)")
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(HERE, "figures", "cs1_hjb_by_state.png"), dpi=160, bbox_inches="tight")
print("wrote cs1_hjb_by_state.png")

# ---------- CS2: FOC errors ----------
FOCS = [("FOC_d", "dirty investment"), ("FOC_g", "green investment"), ("FOC_r", "R&D")]
fig, axes = plt.subplots(3, 4, figsize=(24, 13.5), sharey="row")
for r, (key, nice) in enumerate(FOCS):
    for k, reg in enumerate(REGS):
        a = axes[r, k]
        drawn = False
        for lab, col, ls, mk in RUNS:
            y = series(lab, reg, key)
            m = np.isfinite(y) & (y < 1e5)
            if not m.any():
                continue
            a.loglog(XIS[m], y[m], ls, marker=mk, color=col, lw=2.0, ms=5.5,
                     label=(DISPLAY.get(lab) if (r == 0 and k == 0 and ls == "-") else None))
            drawn = True
        if r == 0:
            a.set_title(TITLES[reg], fontsize=16)
        if not drawn:
            a.text(0.5, 0.5, "no R&D in this jump state", transform=a.transAxes,
                   ha="center", fontsize=12, color="0.45")
            a.set_xscale("log")
        if r == 2:
            a.set_xlabel(r"$\xi$")
        a.grid(alpha=.22, lw=.6, which="both")
    axes[r, 0].set_ylabel(f"{nice}\noptimality error")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.005), fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(HERE, "figures", "cs2_foc_by_state.png"), dpi=160, bbox_inches="tight")
print("wrote cs2_foc_by_state.png")
