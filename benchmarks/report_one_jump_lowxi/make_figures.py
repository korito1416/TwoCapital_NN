"""Figures for the one-jump (pi=1) robustness (xi) sensitivity report.
Source: the CONSISTENT Stage-C NN models (logxi_min=-4.61, cover xi in [0.01,148.6]) for BOTH
adjustment-cost calibrations, deterministic simulation (60 yr, monthly).  Rates = the controls."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 17, "axes.labelsize": 19, "axes.titlesize": 20,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
    "lines.linewidth": 2.4, "figure.dpi": 130, "savefig.bbox": "tight",
})

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_lowxi_001"
FULL = ROOT + "/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m4p61_LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000/SimulationDeterministic"
HALF = ROOT + "/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m4p61_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000/SimulationDeterministic"
OUT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/report_one_jump_lowxi/figures"

XIS = [("148.600", r"$\xi=\infty$ (neutral)"), ("0.100", r"$\xi=0.10$"), ("0.050", r"$\xi=0.05$"),
       ("0.040", r"$\xi=0.04$"), ("0.030", r"$\xi=0.03$"), ("0.020", r"$\xi=0.02$"), ("0.010", r"$\xi=0.01$")]
COLORS = ["0.45", "#6a51a3", "#3b6fb6", "#2c9c8f", "#5aa15a", "#e08a1e", "#c0392b"]  # neutral grey -> averse red
STYLES = ["--", "-", "-", "-", "-", "-", "-"]

def load(base, xi, name):
    p = os.path.join(base, f"SimulationOutputs_ξ_{xi}", name + ".txt")
    return np.loadtxt(p) if os.path.exists(p) else None

def years(a):
    return np.arange(len(a)) / 12.0  # monthly -> years

# ---- Figure 1: 3 quantities x 2 calibrations, full 60yr paths, 5 xi each ----
QUANT = [("DirtyInvestment", r"Dirty investment rate  $i_d$"),
         ("GreenInvestment", r"Green investment rate  $i_g$"),
         ("E", r"Emissions  $\mathcal{E}$")]
fig, ax = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
for c, (base, ctitle) in enumerate([(FULL, "Full adjustment cost\n" + r"($\Gamma=0.060,\ \theta=16.7$)"),
                                     (HALF, "Half adjustment cost\n" + r"($\Gamma=0.12,\ \theta=8.35$)")]):
    for r, (q, ylab) in enumerate(QUANT):
        a = ax[r][c]
        for (xi, lab), col, st in zip(XIS, COLORS, STYLES):
            y = load(base, xi, q)
            if y is None: continue
            a.plot(years(y), y, st, color=col, label=lab)
        if r == 0: a.set_title(ctitle, fontsize=18, pad=10)
        if c == 0: a.set_ylabel(ylab)
        if r == 2: a.set_xlabel("Year")
        a.grid(alpha=0.25)
        if r == 0 and c == 1: a.legend(loc="upper right", framealpha=0.9)
fig.suptitle("One-jump ($\\pi=1$) climate NN model — robustness ($\\xi$) sensitivity of investment & emissions",
             fontsize=19, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(OUT, "fig1_paths_by_xi.png"))
print("saved fig1_paths_by_xi.png")

# ---- Figure 2: robustness pullback summary — i_d rate vs xi (log-x), both calibrations ----
xi_num = [148.6, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01]
def at_year(base, q, yr=5):
    out = []
    for xi, _ in XIS:
        a = load(base, xi, q); out.append(a[min(int(yr*12), len(a)-1)] if a is not None else np.nan)
    return np.array(out)

fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
for k, (q, ylab) in enumerate([("DirtyInvestment", r"Dirty investment rate $i_d$ (year 5)"),
                                ("GreenInvestment", r"Green investment rate $i_g$ (year 5)")]):
    for base, lab, col, mk in [(FULL, "Full adj. cost", "#c0392b", "o"), (HALF, "Half adj. cost", "#3b6fb6", "s")]:
        v = at_year(base, q, 5)
        ax2[k].plot(xi_num, v, mk + "-", color=col, label=lab, markersize=9)
    ax2[k].set_xscale("log"); ax2[k].invert_xaxis()
    ax2[k].set_xlabel(r"$\xi$  (smaller = more robustness aversion $\rightarrow$)")
    ax2[k].set_ylabel(ylab); ax2[k].grid(alpha=0.25); ax2[k].legend()
fig2.suptitle(r"Robustness pullback: lower $\xi$ reduces dirty investment; green investment ~unchanged", fontsize=18)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig2_pullback_vs_xi.png"))
print("saved fig2_pullback_vs_xi.png")

# ---- print the numbers for the report text (rates at year 0 and year 5) ----
print("\n=== rates for report (year 0 / year 5) ===")
for base, tag in [(FULL, "FULL"), (HALF, "HALF")]:
    print(f"-- {tag} --")
    for xi, lab in XIS:
        idd = load(base, xi, "DirtyInvestment"); ig = load(base, xi, "GreenInvestment"); E = load(base, xi, "E")
        g = lambda a, yr: a[min(int(yr*12), len(a)-1)]
        print(f"   xi={xi}: i_d(0)={g(idd,0):.4f} i_d(5)={g(idd,5):.4f} | i_g(0)={g(ig,0):.4f} i_g(5)={g(ig,5):.4f} | E(5)={g(E,5):.4f}")
