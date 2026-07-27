"""fig2 (drift-only): isolated worst-case temperature shift dY(t) = Y_worst - Y_base
= cumulative integral of varsigma*h_y*E (the exact continuous-channel h_y distortion).
1x2 (Full, Half), all stable xi. Replaces the approximate Gaussian-CDF panels with the
exact drift-shift object only (per the revised report)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
FULL = os.path.join(ROOT, "output_lowxi_float64",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_"
    "LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000")
HALF = os.path.join(ROOT, "output_lowxi_float64",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m5p30_"
    "LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000")
HERE = os.path.dirname(os.path.abspath(__file__))
VARSIGMA = 1.2 * 1.86e-3

XIS = ["0.040", "0.030", "0.025", "0.020", "0.010", "0.005"]
LAB = {"0.040": r"$\xi=0.04$", "0.030": r"$\xi=0.03$", "0.025": r"$\xi=0.025$",
       "0.020": r"$\xi=0.02$", "0.010": r"$\xi=0.01$", "0.005": r"$\xi=0.005$"}
COL = {"0.040": "tab:green", "0.030": "tab:olive", "0.025": "tab:blue",
       "0.020": "tab:purple", "0.010": "tab:orange", "0.005": "tab:red"}


def gload(base, xi, name):
    p = os.path.join(base, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}", name)
    return np.atleast_1d(np.loadtxt(p))


def dY(base, xi):
    t = gload(base, xi, "t.txt"); E = gload(base, xi, "E.txt"); h_y = gload(base, xi, "h_y.txt")
    integ = VARSIGMA * h_y * E
    dy = np.zeros_like(t)
    for i in range(1, len(t)):
        dy[i] = dy[i - 1] + 0.5 * (integ[i] + integ[i - 1]) * (t[i] - t[i - 1])
    return t, dy


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
for ax, (cal, base) in zip(axes, [("Full", FULL), ("Half", HALF)]):
    for xi in XIS:
        t, dy = dY(base, xi)
        ax.plot(t, dy, color=COL[xi], label=LAB[xi])
        if abs(t[-1] - 60) < 5 or len(t) > 60:
            k = min(len(t) - 1, int(np.argmin(np.abs(t - 60)))) if t[-1] >= 60 else len(t) - 1
        print(f"{cal} xi={xi}: dY(end={t[-1]:.0f}yr)={dy[-1]:+.4f}")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_title(f"{cal} adj-cost: worst-case temperature shift "
                 r"$\Delta Y(t)=Y_{\rm worst}-Y_{\rm base}$")
    ax.set_xlabel("Year  $t$"); ax.set_ylabel(r"$\Delta Y$  ($^\circ$C, hotter)")
    ax.legend(title="continuous channel"); ax.grid(alpha=0.3)
fig.suptitle(r"Isolated continuous-channel ($h_y$) worst-case temperature shift "
             r"$\Delta Y(t)=\int \varsigma\, h_y\, \mathcal{E}\, dt$  (exact; monotone, ordered by $1/\xi$)")
fig.tight_layout()
out = os.path.join(HERE, "figures", "fig2_worstcase_temperature_drift.png")
fig.savefig(out, dpi=140); print("wrote", out)
