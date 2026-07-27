"""Year-5 R&D investment rate i_r vs xi (both calibrations) for the stable report.
Reads i_r.txt (= I_r/(K^d+K^g), the R&D investment-to-total-capital rate) from the
float64 deterministic sim, index 60 (t=5 yr). Matches the style of fig1."""
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

XIS = ["148.600", "0.040", "0.030", "0.025", "0.020", "0.010", "0.005"]
XLAB = [r"$\infty$", "0.040", "0.030", "0.025", "0.020", "0.010", "0.005"]


def yr5(base, xi, nm="i_r"):
    p = os.path.join(base, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}", nm + ".txt")
    a = np.atleast_1d(np.loadtxt(p))
    return float(a[min(60, len(a) - 1)])


full = [yr5(FULL, xi) for xi in XIS]
half = [yr5(HALF, xi) for xi in XIS]
# x-axis = 1/xi (robustness aversion), TRUE spacing; infinity -> 0
x = np.array([0.0 if s == "148.600" else 1.0 / float(s) for s in XIS])

print("i_r year-5 (rate):")
for xi, f, h in zip(XIS, full, half):
    print(f"  xi={xi:>8}  FULL={f:.5f}  HALF={h:.5f}")

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
ax.plot(x, np.array(full) * 100, "o-", color="tab:red", label="Full adj-cost")
ax.plot(x, np.array(half) * 100, "s-", color="tab:blue", label="Half adj-cost")
ax.set_xticks(x); ax.set_xticklabels(XLAB, rotation=45, fontsize=8)
ax.set_xlabel(r"$1/\xi$  (robustness aversion; $\xi$ labelled; $\xi=\infty$ at $0$)")
ax.set_ylabel("rate (%)")
ax.set_title(r"R&D investment rate $i^r$ (year 5)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "figures", "fig4_rd_investment_vs_xi.png")
fig.savefig(out, dpi=150)
print("wrote", out)
