"""figG — the explicit error via the Jensen / admissibility bound. Clean, large, NO text on figure."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aer_style; aer_style.apply()

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
BASE = {
    "Full": os.path.join(ROOT, "output_lowxi_float64",
        "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_"
        "LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000"),
    "Half": os.path.join(ROOT, "output_lowxi_float64",
        "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m5p30_"
        "LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000"),
}
HERE = os.path.dirname(os.path.abspath(__file__))
XIS = ["148.600", "0.040", "0.030", "0.025", "0.020", "0.015", "0.013", "0.010", "0.005"]
INVXI = np.array([0.0 if s == "148.600" else 1.0 / float(s) for s in XIS])
XI_STAR = 0.025
FULL_C, HALF_C = aer_style.DISTORTED, aer_style.BASELINE


def p60(base, xi):
    p = os.path.join(base, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}", "dmg_jump_prob.txt")
    return float(np.atleast_1d(np.loadtxt(p))[-1]) if os.path.exists(p) else np.nan


fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))

# left: worst-case P(jump, yr60) vs 1/xi + monotone Jensen bound + baseline floor + forbidden band
for cal, col, mk in [("Full", FULL_C, "o-"), ("Half", HALF_C, "s-")]:
    P = np.array([p60(BASE[cal], xi) for xi in XIS])
    axL.plot(INVXI, P, mk, color=col, label=f"{cal}: numerical")
    axL.plot(INVXI, np.maximum.accumulate(P), "--", color=col, alpha=0.7, label=f"{cal}: Jensen bound")
    axL.axhline(P[0], ls=":", color=col, alpha=0.6)
fullbase = p60(BASE["Full"], "148.600")
axL.axhspan(0, fullbase, color=aer_style.FORBIDDEN, zorder=0)
axL.axvline(1.0 / XI_STAR, ls="--", color="0.4")
axL.set_xticks(INVXI); axL.set_xticklabels([r"$\infty$" if s == "148.600" else s for s in XIS], rotation=45)
axL.set_xlabel(r"$1/\xi$"); axL.set_ylabel("worst-case damage-jump prob. (yr 60)")
axL.set_ylim(0, None); axL.legend()

# right: max damage-jump distortion g vs 1/xi (log-y) + monotone bound + g=1 + xi*
g = json.load(open(os.path.join(HERE, "per_xi_g_distortion.json")))
order = ["inf(logxi=5)", "0.1", "0.05", "0.04", "0.03", "0.025", "0.02", "0.01", "0.005"]
gx = np.array([0.0 if k.startswith("inf") else 1.0 / float(k) for k in order])
gmax = np.array([g[k]["dmg_g_max"] for k in order])
axR.plot(gx, gmax, "o-", color=FULL_C, label=r"numerical $\max_\ell g^\ell$")
axR.plot(gx, np.maximum.accumulate(gmax), "--", color=FULL_C, alpha=0.7, label="monotone bound")
axR.axhline(1.0, ls=":", color="0.4")
axR.axvline(1.0 / XI_STAR, ls="--", color="0.4")
axR.set_yscale("log")
axR.set_xticks(gx); axR.set_xticklabels([r"$\infty$" if k.startswith("inf") else k for k in order], rotation=45)
axR.set_xlabel(r"$1/\xi$"); axR.set_ylabel(r"max damage-jump distortion $g^\ell$")
axR.legend()

fig.tight_layout()
out = os.path.join(HERE, "figG_explicit_error.png")
fig.savefig(out); plt.close(fig); print("wrote", out)
