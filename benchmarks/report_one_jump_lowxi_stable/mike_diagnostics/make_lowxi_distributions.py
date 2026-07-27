"""figJ — the worst-case DISTRIBUTIONS themselves, at xi <= 0.025 (the broken regime).
Left: damage-curvature (lambda3) histogram — concentrates to a degenerate point mass.
Right: damage-jump density vs Y — rises then COLLAPSES to zero. Clean, large, no text on figure."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aer_style; aer_style.apply()

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
FULL = os.path.join(ROOT, "output_lowxi_float64",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_"
    "LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000")
HERE = os.path.dirname(os.path.abspath(__file__))

XIS = [("148.600", r"$\xi=\infty$", "C3"),
       ("0.025", r"$\xi=0.025$", "C0"),
       ("0.010", r"$\xi=0.010$", "C1"),
       ("0.005", r"$\xi=0.005$", "C2")]


def L(xi, nm):
    p = os.path.join(FULL, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}", nm + ".txt")
    return np.atleast_1d(np.loadtxt(p)) if os.path.exists(p) else None


fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))

# left: curvature (lambda3) histogram, grouped bars per xi
grid = L("0.010", "lambda3_grid")
xpos = np.arange(len(grid)); w = 0.8 / len(XIS)
for j, (xi, lab, col) in enumerate(XIS):
    wt = L(xi, "lambda3_weights_distorted")
    if wt is not None:
        axL.bar(xpos - 0.4 + (j + 0.5) * w, wt, w, color=col, alpha=0.7, ec="darkgrey", label=lab)
axL.set_xticks(xpos); axL.set_xticklabels([f"{g:.2f}" for g in grid])
axL.set_xlabel(r"damage curvature $\lambda_3$")
axL.set_ylabel("worst-case weight")
axL.legend()

# right: damage-jump density vs Y
for xi, lab, col in XIS:
    dens = L(xi, "dmg_jump_density"); Y = L(xi, "Y")
    if dens is not None and Y is not None:
        axR.plot(Y, dens, color=col, label=lab)
axR.set_xlabel(r"$Y$ (temperature anomaly)")
axR.set_ylabel("worst-case damage-jump density")
axR.legend()

fig.tight_layout()
out = os.path.join(HERE, "figJ_lowxi_distributions.png")
fig.savefig(out); plt.close(fig); print("wrote", out)
