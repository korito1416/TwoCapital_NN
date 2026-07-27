"""OLD (delta=0.025) vs NEW (delta=0.01) damage-curvature distortion. Clean, large, no text on figure."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aer_style; aer_style.apply()

OLD = ("/project/lhansen/Cap_damage/TwoStageTechJump/output/"
       "TechSearch_LR_piecewiseconstant_10e-4,10e-4,10e-4,10e-4_128_neurons_32_"
       "#HiddenLayer_4_num_iterations2000000/PreDamagePreTech")
NEW = ("/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/"
       "TwoStageTech_LR_warmup_cosine_40e-5,40e-5,40e-5,40e-5_128_neurons_32_"
       "#HiddenLayer_4_num_iterations1000000/PreDamagePreTech")
HERE = os.path.dirname(os.path.abspath(__file__))
OLD_C, NEW_C = "0.55", aer_style.DISTORTED


def avail(base):
    return sorted({d.split("ξ_")[-1] for d in glob.glob(os.path.join(base, "SimulationOutputs_ξ_*"))}, key=float)


def weights(base, xi):
    p = os.path.join(base, f"SimulationOutputs_ξ_{xi}", "lambda3_weights_distorted.txt")
    return np.atleast_1d(np.loadtxt(p)) if os.path.exists(p) else None


def grid(base, xi):
    p = os.path.join(base, f"SimulationOutputs_ξ_{xi}", "lambda3_grid.txt")
    return np.atleast_1d(np.loadtxt(p)) if os.path.exists(p) else None


common = [x for x in avail(OLD) if x in avail(NEW)]
g = grid(NEW, common[0])
g = g if g is not None else np.array([0, .083, .167, .25, .333])

fig, axes = plt.subplots(1, len(common) + 1, figsize=(6.0 * (len(common) + 1), 6))
for k, xi in enumerate(sorted(common, key=float, reverse=True)):
    ax = axes[k]; xpos = np.arange(len(g)); w = 0.38
    ax.bar(xpos - w / 2, weights(OLD, xi), w, color=OLD_C, alpha=0.7, ec="darkgrey", label=r"OLD $\delta{=}0.025$")
    ax.bar(xpos + w / 2, weights(NEW, xi), w, color=NEW_C, alpha=0.7, ec="darkgrey", label=r"NEW $\delta{=}0.01$")
    ax.axhline(0.2, ls=":", color="0.5")
    ax.set_xticks(xpos); ax.set_xticklabels([f"{v:.2f}" for v in g])
    ax.set_xlabel(rf"$\lambda_3$   ($\xi={xi}$)")
    if k == 0:
        ax.set_ylabel("worst-case weight")
    ax.legend()
ax = axes[-1]
for base, lab, mk, col in [(OLD, r"OLD $\delta{=}0.025$", "s-", OLD_C), (NEW, r"NEW $\delta{=}0.01$", "o-", NEW_C)]:
    xs = avail(base)
    x = np.array([0.0 if float(s) > 100 else 1.0 / float(s) for s in xs])
    y = np.array([weights(base, s)[-1] * 100 for s in xs])
    o = np.argsort(x)
    ax.plot(x[o], y[o], mk, color=col, label=lab)
ax.axhline(20, ls=":", color="0.5")
ax.set_xlabel(r"$1/\xi$"); ax.set_ylabel(r"weight on most-severe $\lambda_3$ (%)")
ax.legend()
fig.tight_layout()
out = os.path.join(HERE, "figE_old_vs_new_delta.png")
fig.savefig(out); plt.close(fig); print("wrote", out)
