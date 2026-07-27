"""Mike's diagnostic figures — clean, LARGE, project colours, NO text on the figure (caption carries it).
figA investment pathways | figB curvature distortion | figC jump-prob stability | figD drift distortions.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aer_style; aer_style.apply()

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
BASE = {
    "Full adj-cost": os.path.join(ROOT, "output_lowxi_float64",
        "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_"
        "LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000"),
    "Half adj-cost": os.path.join(ROOT, "output_lowxi_float64",
        "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m5p30_"
        "LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000"),
}
HERE = os.path.dirname(os.path.abspath(__file__))
XIS = ["148.600", "0.050", "0.040", "0.030", "0.025", "0.020", "0.010", "0.005"]
INVXI = np.array([0.0 if s == "148.600" else 1.0 / float(s) for s in XIS])
COL = aer_style.xi_colors(len(XIS))
XI_STAR = 0.025


def load(base, xi, name):
    p = os.path.join(base, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}", name + ".txt")
    return np.atleast_1d(np.loadtxt(p)) if os.path.exists(p) else None


def last(base, xi, name):
    a = load(base, xi, name)
    return np.nan if a is None else float(a[-1])


def xilab(s):
    return r"$\xi=\infty$" if s == "148.600" else rf"$\xi={s}$"


# --------------------------------------------------------------------- figA investment pathways
def fig_investment():
    # all three normalized by OUTPUT (Haoyang's convention): GreenInvestment, DirtyInvestment, and RD
    # (= I_r/Output). NOT i_r (= I_r/K_total) — that mixed denominators on one panel and read ~8.7x (=1/ᾱ)
    # too small, which is the "R&D order-of-magnitude" gap Mike flagged.
    fig, axes = plt.subplots(len(BASE), 3, figsize=(19, 11), sharex=True)
    names = [("GreenInvestment", "green investment / output (%)"),
             ("DirtyInvestment", "dirty investment / output (%)"),
             ("RD", "R&D investment / output (%)")]
    for r, (cal, base) in enumerate(BASE.items()):
        t = load(base, "148.600", "t")
        for c, (nm, ylab) in enumerate(names):
            ax = axes[r, c]
            for xi, col in zip(XIS, COL):
                y = load(base, xi, nm)
                if y is not None:
                    ax.plot(t, y * 100, color=col, label=xilab(xi))
            if r == len(BASE) - 1:
                ax.set_xlabel("year")
            ax.set_ylabel(f"{cal}\n{ylab}" if c == 0 else ylab)
    axes[0, 2].legend(ncol=2, loc="best")
    fig.tight_layout()
    _save(fig, "figA_investment_paths.png")


# --------------------------------------------------------------------- figB curvature distortion
def fig_curvature():
    grid = load(BASE["Full adj-cost"], "0.010", "lambda3_grid")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    show_xi = ["0.040", "0.025", "0.010"]
    for k, (cal, base) in enumerate(BASE.items()):
        ax = axes[k]
        xpos = np.arange(len(grid)); w = 0.8 / (len(show_xi) + 1)
        ax.bar(xpos - 0.4 + w / 2, np.full(len(grid), 0.2), w,
               color=aer_style.BASELINE, alpha=0.5, ec="darkgrey", label="baseline")
        for j, xi in enumerate(show_xi):
            wt = load(base, xi, "lambda3_weights_distorted")
            if wt is not None:
                ax.bar(xpos - 0.4 + (j + 1.5) * w, wt, w, alpha=0.7, ec="darkgrey",
                       color=plt.cm.viridis(0.2 + 0.6 * j / len(show_xi)), label=xilab(xi))
        ax.set_xticks(xpos); ax.set_xticklabels([f"{g:.2f}" for g in grid])
        ax.set_xlabel(r"damage curvature $\lambda_3$")
        ax.set_ylabel(f"{cal}\nworst-case weight" if k == 0 else "worst-case weight")
        ax.legend()
    ax = axes[2]
    for cal, base, col, mk in [("Full", BASE["Full adj-cost"], aer_style.DISTORTED, "o-"),
                               ("Half", BASE["Half adj-cost"], aer_style.BASELINE, "s-")]:
        wmax = [last(base, xi, "lambda3_weights_distorted") for xi in XIS]
        ax.plot(INVXI, np.array(wmax) * 100, mk, color=col, label=cal)
    ax.axhline(20, ls=":", color="0.5")
    ax.set_xticks(INVXI); ax.set_xticklabels([("0" if s == "148.600" else str(int(round(1.0 / float(s))))) for s in XIS], rotation=45)
    ax.set_xlabel(r"$1/\xi$"); ax.set_ylabel(r"weight on most-severe $\lambda_3$ (%)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "figB_curvature_distortion.png")


# --------------------------------------------------------------------- figC jump-prob stability
def fig_jump_stability():
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    base = BASE["Full adj-cost"]; t = load(base, "148.600", "t")
    ax = axes[0]
    for xi, col in zip(XIS, COL):
        y = load(base, xi, "dmg_jump_prob")
        if y is not None:
            ax.plot(t, y, color=col, label=xilab(xi))
    ax.set_xlabel("year"); ax.set_ylabel("worst-case damage-jump prob.")
    ax.legend(ncol=2)
    ax = axes[1]
    for cal, b, col, mk in [("Full", BASE["Full adj-cost"], aer_style.DISTORTED, "o-"),
                            ("Half", BASE["Half adj-cost"], aer_style.BASELINE, "s-")]:
        ax.plot(INVXI, [last(b, xi, "dmg_jump_prob") for xi in XIS], mk, color=col, label=cal)
    ax.axvline(1.0 / XI_STAR, ls="--", color="0.4")
    ax.set_xticks(INVXI); ax.set_xticklabels([("0" if s == "148.600" else str(int(round(1.0 / float(s))))) for s in XIS], rotation=45)
    ax.set_xlabel(r"$1/\xi$"); ax.set_ylabel("worst-case damage-jump prob. (yr 60)")
    ax.legend()
    ax = axes[2]
    for cal, b, col, mk in [("Full", BASE["Full adj-cost"], aer_style.DISTORTED, "o-"),
                            ("Half", BASE["Half adj-cost"], aer_style.BASELINE, "s-")]:
        ax.plot(INVXI, [last(b, xi, "tech_jump_prob") for xi in XIS], mk, color=col, label=cal)
    ax.set_xticks(INVXI); ax.set_xticklabels([("0" if s == "148.600" else str(int(round(1.0 / float(s))))) for s in XIS], rotation=45)
    ax.set_xlabel(r"$1/\xi$"); ax.set_ylabel("worst-case tech-jump prob. (yr 60)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "figC_jump_stability.png")


# --------------------------------------------------------------------- figD drift distortions
def fig_drift():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.ravel()
    base = BASE["Full adj-cost"]; t = load(base, "148.600", "t")
    panels = [("h_y", r"climate  $h_y$"), ("h_d", r"dirty capital  $h_d$"),
              ("h_g", r"green capital  $h_g$"), ("h_r", r"knowledge  $h_r$")]
    for c, (nm, ylab) in enumerate(panels):
        ax = axes[c]
        for xi, col in zip(XIS, COL):
            y = load(base, xi, nm)
            if y is not None:
                ax.plot(t, y, color=col, label=xilab(xi))
        ax.axhline(0, ls=":", color="0.5")
        if c >= 2:
            ax.set_xlabel("year")
        ax.set_ylabel(ylab)
    axes[0].legend(ncol=2)
    fig.tight_layout()
    _save(fig, "figD_drift_distortions.png")


def _save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    fig_investment(); fig_curvature(); fig_jump_stability(); fig_drift()
