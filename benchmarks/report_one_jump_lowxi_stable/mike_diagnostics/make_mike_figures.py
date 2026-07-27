"""Mike figures, ALL xi, figure-first: overlaid curvature histograms (baseline vs distorted),
worst-case damage-jump density, per-xi accuracy. Clean, LARGE, project colours, no editorial text."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aer_style; aer_style.apply()

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
FULL = os.path.join(ROOT, "output_lowxi_float64",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_"
    "LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000")
HALF = os.path.join(ROOT, "output_lowxi_float64",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m5p30_"
    "LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000")
HERE = os.path.dirname(os.path.abspath(__file__))
XIS = ["148.600", "0.050", "0.040", "0.030", "0.025", "0.020", "0.010", "0.005"]
LAB = [r"$\xi=\infty$", r"$\xi=0.05$", r"$\xi=0.04$", r"$\xi=0.03$", r"$\xi=0.025$", r"$\xi=0.02$", r"$\xi=0.01$", r"$\xi=0.005$"]
INVXI = np.array([0.0 if s == "148.600" else 1.0 / float(s) for s in XIS])
COL = aer_style.xi_colors(len(XIS))


def L(xi, nm, base=FULL):
    p = os.path.join(base, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}", nm + ".txt")
    return np.atleast_1d(np.loadtxt(p)) if os.path.exists(p) else None


def gbar_mean(xi, base):
    """Path-mean worst-case damage-jump intensity multiplier gbar = lambda_distorted / lambda_baseline.
    Admissibility (Jensen): since V^l <= V, every g^l >= 1, so gbar >= 1. Violation = numerical error."""
    Y, di = L(xi, "Y", base), L(xi, "dmg_jump_intensity", base)
    if Y is None or di is None:
        return np.nan
    Jn = 1.5 * (np.exp(0.36 / 2 * (Y - 1.5) ** 2) - 1) * (Y >= 1.5)   # baseline intensity r1=1.5,r2=0.36,y_lo=1.5
    m = Jn > 1e-9
    return float(np.mean(di[m] / Jn[m])) if m.any() else np.nan


def fig_jensen():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for cal, base, col, mk in [("Full", FULL, aer_style.DISTORTED, "o-"), ("Half", HALF, aer_style.BASELINE, "s-")]:
        gb = [gbar_mean(xi, base) for xi in XIS]
        ax.plot(INVXI, gb, mk, color=col, label=cal)
    ax.axhline(1.0, ls="--", color="0.35")
    ax.axhspan(0, 1.0, color=aer_style.FORBIDDEN, zorder=0)
    ax.axvline(1.0 / 0.025, ls=":", color="0.5")
    ax.set_xticks(INVXI); ax.set_xticklabels([("0" if s == "148.600" else str(int(round(1.0 / float(s))))) for s in XIS], rotation=45)
    ax.set_xlabel(r"$1/\xi$")
    ax.set_ylabel(r"intensity multiplier $\bar g = \lambda_{\rm distorted}/\lambda_{\rm baseline}$")
    ax.legend()
    fig.tight_layout()
    _save(fig, "figJENSEN_admissibility.png")


def _save(fig, name):
    out = os.path.join(HERE, name); fig.savefig(out); plt.close(fig); print("wrote", out)


# --- fig_curv : damage-curvature histogram, baseline (red) vs distorted (blue), one panel per xi (ALL xi)
def fig_curv():
    grid = L("0.010", "lambda3_grid")
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    axes = axes.ravel()
    xpos = np.arange(len(grid))
    for a, (xi, lab) in enumerate(zip(XIS, LAB)):
        ax = axes[a]
        wt = L(xi, "lambda3_weights_distorted")
        ax.bar(xpos - 0.2, np.full(len(grid), 0.2), 0.4, color=aer_style.BASELINE, alpha=0.55,
               ec="darkgrey", label="baseline")
        if wt is not None:
            ax.bar(xpos + 0.2, wt, 0.4, color=aer_style.DISTORTED, alpha=0.75, ec="darkgrey", label="distorted")
        ax.set_title(lab)
        ax.set_xticks(xpos); ax.set_xticklabels([f"{g:.2f}" for g in grid])
        if a % 4 == 0:
            ax.set_ylabel("weight")
        if a >= 4:
            ax.set_xlabel(r"$\lambda_3$")
    axes[0].legend()
    axes[-1].axis("off")
    fig.tight_layout()
    _save(fig, "figCURV_curvature_histograms.png")


# --- fig_density : worst-case damage-jump density vs Y, ALL xi
def fig_density():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for xi, lab, col in zip(XIS, LAB, COL):
        d, Y = L(xi, "dmg_jump_density"), L(xi, "Y")
        if d is not None and Y is not None:
            ax.plot(Y, d, color=col, label=lab)
    ax.set_xlabel(r"$Y$ (temperature anomaly)")
    ax.set_ylabel("worst-case damage-jump density")
    ax.legend()
    fig.tight_layout()
    _save(fig, "figDENS_density.png")


# --- fig_acc : per-xi HJB residual + FOC (left) and jump-exponent amplification (right), ALL xi
def fig_acc():
    R = json.load(open(os.path.join(HERE, "per_xi_residual.json")))
    g = json.load(open(os.path.join(HERE, "per_xi_g_distortion.json")))
    keys = ["inf(logxi=5)", "0.1", "0.05", "0.04", "0.03", "0.025", "0.02", "0.01", "0.005"]
    xis = np.array([float(R["PreDamagePreTech"][k]["xi"]) for k in keys])
    invx = np.where(xis > 100, 0.0, 1.0 / xis)
    o = np.argsort(invx); invx = invx[o]
    pre = np.array([R["PreDamagePreTech"][k]["full_box"]["loss_v_rms"] for k in keys])[o]
    foc = np.array([R["PreDamagePreTech"][k]["full_box"]["FOC_max"] for k in keys])[o]
    term = np.array([R["PostDamagePostTech"][k]["full_box"]["loss_v_rms"] for k in keys])[o]
    dv = np.array([g[k]["dmg_dV_rms"] for k in keys])
    ex = np.array([g[k]["dmg_exp_absmax"] for k in keys])
    gx = np.array([0.0 if k.startswith("inf") else 1.0 / float(k) for k in keys])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
    axL.plot(invx, pre, "o-", color=aer_style.DISTORTED, label="HJB residual (initial regime)")
    axL.plot(invx, term, "s-", color="0.45", label="HJB residual (terminal)")
    axL.plot(invx, foc, "^--", color=aer_style.BASELINE, label="FOC error (max)")
    axL.axhline(1e-3, ls=":", color="0.4")
    axL.set_yscale("log"); axL.set_xlabel(r"$1/\xi$"); axL.set_ylabel("residual / FOC error")
    axL.legend()
    axR.plot(gx, dv, "s-", color="0.45", label=r"value gap $|V^\ell-V|$")
    axR.plot(gx, ex, "o-", color=aer_style.DISTORTED, label=r"jump exponent $\frac{1}{\xi}|V^\ell-V|$")
    axR.axhline(88.7, ls=":", color=aer_style.BASELINE)
    axR.set_yscale("log"); axR.set_xlabel(r"$1/\xi$"); axR.set_ylabel("magnitude")
    axR.legend()
    fig.tight_layout()
    _save(fig, "figACC_accuracy.png")


def fig_money():
    """Money plot: the tiny cross-stage welfare mismatch (~0.1), amplified by 1/xi, grows to dominate
    and drives the worst-case intensity gbar through its floor of 1. Twin axis."""
    g = json.load(open(os.path.join(HERE, "per_xi_g_distortion.json")))
    gk = {"148.600": "inf(logxi=5)", "0.050": "0.05", "0.040": "0.04", "0.030": "0.03", "0.025": "0.025",
          "0.020": "0.02", "0.010": "0.01", "0.005": "0.005"}
    gb = np.array([gbar_mean(x, FULL) for x in XIS])
    expo = np.array([g[gk[x]]["dmg_exp_absmax"] for x in XIS])
    fig, axL = plt.subplots(figsize=(13, 7.5))
    axR = axL.twinx()
    axL.axhspan(0, 1.0, color=aer_style.FORBIDDEN, zorder=0)
    axL.axhline(1.0, ls="--", color="0.35", lw=2)
    l1 = axL.plot(INVXI, gb, "o-", color=aer_style.DISTORTED, ms=10,
                  label=r"worst-case intensity $\bar g$  (must be $\geq 1$)")
    l2 = axR.plot(INVXI, expo, "s--", color=aer_style.BASELINE, ms=9,
                  label=r"amplified cross-regime mismatch  $\frac{1}{\xi}|\Delta V|$")
    axL.axvline(1.0 / 0.025, ls=":", color="0.5", lw=2)
    axR.set_yscale("log")
    axL.set_ylim(0, float(np.nanmax(gb)) * 1.15)
    axL.set_xticks(INVXI); axL.set_xticklabels([("0" if s == "148.600" else str(int(round(1.0 / float(s))))) for s in XIS], rotation=45)
    axL.set_xlabel(r"$1/\xi$  (robustness aversion)")
    axL.set_ylabel(r"worst-case intensity $\bar g$")
    axR.set_ylabel(r"amplified cross-regime mismatch $\frac{1}{\xi}|\Delta V|$")
    ls = l1 + l2
    axL.legend(ls, [x.get_label() for x in ls], loc="center left")
    fig.tight_layout()
    _save(fig, "figMONEY_dominance.png")


if __name__ == "__main__":
    fig_curv(); fig_density(); fig_acc(); fig_jensen(); fig_money()
