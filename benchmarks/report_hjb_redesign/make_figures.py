"""Figures for the one-jump HJB loss/representation redesign experiment.

Follows the house report standard: full-width figures, generous fonts, NO text baked into the
figures (titles/readings live in the note), one ROW per arm with the comparison quantities as
columns, and levels reported alongside the ratio-style diagnostics.

Model source: PostDamagePostTech (3-state terminal regime) of the one-jump (pi=1) system, trained
from scratch in models_onejump_redesign/, 300k iterations, 3 seeds per arm.
"""
import os, sys, glob, json
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/report_one_jump_lowxi_stable/mike_diagnostics")
try:
    import aer_style; aer_style.apply()
except Exception:
    pass
plt.rcParams.update({"axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 13, "lines.linewidth": 2.6})

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
R1 = f"{ROOT}/output_redesign_20260726"              # round 1 (anchor inherited batch xi)
R2 = f"{ROOT}/output_redesign_fixanchor_20260726"    # round 2 (anchor pinned at neutral xi)
OUT = f"{ROOT}/benchmarks/report_hjb_redesign"
SUB = "PostDamagePostTech"
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, f"{ROOT}/models_onejump_redesign")
import tensorflow as tf
from params import PARAMS
from feedforward_subnet import FeedForwardSubNet

LOGK0, Z0, LAM3M = 6.7799, 0.70, 1.0 / 6.0
YHAT, AG2 = PARAMS["y_upper"], PARAMS["A_g_prime_prime"]
V0 = 3.64
XIS = [148.6, 10.0, 1.0, 0.3, 0.1, 0.05]
ARMS = [("A0_baseline", "baseline"), ("A1_nondim", "non-dimensionalised"),
        ("A2_anchor", "level anchor"), ("A3_separable", "separable + anchor"),
        ("A4_theta", r"$1/\xi$ pseudo-state")]
COL = {"A0_baseline": "#444444", "A1_nondim": "#1f77b4", "A2_anchor": "#d62728",
       "A3_separable": "#2ca02c", "A4_theta": "#9467bd", "A5_theta_anchor": "#8c564b"}


def hist(root, arm, seed):
    p = f"{root}/{arm}_seed{seed}/{SUB}/training_history.csv"
    if not os.path.exists(p):
        return None
    rows = [r.strip().split(",") for r in open(p) if r.strip()][1:]
    a = np.array([[float(x) for x in r] for r in rows if len(r) >= 5])
    return a if len(a) else None


def load_phi(root, arm, seed):
    net = FeedForwardSubNet({"num_hiddens": [32]*4, "use_bias": True, "activation": "swish",
                             "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"})
    net.build((None, 7))
    ck = f"{root}/{arm}_seed{seed}/{SUB}/v_nn_checkpoint_{SUB}"
    if not glob.glob(ck + "*"):
        return None
    net.load_weights(ck).expect_partial()
    return net


def xcol(arm, xi):
    return (1.0 / xi) if "theta" in arm else float(np.log(xi))


def vstate(net, arm, lk, z, y, l3, xi, anchored):
    c = xcol(arm, xi)
    s = lambda a, b, cc, d: tf.constant([[a, b, cc, d, AG2, c, c]], tf.float32)
    v = float(net(s(lk, z, y, l3), training=False)[0, 0])
    if anchored:
        v -= float(net(s(LOGK0, Z0, YHAT, LAM3M), training=False)[0, 0]) - V0
    return v


# ---------------------------------------------------------------- figure 1: convergence + level
def fig_convergence():
    fig, axes = plt.subplots(len(ARMS), 2, figsize=(15, 4.0 * len(ARMS)), squeeze=False)
    for i, (arm, lab) in enumerate(ARMS):
        axL, axR = axes[i]
        for seed in (1, 2, 3):
            h = hist(R1, arm, seed)
            if h is None:
                continue
            axL.plot(h[:, 0] / 1000.0, h[:, 1], color=COL[arm], alpha=0.45 + 0.2 * seed)
            axR.plot(h[:, 0] / 1000.0, h[:, 2], color=COL[arm], alpha=0.45 + 0.2 * seed)
        axL.set_yscale("log"); axR.set_yscale("log")
        axL.set_ylabel("HJB residual"); axR.set_ylabel("dirty-investment FOC residual")
        axL.set_ylim(5e-4, 2e-1); axR.set_ylim(1e-5, 2e-1)
        for ax in (axL, axR):
            ax.grid(alpha=0.3)
            ax.annotate(lab, xy=(-0.28, 0.5), xycoords="axes fraction", rotation=90,
                        fontweight="bold", fontsize=15, va="center", ha="center")
        if i == len(ARMS) - 1:
            axL.set_xlabel("training step (thousands)"); axR.set_xlabel("training step (thousands)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_convergence.png"); plt.close(fig)
    print("  fig1_convergence.png")


# ------------------------------------------------------- figure 2: value xi-profile (the anchor bug)
def fig_xi_profile():
    fig, axes = plt.subplots(len(ARMS), 2, figsize=(15, 4.0 * len(ARMS)), squeeze=False)
    for i, (arm, lab) in enumerate(ARMS):
        anchored = ("anchor" in arm) or ("separable" in arm)
        for col, (root, _) in enumerate([(R1, "round1"), (R2, "round2")]):
            ax = axes[i][col]
            got = False
            for seed in (1, 2, 3):
                net = load_phi(root, arm, seed)
                if net is None:
                    continue
                vs = [vstate(net, arm, LOGK0, Z0, YHAT, LAM3M, x, anchored) for x in XIS]
                vn = vs[0]
                ax.plot(XIS, [v - vn for v in vs], "o-", color=COL[arm], alpha=0.45 + 0.2 * seed,
                        markersize=6)
                got = True
            ax.set_xscale("log"); ax.invert_xaxis(); ax.grid(alpha=0.3)
            ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)
            ax.set_ylabel(r"$V(X_0,\xi)-V(X_0,\mathrm{neutral})$")
            if not got:
                ax.set_facecolor("#f5f5f5")
            ax.annotate(lab, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                        fontweight="bold", fontsize=15, va="center", ha="center")
            if i == len(ARMS) - 1:
                ax.set_xlabel(r"$\xi$  (more uncertainty aversion $\rightarrow$)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_xi_profile.png"); plt.close(fig)
    print("  fig2_xi_profile.png")


# ------------------------------------------------ figure 3: measured scale elasticity a(logK)
def fig_elasticity():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    lks = np.linspace(4.5, 7.0, 26)
    for arm, lab in ARMS:
        net = load_phi(R1, arm, 1)
        if net is None:
            continue
        anchored = ("anchor" in arm) or ("separable" in arm)
        a_vals = []
        for lk in lks:
            eps = 1e-2
            vp = vstate(net, arm, lk + eps, Z0, YHAT, LAM3M, 0.05, anchored)
            vm = vstate(net, arm, lk - eps, Z0, YHAT, LAM3M, 0.05, anchored)
            a_vals.append((vp - vm) / (2 * eps))
        axes[0].plot(lks, a_vals, color=COL[arm], label=lab)
        axes[1].plot(np.exp(lks), a_vals, color=COL[arm], label=lab)
    for ax, xl in zip(axes, ["log capital", "capital"]):
        ax.axhline(1.0, color="k", ls="--", lw=1.4, alpha=0.6)
        ax.set_xlabel(xl); ax.set_ylabel(r"marginal value of scale  $\partial V/\partial\log K$")
        ax.grid(alpha=0.3); ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_scale_elasticity.png"); plt.close(fig)
    print("  fig3_scale_elasticity.png")


# ------------------------------------------------------------- figure 4: gate summary (levels+ratios)
def fig_gates():
    names, lvl, res = [], [], []
    base_l = base_r = None
    for arm, lab in ARMS:
        nets = [load_phi(R1, arm, s) for s in (1, 2, 3)]
        nets = [n for n in nets if n is not None]
        hs = [hist(R1, arm, s) for s in (1, 2, 3)]
        hs = [h for h in hs if h is not None]
        if not nets or not hs:
            continue
        anchored = ("anchor" in arm) or ("separable" in arm)
        vals = [vstate(n, arm, 6.0, Z0, YHAT, LAM3M, 0.05, anchored) for n in nets]
        names.append(lab); lvl.append(max(vals) - min(vals))
        res.append(float(np.mean([h[-1, 1] for h in hs])))
        if arm == "A0_baseline":
            base_l, base_r = lvl[-1], res[-1]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    axes[0].bar(x, lvl, color=[COL[a] for a, _ in ARMS][:len(names)])
    axes[0].set_ylabel("cross-seed spread of the value level")
    if base_l:
        axes[0].axhline(base_l, color="k", ls="--", lw=1.4, alpha=0.6)
    axes[1].bar(x, res, color=[COL[a] for a, _ in ARMS][:len(names)])
    axes[1].set_ylabel("HJB residual")
    if base_r:
        axes[1].axhline(base_r, color="k", ls="--", lw=1.4, alpha=0.6)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_gates.png"); plt.close(fig)
    print("  fig4_gates.png")


if __name__ == "__main__":
    print("writing figures ->", OUT)
    fig_convergence(); fig_xi_profile(); fig_elasticity(); fig_gates()
