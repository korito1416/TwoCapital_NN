"""
Evaluate the trained post-damage post-tech DGM network and plot the climate-augmented
solution in the two-capital benchmark format, showing what the climate state Y does.

States: (logK, Z, Y) + pseudo-states (lambda3, logxi). We fix logK, lambda3, logxi and
show controls/marginal-values vs Z (curves indexed by temperature Y), and the temperature
response (controls + marginal value of temperature v_Y) vs Y.

Loads the pretrained weights (output_dgm_001/.../PostDamagePostTech). TF required -> sbatch.
"""
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
sys.path.insert(0, os.path.join(ROOT, "models"))
from feedforward_subnet import FeedForwardSubNet            # noqa: E402
from params import PARAMS, investment_rate_activation       # noqa: E402

CKPT = os.environ.get("PDPT_CKPT") or os.path.join(ROOT, "output_largebatch_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000_LargeBatch1024_Continue_LR_10e-6,10e-4_iters1000000",
    "PostDamagePostTech")
OD = os.path.join(ROOT, "benchmarks", "post_damage_post_tech", "outputs")
os.makedirs(OD, exist_ok=True)

A_d  = PARAMS["A_d"]; A_gpp = PARAMS["A_g_prime_prime"]
y_up = PARAMS["y_upper"]
logK_fix = float(np.log(PARAMS["K0"]))      # ~6.78, the calibrated initial scale
lam3_fix = 1.0/6.0                          # median damage realization
logxi_fix = 5.0                             # xi=148.4 ~ no robustness (isolate climate)


def build_and_load():
    common = dict(num_hiddens=[32, 32, 32, 32], use_bias=True, dim=1)
    v_nn = FeedForwardSubNet({**common, "activation": "swish", "final_activation": "softplus", "nn_name": "v_nn"})
    i_d_nn = FeedForwardSubNet({**common, "activation": "tanh",
                               "final_activation": investment_rate_activation(PARAMS["θ_d"]), "nn_name": "i_d_nn"})
    i_g_nn = FeedForwardSubNet({**common, "activation": "tanh",
                               "final_activation": investment_rate_activation(PARAMS["θ_g"]), "nn_name": "i_g_nn"})
    for net in (v_nn, i_d_nn, i_g_nn):
        net.build((None, 7))
    v_nn.load_weights(os.path.join(CKPT, "v_nn_checkpoint_PostDamagePostTech"))
    i_d_nn.load_weights(os.path.join(CKPT, "i_d_nn_checkpoint_PostDamagePostTech"))
    i_g_nn.load_weights(os.path.join(CKPT, "i_g_nn_checkpoint_PostDamagePostTech"))
    return v_nn, i_d_nn, i_g_nn


def evaluate(v_nn, i_d_nn, i_g_nn, Zv, Yv, logK=logK_fix, lam3=lam3_fix, logxi=logxi_fix):
    """Zv, Yv: 1D arrays of equal length (a set of (Z,Y) points). Returns dict."""
    n = len(Zv)
    Z = tf.constant(Zv.reshape(-1, 1), tf.float32)
    Y = tf.constant(Yv.reshape(-1, 1), tf.float32)
    lK = tf.constant(np.full((n, 1), logK), tf.float32)
    l3 = tf.constant(np.full((n, 1), lam3), tf.float32)
    Agpp = tf.constant(np.full((n, 1), A_gpp), tf.float32)
    lx = tf.constant(np.full((n, 1), logxi), tf.float32)
    with tf.GradientTape(persistent=True) as t:
        t.watch([lK, Z, Y])
        X = tf.concat([lK, Z, Y, l3, Agpp, lx, lx], 1)
        v = v_nn(X, training=False)          # training=False -> BN uses trained moving stats
    v_logK = t.gradient(v, lK); v_Z = t.gradient(v, Z); v_Y = t.gradient(v, Y)
    del t
    X = tf.concat([lK, Z, Y, l3, Agpp, lx, lx], 1)
    i_d = i_d_nn(X, training=False).numpy().ravel(); i_g = i_g_nn(X, training=False).numpy().ravel()
    vlK = v_logK.numpy().ravel(); vZ = v_Z.numpy().ravel(); vY = v_Y.numpy().ravel()
    qd = vlK - Zv * vZ; qg = vlK + (1 - Zv) * vZ          # tilde marginal values
    c = (A_d - i_d) * (1 - Zv) + (A_gpp - i_g) * Zv
    # economic marginal value of temperature V_Y = v_Y - (logN)_Y (regime-consistent slope)
    l1, l2 = PARAMS["λ1"], PARAMS["λ2"]
    lNy = l1 + l2 * Yv + lam3 * (Yv - y_up)
    VY = vY - lNy
    # FOC residuals (should be ~ training loss ~5e-5 if the eval is consistent)
    Gd, td = PARAMS["Γ_d"], PARAMS["θ_d"]; Gg, tg = PARAMS["Γ_g"], PARAMS["θ_g"]
    dl = PARAMS["δ"]
    FOC_d = -dl / np.maximum(c, 1e-8) + Gd * td / (1 + td * i_d) * qd
    FOC_g = -dl / np.maximum(c, 1e-8) + Gg * tg / (1 + tg * i_g) * qg
    return dict(Z=Zv, Y=Yv, i_d=i_d, i_g=i_g, v=v.numpy().ravel(),
                v_logK=vlK, v_Z=vZ, v_Y=vY, V_Y=VY, q_d=qd, q_g=qg, c=c,
                FOC_d=FOC_d, FOC_g=FOC_g)


def main():
    v_nn, i_d_nn, i_g_nn = build_and_load()
    print(f"[loaded] from {CKPT}", flush=True)
    print(f"[fixed] logK={logK_fix:.3f} (K0={PARAMS['K0']}), lambda3={lam3_fix:.3f}, "
          f"logxi={logxi_fix} (xi={np.exp(logxi_fix):.1f})", flush=True)

    Zg = np.linspace(0.1, 0.9, 41)
    Yg = np.linspace(0.0, 4.0, 41)
    Ys_for_Z = [0.0, 1.5, 2.5, 4.0]      # temperature curves on the vs-Z panels
    Zs_for_Y = [0.3, 0.5, 0.7]           # share curves on the vs-Y panels
    cols = plt.cm.plasma(np.linspace(0.1, 0.85, len(Ys_for_Z)))
    colsZ = plt.cm.viridis(np.linspace(0.15, 0.85, len(Zs_for_Y)))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # row 0: controls + marginal values vs Z, curves indexed by Y
    for c, Yv in zip(cols, Ys_for_Z):
        o = evaluate(v_nn, i_d_nn, i_g_nn, Zg, np.full_like(Zg, Yv))
        ax[0, 0].plot(Zg, o["i_d"], color=c, lw=1.8, label=f"Y={Yv:g}")
        ax[0, 1].plot(Zg, o["i_g"], color=c, lw=1.8, label=f"Y={Yv:g}")
        ax[0, 2].plot(Zg, o["q_d"], color=c, lw=1.6, ls="-")
        ax[0, 2].plot(Zg, o["q_g"], color=c, lw=1.6, ls="--")
    ax[0, 0].set_title(r"Dirty investment $i^d(Z)$"); ax[0, 0].set_ylabel(r"$i^d$")
    ax[0, 1].set_title(r"Green investment $i^g(Z)$"); ax[0, 1].set_ylabel(r"$i^g$")
    ax[0, 2].set_title(r"Marginal values $\tilde q_d$ (—), $\tilde q_g$ (--)")
    for a in ax[0]:
        a.set_xlabel("Z (green capital share)"); a.grid(alpha=0.3); a.legend(fontsize=8)

    # row 1: temperature response vs Y, curves indexed by Z
    for c, Zv in zip(colsZ, Zs_for_Y):
        o = evaluate(v_nn, i_d_nn, i_g_nn, np.full_like(Yg, Zv), Yg)
        ax[1, 0].plot(Yg, o["i_d"], color=c, lw=1.8, label=f"Z={Zv:g}")
        ax[1, 1].plot(Yg, o["i_g"], color=c, lw=1.8, label=f"Z={Zv:g}")
        ax[1, 2].plot(Yg, o["V_Y"], color=c, lw=1.8, label=f"Z={Zv:g}")
    for a in ax[1]:
        a.axvline(y_up, color="grey", ls=":", lw=1.0)
        a.set_xlabel("Y (temperature anomaly)"); a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[1, 0].set_title(r"Dirty investment vs temperature $i^d(Y)$"); ax[1, 0].set_ylabel(r"$i^d$")
    ax[1, 1].set_title(r"Green investment vs temperature $i^g(Y)$"); ax[1, 1].set_ylabel(r"$i^g$")
    ax[1, 2].set_title(r"Economic marginal value of temp.\ $V_Y=v_Y-(\log N)_Y$ (should be $\leq 0$)")
    ax[1, 2].set_ylabel(r"$V_Y$"); ax[1, 2].axhline(0, color="r", ls="--", lw=0.8)

    fig.suptitle(f"Post-damage post-tech WITH climate (trained DGM): logK={logK_fix:.2f}, "
                 f"$\\lambda_3$={lam3_fix:.3f}, $\\xi$={np.exp(logxi_fix):.0f}", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OD, "climate_pretrained_overview.png")
    fig.savefig(p, dpi=150); print("saved", p, flush=True)

    # FOC self-check on a grid (should be ~ training loss ~5e-5 if eval is consistent)
    Zc = np.linspace(0.2, 0.8, 13); Yc = np.full_like(Zc, 3.0)
    og = evaluate(v_nn, i_d_nn, i_g_nn, Zc, Yc)
    print(f"[FOC self-check, Y=3] max|FOC_d|={np.max(np.abs(og['FOC_d'])):.2e} "
          f"max|FOC_g|={np.max(np.abs(og['FOC_g'])):.2e} (training loss ~5e-5)", flush=True)
    # readout at the calibrated initial point
    o0 = evaluate(v_nn, i_d_nn, i_g_nn, np.array([0.7]), np.array([3.0]))
    print(f"[at Z=0.7,Y=3.0] i_d={o0['i_d'][0]:+.4f} i_g={o0['i_g'][0]:+.4f} "
          f"V_Y={o0['V_Y'][0]:+.4f} q_d={o0['q_d'][0]:.4f} q_g={o0['q_g'][0]:.4f} c={o0['c'][0]:.4f} "
          f"|FOC_d|={abs(o0['FOC_d'][0]):.2e}", flush=True)


if __name__ == "__main__":
    main()
