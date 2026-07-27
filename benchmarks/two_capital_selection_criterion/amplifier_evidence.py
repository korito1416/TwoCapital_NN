"""AMPLIFIER EVIDENCE (no new pre-damage solve needed).

The worst-case damage-curvature belief reweights the five lambda3 realizations by
w_l = softmax_l(-V^l / xi), where V^l is the post-damage-post-tech value at the jump
entry slice Y = y_upper = 2.5 (the pre-damage value V CANCELS in the normalized
weight). So the distorted belief depends ONLY on the five V^l and xi.

FD gives five consistently-leveled V^l (same PIBYS scheme, each level-pinned by the
delta-forward integral) -> a smooth, bounded, VERIFIABLE distorted belief at every xi,
down to xi=0.005. The NN's five V^l carry per-regime level noise that the 1/xi factor
amplifies, so its distorted belief drifts from the FD reference and breaks down as xi
approaches the documented xi* ~ 0.025 wall. The FD supplies the answer the NN cannot
self-certify: distance-to-FD is the missing criterion, made visible in the belief the
model is actually reporting.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OD = os.path.join(ROOT, "benchmarks", "post_damage_post_tech", "outputs")
sys.path.insert(0, os.path.join(ROOT, "models_warmstart"))
from feedforward_subnet import FeedForwardSubNet

TAGS = ["0000", "0083", "0167", "0250", "0333"]
LAM3 = np.array([0.0, 1/12, 1/6, 1/4, 1/3])
AGPP = 0.1567
REF = (np.log(880.0), 0.7, 2.5)     # entry slice Y = y_upper = 2.5
XIS = np.array([0.3, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005])

RUNA = os.path.join(ROOT, "output_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_"
    "#HiddenLayer_4_num_iterations1000000")
WARMA = os.path.join(ROOT, "output_warmstart_1M_20260714", "nber_s1")

def fd_Vl():
    out = []
    for t in TAGS:
        d = np.load(os.path.join(OD, f"fd_pdpt_v5_stable_lam3_{t}_xi148.npz"))
        f = RGI((d["logK"], d["Z"], d["Y"]), d["v"], bounds_error=False, fill_value=None)
        out.append(float(f(REF)))
    return np.array(out)

def load_v(root):
    cfg = {"num_hiddens": [32]*4, "use_bias": True, "activation": "swish",
           "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    n = FeedForwardSubNet(cfg); n(tf.zeros([1, 7]))
    n.load_weights(f"{root}/PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech").expect_partial()
    return n

def nn_Vl(root, xi):
    """The NN's five post-damage values at the entry slice, at THIS xi (logxi is an input)."""
    net = load_v(root); lx = np.log(xi)
    X = np.array([[REF[0], REF[1], REF[2], g, AGPP, lx, lx] for g in LAM3], dtype=np.float32)
    return net(tf.constant(X), training=False).numpy().ravel().astype(np.float64)

def belief(Vl, xi):
    w = np.exp(-(Vl - Vl.min()) / xi); return w / w.sum()

# ---- assemble distorted beliefs vs xi -------------------------------------------
V_fd = fd_Vl()
print(f"FD V^l spread at entry slice = {V_fd.max()-V_fd.min():.4f}")
series = {"FD (verified)": ("0.1", "-", np.array([belief(V_fd, xi) for xi in XIS]))}
for lab, col, root in [("reference NN", "#0072B2", RUNA), ("warm start A NN", "#D55E00", WARMA)]:
    B = []
    for xi in XIS:
        Vl = nn_Vl(root, xi)
        B.append(belief(Vl, xi))
        if xi == 0.1:
            print(f"{lab}: V^l spread @xi=0.1 = {Vl.max()-Vl.min():.4f}")
    series[lab] = (col, "--", np.array(B))

# ---- figure: distorted belief concentration (max weight) vs xi, + per-l at xi=0.05 --
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
for lab, (col, ls, B) in series.items():
    ax[0].plot(XIS, B.max(axis=1), "o"+ls, color=col, lw=2.4, ms=6, label=lab)
ax[0].axhline(0.2, color="0.6", lw=1, ls=":")  # uniform baseline max
ax[0].axvspan(XIS.min(), 0.025, color="#D55E00", alpha=0.10, lw=0)
ax[0].annotate("documented NN wall\n$\\xi^*\\approx0.025$", xy=(0.025, 0.55), fontsize=10.5, color="0.3")
ax[0].set_xscale("log"); ax[0].invert_xaxis()
ax[0].set_xlabel("ξ (log, more averse →)"); ax[0].set_ylabel("worst-case belief concentration (max $w_l$)")
ax[0].set_title("Distorted damage-curvature belief vs ξ"); ax[0].grid(alpha=.25, which="both", lw=.5)
ax[0].legend(fontsize=11)
# per-l beliefs at xi=0.05
i05 = int(np.argmin(np.abs(XIS - 0.05)))
x = np.arange(5); w = 0.26
for k, (lab, (col, ls, B)) in enumerate(series.items()):
    ax[1].bar(x + (k-1)*w, B[i05], w, color=col, alpha=0.8, label=lab)
ax[1].axhline(0.2, color="0.6", lw=1, ls=":", label="baseline (uniform)")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{g:.2f}" for g in LAM3])
ax[1].set_xlabel(r"$\lambda_3$ (damage curvature)"); ax[1].set_ylabel("distorted weight $w_l$")
ax[1].set_title("Worst-case belief at ξ = 0.05"); ax[1].grid(alpha=.25, axis="y", lw=.5)
ax[1].legend(fontsize=10.5)
fig.tight_layout()
out = os.path.join(HERE, "figures", "amplifier_evidence.png")
fig.savefig(out, dpi=160, bbox_inches="tight"); print("wrote", out)

print("\ndistorted-belief max weight vs xi:")
print(f"{'xi':>8} " + " ".join(f"{lab[:12]:>12}" for lab in series))
for j, xi in enumerate(XIS):
    print(f"{xi:>8} " + " ".join(f"{series[lab][2][j].max():12.3f}" for lab in series))
