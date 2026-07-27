"""Stochastic companion of the four-states figure: welfare V and marginal values by
jump state, evaluated along the STOCHASTIC path states (Brownian shocks on, jumps
off), mean line + 10-90% band per setting; replicate paths pooled. Post-damage jump
states evaluated at their entry temperature Y = y_hat = 2.5, as in the deterministic
figure. Value nets and input layouts via solution_comparison/solution_loader.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "solution_comparison"))
import solution_loader as SL

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNA = os.path.join(ROOT, "output_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_"
    "#HiddenLayer_4_num_iterations1000000")
P1M = os.path.join(ROOT, "output_warmstart_1M_20260714")
WLX = os.path.join(ROOT, "output_warmstart_1M_xi0p005_20260714")
D = np.load(os.path.join(HERE, "..", "policy_rollout", "paths_stoch_xi0p05.npz"))
XI = 0.05

SETTINGS = [
    ("reference",             "0.1",     [("reference", RUNA)]),
    ("warm start A",          "#0072B2", [("nber_s1", f"{P1M}/nber_s1"), ("nber_s2", f"{P1M}/nber_s2")]),
    ("warm start B",          "#009E73", [("perturb_s1", f"{P1M}/perturb_s1"), ("perturb_s2", f"{P1M}/perturb_s2")]),
    ("warm start A, wider ξ", "#CC79A7", [("wlx_s1", f"{WLX}/nber_s1"), ("wlx_s2", f"{WLX}/nber_s2")]),
]
t = D["reference|rec_t"]
T, N = D["reference|rec_lk"].shape

def vals(root, reg, lk, Z, Y, lr):
    """V and marginal values at (flattened) states; returns arrays shaped like lk."""
    net = SL.load_v(root, reg)
    sh = lk.shape
    cols = [tf.constant(a.reshape(-1, 1).astype(np.float32)) for a in (lk, Z, Y, lr)]
    lx = tf.constant(np.full((lk.size, 1), np.log(XI), dtype=np.float32))
    lkT, ZT, YT, lrT = cols
    with tf.GradientTape(persistent=True) as tp:
        tp.watch([lkT, ZT, lrT])
        v = net(SL.build_X(reg, lkT, ZT, YT, lrT, lx), training=False)
    dK, dZ = tp.gradient(v, lkT), tp.gradient(v, ZT)
    dR = tp.gradient(v, lrT) if SL.HASR[reg] else None
    del tp
    r = lambda x: None if x is None else x.numpy().reshape(sh)
    return r(v), r(dK), r(dZ), r(dR)

rows = [("V", r"$V$"), ("VK", r"$V_{\log K}$"), ("VZ", r"$V_Z$"), ("VR", r"$V_{\log R}$")]
allser = {}
for reg in SL.REGS:
    per = {}
    for lab, col, runs in SETTINGS:
        acc = {k: [] for k, _ in rows}
        for rl, root in runs:
            lk, Z = D[f"{rl}|rec_lk"], D[f"{rl}|rec_Z"]
            Y, lr = D[f"{rl}|rec_Y"], D[f"{rl}|rec_lr"]
            Yr = Y if reg.startswith("PreDamage") else np.full_like(Y, SL.Y_ENTRY)
            v, dK, dZv, dR = vals(root, reg, lk, Z, Yr, lr)
            for k, x in zip(("V", "VK", "VZ", "VR"), (v, dK, dZv, dR)):
                if x is not None:
                    acc[k].append(x)
        per[lab] = {k: (np.concatenate(vv, axis=1) if vv else None) for k, vv in acc.items()}
    allser[reg] = per
    print("evaluated", reg)

plt.rcParams.update({"font.size": 16, "axes.linewidth": 1.0,
                     "xtick.labelsize": 13.5, "ytick.labelsize": 13.5})
fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
for c, reg in enumerate(SL.REGS):
    for r_i, (key, ylab) in enumerate(rows):
        a = axes[r_i, c]
        drawn = False
        for lab, col, _ in SETTINGS:
            X = allser[reg][lab][key]
            if X is None:
                continue
            a.plot(t, X.mean(axis=1), color=col, lw=2.2,
                   label=(lab if (r_i == 0 and c == 0) else None))
            a.fill_between(t, np.percentile(X, 10, axis=1), np.percentile(X, 90, axis=1),
                           color=col, alpha=0.18, lw=0)
            drawn = True
        if not drawn:
            a.axis("off")
            continue
        if r_i == 0:
            a.set_title(SL.RLAB[reg], fontsize=16)
        if c == 0:
            a.set_ylabel(ylab)
        if r_i == 3 or (r_i == 2 and allser[reg][SETTINGS[0][0]]["VR"] is None):
            a.set_xlabel("year")
        a.grid(alpha=.22, lw=.6)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.015))
fig.tight_layout()
out = os.path.join(HERE, "figures", "four_states_stochastic_xi0p05.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
