"""Jumps-ON stochastic investment fan: mean line + 10-90% band per setting with the
full stochastic model (Brownian shocks AND damage/tech jump processes, baseline law).
Fourth panel: fraction of paths in which each jump has occurred by year t.
Data: benchmarks/policy_rollout/paths_stoch_jumps_xi0p05.npz (same CRN seed as the
no-jump run, so paths are comparable draw by draw).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "..", "policy_rollout", "paths_stoch_jumps_xi0p05.npz"))

SETTINGS = [
    ("reference",             "0.1",     ["reference"]),
    ("warm start A",          "#0072B2", ["nber_s1", "nber_s2"]),
    ("warm start B",          "#009E73", ["perturb_s1", "perturb_s2"]),
    ("warm start A, wider ξ", "#CC79A7", ["wlx_s1", "wlx_s2"]),
]
QUANT = [("Ig", "green investment $I_g$"), ("Id", "dirty investment $I_d$"),
         ("Ir", "R&D investment $I_r$")]

t = D["reference|rec_t"]
plt.rcParams.update({"font.size": 15, "axes.linewidth": 1.0})
fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.6))
for k, (q, ylab) in enumerate(QUANT):
    a = axes[k]
    for lab, col, runs in SETTINGS:
        X = np.concatenate([D[f"{r}|rec_{q}"] for r in runs], axis=1)
        a.plot(t, X.mean(axis=1), color=col, lw=2.4, label=(lab if k == 0 else None))
        a.fill_between(t, np.percentile(X, 10, axis=1), np.percentile(X, 90, axis=1),
                       color=col, alpha=0.18, lw=0)
    a.set_xlabel("year"); a.set_ylabel(ylab); a.grid(alpha=.25, lw=.6)
    a.set_xlim(0, 60)

a = axes[3]
for lab, col, runs in SETTINGS:
    tech = np.concatenate([D[f"{r}|rec_tech"] for r in runs], axis=1)
    dmg = np.concatenate([D[f"{r}|rec_dmg"] for r in runs], axis=1)
    a.plot(t, (tech == 2).mean(axis=1), color=col, lw=2.4)
    a.plot(t, (dmg == 1).mean(axis=1), color=col, lw=2.0, ls=":")
a.plot([], [], color="0.4", lw=2.4, label="technology jump")
a.plot([], [], color="0.4", lw=2.0, ls=":", label="damage jump")
a.legend(frameon=False, fontsize=12.5, loc="upper left")
a.set_xlabel("year"); a.set_ylabel("fraction of paths jumped")
a.set_xlim(0, 60); a.set_ylim(0, 1); a.grid(alpha=.25, lw=.6)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout()
out = os.path.join(HERE, "figures", "policy_fan_stochastic_jumps_xi0p05.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)

for lab, col, runs in SETTINGS:
    tech = np.concatenate([D[f"{r}|rec_tech"] for r in runs], axis=1)
    dmg = np.concatenate([D[f"{r}|rec_dmg"] for r in runs], axis=1)
    ig = np.concatenate([D[f"{r}|rec_Ig"] for r in runs], axis=1)
    print(f"{lab:24s} P(tech by 60y)={float((tech[-1]==2).mean()):.3f} "
          f"P(dmg by 60y)={float((dmg[-1]==1).mean()):.3f} Ig60 mean={float(ig[-1].mean()):.1f}")
