"""Stochastic companion of the investment-path figure: mean line + 10-90% band per
setting under Brownian shocks (jumps off, the same no-jump scenario as the
deterministic figure). Paths pooled across replicates before taking quantiles.
Data: benchmarks/policy_rollout/paths_stoch_xi0p05.npz (rollout --record-stride).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "..", "policy_rollout", "paths_stoch_xi0p05.npz"))

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
fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0))
for k, (q, ylab) in enumerate(QUANT):
    a = axes[k]
    for lab, col, runs in SETTINGS:
        X = np.concatenate([D[f"{r}|rec_{q}"] for r in runs], axis=1)  # (T, pooled paths)
        mean = X.mean(axis=1)
        lo, hi = np.percentile(X, 10, axis=1), np.percentile(X, 90, axis=1)
        a.plot(t, mean, color=col, lw=2.4, label=(lab if k == 0 else None))
        a.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
    a.set_xlabel("year"); a.set_ylabel(ylab); a.grid(alpha=.25, lw=.6)
    a.set_xlim(0, 60)
axes[0].legend(frameon=False, fontsize=12.5, loc="upper left")
fig.tight_layout()
out = os.path.join(HERE, "figures", "policy_fan_stochastic_xi0p05.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
