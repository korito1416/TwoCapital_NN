"""Arbitration figure: (left) realized discounted utility vs horizon per delivered
solution, with each solution's claimed V(x0) as a dashed line in its color;
(right) claimed level vs realized welfare at the 60-year horizon — the inversion.
Data: benchmarks/policy_rollout/rollout_h_xi148.npz (+ rollout_ext_xi148.npz).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PR = os.path.join(HERE, "..", "policy_rollout")
d1 = np.load(os.path.join(PR, "rollout_h_xi148.npz"))
d2 = np.load(os.path.join(PR, "rollout_ext_xi148.npz"))

def get(d, lab, key):
    k = f"{lab}|{key}"
    return d[k] if k in d.files else None

HORIZONS = [60, 100, 200, 400]
SOLS = [  # label, npz, color, display
    ("reference",     d1, "0.15",    "reference"),
    ("perturb_s1",    d1, "#2e7d32", "warm start B"),
    ("nber_s1",       d1, "#1f4e79", "warm start A"),
    ("paperlr2M_s1",  d2, "#00838f", "cold start"),
]

plt.rcParams.update({"font.size": 14.5, "axes.linewidth": 1.0})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})

for lab, d, col, disp in SOLS:
    js = [float(np.mean(get(d, lab, f"J{h}"))) for h in HORIZONS]
    v0 = float(get(d, lab, "V0"))
    ax.plot(HORIZONS, js, "-o", color=col, lw=2.2, ms=6, label=disp)
    ax.axhline(v0, color=col, ls="--", lw=1.3, alpha=0.8)
ax.set_ylim(-3.5, 6.2)
ax.set_xlabel("horizon (years)")
ax.set_ylabel("realized discounted utility (truncated at horizon)")
ax.legend(frameon=False, fontsize=12.5, loc="lower left")
ax.grid(alpha=0.22, lw=0.6)
ax.text(62, 5.75, "dashed: each solution's claimed $V(x_0)$", fontsize=11.5, color="0.3")

# right panel: claimed vs realized at 60y, all seeds
PTS = [("reference", d1, "0.15", "*", 320), ("perturb_s1", d1, "#2e7d32", "o", 110),
       ("perturb_s2", d1, "#2e7d32", "o", 110), ("nber_s1", d1, "#1f4e79", "o", 110),
       ("nber_s2", d1, "#1f4e79", "o", 110), ("paperlr2M_s1", d2, "#00838f", "D", 100),
       ("paperlr2M_s2", d2, "#00838f", "D", 100), ("paperlr2M_s3", d2, "#00838f", "D", 100),
       ("nberperturb_s1", d2, "#1f4e79", "s", 100), ("noreheat_s1", d2, "#2e7d32", "P", 120)]
seen = set()
for lab, d, col, mk, sz in PTS:
    v0 = get(d, lab, "V0"); j60 = get(d, lab, "J60")
    if v0 is None or j60 is None:
        continue
    fam = (col, mk)
    bx.scatter([float(v0)], [float(np.mean(j60))], c=col, marker=mk, s=sz,
               edgecolors="white", linewidths=1.0, zorder=3,
               label={"0.15": "reference", "#2e7d32": "warm start B family", "#1f4e79": "warm start A family",
                      "#00838f": "cold start"}[col] if col not in seen else None)
    seen.add(col)
bx.set_xlabel("claimed welfare level $V(x_0)$")
bx.set_ylabel("realized welfare, 60-year horizon")
bx.grid(alpha=0.22, lw=0.6)
bx.legend(frameon=False, fontsize=11.5, loc="upper right")
fig.tight_layout()
out = os.path.join(HERE, "figures", "arbiter.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
