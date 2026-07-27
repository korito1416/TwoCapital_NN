"""RCT error figure: per jump state, error in the HJB equation (path RMS, 60-yr
average protocol) across xi, one curve per arm. Same color = same warm-start
method; solid/dashed = the two seeds; gray dotted = reference (base run).
Input: data/cs_warmstart.npy written by solution_comparison/eval_path.py.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSFILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "cs_warmstart.npy")
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
cs = np.load(os.path.join(HERE, CSFILE), allow_pickle=True).item()
D, XIS, REGS = cs["data"], cs["XIS"], cs["REGS"]

METHODS = [("cold", "#c1272d"), ("nber", "#1f4e79"), ("perturb", "#2e7d32"),
           ("analytic", "#e65100"), ("anchor", "#6a1b9a")]
TITLES = {"PreDamagePreTech": "pre-damage, pre-tech",
          "PreDamagePostTech": "pre-damage, post-tech",
          "PostDamagePreTech": "post-damage, pre-tech",
          "PostDamagePostTech": "post-damage, post-tech"}

def curve(lab, reg, key="res"):
    ys = []
    for xi in XIS:
        v = D.get(f"{reg}|{lab}|{xi}", {}).get(key)
        ys.append(np.nan if v is None or v > 1e5 else v)
    return np.array(ys)

plt.rcParams.update({"font.size": 17, "axes.linewidth": 1.1,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})
fig, axes = plt.subplots(1, 4, figsize=(24, 6.0), sharey=True)
for k, reg in enumerate(REGS):
    a = axes[k]
    a.plot(XIS, curve("RUNA(base)", reg), color="0.25", lw=3.2, ls=":",
           label="reference (base run)")
    for meth, col in METHODS:
        for s, ls in [(1, "-"), (2, "--")]:
            y = curve(f"{meth}_s{s}", reg)
            if np.all(np.isnan(y)):
                continue
            a.plot(XIS, y, color=col, lw=2.0, ls=ls, marker="o", ms=4,
                   label=meth if (s == 1 and k == 0) else None)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylim(2e-4, 0.5)   # analytic's 1e4 blow-up exits the top; stated in the caption
    a.set_title(TITLES[reg], fontsize=17)
    a.set_xlabel(r"$\xi$"); a.grid(alpha=.25, lw=.6, which="both")
axes[0].set_ylabel("error in the HJB equation (path RMS)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "figures", f"rct_errors_by_state{SUFFIX}.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
