"""CS1/CS2 figures from an eval_path.py / eval_lhs.py .npy: HJB error by jump
state (2x2) and FOC errors by jump state (3x4), curves = runs, x = xi."""
import argparse, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solution_loader as SL

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out-prefix", default="cs")
ap.add_argument("--xmin", type=float, default=None)
args = ap.parse_args()

D = np.load(args.data, allow_pickle=True).item()
XIS = np.array(D["XIS"]); LABELS = D["labels"]; data = D["data"]
KEEP = XIS >= (args.xmin if args.xmin else -np.inf)
X = XIS[KEEP]
COLORS = ["#c1272d", "#1f4e79", "#2e7d32", "#e65100", "#6a1b9a", "#00838f", "#795548", "#546e7a"]
MARKS = ["o", "s", "^", "v", "D", "P", "X", "*"]

def get(reg, lab, key):
    return np.array([data[f"{reg}|{lab}|{xi}"][key] for xi in XIS], dtype=float)[KEEP]

plt.rcParams.update({"font.size": 17, "axes.linewidth": 1.1, "xtick.labelsize": 14, "ytick.labelsize": 14})

# ---- CS1: HJB error by jump state
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
for k, reg in enumerate(SL.REGS):
    a = axes.flat[k]
    for i, lab in enumerate(LABELS):
        v = get(reg, lab, "res"); m = np.isfinite(v)
        a.loglog(X[m], v[m], MARKS[i % 8] + "-", color=COLORS[i % 8], lw=2.2, ms=6.5, label=lab)
    a.grid(alpha=.22, lw=.6, which="major")
    a.set_title(SL.RLAB[reg], fontsize=17)
    a.set_ylabel("error in the HJB equation"); a.set_xlabel(r"$\xi$")
axes.flat[0].legend(frameon=False, fontsize=13, loc="best")
fig.tight_layout()
fig.savefig(f"{args.out_prefix}1_hjb_by_state.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- CS2: FOC errors by jump state
rows = [("FOC_d", r"dirty FOC$_d$"), ("FOC_g", r"green FOC$_g$"), ("FOC_r", r"R&D FOC$_r$")]
fig, axes = plt.subplots(3, 4, figsize=(19, 12), sharey="row", sharex=True)
for r, (key, ylab) in enumerate(rows):
    for c, reg in enumerate(SL.REGS):
        a = axes[r, c]
        series = [(lab, get(reg, lab, key)) for lab in LABELS]
        if all(not np.isfinite(v).any() for _, v in series):
            a.axis("off"); continue
        for i, (lab, v) in enumerate(series):
            m = np.isfinite(v)
            a.loglog(X[m], v[m], MARKS[i % 8] + "-", color=COLORS[i % 8], lw=2.0, ms=6, label=lab)
        a.grid(alpha=.20, lw=.6, which="major")
        if r == 0: a.set_title(SL.RLAB[reg], fontsize=17)
        if c == 0: a.set_ylabel(ylab + " error")
        a.set_xlabel(r"$\xi$")
axes[0, 0].legend(frameon=False, fontsize=12, loc="best")
fig.tight_layout()
fig.savefig(f"{args.out_prefix}2_foc_by_state.png", dpi=135, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{args.out_prefix}1_hjb_by_state.png", f"{args.out_prefix}2_foc_by_state.png")
