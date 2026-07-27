"""Value function and marginal values by jump state along the simulated path
(the 'four states' grid): rows = V, V_logK, V_Z, V_logR; columns = jump states;
curves = runs. Post-damage columns evaluated at entry temperature Y = 2.5.
Row-shared auto y-limits."""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import solution_loader as SL

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True, help="label=/abs/run/root")
ap.add_argument("--paths-from", required=True)
ap.add_argument("--xi", type=float, default=148.6)
ap.add_argument("--path-xi-dir", default="148.600")
ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--out", default="four_states.png")
args = ap.parse_args()

RUNS = SL.parse_runs(args.run)
lk, Z, Y, lr = SL.path_states(os.path.abspath(args.paths_from), args.path_xi_dir, args.stride)
tt = np.arange(len(lk)) * args.stride / 12.0  # monthly steps -> years
COLORS = ["0.1", "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#6a1b9a", "#795548", "#546e7a"]

def vals(net, reg, Yr):
    lkT, ZT, YT, lrT = (tf.constant(a) for a in (lk, Z, Yr, lr))
    lx = tf.constant(np.full_like(lk, np.log(args.xi)))
    with tf.GradientTape(persistent=True) as tp:
        tp.watch([lkT, ZT, lrT])
        v = net(SL.build_X(reg, lkT, ZT, YT, lrT, lx), training=False)
    dK, dZ = tp.gradient(v, lkT), tp.gradient(v, ZT)
    dR = tp.gradient(v, lrT) if SL.HASR[reg] else None
    del tp
    return (v.numpy().ravel(), dK.numpy().ravel(), dZ.numpy().ravel(),
            None if dR is None else dR.numpy().ravel())

rows = [("V", r"$V$"), ("VK", r"$V_{\log K}$"), ("VZ", r"$V_Z$"), ("VR", r"$V_{\log R}$")]
allser = {}
for reg in SL.REGS:
    Yr = Y if reg.startswith("PreDamage") else np.full_like(Y, SL.Y_ENTRY)
    per = {}
    for lab, root in RUNS:
        # root may be "r1|r2|..." -> average the series across runs (seed mean)
        vs = [vals(SL.load_v(rt, reg), reg, Yr) for rt in root.split("|")]
        agg = lambda i: (None if vs[0][i] is None else np.mean([v[i] for v in vs], axis=0))
        per[lab] = {"V": agg(0), "VK": agg(1), "VZ": agg(2), "VR": agg(3)}
    allser[reg] = per

plt.rcParams.update({"font.size": 16, "axes.linewidth": 1.0, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5})
fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
rowlim = {}
for key, _ in rows:
    vv = [s[key] for reg in SL.REGS for s in allser[reg].values() if s[key] is not None]
    lo, hi = min(v.min() for v in vv), max(v.max() for v in vv)
    pad = 0.06 * (hi - lo)
    rowlim[key] = (lo - pad, hi + pad)
for c, reg in enumerate(SL.REGS):
    for r, (key, ylab) in enumerate(rows):
        a = axes[r, c]
        if allser[reg][RUNS[0][0]][key] is None:
            a.axis("off"); continue
        for i, (lab, _) in enumerate(RUNS):
            a.plot(tt, allser[reg][lab][key], color=COLORS[i % 8], lw=2.0, label=lab)
        a.grid(alpha=.25, lw=.5); a.set_xlim(0, tt[-1]); a.set_ylim(*rowlim[key])
        if r == 0: a.set_title(SL.RLAB[reg], fontsize=16.5)
        if c == 0: a.set_ylabel(ylab, fontsize=18)
        else: a.set_yticklabels([])
        if r == 3 or (r == 2 and allser[reg][RUNS[0][0]]["VR"] is None):
            a.set_xlabel("year")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.015))
fig.tight_layout()
fig.savefig(args.out, dpi=135, bbox_inches="tight")
print("wrote", args.out)
