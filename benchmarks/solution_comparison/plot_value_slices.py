"""Value function one state variable at a time, other coordinates held at the
initial state (K=880, Z=0.7, Y=1.2, R=11.2). Rows = xi values, columns = the
four state variables; curves = runs. Pre-damage/pre-technology jump state."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import solution_loader as SL

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True, help="label=/abs/run/root")
ap.add_argument("--xis", default="0.05,148.6")
ap.add_argument("--reg", default="PreDamagePreTech")
ap.add_argument("--out", default="value_slices.png")
args = ap.parse_args()

RUNS = SL.parse_runs(args.run)
XIS = [float(x) for x in args.xis.split(",")]
BASE = dict(logK=np.log(880.0), Z=0.7, Y=1.2, logR=np.log(11.2))
COORDS = [("logK", 4, 7, r"$\log K$"), ("Z", 0.01, 0.99, r"$Z$"),
          ("Y", 0, 4, r"$Y$"), ("logR", 1, 6, r"$\log R$")]
COLORS = ["#c1272d", "#1f4e79", "#2e7d32", "#e65100", "#6a1b9a", "#00838f", "#795548", "#546e7a"]
M = 200

def slice_v(net, coord, lo, hi, xi):
    x = np.linspace(lo, hi, M)
    st = {k: np.full((M, 1), v, dtype=np.float32) for k, v in BASE.items()}
    st[coord] = x.reshape(-1, 1).astype(np.float32)
    lx = tf.constant(np.full((M, 1), np.log(xi), dtype=np.float32))
    X = SL.build_X(args.reg, tf.constant(st["logK"]), tf.constant(st["Z"]),
                   tf.constant(st["Y"]), tf.constant(st["logR"]), lx)
    return x, net(X, training=False).numpy().ravel()

NETS = {lab: SL.load_v(root, args.reg) for lab, root in RUNS}
plt.rcParams.update({"font.size": 17, "axes.linewidth": 1.0, "xtick.labelsize": 14, "ytick.labelsize": 14})
fig, axes = plt.subplots(len(XIS), 4, figsize=(18, 4.3 * len(XIS)), squeeze=False)
for r, xi in enumerate(XIS):
    for c, (coord, lo, hi, lab_c) in enumerate(COORDS):
        a = axes[r, c]
        for i, (lab, _) in enumerate(RUNS):
            x, y = slice_v(NETS[lab], coord, lo, hi, xi)
            a.plot(x, y, color=COLORS[i % 8], lw=2.2, label=lab)
        a.grid(alpha=.25, lw=.5); a.set_xlim(lo, hi)
        a.set_xlabel(lab_c + r",  $\xi=$" + f"{xi:g}")
    axes[r, 0].set_ylabel(r"$V$", fontsize=19)
axes[0, 0].legend(frameon=False, fontsize=14, loc="upper left")
fig.tight_layout()
fig.savefig(args.out, dpi=140, bbox_inches="tight")
print("wrote", args.out)
