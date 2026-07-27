"""Training-loss composition figures (percent + levels) for the four jump states,
one row per run, from each run's training_history.csv. NOTE: each run's loss is
an average under its OWN sampling distribution — levels are only comparable
across runs whose sampled ranges are identical (see README.md)."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solution_loader as SL

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True, help="label=/abs/run/root")
ap.add_argument("--window", type=int, default=100000, help="last-N-steps window")
ap.add_argument("--out-prefix", default="loss4")
args = ap.parse_args()

RUNS = SL.parse_runs(args.run)
comps = ["loss_v", "loss_FOC_d", "loss_FOC_g", "loss_FOC_r", "loss_dv_dY"]
cols = ["residual (loss$_v$)", "FOC$_d$", "FOC$_g$", "FOC$_r$", "monotonicity"]
colr = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c8c8c"]

def mags(csv):
    df = pd.read_csv(csv)
    end = df["step"].max()
    df = df[df["step"] >= end - args.window].copy()
    M = np.zeros((len(df), len(comps)))
    for j, c in enumerate(comps):
        if c in df.columns:
            M[:, j] = np.clip(df[c].to_numpy(), 0, None)
    w = 6
    M = np.vstack([np.convolve(M[:, j], np.ones(w) / w, mode="same") for j in range(M.shape[1])]).T
    x = (df["step"].to_numpy() - (end - args.window)) / 1e3
    return x[w:-w], M[w:-w]

def mags_avg(root, reg):
    """root may be 'r1|r2|...' -> average the smoothed magnitude series across runs
    (seed mean; runs within a setting share the step grid)."""
    parts = [mags(f"{rt}/{reg}/training_history.csv") for rt in root.split("|")]
    n = min(len(x) for x, _ in parts)
    return parts[0][0][:n], np.mean([M[:n] for _, M in parts], axis=0)

DATA = {(lab, reg): mags_avg(root, reg) for lab, root in RUNS for reg in SL.REGS}
R = len(RUNS)
plt.rcParams.update({"font.size": 19, "xtick.labelsize": 15, "ytick.labelsize": 15})

for mode in ["percent", "levels"]:
    fig, axes = plt.subplots(R, 4, figsize=(18, 4.75 * R), sharey=True, sharex=True, squeeze=False)
    if mode == "levels":
        ymax = 1.05 * max(M.sum(1).max() for _, M in DATA.values())
    for r, (lab, _) in enumerate(RUNS):
        for c, reg in enumerate(SL.REGS):
            x, M = DATA[(lab, reg)]
            S = 100 * M / M.sum(1, keepdims=True) if mode == "percent" else M
            a = axes[r, c]
            a.stackplot(x, S.T, colors=colr, labels=cols)
            a.set_xlim(0, args.window / 1e3)
            a.set_ylim(0, 100 if mode == "percent" else ymax)
            if mode == "levels":
                a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            if r == 0: a.set_title(SL.RLAB[reg], fontsize=18)
            if r == R - 1: a.set_xlabel(f"last {args.window // 1000}k steps")
        axes[r, 0].set_ylabel(lab + ("\nshare of total loss (%)" if mode == "percent" else "\nloss terms"))
    h, l = axes[0, 3].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=16,
               bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout()
    out = f"{args.out_prefix}_{mode}.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
