"""Haoyang-style path figures with the xi curves overlaid, under the worst-case measure.
Multi-panel: temperature, emissions, green productivity (breakthrough), R&D, green/dirty investment
(LEVELS, per the quantity dictionary), consumption, knowledge, and the post-breakthrough share.
Band (10/90) drawn for the most-averse xi. Optionally overlay the reference (physical) run dashed."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# blue=most averse ... to neutral; consistent, colour-blind-safe
XI_COLORS = ["#d1495b", "#edae49", "#1f6feb"]   # 0.05, 0.1, 148.6

PANELS = [
    ("Y",   "temperature  $Y$"),
    ("E",   "emissions  $\\mathcal{E}$"),
    ("A_g", "green productivity  $A_g$ (breakthrough)"),
    ("RD",  "R&D  $I_r/Y$"),
    ("I_g", "green investment  $I_g$ (level)"),
    ("I_d", "dirty investment  $I_d$ (level)"),
    ("C",   "consumption  $C$ (level)"),
    ("R",   "knowledge  $R$"),
    ("posttech", "post-breakthrough share"),
]


def load(npz):
    d = np.load(npz, allow_pickle=True)
    xis = [f"{x:g}" for x in d["xis"]]
    ref = bool(int(d["reference"][0]))
    return d, xis, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="worst-case npz from run_robust_paths.py")
    ap.add_argument("--reference-data", default=None, help="optional reference npz to overlay dashed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, xis, _ = load(a.data)
    dref = None
    if a.reference_data:
        dref = np.load(a.reference_data, allow_pickle=True)

    t = d[f"{xis[0]}::t"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        for ci, xi in enumerate(xis):
            col = XI_COLORS[ci % len(XI_COLORS)]
            if key == "posttech":
                # regime shares over time: PreDPostT(2)+PostDPostT(3)
                r2 = d[f"{xi}::regime2"][:, 0]; r3 = d[f"{xi}::regime3"][:, 0]
                ax.plot(t, r2 + r3, color=col, lw=2, label=f"$\\xi$={xi}")
                if dref is not None:
                    rr = dref[f"{xi}::regime2"][:, 0] + dref[f"{xi}::regime3"][:, 0]
                    ax.plot(t, rr, color=col, lw=1.2, ls="--", alpha=0.7)
                continue
            arr = d[f"{xi}::{key}"]                     # (T,3): mean,p10,p90
            ax.plot(t, arr[:, 0], color=col, lw=2, label=f"$\\xi$={xi}")
            if ci == 0 and key not in ("A_g", "posttech"):   # band on smooth quantities only (skip bimodal)
                ax.fill_between(t, arr[:, 1], arr[:, 2], color=col, alpha=0.13, lw=0)
            if dref is not None:
                ax.plot(t, dref[f"{xi}::{key}"][:, 0], color=col, lw=1.2, ls="--", alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("year"); ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=10, loc="best")
    sub = "worst-case (robust) measure"
    if dref is not None:
        sub += "  ·  dashed = reference (physical) measure"
    fig.suptitle(f"Paths by uncertainty aversion $\\xi$ — {sub}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
