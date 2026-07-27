"""Stochastic-IRF figures on the coupled worst-case simulator: one figure per structural shock,
panels = response of each level quantity, xi curves overlaid. Response = perturbed - baseline (levels)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XI_COLORS = ["#d1495b", "#edae49", "#1f6feb"]     # 0.05, 0.1, 148.6
PANELS = [("Y", "temperature  ΔY"), ("E", "emissions  Δℰ"), ("C", "consumption  ΔC"),
          ("I_g", "green investment  ΔI_g"), ("I_d", "dirty investment  ΔI_d"), ("RD", "R&D  Δ(I_r/Y)"),
          ("A_g", "green productivity  ΔA_g"), ("K", "capital  ΔK"), ("posttech", "post-breakthrough  Δshare")]
SHOCK_TITLE = {"DirtyCapital": "dirty-capital", "GreenCapital": "green-capital",
               "Temperature": "temperature", "Knowledge": "knowledge (R&D)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--measure", default="worst-case (robust)")
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True)
    xis = [f"{x:g}" for x in d["xis"]]
    shocks = [str(s) for s in d["shocks"]]
    outdir = Path(a.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    for shock in shocks:
        t = d[f"{xis[0]}::t"]
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        for ax, (q, title) in zip(axes.ravel(), PANELS):
            for ci, xi in enumerate(xis):
                key = f"{xi}::irf_{shock}_{q}"
                if key not in d.files:
                    continue
                ax.plot(t, d[key], color=XI_COLORS[ci % 3], lw=2, label=f"$\\xi$={xi}")
            ax.axhline(0, color="k", lw=0.6, alpha=0.5)
            ax.set_title(title, fontsize=12); ax.set_xlabel("year"); ax.grid(alpha=0.3)
        axes.ravel()[0].legend(fontsize=10, loc="best")
        fig.suptitle(f"Stochastic IRF to a {SHOCK_TITLE.get(shock, shock)} shock  —  {a.measure} coupled economy",
                     fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = outdir / f"irf_{shock}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
