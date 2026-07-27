#!/usr/bin/env python3
"""Overlay the per-xi stochastic-IRF NPZs (from stochastic_irf.py) into the report figures:
  - one 6-panel STATE/economic IRF figure per shock (response by xi, band on most-averse xi);
  - one PRICED marginal-value decomposition figure (flow i/ii/iii per shock, by xi).
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def smooth(y, w=5):
    y = np.asarray(y, float)
    if w <= 1 or y.size < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")

SHOCKS = ["Capital", "GreenShare", "Temperature", "Technology"]
SHOCK_LAB = {"Capital": "log-capital", "GreenShare": "green share $Z$",
             "Temperature": "temperature $Y$", "Technology": "log-R&D"}
RESP = [("Y", "temperature $Y$"), ("E", "emissions $\\mathcal{E}$"), ("C", "consumption $C$"),
        ("i_g", "green invest rate $i_g$"), ("i_d", "dirty invest rate $i_d$"), ("logR", "log-R&D")]
FLOWS = [("flow_i", "flow i: direct utility", "#999999"),
         ("flow_ii_tech", "flow ii: tech-jump intensity", "#0072B2"),
         ("flow_iii_tech", "flow iii: post-tech value", "#56B4E9"),
         ("flow_ii_damage", "flow ii: damage-jump intensity", "#D55E00"),
         ("flow_iii_damage", "flow iii: post-damage value", "#E69F00")]
PALETTE = ["#D55E00", "#E69F00", "#0072B2", "#000000", "#009E73"]


def load(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "irf_xi_*.npz")),
                   key=lambda f: float(Path(f).stem.split("_")[-1]))
    out = []
    for f in files:
        d = dict(np.load(f))
        out.append((float(d["xi"]), d))
    return out


def plot_state(runs, shock, out_path):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.2)); axf = ax.ravel()
    for j, (key, lab) in enumerate(RESP):
        a = axf[j]
        for i, (xi, d) in enumerate(runs):
            t = d["t"]; arr = d[f"resp_{shock}_{key}"]
            c = PALETTE[i % len(PALETTE)]
            a.plot(t, smooth(arr[:, 0]), color=c, lw=2.2, label=f"$\\xi={xi:g}$" if j == 0 else None)
            if i == 0:
                a.fill_between(t, smooth(arr[:, 1]), smooth(arr[:, 2]), color=c, alpha=0.15, lw=0)
        a.axhline(0, color="k", lw=0.6); a.set_title(lab, fontsize=12)
        a.set_xlabel("year"); a.grid(alpha=.25); a.set_xlim(0, t[-1])
    h, l = axf[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(runs), frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Stochastic impulse response to a {SHOCK_LAB[shock]} shock (robust measure, by $\\xi$)", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def plot_priced(runs, out_path):
    """4 shocks x (diverging-stacked flow decomposition per xi) with a clear net marker,
    consumption-equivalent units (/MU0)."""
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.9), sharey=False)
    xis = [xi for xi, _ in runs]; xpos = np.arange(len(xis))
    for s, shock in enumerate(SHOCKS):
        a = ax[s]
        pos = np.zeros(len(xis)); neg = np.zeros(len(xis))
        for fk, flab, col in FLOWS:
            vals = np.array([d[f"priced_{shock}_{fk}_mean"] / d["MU0"] for _, d in runs])
            base = np.where(vals >= 0, pos, neg)
            a.bar(xpos, vals, 0.62, bottom=base, color=col, edgecolor="white", linewidth=0.4)
            pos += np.where(vals >= 0, vals, 0.0); neg += np.where(vals < 0, vals, 0.0)
        tot = pos + neg
        span = max(1e-9, float(np.nanmax(pos) - np.nanmin(neg)))
        for xp, tv in zip(xpos, tot):
            a.plot([xp - 0.33, xp + 0.33], [tv, tv], color="k", lw=2.6, solid_capstyle="round", zorder=5)
            a.annotate(f"{tv:.0f}", (xp, tv), xytext=(0, 6 if tv >= 0 else -13),
                       textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
        a.margins(y=0.14)
        a.axhline(0, color="k", lw=0.6); a.set_xticks(xpos)
        a.set_xticklabels([f"$\\xi{{=}}{x:g}$" for x in xis]); a.set_title(SHOCK_LAB[shock], fontsize=12)
        a.grid(axis="y", alpha=.25)
        if s == 0:
            a.set_ylabel("marginal value  $DV\\!\\cdot e$  (cons.-equiv.)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in FLOWS] + [Line2D([0], [0], color="k", lw=2.6)]
    labels = [fl for _, fl, _ in FLOWS] + ["net total"]
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Priced stochastic response: marginal value of each shock, decomposed by flow  (by $\\xi$)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--fig-dir", default=None)
    a = p.parse_args()
    runs = load(a.data_dir)
    if not runs:
        raise SystemExit(f"no irf_xi_*.npz in {a.data_dir}")
    fig_dir = Path(a.fig_dir).resolve() if a.fig_dir else Path(a.data_dir).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("loaded xi:", [xi for xi, _ in runs])
    for shock in SHOCKS:
        plot_state(runs, shock, fig_dir / f"stoch_irf_{shock}.png")
    has_priced = any(k.startswith("priced_") for k in runs[0][1])
    if has_priced:
        plot_priced(runs, fig_dir / "stoch_irf_priced.png")
    else:
        print("(no priced_* keys -> skipping FK decomposition figure)")


if __name__ == "__main__":
    main()
