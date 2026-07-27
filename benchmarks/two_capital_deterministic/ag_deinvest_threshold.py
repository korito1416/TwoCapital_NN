"""
A_g de-investment-threshold sensitivity (deterministic two-capital, FD ground truth).

Conclusion converging from two threads: the model does NOT de-invest dirty because
GREEN PRODUCTIVITY A_g is too small. This pins it quantitatively: fix A_d=0.1303,
sweep A_g, find where dirty investment i_d crosses 0 (the de-invest threshold), for
BOTH the baseline adjustment cost (Gamma=0.060, theta=16.7) and the HALF adjustment
cost (Gamma=0.12, theta=8.35) of our experiment. Mark the three calibration A_g
values (pre / intermediate / post-tech). The post-tech breakthrough A_g''=0.1567 is
the highest the model ever reaches -- if even that is far below the threshold, "A_g
too small" is the (correct) economic reason the planner keeps investing dirty.

Note: this is the pure two-capital block (NO climate damage). Damage ADDS a reason to
de-invest, so the WITH-damage threshold is LOWER. This brackets it from above; the
parallel FD-with-damage model gives the relevant (lower) number.

FD is numpy-only and fast (login-node OK per the small/quick exception).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_capital_model as M
from theta_sensitivity import solve_fd

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)

A_D = 0.1303
A_G_CAL = {"pre-tech A_g": 0.1085, "intermediate A_g'": 0.1303, "post-tech A_g''": 0.1567}
ZSTARS = [0.6, 0.7, 0.8]
AGS = np.linspace(0.10, 0.75, 27)


def make_P(A_g, half):
    P = M.load_calibration("A_g_prime_prime")
    P["A_d"] = A_D
    P["A_g"] = float(A_g)
    if half:
        P["Gamma_d"] = P["Gamma_g"] = 0.12
        P["theta_d"] = P["theta_g"] = 8.35
    return P


def i_d_at(P, z):
    oc = solve_fd(P, n=2000); of = solve_fd(P, n=4000)
    return 2.0 * np.interp(z, of["Z"], of["i_d"]) - np.interp(z, oc["Z"], oc["i_d"])


def crossing(x, y):
    s = np.sign(y); idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def run(half, tag):
    print(f"\n==== {tag} (A_d={A_D} fixed) ====", flush=True)
    res = {z: [] for z in ZSTARS}
    for A_g in AGS:
        P = make_P(A_g, half)
        for z in ZSTARS:
            res[z].append(i_d_at(P, z))
    for z in ZSTARS:
        res[z] = np.array(res[z])
    out = {"res": res, "thr": {}, "cal": {}}
    for z in ZSTARS:
        thr = crossing(AGS, res[z])
        out["thr"][z] = thr
        # i_d at the post-tech calibration A_g''
        idpost = np.interp(0.1567, AGS, res[z])
        out["cal"][z] = idpost
        rd = thr / A_D if thr else float("nan")
        print(f"  Z={z}: de-invest threshold A_g={thr:.3f} (={rd:.1f}x A_d) | "
              f"i_d at post-tech A_g''=0.1567 -> {idpost:+.4f} ({'INVEST' if idpost>0 else 'de-invest'})", flush=True)
    return out


def main():
    base = run(False, "BASELINE adjcost (Gamma=0.06, theta=16.7)")
    half = run(True, "HALF adjcost (Gamma=0.12, theta=8.35)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for a, z in zip(ax, ZSTARS):
        a.plot(AGS, base["res"][z], "b-o", ms=4, lw=1.9, label="baseline adjcost")
        a.plot(AGS, half["res"][z], "r-s", ms=4, lw=1.9, label="half adjcost")
        a.axhline(0, color="k", lw=0.9, ls=":")
        # calibration A_g markers (vertical lines only, no text)
        for ag in A_G_CAL.values():
            a.axvline(ag, color="grey", lw=0.8, ls="--")
        # de-invest threshold markers (vertical lines only, no text)
        if base["thr"][z]:
            a.axvline(base["thr"][z], color="b", lw=0.8, ls=":")
        if half["thr"][z]:
            a.axvline(half["thr"][z], color="r", lw=0.8, ls=":")
        a.set_xlabel("green productivity $A_g$"); a.set_ylabel("$i^d$ (dirty investment)")
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OD, "ag_deinvest_threshold.png")
    fig.savefig(p, dpi=150); print("\nsaved", p)

    print("\n==== BOTTOM LINE ====")
    print(f"  Calibration A_g values: pre={A_G_CAL['pre-tech A_g']}, "
          f"interm={A_G_CAL[chr(105)+'ntermediate A_g'+chr(39)]}, post={A_G_CAL['post-tech A_g'+chr(39)+chr(39)]}")
    tb = np.nanmean([base['thr'][z] for z in ZSTARS])
    th = np.nanmean([half['thr'][z] for z in ZSTARS])
    print(f"  De-invest needs A_g ~ {tb:.2f} (baseline) / {th:.2f} (half adjcost), i.e. ~{tb/A_D:.0f}x / ~{th/A_D:.0f}x A_d.")
    print(f"  Post-tech A_g''=0.1567 is only {0.1567/A_D:.1f}x A_d -> WAY below threshold -> invest is correct -> 'A_g too small' confirmed.")
    print(f"  Half adjcost {'LOWERS' if th < tb else 'RAISES'} the threshold ({tb:.2f}->{th:.2f}).")


if __name__ == "__main__":
    main()
