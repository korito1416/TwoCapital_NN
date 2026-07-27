"""
Green-productivity (A_g) sensitivity of the deterministic two-capital model
(no shocks), FD only — the symmetric mirror of ad_sensitivity.py.

Lars's follow-up: hold the dirty productivity fixed at the base calibration
A_d=0.1303 and instead sweep the GREEN productivity A_g, to see how big A_g must
become (i.e. how large a tech jump in A_g) before the planner DE-INVESTS in dirty
capital (i^d<0). Original adjustment-cost function throughout (theta=16.7,
Gamma=0.06, alpha=-0.035). Three fixed green shares Z=0.6, 0.7, 0.8.

Economics / symmetry test. In ad_sensitivity.py (A_d swept, A_g=0.1567 fixed) the
dirty investment i^d crossed 0 at A_d*~0.073, i.e. at the ratio A_d*/A_g~0.466
(A_g ~ 2.1x A_d at the crossing). The symmetric conjecture is that holding
A_d=0.1303 fixed, i^d should turn negative once A_g exceeds ~A_d/0.466 ~ 0.28
(~2.1x A_d). We also report where the GREEN investment i^g crosses 0 (expected near
A_g ~ 0.466*A_d ~ 0.061), so the figure shows both crossings and the symmetric
point A_g=A_d.

Investment may be NEGATIVE (de-investment allowed); only K>=0 (1+theta*i>=0) is
enforced (slope clamp inside solve_fd). i^d, i^g at each target Z are Richardson-
extrapolated from grids n=2000,4000 to remove the O(dZ) upwind bias.

Runs on the login node (numpy/scipy only): module load python/anaconda-2021.05.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_capital_model as M
from theta_sensitivity import solve_fd, at_Z

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)

A_D = 0.1303                              # dirty productivity held FIXED (base calib)
AGS = np.linspace(0.05, 0.65, 31)         # green productivity sweep (wide: crossing ~0.5)
ZSTARS = [0.6, 0.7, 0.8]


def make_P(A_g):
    """Base calibration (original adjustment cost) with A_d fixed, A_g varied."""
    P = M.load_calibration("A_g_prime_prime")  # gives the base dict; we overwrite A's
    P["A_d"] = A_D
    P["A_g"] = float(A_g)
    return P


def richardson(P, z, out_c, out_f, key):
    fc = at_Z(out_c, key, z)
    ff = at_Z(out_f, key, z)
    return 2.0 * ff - fc


def zero_crossing(x, y):
    """First sign change of y(x), linearly interpolated; None if no crossing."""
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def main():
    res = {z: {"i_d": [], "i_g": []} for z in ZSTARS}
    for A_g in AGS:
        P = make_P(A_g)
        out_c = solve_fd(P, n=2000)
        out_f = solve_fd(P, n=4000)
        line = f"A_g={A_g:.4f} | "
        for z in ZSTARS:
            i_d = richardson(P, z, out_c, out_f, "i_d")
            i_g = richardson(P, z, out_c, out_f, "i_g")
            res[z]["i_d"].append(i_d)
            res[z]["i_g"].append(i_g)
            line += f"Z={z}: i_d={i_d:+.5f} i_g={i_g:+.5f}  "
        print(line, flush=True)
    for z in ZSTARS:
        for k in res[z]:
            res[z][k] = np.array(res[z][k])

    # crossings
    print("\n--- where investment turns negative (zero crossings in A_g) ---")
    cross = {}
    for z in ZSTARS:
        agd = zero_crossing(AGS, res[z]["i_d"])   # i^d turns negative ABOVE this A_g
        agg = zero_crossing(AGS, res[z]["i_g"])    # i^g turns negative BELOW this A_g
        cross[z] = (agd, agg)
        sd = f"A_g={agd:.4f} ( = {agd/A_D:.2f} x A_d )" if agd else "none in range"
        sg = f"A_g={agg:.4f}" if agg else "none in range"
        print(f"Z={z}: i^d=0 at {sd}   i^g=0 at {sg}")

    np.savez(
        os.path.join(OD, "ag_sensitivity.npz"),
        A_g=AGS, A_d=A_D,
        **{f"i_d_Z{z}": res[z]["i_d"] for z in ZSTARS},
        **{f"i_g_Z{z}": res[z]["i_g"] for z in ZSTARS},
    )

    # ---- Three figures: one per fixed Z, i^d (and i^g) vs A_g ----------------
    for z in ZSTARS:
        agd, agg = cross[z]
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        ax.plot(AGS, res[z]["i_d"], "b-o", lw=2.0, ms=5, label=r"$i^d$ (dirty)")
        ax.plot(AGS, res[z]["i_g"], "g--s", lw=1.6, ms=4, alpha=0.8,
                label=r"$i^g$ (green)")
        ax.axhline(0, color="k", lw=0.9, ls=":")
        ax.axvline(A_D, color="grey", lw=1.0, ls="--",
                   label=fr"$A_d={A_D}$ (fixed)")
        if agd:
            ax.axvline(agd, color="b", lw=1.0, ls=":")
            ax.annotate(fr"$i^d=0$ at $A_g={agd:.3f}$" "\n" fr"$\approx{agd/A_D:.1f}\,A_d$",
                        xy=(agd, 0), xytext=(agd - 0.10, 0.045),
                        fontsize=9, color="b",
                        arrowprops=dict(arrowstyle="->", color="b", lw=0.8))
        ax.set_xlabel(r"green productivity $A_g$  (size of the tech jump)")
        ax.set_ylabel("optimal investment rate")
        ax.set_title(fr"Two-capital (deterministic), fixed $Z={z}$, $A_d={A_D}$:"
                     fr" investment vs $A_g$")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(OD, f"ag_sensitivity_Z{z}.png")
        fig.savefig(p, dpi=150)
        print("saved", p, flush=True)

    # ---- Combined overlay: i^d vs A_g for the three Z -----------------------
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    cols = {0.6: "#1f77b4", 0.7: "#ff7f0e", 0.8: "#2ca02c"}
    for z in ZSTARS:
        ax.plot(AGS, res[z]["i_d"], "-o", color=cols[z], lw=2.0, ms=5,
                label=fr"$Z={z}$")
    ax.axhline(0, color="k", lw=0.9, ls=":")
    ax.axvline(A_D, color="grey", lw=1.0, ls="--", label=fr"$A_d={A_D}$ (fixed)")
    agd07 = cross[0.7][0]
    if agd07:
        ax.axvline(agd07, color="red", lw=1.0, ls=":",
                   label=fr"$i^d=0$ at $A_g\approx{agd07:.2f}\,(\approx{agd07/A_D:.1f}A_d)$")
    ax.set_xlabel(r"green productivity $A_g$  (size of the tech jump)")
    ax.set_ylabel(r"optimal dirty investment $i^d$")
    ax.set_title(r"How large must $A_g$ be to push dirty investment $i^d<0$?"
                 "\n" r"(deterministic two-capital, $A_d$ fixed)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OD, "ag_sensitivity_id_combined.png")
    fig.savefig(p, dpi=150)
    print("saved", p, flush=True)


if __name__ == "__main__":
    main()
