"""
Dirty-productivity (A_d) sensitivity of the deterministic two-capital model
(no shocks), FD only.

Lars's test: keep the *original* adjustment-cost function unchanged
(phi_j(i)=alpha_j+Gamma_j log(1+theta_j i) with the base calibration
theta=16.7, Gamma=0.06, alpha=-0.035), and sweep the dirty productivity A_d
from 0.01 to 0.13. Green productivity is held at the post-tech-jump value
A_g=A_g''=0.1567. For three fixed green shares Z=0.6, 0.7, 0.8 we read off the
optimal dirty investment rate i^d and plot it against A_d (one figure per Z),
plus a combined overlay.

Investment may be NEGATIVE (de-investment is allowed); the only restriction is
that capital cannot go negative, i.e. 1+theta_j i^j>=0 (q_j>=0), enforced via the
slope clamp inside solve_fd. When A_d is small the dirty capital is unproductive,
so the planner de-invests dirty (i^d<0); as A_d rises toward A_g, i^d increases.

Solver: the same semi-implicit upwind false-transient used in theta_sensitivity.py
(robust M-matrix contraction). The upwind scheme carries O(dZ) numerical diffusion,
so i^d at the target Z is Richardson-extrapolated from grids n and 2n; the grid gap
is reported as an accuracy check. Interior HJB residual at Z in [0.5,0.9] is ~4e-6.

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

ADS = np.linspace(0.01, 0.13, 25)        # dirty productivity sweep
ZSTARS = [0.6, 0.7, 0.8]                  # fixed green shares
A_G = None                                # filled from calibration below


def make_P(A_d):
    """Base calibration (original adjustment cost) with dirty productivity A_d."""
    P = M.load_calibration("A_g_prime_prime")  # A_g = A_g'' = 0.1567 (post jump)
    P["A_d"] = float(A_d)
    return P


def richardson(P, z, out_c, out_f, key):
    """2*f(2n) - f(n): remove the leading O(dZ) upwind bias at a single Z."""
    fc = at_Z(out_c, key, z)
    ff = at_Z(out_f, key, z)
    return 2.0 * ff - fc, abs(ff - fc)


def main():
    global A_G
    A_G = make_P(0.0)["A_g"]
    # results[z] = dict of arrays over ADS
    res = {z: {"i_d": [], "i_g": [], "c": [], "gap": []} for z in ZSTARS}
    resid = []
    for A_d in ADS:
        P = make_P(A_d)
        out_c = solve_fd(P, n=2000)   # coarse
        out_f = solve_fd(P, n=4000)   # fine
        resid.append(out_f["max_abs_residual"])
        line = f"A_d={A_d:.4f} | "
        for z in ZSTARS:
            i_d, gap = richardson(P, z, out_c, out_f, "i_d")
            i_g, _ = richardson(P, z, out_c, out_f, "i_g")
            c, _ = richardson(P, z, out_c, out_f, "c")
            res[z]["i_d"].append(i_d)
            res[z]["i_g"].append(i_g)
            res[z]["c"].append(c)
            res[z]["gap"].append(gap)
            line += f"Z={z}: i_d={i_d:+.5f} i_g={i_g:+.5f}  "
        print(line, flush=True)
    for z in ZSTARS:
        for k in res[z]:
            res[z][k] = np.array(res[z][k])

    np.savez(
        os.path.join(OD, "ad_sensitivity.npz"),
        A_d=ADS, A_g=A_G,
        **{f"i_d_Z{z}": res[z]["i_d"] for z in ZSTARS},
        **{f"i_g_Z{z}": res[z]["i_g"] for z in ZSTARS},
        **{f"c_Z{z}": res[z]["c"] for z in ZSTARS},
        resid=np.array(resid),
    )

    # ---- Three figures: one per fixed Z, i^d vs A_d --------------------------
    for z in ZSTARS:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.plot(ADS, res[z]["i_d"], "b-o", lw=2.0, ms=5, label=r"$i^d$ (dirty)")
        ax.plot(ADS, res[z]["i_g"], "g--s", lw=1.5, ms=4, alpha=0.7,
                label=r"$i^g$ (green, reference)")
        ax.axhline(0, color="k", lw=0.9, ls=":")
        ax.axvline(A_G, color="grey", lw=1.0, ls="--",
                   label=fr"$A_g={A_G:.4f}$")
        ax.set_xlabel(r"dirty productivity $A_d$")
        ax.set_ylabel("optimal investment rate")
        ax.set_title(fr"Two-capital (deterministic), fixed $Z={z}$:"
                     fr" investment vs $A_d$")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(OD, f"ad_sensitivity_Z{z}.png")
        fig.savefig(p, dpi=150)
        print("saved", p, flush=True)

    # ---- Combined overlay: i^d vs A_d for the three Z ------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    cols = {0.6: "#1f77b4", 0.7: "#ff7f0e", 0.8: "#2ca02c"}
    for z in ZSTARS:
        ax.plot(ADS, res[z]["i_d"], "-o", color=cols[z], lw=2.0, ms=5,
                label=fr"$Z={z}$")
    ax.axhline(0, color="k", lw=0.9, ls=":")
    ax.axvline(A_G, color="grey", lw=1.0, ls="--", label=fr"$A_g={A_G:.4f}$")
    ax.set_xlabel(r"dirty productivity $A_d$")
    ax.set_ylabel(r"optimal dirty investment $i^d$")
    ax.set_title(r"Dirty investment $i^d$ vs $A_d$ at fixed green shares "
                 r"(de-investment allowed)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OD, "ad_sensitivity_id_combined.png")
    fig.savefig(p, dpi=150)
    print("saved", p, flush=True)

    print(f"\nmax interior HJB residual over sweep: {np.max(resid):.2e}")


if __name__ == "__main__":
    main()
