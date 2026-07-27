"""
Theta sensitivity of the deterministic two-capital model (no shocks), FD only.

We vary the adjustment-cost curvature theta over a family with theta*Gamma=1, so
every adjustment cost phi_j(i)=alpha_j+Gamma_j log(1+theta_j i) shares the same
marginal product of investment at i=0 (phi'(0)=Gamma*theta=1) but differs in
curvature. theta runs from small (near-linear, cheap to adjust) to large (stiff).
For each theta we solve the HJB and read the optimal investment rates i^d, i^g at
the fixed share Z=0.7.

Investment may be NEGATIVE (de-investment is allowed). The only restriction is that
capital cannot go negative: the capital-growth factor 1+theta_j i^j>=0 (equiv.
q_j>=0), enforced through the slope clamp q_d,q_g>0. We report 1+theta i^j at Z=0.7
to confirm K stays positive even when i^j<0.

Solver: semi-implicit upwind false-transient (an M-matrix contraction, robust for
the whole theta range). The upwind scheme carries O(dZ) numerical diffusion, so the
investment rates at Z=0.7 are Richardson-extrapolated from grids n and 2n to remove
the leading bias; the grid gap is reported as an accuracy check.

Runs on the login node (numpy/scipy only): module load python/anaconda-2021.05.
"""

import os
import numpy as np
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_capital_model as M

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)

THETAS = [0.1, 0.2, 0.35, 0.6, 1.0, 1.7, 3.0, 5.0, 8.5, 16.7, 30.0, 55.0, 100.0]
ZSTAR = 0.7


def make_P(theta):
    P = M.load_calibration("A_g_prime_prime")
    P["theta_d"] = P["theta_g"] = float(theta)
    P["Gamma_d"] = P["Gamma_g"] = 1.0 / float(theta)
    return P


def _clamp(p, Z, margin=1e-7):
    """Keep q_d=1-Zp>0, q_g=1+(1-Z)p>0  <=>  1+theta i>0  <=>  K-growth factor>0."""
    lo = -1.0 / np.maximum(1.0 - Z, 1e-9) + margin
    hi = 1.0 / np.maximum(Z, 1e-9) - margin
    return np.clip(p, lo, hi)


def _tri(sub, diag, sup, rhs):
    m = len(diag)
    ab = np.zeros((3, m))
    ab[0, 1:] = sup[:-1]
    ab[1, :] = diag
    ab[2, :-1] = sub[1:]
    return solve_banded((1, 1), ab, rhs)


def solve_fd(P, n=4000, dtau=2.0, max_iter=200000, tol=1e-12):
    """Semi-implicit upwind false transient for v(Z) (de-investment allowed)."""
    Z = np.linspace(0.0, 1.0, n + 1); dZ = 1.0 / n
    v0, vN = M.boundary_values(P)
    Abar = M.A_bar(Z, P)
    c_sym = P["delta"] * (1.0 + P["theta_d"] * Abar) / (P["theta_d"] * (P["delta"] + P["Gamma_d"]))
    v = (np.log(c_sym) + ((1 - Z) * P["alpha_d"] + Z * P["alpha_g"]) / P["delta"]
         + (P["Gamma_d"] / P["delta"]) * np.log(P["Gamma_d"] * P["theta_d"] * c_sym / P["delta"]))
    v[0], v[-1] = v0, vN
    idx = np.arange(1, n)
    it = 0
    for it in range(max_iter):
        p = np.empty_like(v)
        p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
        p[0] = (v[1] - v[0]) / dZ; p[-1] = (v[-1] - v[-2]) / dZ
        p = _clamp(p, Z)
        i_d, i_g, c = M.controls(Z, p, P)
        phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        mu = Z * (1.0 - Z) * (phi_g - phi_d)
        flow = P["delta"] * np.log(np.maximum(c, 1e-300)) + (1.0 - Z) * phi_d + Z * phi_g
        mi = mu[idx]; fwd = mi > 0.0; coef = mi / dZ
        diag = np.full(n - 1, 1.0 / dtau + P["delta"]); sub = np.zeros(n - 1); sup = np.zeros(n - 1)
        diag[fwd] += coef[fwd]; sup[fwd] -= coef[fwd]
        diag[~fwd] -= coef[~fwd]; sub[~fwd] += coef[~fwd]
        rhs = v[idx] / dtau + flow[idx]; rhs[0] -= sub[0] * v0; rhs[-1] -= sup[-1] * vN
        v_new = _tri(sub, diag, sup, rhs)
        step = np.max(np.abs(v_new - v[idx]))
        v[idx] = v_new
        if step < tol:
            break
    p = np.empty_like(v)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
    p[0] = (v[1] - v[0]) / dZ; p[-1] = (v[-1] - v[-2]) / dZ
    slope = _clamp(p, Z)
    i_d, i_g, c = M.controls(Z, slope, P)
    resid = M.hjb_residual(Z, v, slope, P)
    return {"Z": Z, "v": v, "slope": slope, "i_d": i_d, "i_g": i_g, "c": c, "iters": it + 1,
            "max_abs_residual": float(np.max(np.abs(resid[2:-2])))}


def at_Z(out, key, z=ZSTAR):
    return float(np.interp(z, out["Z"], out[key]))


def main():
    rows = []
    profiles = {}
    keep_profiles = {0.1, 1.0, 16.7, 100.0}
    for th in THETAS:
        P = make_P(th)
        out_c = solve_fd(P, n=2000)   # coarse
        out = solve_fd(P, n=4000)     # fine (used for profiles)
        # Richardson extrapolation of slope at Z=0.7 (upwind error ~ C*dZ):
        pc = at_Z(out_c, "slope"); pf = at_Z(out, "slope")
        p_ext = 2.0 * pf - pc
        gap = abs(pf - pc)  # grid-convergence gap (accuracy proxy)
        # controls from the extrapolated slope at Z=0.7
        q_d = 1.0 - ZSTAR * p_ext; q_g = 1.0 + (1.0 - ZSTAR) * p_ext
        c07 = at_Z(out, "c")  # consumption barely grid-sensitive; use fine grid
        # recompute c consistently with extrapolated slope via closed form
        Abar07 = (1 - ZSTAR) * P["A_d"] + ZSTAR * P["A_g"]
        num = P["delta"] * (Abar07 + (1 - ZSTAR) / P["theta_d"] + ZSTAR / P["theta_g"])
        den = P["delta"] + (1 - ZSTAR) * P["Gamma_d"] * q_d + ZSTAR * P["Gamma_g"] * q_g
        c07 = num / den
        idv = P["Gamma_d"] * c07 * q_d / P["delta"] - 1.0 / P["theta_d"]
        igv = P["Gamma_g"] * c07 * q_g / P["delta"] - 1.0 / P["theta_g"]
        kd = 1.0 + th * idv; kg = 1.0 + th * igv
        rows.append((th, idv, igv, p_ext, kd, kg, gap, out["max_abs_residual"]))
        if th in keep_profiles:
            profiles[th] = out
        print(f"theta={th:6.2f} Gamma={1/th:7.4f} | i_d={idv:+.5f} i_g={igv:+.5f} v'={p_ext:+.4f} "
              f"| 1+θi_d={kd:.4f} 1+θi_g={kg:.4f} | grid_gap(v')={gap:.1e} resid4k={out['max_abs_residual']:.1e}",
              flush=True)

    rows = np.array(rows)
    np.savez(os.path.join(OD, "theta_sensitivity.npz"),
             theta=rows[:, 0], i_d=rows[:, 1], i_g=rows[:, 2], slope=rows[:, 3],
             Kfac_d=rows[:, 4], Kfac_g=rows[:, 5], grid_gap=rows[:, 6], resid=rows[:, 7])

    th = rows[:, 0]
    # ---- Figure 1: i_d, i_g vs theta at Z=0.7 + capital-growth factor --------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].plot(th, rows[:, 1], "b-o", lw=1.9, ms=6, label=r"$i^d$ (dirty)")
    ax[0].plot(th, rows[:, 2], "g-s", lw=1.9, ms=6, label=r"$i^g$ (green)")
    ax[0].axhline(0, color="k", lw=0.9, ls=":")
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"adjustment-cost curvature $\theta$  ($\Gamma=1/\theta$)")
    ax[0].set_ylabel("investment rate"); ax[0].set_title(r"Investment at $Z=0.7$ vs $\theta$")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

    ax[1].plot(th, rows[:, 4], "b-o", lw=1.9, ms=6, label=r"$1+\theta i^d$ (dirty)")
    ax[1].plot(th, rows[:, 5], "g-s", lw=1.9, ms=6, label=r"$1+\theta i^g$ (green)")
    ax[1].axhline(0, color="r", lw=1.0, ls="--", label=r"$K=0$ floor")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"adjustment-cost curvature $\theta$")
    ax[1].set_ylabel(r"capital-growth factor $1+\theta i$")
    ax[1].set_title(r"Capital stays positive: $1+\theta i^j\geq0$ at $Z=0.7$")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle(r"Two-capital (deterministic), $\theta\Gamma=1$: investment can be negative,"
                 r" capital cannot ($1+\theta i\geq0$)", fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(OD, "theta_sensitivity_Z0.7.png")
    fig.savefig(p1, dpi=150); print("saved", p1)

    # ---- Figure 2: full i^d(Z), i^g(Z) profiles for selected thetas ----------
    if profiles:
        ths = sorted(profiles)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        cols = plt.cm.viridis(np.linspace(0, 0.85, len(ths)))
        for c, t in zip(cols, ths):
            out = profiles[t]
            mlo = (out["Z"] >= 0.05) & (out["Z"] <= 0.95)
            ax[0].plot(out["Z"][mlo], out["i_d"][mlo], "-", color=c, lw=1.8, label=f"$\\theta$={t:g}")
            ax[1].plot(out["Z"][mlo], out["i_g"][mlo], "-", color=c, lw=1.8, label=f"$\\theta$={t:g}")
        for a, ti, yl in zip(ax, [r"Dirty $i^d(Z)$", r"Green $i^g(Z)$"], [r"$i^d$", r"$i^g$"]):
            a.axhline(0, color="k", lw=0.9, ls=":"); a.axvline(ZSTAR, color="grey", lw=0.9, ls=":")
            a.set_xlabel("Z (green capital share)"); a.set_ylabel(yl); a.set_title(ti)
            a.legend(fontsize=9); a.grid(alpha=0.3)
        fig.suptitle("Investment profiles across $Z$ (de-investment allowed)", fontsize=13)
        fig.tight_layout()
        p2 = os.path.join(OD, "theta_sensitivity_profiles.png")
        fig.savefig(p2, dpi=150); print("saved", p2)


if __name__ == "__main__":
    main()
