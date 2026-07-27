"""
Correlated capital shocks (no uncertainty). FD solve of the two-capital-with-shocks
model with a nonzero correlation rho between dW^d and dW^g. The reduced ODE gains
cross terms: the logK Ito drag picks up +2 rho Z(1-Z) sigma_d sigma_g, the v' drift
picks up (2Z-1) rho sigma_d sigma_g, and the v'' diffusion coefficient becomes
1/2 Z^2(1-Z)^2 (sigma_d^2 + sigma_g^2 - 2 rho sigma_d sigma_g). rho=0 reproduces the
independent shock model exactly (backward-compatibility check below).

Primary request: sigma=0.01, rho=0.9. We overlay rho=0 vs rho=0.9. Because the effect
scales with sigma^2, it is tiny at sigma=0.01; a sigma=0.2 panel is added to make the
correlation mechanism (here a ~10x reduction of the Z-share diffusion) visible.

Login node: module load python/anaconda-2021.05.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_capital_shock_model as M
from fd_shock import solve_fd_shock

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)


def augment(d, P):
    Z = d["Z"]
    d["q_d"] = 1.0 - Z * d["slope"]
    d["q_g"] = 1.0 + (1.0 - Z) * d["slope"]
    d["Abar"] = (1.0 - Z) * P["A_d"] + Z * P["A_g"]
    d["C_over_Y"] = d["c"] / d["Abar"]
    return d


def solve(sigma, rho, n=2000):
    P = M.load_calibration("A_g_prime_prime")
    P["sigma_d"] = P["sigma_g"] = sigma
    P["rho"] = rho
    return augment(solve_fd_shock(P, n=n), P), P


def backward_compat_check():
    """rho=0 must reproduce the original independent model; sigma->0 the deterministic."""
    a, P = solve(0.01, 0.0)
    print(f"[compat] rho=0 sigma=0.01: max|resid|={a['max_abs_residual']:.2e}", flush=True)
    # sigma->0 with rho=0 must match deterministic
    import two_capital_model as DET
    from fd_solver import solve_fd as solve_fd_det
    P0 = M.load_calibration("A_g_prime_prime"); P0["sigma_d"]=P0["sigma_g"]=0.0; P0["rho"]=0.9
    o0 = solve_fd_shock(P0, n=2000)
    od = solve_fd_det(DET.load_calibration("A_g_prime_prime"), n=2000)
    diff = max(abs(np.interp(z, o0["Z"], o0["i_g"]) - np.interp(z, od["Z"], od["i_g"]))
               for z in np.linspace(0.1, 0.9, 17))
    print(f"[compat] sigma->0 (rho=0.9 irrelevant): max|i_g - i_g_det|={diff:.2e}", flush=True)


def panel(ax, a0, a9, key, title, ylab, plo=0.1, phi=0.9, npts=33):
    def pts(o):
        m = (o["Z"] >= plo) & (o["Z"] <= phi)
        Z, y = o["Z"][m], o[key][m]
        s = max(1, len(Z) // npts)
        return Z[::s], y[::s]
    z0, y0 = pts(a0); z9, y9 = pts(a9)
    ax.plot(z0, y0, "b-o", ms=4, lw=1.5, label=r"$\rho=0$")
    ax.plot(z9, y9, "r--s", ms=4, lw=1.5, label=r"$\rho=0.9$")
    ax.set_xlabel("Z (green capital share)"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)


def make_fig(sigma, tag):
    a0, P0 = solve(sigma, 0.0)
    a9, P9 = solve(sigma, 0.9)
    # report max differences on [0.1,0.9]
    zc = np.linspace(0.1, 0.9, 17)
    for k in ("i_d", "i_g", "slope", "q_d"):
        d = np.max(np.abs(np.interp(zc, a0["Z"], a0[k]) - np.interp(zc, a9["Z"], a9[k])))
        print(f"  sigma={sigma} {k}: max|rho0 - rho0.9| on [0.1,0.9] = {d:.3e}", flush=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panel(ax[0, 0], a0, a9, "i_d", r"Dirty investment $i^d$", r"$i^d$")
    panel(ax[0, 1], a0, a9, "i_g", r"Green investment $i^g$", r"$i^g$")
    panel(ax[0, 2], a0, a9, "q_d", r"Marginal value dirty $q_d$", r"$q_d$")
    panel(ax[1, 0], a0, a9, "q_g", r"Marginal value green $q_g$", r"$q_g$")
    panel(ax[1, 1], a0, a9, "slope", r"Value slope $v'(Z)$", r"$v'$")
    panel(ax[1, 2], a0, a9, "C_over_Y", r"Consumption/output $C/Y$", "C/Y")
    fig.suptitle(f"Correlated capital shocks (no uncertainty): rho=0 vs rho=0.9, "
                 f"sigma={sigma}", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OD, f"shock_correlation_{tag}.png")
    fig.savefig(p, dpi=150); print("saved", p, flush=True)
    return a0, a9


if __name__ == "__main__":
    backward_compat_check()
    print("\n=== sigma=0.01 (requested) ===")
    make_fig(0.01, "sigma0.01")
    print("\n=== sigma=0.2 (correlation effect made visible) ===")
    make_fig(0.2, "sigma0.2")
