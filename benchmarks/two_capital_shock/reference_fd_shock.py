"""
High-accuracy reference solution for the two-capital-WITH-shocks HJB (2nd order).

The production solver fd_shock.py is a semi-implicit false-transient with an UPWIND
drift term: that upwind carries O(dZ) numerical diffusion, so its residual measured
with a central stencil grows with sigma (the larger the drift v1, the larger the
upwind bias). At sigma=0.2 that shows up as max|resid| ~ 2e-2, which is too coarse
to serve as ground truth for judging the neural net.

Here we solve the SAME reduced ODE with second-order CENTRAL differences for both
v' and v'' and a damped Newton iteration with the analytic tridiagonal Jacobian.
The interior residual R_n = hjb_residual(Z_n, v_n, p_n, q_n) with
  p_n = (v_{n+1}-v_{n-1})/(2 dZ),  q_n = (v_{n+1}-2 v_n + v_{n-1})/dZ^2
depends only on (v_{n-1}, v_n, v_{n+1}), so dR/dv is tridiagonal and Newton
converges quadratically to ~1e-9 -- a faithful "truth" at every sigma.

A central-difference damped policy iteration (solve_central) is provided as an
independent cross-check, and sigma->0 must reproduce the deterministic solution.
"""

import numpy as np
from scipy.linalg import solve_banded

import two_capital_shock_model as M


def _tridiag(sub, diag, sup, rhs):
    """LAPACK tridiagonal solve. sub/sup have length len(diag)-1."""
    m = len(diag)
    ab = np.zeros((3, m))
    ab[0, 1:] = sup
    ab[1, :] = diag
    ab[2, :-1] = sub
    return solve_banded((1, 1), ab, rhs)


def _clamp(p, Z, margin=1e-7):
    lo = -1.0 / np.maximum(1.0 - Z, 1e-9) + margin
    hi = 1.0 / np.maximum(Z, 1e-9) - margin
    return np.clip(p, lo, hi)


def _G(p, Z, P):
    """Slope-dependent part of the HJB RHS: everything except -delta*v and the
    v'' diffusion term, i.e. delta*log c + logK_drift + v1(p,Z)*p.
    Equals hjb_residual(Z, v=0, slope=p, curv=0)."""
    return M.hjb_residual(Z, 0.0, p, 0.0, P)


def _v2(Z, P):
    sd2, sg2 = P["sigma_d"] ** 2, P["sigma_g"] ** 2
    return 0.5 * Z ** 2 * (1.0 - Z) ** 2 * (sd2 + sg2)


def _init_v(Z, P):
    Abar = (1.0 - Z) * P["A_d"] + Z * P["A_g"]
    c_sym = P["delta"] * (1.0 + P["theta_d"] * Abar) / (P["theta_d"] * (P["delta"] + P["Gamma_d"]))
    v = (np.log(c_sym) + ((1 - Z) * P["alpha_d"] + Z * P["alpha_g"]) / P["delta"]
         + (P["Gamma_d"] / P["delta"]) * np.log(P["Gamma_d"] * P["theta_d"] * c_sym / P["delta"]))
    v0, vN = M.boundary_values(P)
    v[0], v[-1] = v0, vN
    return v


def _newton_resid(v, idx, Zc, v2c, dZ, P):
    p = (v[2:] - v[:-2]) / (2.0 * dZ)
    q = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / dZ ** 2
    pcl = _clamp(p, Zc)
    return _G(pcl, Zc, P) - P["delta"] * v[idx] + v2c * q, pcl


def solve_newton(P, n=2000, max_iter=200, tol=1e-11, warm=None, verbose=False):
    """Central-difference Newton with analytic tridiagonal Jacobian and a
    backtracking line search. Warm-started from the upwind solution (which is
    close), so the globalized step converges where a raw full Newton step blows up."""
    from fd_shock import solve_fd_shock
    Z = np.linspace(0.0, 1.0, n + 1)
    dZ = 1.0 / n
    v0, vN = M.boundary_values(P)
    if warm is None:
        v = np.interp(Z, *(lambda u: (u["Z"], u["v"]))(solve_fd_shock(P, n=n))).copy()
    else:
        v = np.interp(Z, warm["Z"], warm["v"]).copy()
    v[0], v[-1] = v0, vN
    idx = np.arange(1, n)
    Zc = Z[idx]
    v2c = _v2(Zc, P)
    eps = 1e-7
    success, nit, m = False, max_iter, np.inf
    for it in range(max_iter):
        R, pcl = _newton_resid(v, idx, Zc, v2c, dZ, P)
        m = float(np.max(np.abs(R)))
        if verbose and (it % 5 == 0 or m < tol):
            print(f"  [newton] it {it:4d} max|R|={m:.3e}", flush=True)
        if m < tol:
            success, nit = True, it
            break
        Gp = (_G(pcl + eps, Zc, P) - _G(pcl - eps, Zc, P)) / (2.0 * eps)
        sub = Gp * (-1.0 / (2.0 * dZ)) + v2c / dZ ** 2
        diag = -P["delta"] - 2.0 * v2c / dZ ** 2
        sup = Gp * (1.0 / (2.0 * dZ)) + v2c / dZ ** 2
        dv = _tridiag(sub[1:], diag, sup[:-1], -R)
        # backtracking line search on max|R|
        step = 1.0
        for _ in range(40):
            vt = v.copy(); vt[idx] += step * dv
            Rt, _ = _newton_resid(vt, idx, Zc, v2c, dZ, P)
            if np.all(np.isfinite(Rt)) and float(np.max(np.abs(Rt))) < m:
                v = vt
                break
            step *= 0.5
        else:
            break  # no decrease found -> stop
        nit = it + 1
    return _finalize(Z, v, dZ, P, success, nit, m)


def solve_central(P, n=2000, omega=0.6, max_iter=200000, tol=1e-13, verbose=False):
    """Independent cross-check: central-difference implicit damped policy iteration
    (no upwind). Diffusion + advection both central + implicit, controls frozen."""
    Z = np.linspace(0.0, 1.0, n + 1)
    dZ = 1.0 / n
    v0, vN = M.boundary_values(P)
    v = _init_v(Z, P)
    sd2, sg2 = P["sigma_d"] ** 2, P["sigma_g"] ** 2
    idx = np.arange(1, n)
    nit, step = max_iter, np.inf
    for it in range(max_iter):
        p = np.empty_like(v)
        p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
        p[0] = (v[1] - v[0]) / dZ
        p[-1] = (v[-1] - v[-2]) / dZ
        p = _clamp(p, Z)
        i_d, i_g, c = M.controls(Z, p, P)
        phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        v1 = (phi_g - phi_d + (1.0 - Z) * sd2 - Z * sg2) * Z * (1.0 - Z)
        v2 = 0.5 * Z ** 2 * (1.0 - Z) ** 2 * (sd2 + sg2)
        flow = (P["delta"] * np.log(np.maximum(c, 1e-12)) + (1.0 - Z) * phi_d + Z * phi_g
                - 0.5 * (sd2 * (1.0 - Z) ** 2 + sg2 * Z ** 2))
        a1, a2, fl = v1[idx], v2[idx], flow[idx]
        sub = a1 / (2.0 * dZ) - a2 / dZ ** 2
        diag = np.full(n - 1, P["delta"]) + 2.0 * a2 / dZ ** 2
        sup = -a1 / (2.0 * dZ) - a2 / dZ ** 2
        rhs = fl.copy()
        rhs[0] -= sub[0] * v0
        rhs[-1] -= sup[-1] * vN
        v_new = _tridiag(sub[1:], diag, sup[:-1], rhs)
        v_int = (1.0 - omega) * v[idx] + omega * v_new
        step = float(np.max(np.abs(v_int - v[idx])))
        v[idx] = v_int
        nit = it + 1
        if verbose and it % 5000 == 0:
            print(f"  [central] it {it:6d} max|dv|={step:.3e}", flush=True)
        if step < tol:
            break
    return _finalize(Z, v, dZ, P, step < tol, nit, step)


def _finalize(Z, v, dZ, P, success, nit, conv):
    p = np.empty_like(v)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
    p[0] = (v[1] - v[0]) / dZ
    p[-1] = (v[-1] - v[-2]) / dZ
    slope = _clamp(p, Z)
    vpp = np.zeros_like(v)
    vpp[1:-1] = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / dZ ** 2
    resid = M.hjb_residual(Z, v, slope, vpp, P)
    i_d, i_g, c = M.controls(Z, slope, P)
    return {"Z": Z, "v": v, "slope": slope, "i_d": i_d, "i_g": i_g, "c": c,
            "residual": resid, "iters": nit, "success": bool(success), "conv": float(conv),
            "max_abs_residual": float(np.max(np.abs(resid[2:-2]))),
            "l2_residual": float(np.sqrt(np.mean(resid[2:-2] ** 2)))}


def solve_reference(P, n=2000):
    """Newton primary; fall back to central if Newton does not reach tol."""
    out = solve_newton(P, n=n)
    if not out["success"]:
        out = solve_central(P, n=n)
    return out


if __name__ == "__main__":
    import two_capital_model as DET
    from fd_shock import solve_fd_shock

    for sigma in (0.01, 0.1, 0.2):
        P = M.load_calibration("A_g_prime_prime")
        P["sigma_d"] = sigma; P["sigma_g"] = sigma
        nw = solve_newton(P, n=2000)
        ct = solve_central(P, n=2000)
        up = solve_fd_shock(P, n=2000)
        # agreement between the two independent references
        dd = max(abs(np.interp(z, nw["Z"], nw["v"]) - np.interp(z, ct["Z"], ct["v"]))
                 for z in np.linspace(0.05, 0.95, 19))
        # how far the upwind production FD sits from the accurate reference
        du = max(abs(np.interp(z, up["Z"], up["slope"]) - np.interp(z, nw["Z"], nw["slope"]))
                 for z in np.linspace(0.1, 0.9, 17))
        print(f"sigma={sigma}: newton success={nw['success']} it={nw['iters']} "
              f"max|R|={nw['max_abs_residual']:.2e} L2={nw['l2_residual']:.2e} | "
              f"central max|R|={ct['max_abs_residual']:.2e} | "
              f"|v_newton-v_central|={dd:.2e} | "
              f"upwind max|R|={up['max_abs_residual']:.2e}  |slope_upwind-slope_ref|={du:.2e}")

    # sigma -> 0 must recover the deterministic reference
    from reference_solver import solve_newton as det_newton
    P0 = M.load_calibration("A_g_prime_prime"); P0["sigma_d"] = 0.0; P0["sigma_g"] = 0.0
    o0 = solve_newton(P0, n=2000)
    Zd, vd, pd, sold = det_newton(DET.load_calibration("A_g_prime_prime"), n=2000)
    diff = max(abs(np.interp(z, o0["Z"], o0["slope"]) - np.interp(z, Zd, pd))
               for z in np.linspace(0.1, 0.9, 17))
    print(f"\n[sigma->0] max|slope(shock newton) - slope(deterministic newton)| = {diff:.2e}")
