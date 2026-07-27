"""
High-accuracy reference solution for the deterministic two-capital HJB, used to
settle which discretization is correct (the first-order upwind scheme carries
O(dZ) numerical diffusion and can bias the slope v'(Z)).

Two independent second-order references:
  (1) central-difference Newton/damped fixed point for v(Z);
  (2) scipy.integrate.solve_bvp on the second-order form
        v'' = (delta*v' - F_Z) / F_p,   delta*v = F(v', Z).

Both should agree; their common v'(Z) is the ground truth against which the
upwind FD and the DGM neural net are judged. The "loss" of any solution is the
HJB residual R(Z) (the same object the NN minimizes), reported in L2 and max
norm over the interior grid using an accurate central derivative.
"""

import numpy as np
from scipy.linalg import solve_banded

import two_capital_model as M


def _tridiag(sub, diag, sup, rhs):
    """Fast tridiagonal solve via LAPACK. sub/sup length m-1, diag/rhs length m."""
    m = len(diag)
    ab = np.zeros((3, m))
    ab[0, 1:] = sup
    ab[1, :] = diag
    ab[2, :-1] = sub
    return solve_banded((1, 1), ab, rhs)


def F_rhs(p, Z, P):
    """HJB right-hand side delta*v target: F(p,Z) with optimal controls."""
    i_d, i_g, c = M.controls(Z, p, P)
    phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    mu = Z * (1.0 - Z) * (phi_g - phi_d)
    return P["delta"] * np.log(np.maximum(c, 1e-12)) + (1.0 - Z) * phi_d + Z * phi_g + mu * p


def _clamp(p, Z, m=1e-7):
    lo = -1.0 / np.maximum(1.0 - Z, 1e-9) + m
    hi = 1.0 / np.maximum(Z, 1e-9) - m
    return np.clip(p, lo, hi)


def solve_central(P, n=2000, omega=0.5, max_iter=200000, tol=1e-12):
    """Second-order central-difference damped fixed point for v(Z)."""
    Z = np.linspace(0.0, 1.0, n + 1)
    dZ = 1.0 / n
    v0, vN = M.boundary_values(P)
    Abar = M.A_bar(Z, P)
    alphabar = (1.0 - Z) * P["alpha_d"] + Z * P["alpha_g"]
    c_sym = P["delta"] * (1.0 + P["theta_d"] * Abar) / (P["theta_d"] * (P["delta"] + P["Gamma_d"]))
    v = (np.log(c_sym) + alphabar / P["delta"]
         + (P["Gamma_d"] / P["delta"]) * np.log(P["Gamma_d"] * P["theta_d"] * c_sym / P["delta"]))
    v[0], v[-1] = v0, vN
    idx = np.arange(1, n)
    for it in range(max_iter):
        p = np.empty_like(v)
        p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
        p[0] = (v[1] - v[0]) / dZ
        p[-1] = (v[-1] - v[-2]) / dZ
        p = _clamp(p, Z)
        i_d, i_g, c = M.controls(Z, p, P)
        phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        mu = Z * (1.0 - Z) * (phi_g - phi_d)
        flow = P["delta"] * np.log(np.maximum(c, 1e-12)) + (1.0 - Z) * phi_d + Z * phi_g
        # delta v_n - mu_n (v_{n+1}-v_{n-1})/(2dZ) = flow_n   (central, frozen mu/flow)
        a = -mu[idx] / (2.0 * dZ)   # coef v_{n-1}  (sub)
        b = np.full(n - 1, P["delta"])
        cc = mu[idx] / (2.0 * dZ)   # coef v_{n+1}  (super)
        rhs = flow[idx].copy()
        rhs[0] -= a[0] * v0
        rhs[-1] -= cc[-1] * vN
        v_new = _tridiag(a[1:], b, cc[:-1], rhs)
        v_int = (1 - omega) * v[idx] + omega * v_new
        step = np.max(np.abs(v_int - v[idx]))
        v[idx] = v_int
        if step < tol:
            break
    p = np.gradient(v, Z)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
    return Z, v, _clamp(p, Z), it + 1


def _thomas(a, b, c, d):
    m = len(b)
    cp = np.zeros(m - 1); dp = np.zeros(m)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, m - 1):
        den = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / den
    dp[m - 1] = (d[m - 1] - a[m - 2] * dp[m - 2]) / (b[m - 1] - a[m - 2] * cp[m - 2])
    x = np.zeros(m); x[-1] = dp[-1]
    for i in range(m - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class _Sol:
    def __init__(self, success, nit, res):
        self.success = success; self.nit = nit; self.max_res = res


def _f_part(p, Zc, P):
    """The slope-dependent part of the HJB RHS (everything except -delta*v)."""
    i_d, i_g, c = M.controls(Zc, p, P)
    phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    mu = Zc * (1.0 - Zc) * (phi_g - phi_d)
    return P["delta"] * np.log(np.maximum(c, 1e-12)) + (1.0 - Zc) * phi_d + Zc * phi_g + mu * p


def solve_newton(P, n=600, max_iter=200, tol=1e-11, damp=1.0):
    """Central-difference damped Newton with the analytic tridiagonal Jacobian.

    The central residual R_n = f(p_n,Z_n) - delta*v_n with p_n=(v_{n+1}-v_{n-1})/(2dZ)
    depends only on (v_{n-1}, v_n, v_{n+1}), so dR/dv is tridiagonal and Newton
    converges quadratically to ~1e-10 (the FD 'ground truth'). Returns Z, v, slope, sol.
    """
    from fd_solver import solve_fd
    Z = np.linspace(0.0, 1.0, n + 1)
    dZ = 1.0 / n
    v0, vN = M.boundary_values(P)
    v = solve_fd(P, n=n)["v"].copy()  # warm start from the upwind solution
    v[0], v[-1] = v0, vN
    idx = np.arange(1, n)
    Zc = Z[idx]
    eps = 1e-7
    sol = _Sol(False, max_iter, np.inf)
    for it in range(max_iter):
        p = (v[2:] - v[:-2]) / (2.0 * dZ)          # central interior slopes (len n-1)
        R = _f_part(p, Zc, P) - P["delta"] * v[idx]
        m = float(np.max(np.abs(R)))
        if m < tol:
            sol = _Sol(True, it, m); break
        dfdp = (_f_part(p + eps, Zc, P) - _f_part(p - eps, Zc, P)) / (2.0 * eps)
        diag = np.full(n - 1, -P["delta"])
        sub = dfdp * (-1.0 / (2.0 * dZ))           # dR_n/dv_{n-1}
        sup = dfdp * (1.0 / (2.0 * dZ))            # dR_n/dv_{n+1}
        # interior unknowns Δv_1..Δv_{n-1}; Δv_0=Δv_N=0 (boundaries fixed)
        dv = _tridiag(sub[1:], diag, sup[:-1], -R)
        v[idx] += damp * dv
        sol = _Sol(False, it + 1, m)
    p_full = np.empty(n + 1)
    p_full[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
    p_full[0] = (v[1] - v[0]) / dZ
    p_full[-1] = (v[-1] - v[-2]) / dZ
    return Z, v, _clamp(p_full, Z), sol


def solve_bvp_ref(P, n=400):
    from scipy.integrate import solve_bvp
    v0, vN = M.boundary_values(P)
    eps = 1e-6

    def odes(Z, y):
        p = y[1]
        Fp = (F_rhs(p + eps, Z, P) - F_rhs(p - eps, Z, P)) / (2 * eps)
        Zp = np.minimum(Z + eps, 1.0 - 1e-9)
        Zm = np.maximum(Z - eps, 1e-9)
        FZ = (F_rhs(p, Zp, P) - F_rhs(p, Zm, P)) / (Zp - Zm)
        vpp = (P["delta"] * p - FZ) / Fp
        return np.vstack([p, vpp])

    def bc(ya, yb):
        return np.array([ya[0] - v0, yb[0] - vN])

    Zm = np.linspace(1e-3, 1 - 1e-3, n)
    guess = np.vstack([np.linspace(v0, vN, n), M.perturbation_slope(Zm, P)])
    sol = solve_bvp(odes, bc, Zm, guess, max_nodes=40000, tol=1e-9, verbose=0)
    return sol


def true_residual(Z, v, P):
    """HJB residual using an accurate central derivative (the 'loss' of a solution)."""
    dZ = Z[1] - Z[0]
    p = np.empty_like(v)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
    p[0] = (v[1] - v[0]) / dZ
    p[-1] = (v[-1] - v[-2]) / dZ
    return M.hjb_residual(Z, v, _clamp(p, Z), P)


if __name__ == "__main__":
    P = M.load_calibration("A_g_prime_prime")
    print("PARAM SOURCE = parent models/params.py")
    print(f"  A_d={P['A_d']}  A_g={P['A_g']} (A_g_prime_prime, post tech jump)  "
          f"delta={P['delta']}  Gamma={P['Gamma_d']}  theta={P['theta_d']}  alpha={P['alpha_d']}")

    # 1) solve_bvp: the gold-standard adaptive BVP solver (primary ground truth)
    try:
        sol = solve_bvp_ref(P)
        rms = float(getattr(sol, "rms_residuals", np.array([np.nan])).max())
        print(f"\n[solve_bvp]  success={sol.success}  nodes={sol.x.size}  max_rms_residual={rms:.2e}")
        bvp_ok = sol.success
    except Exception as e:
        print(f"\n[solve_bvp] failed: {e}")
        bvp_ok = False

    # 2) central 2nd-order FD (independent cross-check + its HJB 'loss')
    Zc, vc, pc, itc = solve_central(P)
    rc = true_residual(Zc, vc, P)
    print(f"[central-2nd-order] iters={itc}  L2(resid)={np.sqrt(np.mean(rc[2:-2]**2)):.2e}  "
          f"max|resid|={np.max(np.abs(rc[2:-2])):.2e}")

    # 3) upwind 1st-order FD (the biased one) + its loss
    from fd_solver import solve_fd
    up = solve_fd(P)
    ru = true_residual(up["Z"], up["v"], P)
    print(f"[upwind-1st-order]  iters={up['iters']}  L2(resid)={np.sqrt(np.mean(ru[2:-2]**2)):.2e}  "
          f"max|resid|={np.max(np.abs(ru[2:-2])):.2e}")

    print("\nv'(Z) by method  (solve_bvp = ground truth):")
    print(f"{'Z':>5} {'solve_bvp':>10} {'central':>9} {'upwind':>9} {'perturb':>9}")
    for z in (0.1, 0.3, 0.5, 0.7, 0.9):
        pb = float(sol.sol(z)[1]) if bvp_ok else float("nan")
        kc = int(z * (len(Zc) - 1))
        ku = int(z * (len(up["Z"]) - 1))
        pp = float(M.perturbation_slope(np.array([z]), P)[0])
        print(f"{z:>5.1f} {pb:>10.4f} {pc[kc]:>9.4f} {up['slope'][ku]:>9.4f} {pp:>9.4f}")
