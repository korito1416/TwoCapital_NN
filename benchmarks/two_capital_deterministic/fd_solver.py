"""
Finite-difference solver for the deterministic two-capital HJB.

Solves the first-order nonlinear ODE for v(Z) on Z in [0,1] by a semi-implicit
false-transient (pseudo-time) iteration with upwind treatment of the Z-drift
(adjustment_cost_no_random.tex, Section 11). The pseudo-transient

    dv/dtau = delta*log c + (1-Z)phi_d + Z phi_g + mu_Z(Z) v' - delta*v

is stepped to steady state. The controls/consumption are closed form given the
value-function slope p = v'(Z); the slope is clamped to the admissibility region
q_d>0, q_g>0 (i.e. -1/(1-Z) < p < 1/Z) so the controls stay in their domain. The
drift mu_Z = Z(1-Z)(phi_g - phi_d) vanishes at both ends, so the boundaries reduce
to the one-capital values (imposed for robustness). The implicit transport uses
forward differencing where mu_Z>0 and backward where mu_Z<0, which yields an
M-matrix and is unconditionally stable.
"""

import numpy as np

import two_capital_model as M


def _slope_clamped(v, Z, dZ, margin=1e-6):
    p = np.empty_like(v)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
    p[0] = (v[1] - v[0]) / dZ
    p[-1] = (v[-1] - v[-2]) / dZ
    lo = -1.0 / np.maximum(1.0 - Z, 1e-9) + margin
    hi = 1.0 / np.maximum(Z, 1e-9) - margin
    return np.clip(p, lo, hi)


def solve_fd(p, n=2000, dtau=2.0, max_iter=200000, tol=1e-12, verbose=False):
    Z = np.linspace(0.0, 1.0, n + 1)
    dZ = 1.0 / n
    v0, vN = M.boundary_values(p)

    # Perturbation initial guess.
    Abar = M.A_bar(Z, p)
    alphabar = (1.0 - Z) * p["alpha_d"] + Z * p["alpha_g"]
    c_sym = p["delta"] * (1.0 + p["theta_d"] * Abar) / (p["theta_d"] * (p["delta"] + p["Gamma_d"]))
    v = (np.log(c_sym) + alphabar / p["delta"]
         + (p["Gamma_d"] / p["delta"]) * np.log(p["Gamma_d"] * p["theta_d"] * c_sym / p["delta"]))
    v[0], v[-1] = v0, vN

    idx = np.arange(1, n)
    last = None
    for it in range(max_iter):
        slope = _slope_clamped(v, Z, dZ)
        i_d, i_g, c = M.controls(Z, slope, p)
        phi_d = M.phi(i_d, p["alpha_d"], p["Gamma_d"], p["theta_d"])
        phi_g = M.phi(i_g, p["alpha_g"], p["Gamma_g"], p["theta_g"])
        mu = Z * (1.0 - Z) * (phi_g - phi_d)
        flow = p["delta"] * np.log(np.maximum(c, 1e-12)) + (1.0 - Z) * phi_d + Z * phi_g

        mi = mu[idx]
        fwd = mi > 0.0  # forward where mu>0 (M-matrix), backward where mu<0
        coef = mi / dZ

        # (1/dtau + delta) v_n - mu * D^up v_n = v_n^old/dtau + flow_n
        diag = np.full(n - 1, 1.0 / dtau + p["delta"])
        sub = np.zeros(n - 1)   # coef of v_{n-1}
        sup = np.zeros(n - 1)   # coef of v_{n+1}
        # forward: -mu*(v_{n+1}-v_n)/dZ = +coef*v_n - coef*v_{n+1}
        diag[fwd] += coef[fwd]
        sup[fwd] -= coef[fwd]
        # backward: -mu*(v_n-v_{n-1})/dZ = -coef*v_n + coef*v_{n-1}
        diag[~fwd] -= coef[~fwd]
        sub[~fwd] += coef[~fwd]

        rhs = v[idx] / dtau + flow[idx]
        rhs[0] -= sub[0] * v0
        rhs[-1] -= sup[-1] * vN

        v_new = _thomas(sub[1:], diag, sup[:-1], rhs)
        step = np.max(np.abs(v_new - v[idx]))
        v[idx] = v_new
        if verbose and (it % 2000 == 0):
            print(f"  [FD] iter {it:6d}  max|dv|={step:.3e}")
        if step < tol:
            last = it
            break
        last = it

    slope = _slope_clamped(v, Z, dZ)
    i_d, i_g, c = M.controls(Z, slope, p)
    resid = M.hjb_residual(Z, v, slope, p)
    return {
        "method": "FD", "Z": Z, "v": v, "slope": slope,
        "i_d": i_d, "i_g": i_g, "c": c, "ratio": i_g / i_d,
        "residual": resid, "iters": last + 1,
        "max_abs_residual": float(np.max(np.abs(resid[1:-1]))),
    }


def _thomas(a, b, c, d):
    """Tridiagonal solve. a: sub (m-1), b: diag (m), c: super (m-1), d: rhs (m)."""
    m = len(b)
    cp = np.zeros(m - 1)
    dp = np.zeros(m)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, m - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[m - 1] = (d[m - 1] - a[m - 2] * dp[m - 2]) / (b[m - 1] - a[m - 2] * cp[m - 2])
    x = np.zeros(m)
    x[-1] = dp[-1]
    for i in range(m - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


if __name__ == "__main__":
    P = M.load_calibration("A_g_prime_prime")
    out = solve_fd(P, verbose=True)
    print(f"FD done in {out['iters']} iters, max|residual|={out['max_abs_residual']:.3e}")
    for z in (0.1, 0.3, 0.5, 0.7, 0.9):
        k = int(z * (len(out["Z"]) - 1))
        print(f"  Z={z:.1f}  v'={out['slope'][k]:.4f}  i_d={out['i_d'][k]:.5f}  "
              f"i_g={out['i_g'][k]:.5f}  i_g/i_d={out['ratio'][k]:.4f}")
