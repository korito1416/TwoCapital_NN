"""
Finite-difference solver for the two-capital-with-shocks HJB (second-order BVP).

Semi-implicit false-transient: the diffusion term 1/2 Z^2(1-Z)^2(sigma_d^2+sigma_g^2) v''
is central + implicit, the drift (v') term is upwind + implicit (guaranteeing an
M-matrix), and the controls/flow are frozen from the previous iterate (policy
iteration). Dirichlet data v(0)=v_d, v(1)=v_g include the Ito drag -sigma_j^2/(2 delta).
Setting sigma=0 must recover the deterministic solution.
"""

import numpy as np
from scipy.linalg import solve_banded

import two_capital_shock_model as M


def _tri(sub, diag, sup, rhs):
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = sup[:-1]
    ab[1, :] = diag
    ab[2, :-1] = sub[1:]
    return solve_banded((1, 1), ab, rhs)


def _slope(v, Z, dZ, margin=1e-7):
    p = np.empty_like(v)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
    p[0] = (v[1] - v[0]) / dZ
    p[-1] = (v[-1] - v[-2]) / dZ
    lo = -1.0 / np.maximum(1.0 - Z, 1e-9) + margin
    hi = 1.0 / np.maximum(Z, 1e-9) - margin
    return np.clip(p, lo, hi)


def solve_fd_shock(P, n=2000, dtau=2.0, max_iter=200000, tol=1e-12, verbose=False):
    Z = np.linspace(0.0, 1.0, n + 1)
    dZ = 1.0 / n
    v0, vN = M.boundary_values(P)
    sd2, sg2 = P["sigma_d"] ** 2, P["sigma_g"] ** 2
    sdg = P.get("rho", 0.0) * P["sigma_d"] * P["sigma_g"]   # cross-covariance of the shocks

    # perturbation-style init (deterministic symmetric value, shifted)
    Abar = (1.0 - Z) * P["A_d"] + Z * P["A_g"]
    c_sym = P["delta"] * (1.0 + P["theta_d"] * Abar) / (P["theta_d"] * (P["delta"] + P["Gamma_d"]))
    v = (np.log(c_sym) + ((1 - Z) * P["alpha_d"] + Z * P["alpha_g"]) / P["delta"]
         + (P["Gamma_d"] / P["delta"]) * np.log(P["Gamma_d"] * P["theta_d"] * c_sym / P["delta"]))
    v[0], v[-1] = v0, vN
    idx = np.arange(1, n)
    last = 0
    for it in range(max_iter):
        slope = _slope(v, Z, dZ)
        i_d, i_g, c = M.controls(Z, slope, P)
        phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        v1 = (phi_g - phi_d + (1.0 - Z) * sd2 - Z * sg2
              + (2.0 * Z - 1.0) * sdg) * Z * (1.0 - Z)                      # v' coefficient
        v2 = 0.5 * Z ** 2 * (1.0 - Z) ** 2 * (sd2 + sg2 - 2.0 * sdg)        # v'' coefficient
        flow = (P["delta"] * np.log(np.maximum(c, 1e-12)) + (1.0 - Z) * phi_d + Z * phi_g
                - 0.5 * (sd2 * (1.0 - Z) ** 2 + sg2 * Z ** 2 + 2.0 * sdg * Z * (1.0 - Z))
                + M.robustness_drag(Z, slope, P))   # xi-> inf gives 0 (no robustness)

        a1, a2, fl = v1[idx], v2[idx], flow[idx]
        diff = a2 / dZ ** 2
        fwd = a1 > 0.0
        adv = np.abs(a1) / dZ
        diag = np.full(n - 1, 1.0 / dtau + P["delta"]) + 2.0 * diff + adv
        sub = -diff.copy()
        sup = -diff.copy()
        sub[~fwd] += a1[~fwd] / dZ     # backward where v1<0: +a1/dZ (negative) on sub
        sup[fwd] += -a1[fwd] / dZ      # forward where v1>0: -a1/dZ on sup
        rhs = v[idx] / dtau + fl
        rhs[0] -= sub[0] * v0
        rhs[-1] -= sup[-1] * vN

        v_new = _tri(sub, diag, sup, rhs)
        step = np.max(np.abs(v_new - v[idx]))
        v[idx] = v_new
        last = it
        if verbose and it % 5000 == 0:
            print(f"  [FD-shock] it {it:6d} max|dv|={step:.3e}", flush=True)
        if step < tol:
            break

    slope = _slope(v, Z, dZ)
    i_d, i_g, c = M.controls(Z, slope, P)
    # second-order residual using central v', v''
    vpp = np.zeros_like(v)
    vpp[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dZ ** 2
    resid = M.hjb_residual(Z, v, slope, vpp, P)
    return {"Z": Z, "v": v, "slope": slope, "i_d": i_d, "i_g": i_g, "c": c,
            "ratio": i_g / i_d, "residual": resid, "iters": last + 1,
            "max_abs_residual": float(np.max(np.abs(resid[2:-2])))}


if __name__ == "__main__":
    P = M.load_calibration("A_g_prime_prime")
    print(f"sigma_d={P['sigma_d']}, sigma_g={P['sigma_g']}")
    out = solve_fd_shock(P, verbose=True)
    print(f"FD-shock: iters={out['iters']} max|resid|={out['max_abs_residual']:.2e}")
    for z in (0.1, 0.3, 0.5, 0.7, 0.9):
        k = int(z * (len(out["Z"]) - 1))
        print(f"  Z={z:.1f} v'={out['slope'][k]:.4f} i_d={out['i_d'][k]:.5f} i_g={out['i_g'][k]:.5f}")

    # sigma -> 0 must recover the deterministic solution
    import two_capital_model as DET
    from fd_solver import solve_fd as solve_fd_det
    P0 = M.load_calibration("A_g_prime_prime"); P0["sigma_d"] = 0.0; P0["sigma_g"] = 0.0
    o0 = solve_fd_shock(P0)
    od = solve_fd_det(DET.load_calibration("A_g_prime_prime"))
    diff = max(abs(np.interp(z, o0["Z"], o0["i_g"]) - np.interp(z, od["Z"], od["i_g"]))
               for z in np.linspace(0.1, 0.9, 17))
    print(f"\n[sigma->0 check] max|i_g(shock,sigma=0) - i_g(deterministic)| on [0.1,0.9] = {diff:.2e}")
