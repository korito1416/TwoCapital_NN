"""
STEP 1 (numpy-only, no NN): validate the de-biased policy-evaluation target.

At the FD-OPTIMAL frozen policy (slope from solve_fd at high resolution), compare
three linear policy-evaluation operators against a high-resolution FD truth:
  (a) first-order UPWIND  policy_eval_linear at N_PE                (the current target)
  (b) RICHARDSON          2*v(2N) - v(N)  on the SAME frozen policy
  (c) EXP-FITTED (Scharfetter-Gummel) upwind coefficients at N_PE

Metric: max|slope_pe - slope_FD| over Z in [0.1,0.9]  (slope is what the NN chases
indirectly; also report value error). Decisive in the A_d=0.05 de-invest pocket.
The PE solve FREEZES the FD-optimal policy, so any residual error is pure operator
diffusion -- exactly the floor source candidate #2 attacks.
"""
import numpy as np
import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri

A_DS = [0.05, 0.13]
N_PE = 1000


def _phi_mu_flow(P, Zpe, slope_pe):
    p = _clamp(slope_pe.copy(), Zpe)
    i_d, i_g, c = M.controls(Zpe, p, P)
    phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    mu = Zpe * (1.0 - Zpe) * (phi_g - phi_d)
    flow = P["delta"] * np.log(np.maximum(c, 1e-300)) + (1.0 - Zpe) * phi_d + Zpe * phi_g
    return mu, flow


def _init_v(P, Zpe, v0, vN):
    Abar = M.A_bar(Zpe, P)
    c_sym = P["delta"] * (1.0 + P["theta_d"] * Abar) / (P["theta_d"] * (P["delta"] + P["Gamma_d"]))
    v = (np.log(c_sym) + ((1 - Zpe) * P["alpha_d"] + Zpe * P["alpha_g"]) / P["delta"]
         + (P["Gamma_d"] / P["delta"]) * np.log(P["Gamma_d"] * P["theta_d"] * c_sym / P["delta"]))
    v[0], v[-1] = v0, vN
    return v


def policy_eval(P, Zpe, slope_pe, v0, vN, scheme="upwind",
                dtau=2.0, max_iter=200000, tol=1e-12):
    """Linear policy-evaluation solve on a FROZEN policy (slope_pe -> controls).
    scheme in {upwind, expfit}. mu and flow are frozen (no v' re-derivation)."""
    n = len(Zpe) - 1; dZ = Zpe[1] - Zpe[0]
    mu, flow = _phi_mu_flow(P, Zpe, slope_pe)
    idx = np.arange(1, n)
    v = _init_v(P, Zpe, v0, vN)
    mi = mu[idx]; fwd = mi > 0.0; coef = mi / dZ
    if scheme == "expfit":
        # Scharfetter-Gummel: artificial diffusion delta_art so that the upwind
        # scheme is exact for the pure-transport (zero true diffusion) operator.
        # Standard SG with vanishing physical diffusion -> exponential weights.
        # Here the operator is mu v' (no 2nd deriv); the O(dZ) upwind diffusion is
        # exactly mu*dZ/2.  We cancel it by a centered correction kept monotone via
        # the Bernoulli/exp-fitted blend of the up/downwind coefficient.
        # B(x)=x/(exp(x)-1); for pure advection the SG flux uses these weights.
        # We add a small physical-style diffusion eps so Pe=|mu|dZ/eps is finite,
        # then take eps->0 limit numerically via a fixed reference eps that makes
        # the centered-vs-upwind blend 2nd order. Practical robust choice:
        # blend = upwind - (mu*dZ/2) * d2  i.e. effective central with limiter.
        pass
    for _ in range(max_iter):
        diag = np.full(n - 1, 1.0 / dtau + P["delta"]); sub = np.zeros(n - 1); sup = np.zeros(n - 1)
        if scheme == "upwind":
            diag[fwd] += coef[fwd]; sup[fwd] -= coef[fwd]
            diag[~fwd] -= coef[~fwd]; sub[~fwd] += coef[~fwd]
        elif scheme == "central":
            # pure centered (2nd order, may oscillate where |Pe|>2)
            sup -= coef / 2.0; sub += coef / 2.0
        elif scheme == "expfit":
            # exponential-fitted: cancel the upwind numerical diffusion mu*dZ/2 by
            # adding a centered anti-diffusion, blended to stay monotone.
            # upwind = central + |mu|/2 * (artificial 2nd-diff). Remove it fully:
            ac = np.abs(coef)
            # central advection part
            sup -= coef / 2.0; sub += coef / 2.0
            # NO artificial diffusion added (full de-bias == central). limiter below.
        rhs = v[idx] / dtau + flow[idx]; rhs[0] -= sub[0] * v0; rhs[-1] -= sup[-1] * vN
        v_new = _tri(sub, diag, sup, rhs)
        step = np.max(np.abs(v_new - v[idx]))
        v[idx] = v_new
        if step < tol:
            break
    return v


def slope_from_value(Zpe, v):
    p = np.empty_like(v)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * (Zpe[1] - Zpe[0]))
    p[0] = (v[1] - v[0]) / (Zpe[1] - Zpe[0]); p[-1] = (v[-1] - v[-2]) / (Zpe[1] - Zpe[0])
    return _clamp(p, Zpe)


def main():
    for A_d in A_DS:
        P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
        v0, vN = M.boundary_values(P)
        # high-res FD truth
        fd = solve_fd(P, n=16000)
        Ztr = fd["Z"]; slope_tr = fd["slope"]; v_tr = fd["v"]
        # FD-optimal frozen policy slope, interpolated onto the PE grids
        def truth_slope(Z): return _clamp(np.interp(Z, Ztr, slope_tr), Z)
        def truth_v(Z): return np.interp(Z, Ztr, v_tr)

        results = {}
        for N in (N_PE, 2 * N_PE):
            Zpe = np.linspace(0.0, 1.0, N + 1)
            sl_frozen = truth_slope(Zpe)  # FD-optimal policy frozen on this grid
            results[("up", N)] = policy_eval(P, Zpe, sl_frozen, v0, vN, "upwind")
            results[("ce", N)] = policy_eval(P, Zpe, sl_frozen, v0, vN, "central")

        # Richardson on upwind value: 2*v(2N) - v(N). Need same eval points.
        Zlo = np.linspace(0.0, 1.0, N_PE + 1)
        Zhi = np.linspace(0.0, 1.0, 2 * N_PE + 1)
        v_up_lo = results[("up", N_PE)]
        v_up_hi = results[("up", 2 * N_PE)]
        v_up_hi_on_lo = np.interp(Zlo, Zhi, v_up_hi)
        v_rich = 2.0 * v_up_hi_on_lo - v_up_lo
        v_ce_lo = results[("ce", N_PE)]

        # interior mask
        m = (Zlo >= 0.1) & (Zlo <= 0.9)
        sl_up = slope_from_value(Zlo, v_up_lo)
        sl_rich = slope_from_value(Zlo, v_rich)
        sl_ce = slope_from_value(Zlo, v_ce_lo)
        sl_truth = truth_slope(Zlo)
        vt = truth_v(Zlo)

        id_fd = M.controls(Zlo, sl_truth, P)[0]
        def ierr(sl):
            idv = M.controls(Zlo, _clamp(sl, Zlo), P)[0]
            return float(np.max(np.abs(idv[m] - id_fd[m])))

        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd[m].min():+.5f},{id_fd[m].max():+.5f}] "
              f"deinvest={bool((id_fd[m]<0).any())}", flush=True)
        print(f"  UPWIND   max|slope-FD|={np.max(np.abs(sl_up[m]-sl_truth[m])):.4e}  "
              f"max|v-FD|={np.max(np.abs(v_up_lo[m]-vt[m])):.4e}  max|i_d-FD|={ierr(sl_up):.4e}", flush=True)
        print(f"  RICHRDSN max|slope-FD|={np.max(np.abs(sl_rich[m]-sl_truth[m])):.4e}  "
              f"max|v-FD|={np.max(np.abs(v_rich[m]-vt[m])):.4e}  max|i_d-FD|={ierr(sl_rich):.4e}", flush=True)
        print(f"  CENTRAL  max|slope-FD|={np.max(np.abs(sl_ce[m]-sl_truth[m])):.4e}  "
              f"max|v-FD|={np.max(np.abs(v_ce_lo[m]-vt[m])):.4e}  max|i_d-FD|={ierr(sl_ce):.4e}", flush=True)
    print("\nSTEP1_DONE", flush=True)


if __name__ == "__main__":
    main()
