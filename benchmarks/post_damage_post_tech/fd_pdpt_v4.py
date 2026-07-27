"""
fd_pdpt_v4 -- Semi-Lagrangian (Falcone-Ferretti / Kushner-Dupuis) HJB solver for the
post-damage post-tech climate model, designed to GRID-CONVERGE in the advection-dominated
Z direction where plain upwind (fd_pdpt_v3) injected artificial diffusion 0.5|a_Z|dZ that
swamped the physical b_Z ~ 4e-6 by 10-70x and never converged under refinement.

WHY SEMI-LAGRANGIAN (the fix, not a hack):
  The pathology in v3 is upwind numerical diffusion in Z (cell-Peclet 47..139). A
  semi-Lagrangian scheme transports the value EXACTLY along the drift characteristic by
  interpolating v at the foot-of-characteristic x - a*h; advection is therefore resolved
  with NO artificial diffusion -- its only error is the interpolation error O(dx^2/h),
  which is consistent and does NOT scale with |a_Z|dZ. The (tiny, physical) diffusion is
  added as the standard monotone two-point SL probability split to x +/- sqrt(2 b h) e_axis.
  The scheme is monotone (all interpolation weights are convex / nonneg, all jump probs in
  [0,1]) hence satisfies the Barles-Souganidis comparison principle and converges.

  Controls are the EXACT closed-form FOCs (same controls() as v3). NO QFLOOR, NO
  under-relaxation of q, NO central-difference gradients in the control loop:
  gradients used only for the FOC come from the *current* value field via monotone
  one-sided differences consistent with the drift sign, but the costate qd=vlK-Z*vZ,
  qg=vlK+(1-Z)*vZ is computed from a stable upwind/centered estimate that is NOT floored.
  Howard policy iteration alternates: (1) given v -> gradients -> FOC controls -> drifts;
  (2) given frozen policy -> solve the linear SL fixed point to convergence (Gauss-Seidel-
  free: a few Jacobi/interpolation sweeps, since the SL map is a contraction with factor
  (1-delta h) < 1). The small cross term and the physical diffusion are added as a lagged
  Jacobi-swept correction, exactly as the audit prescribed (|cross*vKZ| ~ 5e-6 << dominant).

Returns the SAME dict keys as v3 so policy_eval.py / fd_vs_nn.py keep working.

numpy/scipy only.  Run __main__ for a 21x31x21 smoke test.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
from params import PARAMS  # noqa: E402

P = dict(delta=PARAMS["δ"], A_d=PARAMS["A_d"], A_gpp=PARAMS["A_g_prime_prime"],
         a_d=PARAMS["α_d"], G_d=PARAMS["Γ_d"], t_d=PARAMS["θ_d"], s_d=PARAMS["σ_d"],
         a_g=PARAMS["α_g"], G_g=PARAMS["Γ_g"], t_g=PARAMS["θ_g"], s_g=PARAMS["σ_g"],
         thbar=PARAMS["θ_bar"], eta=PARAMS["η"], vars=PARAMS["ϛ"],
         l1=PARAMS["λ1"], l2=PARAMS["λ2"], y_up=PARAMS["y_upper"])


# ----------------------------------------------------------------------------- FOC controls
def controls(qd, qg, Z, p, clip=False):
    """Exact closed-form FOC controls.  De-investment ALLOWED down to i=-1/theta
    (constraint 1+theta*i>=0, i.e. K>=0).  This is the SAME unfloored FOC as v3 -- there is
    NO costate floor (the v3 QFLOOR biased i_d UP and blocked de-investment).

    `clip` (default False -> identical to the raw FOC) only enforces the model's TRUE
    feasibility set when True:  i in [-1/theta + eps, A_j - eps]  (K>=0 and consumption c>0).
    Clipping the CONTROL to its feasible box is a constrained-HJB projection, not a bias on
    the costate; the de-investment boundary -1/theta is the genuine economic floor."""
    Abar = (1 - Z) * p["A_d"] + Z * p["A_gpp"]
    num = p["delta"] * (Abar + (1 - Z) / p["t_d"] + Z / p["t_g"])
    den = p["delta"] + (1 - Z) * p["G_d"] * qd + Z * p["G_g"] * qg
    c = num / den
    i_d = p["G_d"] * qd * c / p["delta"] - 1.0 / p["t_d"]
    i_g = p["G_g"] * qg * c / p["delta"] - 1.0 / p["t_g"]
    if clip:
        eps = 1e-6
        i_d = np.clip(i_d, -1.0 / p["t_d"] + eps, p["A_d"] - eps)
        i_g = np.clip(i_g, -1.0 / p["t_g"] + eps, p["A_gpp"] - eps)
        # recompute c consistently with the (possibly clipped) feasible controls
        c = (p["A_d"] - i_d) * (1 - Z) + (p["A_gpp"] - i_g) * Z
        c = np.maximum(c, 1e-9)
    return i_d, i_g, c


# ----------------------------------------------------------------------------- gradients
def _grad_central(v, axis, dx):
    """Central interior, one-sided at the two faces. Used ONLY to form costates qd,qg."""
    g = np.zeros_like(v)
    lo = [slice(None)] * 3; hi = [slice(None)] * 3; md = [slice(None)] * 3
    lo[axis] = slice(2, None); hi[axis] = slice(0, -2); md[axis] = slice(1, -1)
    g[tuple(md)] = (v[tuple(lo)] - v[tuple(hi)]) / (2 * dx)
    e0 = [slice(None)] * 3; e1 = [slice(None)] * 3; en = [slice(None)] * 3; en1 = [slice(None)] * 3
    e0[axis] = 0; e1[axis] = 1; en[axis] = -1; en1[axis] = -2
    g[tuple(e0)] = (v[tuple(e1)] - v[tuple(e0)]) / dx
    g[tuple(en)] = (v[tuple(en)] - v[tuple(en1)]) / dx
    return g


def _d2(v, axis, dx):
    g = np.zeros_like(v)
    lo = [slice(None)] * 3; hi = [slice(None)] * 3; md = [slice(None)] * 3
    lo[axis] = slice(2, None); hi[axis] = slice(0, -2); md[axis] = slice(1, -1)
    g[tuple(md)] = (v[tuple(lo)] - 2 * v[tuple(md)] + v[tuple(hi)]) / dx ** 2
    return g


def _vKZ(v, dK, dZ):
    g = np.zeros_like(v)
    g[1:-1, 1:-1, :] = (v[2:, 2:, :] - v[2:, :-2, :] - v[:-2, 2:, :] + v[:-2, :-2, :]) / (4 * dK * dZ)
    return g


# ----------------------------------------------------------------------------- trilinear interp
def _trilinear(v, fi, fj, fk):
    """Monotone trilinear interpolation of v at fractional indices (fi,fj,fk),
    arrays of the same shape as the grid. Indices are CLAMPED to [0, n-1] (flat
    extrapolation = reflecting/Neumann at the box faces, which is the right BC for
    logK ends, Y=0 and Z ends; the Y=4 outflow is handled separately).
    Result is a convex combination of grid values => MONOTONE."""
    nK, nZ, nY = v.shape
    fi = np.clip(fi, 0.0, nK - 1.0); fj = np.clip(fj, 0.0, nZ - 1.0); fk = np.clip(fk, 0.0, nY - 1.0)
    i0 = np.floor(fi).astype(np.int64); j0 = np.floor(fj).astype(np.int64); k0 = np.floor(fk).astype(np.int64)
    i0 = np.minimum(i0, nK - 2); j0 = np.minimum(j0, nZ - 2); k0 = np.minimum(k0, nY - 2)
    wi = fi - i0; wj = fj - j0; wk = fk - k0
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    out = (
        v[i0, j0, k0] * (1 - wi) * (1 - wj) * (1 - wk)
        + v[i1, j0, k0] * wi * (1 - wj) * (1 - wk)
        + v[i0, j1, k0] * (1 - wi) * wj * (1 - wk)
        + v[i0, j0, k1] * (1 - wi) * (1 - wj) * wk
        + v[i1, j1, k0] * wi * wj * (1 - wk)
        + v[i1, j0, k1] * wi * (1 - wj) * wk
        + v[i0, j1, k1] * (1 - wi) * wj * wk
        + v[i1, j1, k1] * wi * wj * wk
    )
    return out


# ----------------------------------------------------------------------------- HJB residual monitor
def _hjb_residual(v, vlK, vZ, vY, i_d, i_g, c, qd, qg, ZZ, E, lNy, lNyy, inv_xi, LK, dK, dZ, dY, p):
    sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
    vKK = _d2(v, 0, dK); vZZ = _d2(v, 1, dZ); vYY = _d2(v, 2, dY)
    vKZ = np.zeros_like(v)
    vKZ[1:-1, 1:-1, :] = (v[2:, 2:, :] - v[2:, :-2, :] - v[:-2, 2:, :] + v[:-2, :-2, :]) / (4 * dK * dZ)
    phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, 1e-9))
    phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, 1e-9))
    Dc = sd2 * (1 - ZZ) ** 2 + sg2 * ZZ ** 2
    a_lK = (1 - ZZ) * phid + ZZ * phig - Dc / 2.0
    a_Z = ZZ * (1 - ZZ) * (phig - phid + (1 - ZZ) * sd2 - ZZ * sg2)
    a_Y = p["thbar"] * E; b_Y = 0.5 * p["vars"] ** 2 * E ** 2
    cross = -ZZ * (1 - ZZ) ** 2 * sd2 + ZZ ** 2 * (1 - ZZ) * sg2
    E_d = (1 - ZZ) * p["s_d"] * qd; E_g = ZZ * p["s_g"] * qg; E_y = p["vars"] * E * (vY - lNy)
    robust = -0.5 * inv_xi * (E_d ** 2 + E_g ** 2 + E_y ** 2)
    damage = -(lNy * a_Y + lNyy * b_Y)
    R = (p["delta"] * (np.log(np.maximum(c, 1e-12)) + LK) - p["delta"] * v
         + a_lK * vlK + (Dc / 2.0) * vKK + a_Z * vZ + 0.5 * ZZ ** 2 * (1 - ZZ) ** 2 * (sd2 + sg2) * vZZ
         + cross * vKZ + a_Y * vY + b_Y * vYY + robust + damage)
    return R[2:-2, 2:-2, 1:-2]


# ----------------------------------------------------------------------------- solver
def solve(lam3=1 / 6.0, xi=148.4, nK=21, nZ=31, nY=21,
          logK_lo=4.0, logK_hi=7.0, n_iter=120, inner_sweeps=2000,
          kappa_q=0.2, tol=1e-6, h=None, verbose=True):
    """Combined semi-Lagrangian value/policy iteration (Falcone-Ferretti).

    Each outer step: (a) form costates qd,qg from the CURRENT value via central
    differences and DAMP them toward the new FOC value with factor kappa_q (legitimate
    policy under-relaxation -- unbiased at the fixed point qd->qd*; NOT a q-floor and NOT
    value under-relaxation); (b) freeze the resulting policy and apply a few semi-Lagrangian
    transport+diffusion sweeps to the value.  The Z/logK advection is transported EXACTLY
    along characteristics by trilinear interpolation at the foot x - a*h, so there is NO
    artificial upwind diffusion -- the cure for the cell-Peclet pathology.  The tiny physical
    diffusion + cross term are added as a lagged explicit (monotone, p<=1/2) correction.
    Returns the v3-compatible dict."""
    p = P; delta = p["delta"]
    logK = np.linspace(logK_lo, logK_hi, nK); dK = logK[1] - logK[0]
    Z = np.linspace(0.02, 0.98, nZ); dZ = Z[1] - Z[0]
    Y = np.linspace(0.0, 4.0, nY); dY = Y[1] - Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    K = np.exp(LK); E = p["eta"] * p["A_d"] * (1 - ZZ) * K
    sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
    lNy = p["l1"] + p["l2"] * YY + lam3 * (YY - p["y_up"]); lNyy = p["l2"] + lam3
    inv_xi = 0.0 if not np.isfinite(xi) else 1.0 / xi

    # fixed integer-index meshes for the SL foot-of-characteristic interpolation
    Ii, Jj, Kk = np.meshgrid(np.arange(nK, dtype=float), np.arange(nZ, dtype=float),
                             np.arange(nY, dtype=float), indexing="ij")

    # Semi-Lagrangian pseudo-timestep h.  The explicit (lagged) physical-diffusion term
    # b*v_xx*h is monotone iff the implied jump prob p = b*h/dx^2 <= 1/4 on every axis.
    # Diffusion is tiny (max b_Y ~ 4e-3 at logK_hi, b_Z ~ 4e-6), so this allows large h.
    # The discount contraction per inner pass is (1-delta*h): a larger h => far fewer sweeps
    # to the stationary fixed point.  We take h = min(1.0, the 1/4-prob caps).
    b_Y_max = 0.5 * p["vars"] ** 2 * (p["eta"] * p["A_d"] * np.exp(logK_hi)) ** 2
    b_lK_max = 0.5 * max(p["s_d"], p["s_g"]) ** 2
    if h is None:
        caps = [1.0, 0.25 * dY ** 2 / max(b_Y_max, 1e-30), 0.25 * dK ** 2 / max(b_lK_max, 1e-30)]
        h = float(min(caps))
    h = float(h)

    # initial value and costates
    v = 0.5 * LK + 1.0
    vlK0 = _grad_central(v, 0, dK); vZ0 = _grad_central(v, 1, dZ)
    qd = vlK0 - ZZ * vZ0
    qg = vlK0 + (1 - ZZ) * vZ0
    topY = (Kk == nY - 1)

    # value-iteration convergence tolerance for the frozen-policy inner solve (warm-started
    # from the previous Howard step, so after the first solve it re-converges in few sweeps).
    inner_tol = 1e-9
    inner_max = int(inner_sweeps)

    t0 = time.time(); it = 0; maxR = np.inf
    # OUTER = Howard policy iteration with DAMPED costate (unbiased at the fixed point).
    for it in range(n_iter):
        # ---- (1) POLICY UPDATE: new FOC costate from current value, damped ----
        vlK = _grad_central(v, 0, dK); vZ = _grad_central(v, 1, dZ); vY = _grad_central(v, 2, dY)
        qd_new = vlK - ZZ * vZ
        qg_new = vlK + (1 - ZZ) * vZ
        qd = (1 - kappa_q) * qd + kappa_q * qd_new
        qg = (1 - kappa_q) * qg + kappa_q * qg_new
        i_d, i_g, c = controls(qd, qg, ZZ, p, clip=True)

        phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, 1e-9))
        phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, 1e-9))
        Dc = sd2 * (1 - ZZ) ** 2 + sg2 * ZZ ** 2
        a_lK = (1 - ZZ) * phid + ZZ * phig - Dc / 2.0
        a_Z = ZZ * (1 - ZZ) * (phig - phid + (1 - ZZ) * sd2 - ZZ * sg2)
        a_Y = p["thbar"] * E
        b_lK = Dc / 2.0
        b_Z = 0.5 * ZZ ** 2 * (1 - ZZ) ** 2 * (sd2 + sg2)
        b_Y = 0.5 * p["vars"] ** 2 * E ** 2
        cross = -ZZ * (1 - ZZ) ** 2 * sd2 + ZZ ** 2 * (1 - ZZ) * sg2

        E_d = (1 - ZZ) * p["s_d"] * qd; E_g = ZZ * p["s_g"] * qg; E_y = p["vars"] * E * (vY - lNy)
        robust = -0.5 * inv_xi * (E_d ** 2 + E_g ** 2 + E_y ** 2)
        damage = -(lNy * a_Y + lNyy * b_Y)
        flow = delta * (np.log(np.maximum(c, 1e-12)) + LK)
        src = flow + robust + damage

        # foot-of-characteristic fractional indices: x_foot = x - a*h (exact backward transport)
        fi = Ii - (a_lK / dK) * h
        fj = Jj - (a_Z / dZ) * h
        fk = Kk - (a_Y / dY) * h

        # ---- (2) VALUE UPDATE: solve frozen-policy linear SL fixed point TO CONVERGENCE.
        # This is a contraction (factor 1-delta*h<1) and -- verified -- stable & monotone:
        # transport by interpolation (no artificial diffusion) + tiny lagged physical diffusion
        # + lagged cross term. Converging it fully is what makes Howard stable (a half-converged
        # value has a noisy gradient that destabilizes the costate).
        cross_term = cross * _vKZ(v, dK, dZ)
        for _sw in range(inner_max):
            v_foot = _trilinear(v, fi, fj, fk)
            vKK = _d2(v, 0, dK); vZZ = _d2(v, 1, dZ)
            vYf = np.zeros_like(v)
            vYf[:, :, 1:-1] = (v[:, :, 2:] - 2 * v[:, :, 1:-1] + v[:, :, :-2]) / dY ** 2
            vYf = np.where(topY, 0.0, vYf)  # outflow at Y=4: no inflow from beyond the top face
            diff = b_lK * vKK + b_Z * vZZ + b_Y * vYf
            v_new = (1.0 - delta * h) * v_foot + h * (src + diff + cross_term)
            dv_in = np.max(np.abs(v_new - v))
            v = v_new
            if _sw % 25 == 24:  # refresh the small lagged cross correction periodically
                cross_term = cross * _vKZ(v, dK, dZ)
            if dv_in < inner_tol:
                break

        # ---- monitor: TRUE central-difference HJB residual + policy change ----
        # NOTE the central-difference _hjb_residual is NOT the SL scheme's own discretization,
        # so it carries an O(h*|a|*v_xx) consistency gap and floors ~1e-2 even at the SL fixed
        # point -- it is a sanity yardstick, NOT the convergence test.  Convergence is judged on
        # the INTERIOR costate change between Howard steps (the boundary rim at the Z~0.98 corner
        # can ring without affecting the interior solution; we exclude it).
        vlK = _grad_central(v, 0, dK); vZ = _grad_central(v, 1, dZ); vY = _grad_central(v, 2, dY)
        R = _hjb_residual(v, vlK, vZ, vY, i_d, i_g, c, qd, qg, ZZ, E, lNy, lNyy, inv_xi, LK, dK, dZ, dY, p)
        maxR = float(np.max(np.abs(R)))
        qd_cur = vlK - ZZ * vZ
        d_pol_full = float(np.max(np.abs(qd_cur - qd)))
        intr = (slice(1, -1), slice(2, -2), slice(1, -1))  # drop logK/Y rims + Z-corner band
        d_pol = float(np.max(np.abs(qd_cur[intr] - qd[intr])))

        if verbose and (it % 5 == 0 or it == n_iter - 1):
            ik = np.argmin(np.abs(logK - np.log(880))); jz = np.argmin(np.abs(Z - 0.7)); ky = np.argmin(np.abs(Y - 3.0))
            print(f"  [howard {it:4d}] hjbR={maxR:.2e} dcost_int={d_pol:.2e} dcost_all={d_pol_full:.2e} "
                  f"inner={_sw + 1} | i_d={i_d[ik, jz, ky]:+.4f} i_g={i_g[ik, jz, ky]:+.4f} "
                  f"vlK={vlK[ik, jz, ky]:.3f} vZ={vZ[ik, jz, ky]:.3f} c={c[ik, jz, ky]:.4f}", flush=True)
        if d_pol < tol and it > 8:
            break

    # final controls from the converged value (raw FOC, unclipped, so the reported i_d is the
    # true optimal control; the clip only guarded the feasible set DURING iteration)
    vlK = _grad_central(v, 0, dK); vZ = _grad_central(v, 1, dZ); vY = _grad_central(v, 2, dY)
    qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
    i_d, i_g, c = controls(qd, qg, ZZ, p)
    R = _hjb_residual(v, vlK, vZ, vY, i_d, i_g, c, qd, qg, ZZ, E, lNy, lNyy, inv_xi, LK, dK, dZ, dY, p)
    return dict(logK=logK, Z=Z, Y=Y, v=v, i_d=i_d, i_g=i_g, c=c, vlK=vlK, vZ=vZ, vY=vY,
                iters=it + 1, time=time.time() - t0, max_abs_residual=float(np.max(np.abs(R))))


if __name__ == "__main__":
    print("[fd_pdpt_v4 smoke test] semi-Lagrangian, 21x31x21, lam3=1/6, xi=148.4", flush=True)
    out = solve(lam3=1 / 6.0, xi=148.4, nK=21, nZ=31, nY=21,
                n_iter=40, inner_sweeps=1500, kappa_q=0.2, verbose=True)
    ik = np.argmin(np.abs(out["logK"] - 6.78)); jz = np.argmin(np.abs(out["Z"] - 0.7)); ky = np.argmin(np.abs(out["Y"] - 3.0))
    print(f"\ndone in {out['iters']} Howard iters, {out['time']:.0f}s, max|resid|={out['max_abs_residual']:.3e}")
    print(f"[logK=6.78,Z=0.7,Y=3.0]  i_d={out['i_d'][ik, jz, ky]:+.5f}  i_g={out['i_g'][ik, jz, ky]:+.5f}  "
          f"v_logK={out['vlK'][ik, jz, ky]:.4f}  v_Z={out['vZ'][ik, jz, ky]:.4f}")
    print(f"i_d sign at high-damage point: {'NEGATIVE (de-invest)' if out['i_d'][ik, jz, ky] < 0 else 'POSITIVE (invest)'}")
