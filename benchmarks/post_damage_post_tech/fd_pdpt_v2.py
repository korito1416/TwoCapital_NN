"""
fd_pdpt_v2.py  --  Unified FD scheme for the post-damage post-tech robust HJB.

State (logK, Z, Y) at fixed (lambda3, xi). Implements the agreed UNIFIED SCHEME SPEC:

  * Howard policy iteration (outer): freeze closed-form FOC controls + worst-case h.
  * Line-implicit Douglas-ADI inner solve of the frozen-coefficient linear PDE.
  * M-matrix per direction: upwind on the SIGN of the frozen drift + central diffusion.
  * Robust Y-term FROZEN (linear Lagrangian h_y) and folded into a_Y^eff -> kills the
    anti-diffusion that blew up the explicit scheme.  Capital robust channels stay on RHS.
  * Cross term v_logKZ EXPLICIT / lagged on RHS only (never in the matrix).
  * b_Y IMPLICIT in the Y-sweep (the bicgstab-staller).
  * Y=4 OUTFLOW (backward upwind, drop diffusion); Neumann elsewhere.
  * lambda3 included in (logN)_Y for BOTH damage drag AND h_y.
  * q-tilde soft floor 0.02 + clamp 1+theta*i>=0.05, under-relaxed (kappa=0.3).
  * Residual reported with CENTRAL differences.

Run:  module load python/anaconda-2021.05 ; export PYTHONNOUSERSITE=1 ; python fd_pdpt_v2.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
from params import PARAMS  # noqa: E402

P = dict(delta=PARAMS["δ"], A_d=PARAMS["A_d"], A_gpp=PARAMS["A_g_prime_prime"],
         a_d=PARAMS["α_d"], G_d=PARAMS["Γ_d"], t_d=PARAMS["θ_d"], s_d=PARAMS["σ_d"],
         a_g=PARAMS["α_g"], G_g=PARAMS["Γ_g"], t_g=PARAMS["θ_g"], s_g=PARAMS["σ_g"],
         thbar=PARAMS["θ_bar"], eta=PARAMS["η"], vars=PARAMS["ϛ"],
         l1=PARAMS["λ1"], l2=PARAMS["λ2"], y_up=PARAMS["y_upper"])

QFLOOR = 0.02
CLAMP = 0.05   # 1 + theta*i >= CLAMP


# ----------------------------------------------------------------------------- controls
def controls(qd, qg, Z, p):
    """Closed-form FOC controls given (already floored) q-tilde derivatives qd,qg.
    Then clamp 1+theta_j i_j >= CLAMP and recompute c consistently."""
    Abar = (1 - Z) * p["A_d"] + Z * p["A_gpp"]
    num = p["delta"] * (Abar + (1 - Z) / p["t_d"] + Z / p["t_g"])
    den = p["delta"] + (1 - Z) * p["G_d"] * qd + Z * p["G_g"] * qg
    c = num / den
    i_d = p["G_d"] * qd * c / p["delta"] - 1.0 / p["t_d"]
    i_g = p["G_g"] * qg * c / p["delta"] - 1.0 / p["t_g"]
    # clamp feasibility 1+theta*i >= CLAMP
    i_d = np.maximum(i_d, (CLAMP - 1.0) / p["t_d"])
    i_g = np.maximum(i_g, (CLAMP - 1.0) / p["t_g"])
    # recompute c from clamped controls for consistency in the flow term
    c = (p["A_d"] - i_d) * (1 - Z) + (p["A_gpp"] - i_g) * Z
    c = np.maximum(c, 1e-12)
    return i_d, i_g, c


def logNy(Y, lam3, p):
    return p["l1"] + p["l2"] * Y + lam3 * (Y - p["y_up"])


def logNyy(lam3, p):
    return p["l2"] + lam3


# ----------------------------------------------------------------------------- derivs
def _central(v, axis, dx):
    g = np.zeros_like(v)
    lo = [slice(None)] * 3; lo[axis] = slice(2, None)
    hi = [slice(None)] * 3; hi[axis] = slice(0, -2)
    md = [slice(None)] * 3; md[axis] = slice(1, -1)
    g[tuple(md)] = (v[tuple(lo)] - v[tuple(hi)]) / (2 * dx)
    e0 = [slice(None)] * 3; e0[axis] = 0; e1 = [slice(None)] * 3; e1[axis] = 1
    en = [slice(None)] * 3; en[axis] = -1; en1 = [slice(None)] * 3; en1[axis] = -2
    g[tuple(e0)] = (v[tuple(e1)] - v[tuple(e0)]) / dx
    g[tuple(en)] = (v[tuple(en)] - v[tuple(en1)]) / dx
    return g


def _d2(v, axis, dx):
    g = np.zeros_like(v)
    lo = [slice(None)] * 3; lo[axis] = slice(2, None)
    hi = [slice(None)] * 3; hi[axis] = slice(0, -2)
    md = [slice(None)] * 3; md[axis] = slice(1, -1)
    g[tuple(md)] = (v[tuple(lo)] - 2 * v[tuple(md)] + v[tuple(hi)]) / dx ** 2
    return g


def _cross(v, dK, dZ):
    g = np.zeros_like(v)
    g[1:-1, 1:-1, :] = (v[2:, 2:, :] - v[2:, :-2, :] - v[:-2, 2:, :] + v[:-2, :-2, :]) / (4 * dK * dZ)
    return g


# ----------------------------------------------------------------------------- tridiag
def thomas_batch(lo, di, up, rhs):
    """Solve tridiagonal systems batched along axis 0 of (lo,di,up,rhs) with shape (n, m).
    lo[0] and up[-1] are ignored. Returns x of shape (n, m)."""
    n, m = rhs.shape
    cp = np.empty((n, m)); dp = np.empty((n, m))
    cp[0] = up[0] / di[0]
    dp[0] = rhs[0] / di[0]
    for i in range(1, n):
        denom = di[i] - lo[i] * cp[i - 1]
        cp[i] = up[i] / denom
        dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / denom
    x = np.empty((n, m))
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ----------------------------------------------------------------------------- operators
def build_K_coeffs(a_lK, b_lK, dK, nK):
    """M-matrix tridiagonal coefficients (lo,di,up) for L_K along axis 0.
    Neumann (zero-flux) at logK ends: drop diffusion ghost, one-sided drift inward."""
    ap = np.maximum(a_lK, 0.0); am = np.minimum(a_lK, 0.0)
    lo = am / dK - b_lK / dK ** 2
    di = (ap - am) / dK + 2 * b_lK / dK ** 2
    up = -ap / dK - b_lK / dK ** 2
    # boundary k=0: drop diffusion, drift forward (inward) only
    lo0 = np.zeros_like(lo[0]); di0 = ap[0] / dK; up0 = -ap[0] / dK
    lo[0], di[0], up[0] = lo0, di0, up0
    # boundary k=nK-1: drop diffusion, drift backward (inward) only
    di[-1] = -am[-1] / dK; lo[-1] = am[-1] / dK; up[-1] = np.zeros_like(up[-1])
    return lo, di, up


def build_Z_coeffs(a_Z, b_Z, dZ, nZ):
    ap = np.maximum(a_Z, 0.0); am = np.minimum(a_Z, 0.0)
    lo = am / dZ - b_Z / dZ ** 2
    di = (ap - am) / dZ + 2 * b_Z / dZ ** 2
    up = -ap / dZ - b_Z / dZ ** 2
    lo[0] = 0.0; di[0] = ap[0] / dZ; up[0] = -ap[0] / dZ
    di[-1] = -am[-1] / dZ; lo[-1] = am[-1] / dZ; up[-1] = 0.0
    return lo, di, up


def build_Y_coeffs(a_Y, b_Y, dY, nY, ytop="outflow"):
    """L_Y with b_Y implicit. Y=0 Neumann (drift up, inward forward).
    ytop controls the Y=4 boundary:
      'outflow' : backward upwind, drop diffusion (pure advective exit).
      'reflect' : zero-flux on v (vY=0 ghost) -> keep diffusion; caps the steep cliff
                  the pure-outflow BC induces where b_Y is large (high logK).
    """
    ap = np.maximum(a_Y, 0.0); am = np.minimum(a_Y, 0.0)
    lo = am / dY - b_Y / dY ** 2
    di = (ap - am) / dY + 2 * b_Y / dY ** 2
    up = -ap / dY - b_Y / dY ** 2
    # bottom k=0: drop diffusion ghost, inward
    lo[0] = 0.0; di[0] = ap[0] / dY; up[0] = -ap[0] / dY
    apt = ap[-1]; amt = am[-1]
    if ytop == "reflect":
        # ghost v[n]=v[n-1]: vYY=(v[n-1]-2v[n-1]+v[n-2])/dY^2=(v[n-2]-v[n-1])/dY^2,
        # vY backward = (v[n-1]-v[n-2])/dY. M-matrix preserving:
        di[-1] = apt / dY + b_Y[-1] / dY ** 2
        lo[-1] = -apt / dY - b_Y[-1] / dY ** 2
        up[-1] = np.zeros_like(up[-1])
    else:  # outflow
        di[-1] = apt / dY - amt / dY
        lo[-1] = -apt / dY
        up[-1] = np.zeros_like(up[-1])
    return lo, di, up


# ----------------------------------------------------------------------------- residual
def residual_central(v, lam3, xi, ZZ, K, E, lNy, lNyy, dK, dZ, dY, p):
    inv_xi = 0.0 if not np.isfinite(xi) else 1.0 / xi
    sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
    vlK = _central(v, 0, dK); vZ = _central(v, 1, dZ); vY = _central(v, 2, dY)
    vKK = _d2(v, 0, dK); vZZ = _d2(v, 1, dZ); vYY = _d2(v, 2, dY)
    vKZ = _cross(v, dK, dZ)
    qd = np.maximum(vlK - ZZ * vZ, QFLOOR)
    qg = np.maximum(vlK + (1 - ZZ) * vZ, QFLOOR)
    i_d, i_g, c = controls(qd, qg, ZZ, p)
    phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, CLAMP))
    phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, CLAMP))
    Dc = sd2 * (1 - ZZ) ** 2 + sg2 * ZZ ** 2
    a_lK = (1 - ZZ) * phid + ZZ * phig - Dc / 2.0
    a_Z = ZZ * (1 - ZZ) * (phig - phid + (1 - ZZ) * sd2 - ZZ * sg2)
    a_Y = p["thbar"] * E
    b_Y = 0.5 * p["vars"] ** 2 * E ** 2
    cross = -ZZ * (1 - ZZ) ** 2 * sd2 + ZZ ** 2 * (1 - ZZ) * sg2
    E_d = (1 - ZZ) * p["s_d"] * qd; E_g = ZZ * p["s_g"] * qg
    E_y = p["vars"] * E * (vY - lNy)
    robust = -0.5 * inv_xi * (E_d ** 2 + E_g ** 2 + E_y ** 2)
    # h_y folds into the drift: v_y^drift = (thbar + vars*h_y)*E,  h_y = -inv_xi*vars*E*(vY-lNy)
    h_y = -inv_xi * p["vars"] * E * (vY - lNy)
    vy_drift = (p["thbar"] + p["vars"] * h_y) * E
    damage = -(lNy * vy_drift + lNyy * b_Y)
    LK = np.log(K)
    R = (p["delta"] * (np.log(c) + LK) - p["delta"] * v
         + a_lK * vlK + (Dc / 2.0) * vKK + a_Z * vZ
         + 0.5 * ZZ ** 2 * (1 - ZZ) ** 2 * (sd2 + sg2) * vZZ
         + cross * vKZ + vy_drift * vY + b_Y * vYY + robust + damage)
    return R[2:-2, 2:-2, 2:-2]


# ----------------------------------------------------------------------------- solve
def solve(lam3=1/6.0, xi=148.4, nK=25, nZ=30, nY=25,
          outer_max=120, inner_max=600, dtau=20.0, omega=0.5, kappa=1.0,
          dtau_ramp=None, ramp_outer=200, ytop="outflow", v0=None,
          avg_tail=0, verbose=True):
    p = P
    logK = np.linspace(4.0, 7.0, nK); dK = logK[1] - logK[0]
    Z = np.linspace(0.02, 0.98, nZ); dZ = Z[1] - Z[0]
    Y = np.linspace(0.0, 4.0, nY); dY = Y[1] - Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    K = np.exp(LK)
    E = p["eta"] * p["A_d"] * (1 - ZZ) * K
    sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
    lNy = logNy(YY, lam3, p); lNyy = logNyy(lam3, p)
    inv_xi = 0.0 if not np.isfinite(xi) else 1.0 / xi
    delta = p["delta"]

    v = (0.5 * LK + 1.0) if v0 is None else v0.copy()
    tail_buf = []  # for limit-cycle averaging
    # relaxed control-feeding derivatives (state across outer iters)
    vlK = _central(v, 0, dK); vZ = _central(v, 1, dZ)
    qd_s = np.maximum(vlK - ZZ * vZ, QFLOOR)
    qg_s = np.maximum(vlK + (1 - ZZ) * vZ, QFLOOR)

    i_d_prev = np.zeros_like(v); i_g_prev = np.zeros_like(v)
    t0 = time.time()

    for outer in range(outer_max):
        # dtau annealing: gentle (stable) early, large (fast steady-state) later
        if dtau_ramp is not None:
            frac = min(1.0, outer / float(ramp_outer))
            dtau = dtau_ramp[0] * (dtau_ramp[1] / dtau_ramp[0]) ** frac
        # ---------- (A) freeze policy ----------
        vlK = _central(v, 0, dK); vZ = _central(v, 1, dZ); vY = _central(v, 2, dY)
        qd_new = np.maximum(vlK - ZZ * vZ, QFLOOR)
        qg_new = np.maximum(vlK + (1 - ZZ) * vZ, QFLOOR)
        qd_s = (1 - kappa) * qd_s + kappa * qd_new
        qg_s = (1 - kappa) * qg_s + kappa * qg_new
        i_d, i_g, c = controls(qd_s, qg_s, ZZ, p)
        phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, CLAMP))
        phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, CLAMP))

        Dc = sd2 * (1 - ZZ) ** 2 + sg2 * ZZ ** 2
        a_lK = (1 - ZZ) * phid + ZZ * phig - Dc / 2.0
        a_Z = ZZ * (1 - ZZ) * (phig - phid + (1 - ZZ) * sd2 - ZZ * sg2)
        # frozen worst-case h_y (uses CONSISTENT lNy with lambda3)
        h_y = -inv_xi * p["vars"] * E * (vY - lNy)
        a_Y_eff = (p["thbar"] + p["vars"] * h_y) * E
        b_lK = Dc / 2.0
        b_Z = 0.5 * ZZ ** 2 * (1 - ZZ) ** 2 * (sd2 + sg2)
        b_Y = 0.5 * p["vars"] ** 2 * E ** 2
        cross = -ZZ * (1 - ZZ) ** 2 * sd2 + ZZ ** 2 * (1 - ZZ) * sg2

        # source (frozen/explicit, EXCLUDING cross which is re-lagged each inner sweep)
        flow = delta * (np.log(c) + LK)
        E_d = (1 - ZZ) * p["s_d"] * qd_s; E_g = ZZ * p["s_g"] * qg_s
        robust = -0.5 * inv_xi * (E_d ** 2 + E_g ** 2)          # capital channels only
        robust_Yconst = 0.5 * (xi if np.isfinite(xi) else 0.0) * h_y ** 2
        # damage drag uses a_Y_eff (consistent w/ a_Y_eff in L_Y)
        damage = -(lNy * a_Y_eff + lNyy * b_Y)
        source0 = flow + robust + robust_Yconst + damage

        # ---------- build frozen tridiagonal operators ----------
        # K along axis0: coeffs shape (nK, nZ, nY) -> reshape to (nK, nZ*nY) for batch thomas
        loK, diK, upK = build_K_coeffs(a_lK, b_lK, dK, nK)
        loZ, diZ, upZ = build_Z_coeffs(a_Z, b_Z, dZ, nZ)
        loY, diY, upY = build_Y_coeffs(a_Y_eff, b_Y, dY, nY, ytop=ytop)

        # We solve the frozen-policy linear system  A v = b  with
        #   A = delta*I + (-L_K) + (-L_Z) + (-L_Y),   b = source0 + cross*vKZ.
        # Split A = A_K + A_Z + A_Y with  A_dim = (-L_dim) + (delta/3) I  (each an M-matrix,
        # row-sum delta/3 > 0). (lo,di,up) already hold the rows of (-L_dim); add delta/3 to di.
        d3 = delta / 3.0
        diKa = diK + d3; diZa = diZ + d3; diYa = diY + d3

        def applyA_K(w):                 # A_K w  =  (-L_K + d3 I) w
            r = diKa * w
            r[1:] += loK[1:] * w[:-1]
            r[:-1] += upK[:-1] * w[1:]
            return r
        def applyA_Z(w):
            r = diZa * w
            r[:, 1:] += loZ[:, 1:] * w[:, :-1]
            r[:, :-1] += upZ[:, :-1] * w[:, 1:]
            return r
        def applyA_Y(w):
            r = diYa * w
            r[:, :, 1:] += loY[:, :, 1:] * w[:, :, :-1]
            r[:, :, :-1] += upY[:, :, :-1] * w[:, :, 1:]
            return r

        # Pre-build the (I + dtau*A_dim) tridiagonals for the Thomas sweeps.
        AK_lo = (dtau * loK).reshape(nK, -1)
        AK_di = (1.0 + dtau * diKa).reshape(nK, -1)
        AK_up = (dtau * upK).reshape(nK, -1)
        AZ_lo = np.moveaxis(dtau * loZ, 1, 0).reshape(nZ, -1)
        AZ_di = np.moveaxis(1.0 + dtau * diZa, 1, 0).reshape(nZ, -1)
        AZ_up = np.moveaxis(dtau * upZ, 1, 0).reshape(nZ, -1)
        AY_lo = np.moveaxis(dtau * loY, 2, 0).reshape(nY, -1)
        AY_di = np.moveaxis(1.0 + dtau * diYa, 2, 0).reshape(nY, -1)
        AY_up = np.moveaxis(dtau * upY, 2, 0).reshape(nY, -1)

        v_pol = v.copy()
        # ---------- (B) inner: Douglas-Rachford ADI to steady state of  A v = b ----------
        for inner in range(inner_max):
            vKZ = _cross(v, dK, dZ)
            b = source0 + cross * vKZ
            # full residual predictor:  y0 = v - dtau*(A v - b) = v + dtau*(b - A v)
            Av = applyA_K(v) + applyA_Z(v) + applyA_Y(v)
            y0 = v + dtau * (b - Av)
            # correction sweeps:  (I + dtau A_dim) y = y_prev + dtau A_dim v
            r1 = (y0 + dtau * applyA_K(v)).reshape(nK, -1)
            y1 = thomas_batch(AK_lo, AK_di, AK_up, r1).reshape(nK, nZ, nY)
            r2 = np.moveaxis(y1 + dtau * applyA_Z(v), 1, 0).reshape(nZ, -1)
            y2 = np.moveaxis(thomas_batch(AZ_lo, AZ_di, AZ_up, r2).reshape(nZ, nK, nY), 0, 1)
            r3 = np.moveaxis(y2 + dtau * applyA_Y(v), 2, 0).reshape(nY, -1)
            vstar = np.moveaxis(thomas_batch(AY_lo, AY_di, AY_up, r3).reshape(nY, nK, nZ), 0, 2)

            step = np.max(np.abs(vstar - v))
            v = vstar
            if step < 1e-10:
                break

        # ---------- (C) damp + test outer ----------
        v = v_pol + omega * (v - v_pol)
        # limit-cycle averaging: the weak-identification stall makes v oscillate
        # (roughly) symmetrically about the true fixed point; averaging the tail of the
        # iterate sequence cancels the oscillation. Accumulated WITHOUT feeding back.
        if avg_tail and outer >= outer_max - avg_tail:
            tail_buf.append(v.copy())
        R = residual_central(v, lam3, xi, ZZ, K, E, lNy, lNyy, dK, dZ, dY, p)
        maxR = float(np.max(np.abs(R)))
        dv_pol = float(np.max(np.abs(v - v_pol)))
        di_ctrl = float(max(np.max(np.abs(i_d - i_d_prev)), np.max(np.abs(i_g - i_g_prev))))
        i_d_prev, i_g_prev = i_d.copy(), i_g.copy()
        if verbose:
            ik = np.argmin(np.abs(logK - np.log(880)))
            jz = np.argmin(np.abs(Z - 0.7)); ky = np.argmin(np.abs(Y - 3.0))
            print(f"  [outer {outer:3d}] inner={inner+1:3d} maxR={maxR:.3e} "
                  f"d(v_pol)={dv_pol:.2e} d(ctrl)={di_ctrl:.2e} | "
                  f"i_d={i_d[ik,jz,ky]:+.4f} i_g={i_g[ik,jz,ky]:+.4f} "
                  f"vlK={vlK[ik,jz,ky]:.3f}", flush=True)
        if maxR < 1e-5 and dv_pol < 1e-9 and di_ctrl < 1e-7:
            break

    # limit-cycle averaging applied to the FINAL value only
    if tail_buf:
        v = sum(tail_buf) / len(tail_buf)

    # final readout
    vlK = _central(v, 0, dK); vZ = _central(v, 1, dZ); vY = _central(v, 2, dY)
    qd = np.maximum(vlK - ZZ * vZ, QFLOOR); qg = np.maximum(vlK + (1 - ZZ) * vZ, QFLOOR)
    i_d, i_g, c = controls(qd, qg, ZZ, p)
    R = residual_central(v, lam3, xi, ZZ, K, E, lNy, lNyy, dK, dZ, dY, p)
    # bulk residual excludes the steep Y-top boundary layer (Y>3.2 at high logK is a
    # genuine first-order corner singularity that does not converge under refinement).
    kY = np.argmin(np.abs(Y - 3.2))
    bulkR = float(np.max(np.abs(R[:, :, :max(1, kY - 2)])))
    ik = np.argmin(np.abs(logK - np.log(880)))
    jz = np.argmin(np.abs(Z - 0.7)); ky = np.argmin(np.abs(Y - 3.0))
    Rval = float(abs(R[ik - 2, jz - 2, ky - 2])) if (ik >= 2 and jz >= 2 and ky >= 2) else float("nan")
    return dict(logK=logK, Z=Z, Y=Y, v=v, i_d=i_d, i_g=i_g, c=c, vlK=vlK, vZ=vZ, vY=vY,
                lNy=lNy, qd=qd, qg=qg, outer=outer + 1, runtime=time.time() - t0,
                max_abs_residual=float(np.max(np.abs(R))),
                bulk_residual=bulkR, residual_at_validation=Rval)


if __name__ == "__main__":
    # Stable false-transient (inner=4 ADI sweeps/outer, modest dtau) with the q-floor
    # under-relaxation; reflect Y-top BC (keeps b_Y diffusion -> removes the steep cliff
    # the pure-outflow BC induces at high logK). Converges to the NN interior solution.
    out = solve(lam3=1/6.0, xi=148.4, nK=25, nZ=30, nY=25,
                outer_max=800, inner_max=4, dtau=4.0, omega=0.85, kappa=0.06,
                ytop="reflect", verbose=True)
    print(f"\nDONE: outer={out['outer']} runtime={out['runtime']:.1f}s")
    print(f"  max|resid| (full interior)     = {out['max_abs_residual']:.3e}  "
          f"(dominated by Y->4 / high-logK corner layer)")
    print(f"  max|resid| (bulk, Y<=3.2)      = {out['bulk_residual']:.3e}")
    print(f"  |resid| at validation node     = {out['residual_at_validation']:.3e}")
    OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OD, exist_ok=True)
    np.savez(os.path.join(OD, "fd_pdpt_v2_lam3_0167_xi148.npz"),
             **{k: out[k] for k in ("logK", "Z", "Y", "v", "i_d", "i_g", "c", "vlK", "vZ", "vY")})
    ik = np.argmin(np.abs(out["logK"] - np.log(880)))
    jz = np.argmin(np.abs(out["Z"] - 0.7))
    ky = np.argmin(np.abs(out["Y"] - 3.0))
    VY = out["vY"][ik, jz, ky] - out["lNy"][ik, jz, ky]
    print(f"[readout logK={out['logK'][ik]:.2f} Z={out['Z'][jz]:.2f} Y={out['Y'][ky]:.1f}]")
    print(f"  i_d={out['i_d'][ik,jz,ky]:+.4f} (NN 0.040)   i_g={out['i_g'][ik,jz,ky]:+.4f} (NN 0.104)")
    print(f"  vlK={out['vlK'][ik,jz,ky]:.3f} (NN 0.54)   V_Y={VY:+.3f} (NN -0.16)   c={out['c'][ik,jz,ky]:.4f} (NN 0.064)")
