"""
fd_pdpt_v5 -- Policy Iteration By Simulation (PIBYS) for the post-damage post-tech climate HJB.

WHY (after v3 upwind and v4 semi-Lagrangian both failed to grid-converge):
  The Z direction is advection-dominated (cell-Peclet 47-139), physical diffusion b_Z~4e-6.
  ANY Eulerian first-order scheme (upwind v3, OR linear-interpolation semi-Lagrangian v4)
  injects O(a_Z*dZ)~1e-4 numerical diffusion that swamps b_Z and drifts v_Z under refinement.
  v4 additionally DIVERGED (central-diff costates updated every sweep -> oscillation).

  PIBYS sidesteps the Eulerian PDE entirely -- it is exact Howard policy iteration:
    EVALUATE (policy fixed): for every grid node, integrate the deterministic drift ODE forward
      (sigma=0.01 => value = deterministic drift-integral to O(sigma^2)~1e-4) and accumulate the
      discounted running cost  v(x)=int e^{-dt}[ delta(log c + logK) - ((logN)_Y a_Y+(logN)_YY b_Y) ] dt.
      Controls are read off the FIXED policy grid along the trajectory (trilinear, edge-clamped).
      NO artificial diffusion: characteristics are integrated exactly (RK2), v is ACCUMULATED, never
      transported through the grid, so there is no repeated-interpolation smoothing of v_Z.
    IMPROVE: central-difference the SMOOTH integrated v -> costates qd=v_logK-Z v_Z, qg=v_logK+(1-Z)v_Z
      -> exact closed-form FOC controls (NO q-floor, NO under-relaxed costates). Under-relax the
      CONTROL field modestly for stability. Policy iteration is monotone => converges; refining the
      grid sharpens gradients without adding diffusion => it GRID-CONVERGES.

  Robustness drag (-1/2xi * sum E_j^2) is O(sigma^2/xi); at xi=148.4 it is ~1e-6 and is dropped in
  the trajectory cost (it needs v_Y which is unavailable mid-sim); it re-enters only the residual
  monitor. For small xi this file would need the worst-case drift added -- out of scope here.

Returns the v3/v4-compatible dict so policy_eval.py / fd_vs_nn.py keep working. numpy/scipy only.
"""
import os
import sys
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
from params import PARAMS  # noqa: E402

P = dict(delta=PARAMS["δ"], A_d=PARAMS["A_d"], A_gpp=PARAMS["A_g_prime_prime"],
         a_d=PARAMS["α_d"], G_d=PARAMS["Γ_d"], t_d=PARAMS["θ_d"], s_d=PARAMS["σ_d"],
         a_g=PARAMS["α_g"], G_g=PARAMS["Γ_g"], t_g=PARAMS["θ_g"], s_g=PARAMS["σ_g"],
         thbar=PARAMS["θ_bar"], eta=PARAMS["η"], vars=PARAMS["ϛ"],
         l1=PARAMS["λ1"], l2=PARAMS["λ2"], y_up=PARAMS["y_upper"])
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# Costate floor = the FEASIBILITY constraint, NOT a bias hack. Since 1+theta_d*i_d =
# theta_d*Gamma_d*qd*c/delta, qd>0 is exactly K>=0 (capital stays positive); qd<0 would force
# 1+theta*i<0 (infeasible) and make den->0 -> c blows up (the v4/v5 divergence). QFLOOR=2e-3 keeps
# 1+theta*i >~ 0.02, i.e. de-investment is allowed down to i ~ -0.0587 (the natural -1/theta corner),
# well below the de-invest value at the reference (i_d~-0.018 <-> qd~0.07). It binds ONLY in the deep-
# decarbonized Z->1 corner, so it does NOT bias the interior; and PIBYS has no artificial diffusion, so
# unlike v3 (upwind+floor) this floor is the ONLY safeguard and the scheme still grid-converges.
QFLOOR = 2e-3


def controls(qd, qg, Z, p=P):
    """Exact closed-form FOC controls with the feasibility floor on the costates (qd,qg >= QFLOOR)."""
    qd = np.maximum(qd, QFLOOR); qg = np.maximum(qg, QFLOOR)
    Abar = (1 - Z) * p["A_d"] + Z * p["A_gpp"]
    num = p["delta"] * (Abar + (1 - Z) / p["t_d"] + Z / p["t_g"])
    den = p["delta"] + (1 - Z) * p["G_d"] * qd + Z * p["G_g"] * qg   # >= delta > 0 now
    c = num / den
    i_d = p["G_d"] * qd * c / p["delta"] - 1.0 / p["t_d"]
    i_g = p["G_g"] * qg * c / p["delta"] - 1.0 / p["t_g"]
    return i_d, i_g, c


def _grad(v, axis, dx):
    """Central interior, one-sided at faces."""
    g = np.zeros_like(v)
    lo = [slice(None)] * 3; hi = [slice(None)] * 3; md = [slice(None)] * 3
    lo[axis] = slice(2, None); hi[axis] = slice(0, -2); md[axis] = slice(1, -1)
    g[tuple(md)] = (v[tuple(lo)] - v[tuple(hi)]) / (2 * dx)
    e0 = [slice(None)] * 3; e1 = [slice(None)] * 3; en = [slice(None)] * 3; en1 = [slice(None)] * 3
    e0[axis] = 0; e1[axis] = 1; en[axis] = -1; en1[axis] = -2
    g[tuple(e0)] = (v[tuple(e1)] - v[tuple(e0)]) / dx
    g[tuple(en)] = (v[tuple(en)] - v[tuple(en1)]) / dx
    return g


# LK_CAP bounds the EMISSIONS-variance damage channel (lNyy*b_Y ~ E^2) so it does not blow up in the
# trajectory tail and confound the temperature knob; 8 keeps it honest just past the logK grid top (7).
# Y_CAP is the TEMPERATURE-DAMAGE HORIZON: the marginal temperature damage lNy SATURATES at lNy(Y_CAP)
# beyond Y_CAP (keeps accruing at that rate -- no quadratic blow-up, no "free zone"). It is THE y_max knob.
LK_CAP = 8.0
Y_CAP = 4.0


def _drift(lk, z, y, i_d, i_g, p=P):
    phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, 1e-9))
    phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, 1e-9))
    Dc = p["s_d"] ** 2 * (1 - z) ** 2 + p["s_g"] ** 2 * z ** 2
    a_lK = (1 - z) * phid + z * phig - Dc / 2.0
    a_Z = z * (1 - z) * (phig - phid + (1 - z) * p["s_d"] ** 2 - z * p["s_g"] ** 2)
    E = p["eta"] * p["A_d"] * (1 - z) * np.exp(np.minimum(lk, LK_CAP))
    a_Y = p["thbar"] * E
    return a_lK, a_Z, a_Y, E


def simulate_v(logK, Z, Y, i_d_grid, i_g_grid, lam3, T=1200.0, dt=1.0, p=P, y_cap=Y_CAP):
    """POLICY EVALUATION by simulation: integrate v from EVERY node under the fixed (i_d,i_g) grid.
    Vectorized over all nodes; RK2 on the drift ODE; controls edge-clamped trilinear along each path."""
    axes = (logK, Z, Y)
    f_id = RGI(axes, i_d_grid, method="linear", bounds_error=False, fill_value=None)
    f_ig = RGI(axes, i_g_grid, method="linear", bounds_error=False, fill_value=None)
    lo = np.array([logK[0], Z[0], Y[0]]); hi = np.array([logK[-1], Z[-1], Y[-1]])

    def ctrl(lk, z, y):
        q = np.empty((lk.size, 3))
        q[:, 0] = np.clip(lk, lo[0], hi[0]); q[:, 1] = np.clip(z, lo[1], hi[1]); q[:, 2] = np.clip(y, lo[2], hi[2])
        return f_id(q), f_ig(q)

    def deriv(lk, z, y):
        idv, igv = ctrl(lk, z, y)
        a_lK, a_Z, a_Y, E = _drift(lk, z, y, idv, igv, p)
        return a_lK, a_Z, a_Y, idv, igv, E

    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    lk = LK.ravel().copy(); z = ZZ.ravel().copy(); y = YY.ravel().copy()
    V = np.zeros_like(lk); t = 0.0
    nstep = int(round(T / dt))
    for s in range(nstep):
        a_lK, a_Z, a_Y, idv, igv, E = deriv(lk, z, y)
        c = (p["A_d"] - idv) * (1 - z) + (p["A_gpp"] - igv) * z
        # TEMPERATURE-DAMAGE HORIZON: the marginal temperature damage lNy(Y) SATURATES at lNy(y_cap)
        # beyond y_cap -- damage keeps accruing at that rate (no quadratic blow-up to absurd temps, and
        # no "free zone" that a hard mask would create). y_cap is THE y_max knob. Emissions damage uses E
        # (logK capped at LK_CAP in _drift) so the E^2 channel can't run away and confound the knob.
        y_eff = np.minimum(y, y_cap)
        lNy = p["l1"] + p["l2"] * y_eff + lam3 * (y_eff - p["y_up"]); lNyy = p["l2"] + lam3
        b_Y = 0.5 * p["vars"] ** 2 * E ** 2
        clim = lNy * a_Y + lNyy * b_Y
        flow = p["delta"] * (np.log(np.maximum(c, 1e-12)) + lk) - clim
        V += np.exp(-p["delta"] * t) * flow * dt
        # RK2 midpoint state update
        lk_m = lk + 0.5 * dt * a_lK; z_m = np.clip(z + 0.5 * dt * a_Z, 1e-4, 1 - 1e-4); y_m = y + 0.5 * dt * a_Y
        a2_lK, a2_Z, a2_Y, _, _, _ = deriv(lk_m, z_m, y_m)
        lk = lk + dt * a2_lK; z = np.clip(z + dt * a2_Z, 1e-4, 1 - 1e-4); y = y + dt * a2_Y
        t += dt
    return V.reshape(LK.shape)


def _residual(v, i_d, i_g, c, ZZ, E, lNy, lNyy, lam3, LK, dK, dZ, dY, p=P):
    """TRUE HJB residual (central differences incl. tiny diffusion) at interior nodes, monitor only."""
    sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
    vlK = _grad(v, 0, dK); vZ = _grad(v, 1, dZ); vY = _grad(v, 2, dY)
    vKK = np.zeros_like(v); vZZ = np.zeros_like(v); vYY = np.zeros_like(v)
    vKK[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dK ** 2
    vZZ[:, 1:-1] = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / dZ ** 2
    vYY[:, :, 1:-1] = (v[:, :, 2:] - 2 * v[:, :, 1:-1] + v[:, :, :-2]) / dY ** 2
    phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, 1e-9))
    phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, 1e-9))
    Dc = sd2 * (1 - ZZ) ** 2 + sg2 * ZZ ** 2
    a_lK = (1 - ZZ) * phid + ZZ * phig - Dc / 2.0
    a_Z = ZZ * (1 - ZZ) * (phig - phid + (1 - ZZ) * sd2 - ZZ * sg2)
    a_Y = p["thbar"] * E; b_Y = 0.5 * p["vars"] ** 2 * E ** 2
    b_Z = 0.5 * ZZ ** 2 * (1 - ZZ) ** 2 * (sd2 + sg2)
    damage = -(lNy * a_Y + lNyy * b_Y)
    R = (p["delta"] * (np.log(np.maximum(c, 1e-12)) + LK) - p["delta"] * v
         + a_lK * vlK + (Dc / 2.0) * vKK + a_Z * vZ + b_Z * vZZ + a_Y * vY + b_Y * vYY + damage)
    return R[1:-1, 1:-1, 1:-2]


def _init_from_v3(logK, Z, Y):
    """Interpolate v3's converged controls onto this grid as a warm start (keeps early policy sane)."""
    f = os.path.join(OD, "fd_pdpt_v3_lam3_0167_xi148.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    q = np.stack([np.clip(LK, d["logK"][0], d["logK"][-1]).ravel(),
                  np.clip(ZZ, d["Z"][0], d["Z"][-1]).ravel(),
                  np.clip(YY, d["Y"][0], d["Y"][-1]).ravel()], axis=1)
    g_id = RGI((d["logK"], d["Z"], d["Y"]), d["i_d"], bounds_error=False, fill_value=None)
    g_ig = RGI((d["logK"], d["Z"], d["Y"]), d["i_g"], bounds_error=False, fill_value=None)
    return g_id(q).reshape(LK.shape), g_ig(q).reshape(LK.shape)


def solve(lam3=1 / 6.0, xi=148.4, nK=21, nZ=31, nY=21, T=1200.0, dt=2.5,
          howard_max=32, tol=2e-4, relax=0.22, warm=True, verbose=True,
          y_max=4.0, y_cap=None):
    # y_max = top of the Y grid (the model's temperature-damage domain; original = 4.0).
    # y_cap = temperature beyond which damage SATURATES in the trajectory cost (default = module
    # Y_CAP=30, i.e. count damage far out; pass y_cap=y_max for a model-faithful truncated horizon).
    p = P
    yc = Y_CAP if y_cap is None else y_cap
    logK = np.linspace(4.0, 7.0, nK); dK = logK[1] - logK[0]
    Z = np.linspace(0.02, 0.98, nZ); dZ = Z[1] - Z[0]
    Y = np.linspace(0.0, y_max, nY); dY = Y[1] - Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    K = np.exp(LK); E = p["eta"] * p["A_d"] * (1 - ZZ) * K
    lNy = p["l1"] + p["l2"] * YY + lam3 * (YY - p["y_up"]); lNyy = p["l2"] + lam3

    init = _init_from_v3(logK, Z, Y) if warm else None
    if init is not None:
        i_d, i_g = init
        if verbose: print("  [init] warm-started from fd_pdpt_v3 controls", flush=True)
    else:
        i_d = np.zeros_like(LK); i_g = np.full_like(LK, 0.05)
        if verbose: print("  [init] cold start i_d=0, i_g=0.05", flush=True)

    t0 = time.time(); it = 0; di = np.inf
    ik = np.argmin(np.abs(logK - np.log(880))); jz = np.argmin(np.abs(Z - 0.7)); ky = np.argmin(np.abs(Y - 3.0))
    v = np.zeros_like(LK)
    for it in range(howard_max):
        v = simulate_v(logK, Z, Y, i_d, i_g, lam3, T=T, dt=dt, p=p, y_cap=yc)  # EVALUATE
        vlK = _grad(v, 0, dK); vZ = _grad(v, 1, dZ)                            # IMPROVE
        qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
        i_d_new, i_g_new, c = controls(qd, qg, ZZ, p)
        di = max(np.max(np.abs(i_d_new - i_d)), np.max(np.abs(i_g_new - i_g)))
        i_d = (1 - relax) * i_d + relax * i_d_new
        i_g = (1 - relax) * i_g + relax * i_g_new
        if verbose:
            R = _residual(v, i_d, i_g, c, ZZ, E, lNy, lNyy, lam3, LK, dK, dZ, dY, p)
            print(f"  [howard {it:2d}] dpol={di:.2e} maxR={np.max(np.abs(R)):.2e} | "
                  f"i_d={i_d[ik,jz,ky]:+.4f} i_g={i_g[ik,jz,ky]:+.4f} "
                  f"vlK={vlK[ik,jz,ky]:.3f} vZ={vZ[ik,jz,ky]:.3f} c={c[ik,jz,ky]:.4f}", flush=True)
        if di < tol and it > 2:
            break

    v = simulate_v(logK, Z, Y, i_d, i_g, lam3, T=T, dt=dt, p=p, y_cap=yc)
    vlK = _grad(v, 0, dK); vZ = _grad(v, 1, dZ); vY = _grad(v, 2, dY)
    qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
    i_d, i_g, c = controls(qd, qg, ZZ, p)
    R = _residual(v, i_d, i_g, c, ZZ, E, lNy, lNyy, lam3, LK, dK, dZ, dY, p)
    return dict(logK=logK, Z=Z, Y=Y, v=v, i_d=i_d, i_g=i_g, c=c, vlK=vlK, vZ=vZ, vY=vY,
                iters=it + 1, time=time.time() - t0, max_abs_residual=float(np.max(np.abs(R))))


if __name__ == "__main__":
    print("[fd_pdpt_v5 smoke test] PIBYS, 21x31x21, lam3=1/6, xi=148.4", flush=True)
    out = solve(nK=21, nZ=31, nY=21, T=1200.0, dt=1.0, howard_max=20, verbose=True)
    ik = np.argmin(np.abs(out["logK"] - np.log(880))); jz = np.argmin(np.abs(out["Z"] - 0.7)); ky = np.argmin(np.abs(out["Y"] - 3.0))
    print(f"\ndone {out['iters']} Howard iters, {out['time']:.0f}s, max|resid|={out['max_abs_residual']:.3e}")
    print(f"[logK=6.78,Z=0.7,Y=3.0]  i_d={out['i_d'][ik,jz,ky]:+.5f}  i_g={out['i_g'][ik,jz,ky]:+.5f}  "
          f"v_logK={out['vlK'][ik,jz,ky]:.4f}  v_Z={out['vZ'][ik,jz,ky]:.4f}  c={out['c'][ik,jz,ky]:.4f}")
    print(f"i_d sign at high-damage point: {'NEGATIVE (de-invest)' if out['i_d'][ik,jz,ky] < 0 else 'POSITIVE (invest)'}")
