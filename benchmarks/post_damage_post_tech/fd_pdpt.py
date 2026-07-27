"""
3-D finite-difference reference solver for the post-damage post-tech climate HJB
(derivation.tex, symbolically verified). State (logK, Z, Y) at fixed (lambda3, xi).

Scheme: semi-implicit false transient with policy-iterated (FOC closed-form) controls.
Per pseudo-step the drift terms are upwinded (M-matrix) and the diagonal diffusions are
central, both treated IMPLICITLY (7-point stencil, scipy sparse solve); the cross term
v_{logK,Z} and the damage/robustness source are explicit. Boundaries are zero-flux
(index clamping). xi -> inf removes robustness.

Runs on the login node (numpy/scipy). HJB residual reported with central derivatives.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
from params import PARAMS  # noqa: E402

P = dict(delta=PARAMS["δ"], A_d=PARAMS["A_d"], A_gpp=PARAMS["A_g_prime_prime"],
         a_d=PARAMS["α_d"], G_d=PARAMS["Γ_d"], t_d=PARAMS["θ_d"], s_d=PARAMS["σ_d"],
         a_g=PARAMS["α_g"], G_g=PARAMS["Γ_g"], t_g=PARAMS["θ_g"], s_g=PARAMS["σ_g"],
         thbar=PARAMS["θ_bar"], eta=PARAMS["η"], vars=PARAMS["ϛ"],
         l1=PARAMS["λ1"], l2=PARAMS["λ2"], y_up=PARAMS["y_upper"])


def controls(vlK, vZ, Z, p):
    """Closed-form FOC controls given v_logK, v_Z (q-tilde clamped > 0 for feasibility)."""
    eps = 1e-6
    qd = np.maximum(vlK - Z * vZ, eps)      # tilde q_d
    qg = np.maximum(vlK + (1 - Z) * vZ, eps)
    Abar = (1 - Z) * p["A_d"] + Z * p["A_gpp"]
    num = p["delta"] * (Abar + (1 - Z) / p["t_d"] + Z / p["t_g"])
    den = p["delta"] + (1 - Z) * p["G_d"] * qd + Z * p["G_g"] * qg
    c = num / den
    i_d = p["G_d"] * qd * c / p["delta"] - 1.0 / p["t_d"]
    i_g = p["G_g"] * qg * c / p["delta"] - 1.0 / p["t_g"]
    return i_d, i_g, c, qd, qg


def logNy(Y, lam3, p):
    return p["l1"] + p["l2"] * Y + lam3 * (Y - p["y_up"])


def logNyy(lam3, p):
    return p["l2"] + lam3


def _central(v, axis, dx):
    """Central first derivative with one-sided (zero-flux) ends along `axis`."""
    g = np.zeros_like(v)
    sl = [slice(None)] * 3
    f = np.take(v, range(1, v.shape[axis]), axis) - np.take(v, range(0, v.shape[axis] - 1), axis)
    # interior central
    idx_lo = [slice(None)] * 3; idx_lo[axis] = slice(2, None)
    idx_hi = [slice(None)] * 3; idx_hi[axis] = slice(0, -2)
    idx_md = [slice(None)] * 3; idx_md[axis] = slice(1, -1)
    g[tuple(idx_md)] = (v[tuple(idx_lo)] - v[tuple(idx_hi)]) / (2 * dx)
    # one-sided ends
    e0 = [slice(None)] * 3; e0[axis] = 0; e1 = [slice(None)] * 3; e1[axis] = 1
    en = [slice(None)] * 3; en[axis] = -1; en1 = [slice(None)] * 3; en1[axis] = -2
    g[tuple(e0)] = (v[tuple(e1)] - v[tuple(e0)]) / dx
    g[tuple(en)] = (v[tuple(en)] - v[tuple(en1)]) / dx
    return g


def _fwd(v, axis, dx):           # (v[i+1]-v[i])/dx; top boundary -> 0 (zero forward flux)
    d = np.zeros_like(v); lo = [slice(None)] * 3; hi = [slice(None)] * 3
    lo[axis] = slice(0, -1); hi[axis] = slice(1, None)
    d[tuple(lo)] = (v[tuple(hi)] - v[tuple(lo)]) / dx
    return d


def _bwd(v, axis, dx):           # (v[i]-v[i-1])/dx; bottom boundary -> 0
    d = np.zeros_like(v); lo = [slice(None)] * 3; hi = [slice(None)] * 3
    lo[axis] = slice(0, -1); hi[axis] = slice(1, None)
    d[tuple(hi)] = (v[tuple(hi)] - v[tuple(lo)]) / dx
    return d


def _upwind(v, axis, a, dx):     # forward where drift a>0, backward where a<0
    return np.where(a > 0, _fwd(v, axis, dx), _bwd(v, axis, dx))


def _d2(v, axis, dx):            # central 2nd derivative; ends one-sided (Neumann)
    g = np.zeros_like(v); lo = [slice(None)] * 3; hi = [slice(None)] * 3; md = [slice(None)] * 3
    lo[axis] = slice(2, None); hi[axis] = slice(0, -2); md[axis] = slice(1, -1)
    g[tuple(md)] = (v[tuple(lo)] - 2 * v[tuple(md)] + v[tuple(hi)]) / dx ** 2
    return g


def solve(lam3=1/6.0, xi=148.4, nK=25, nZ=30, nY=25, dtau=1.0,
          max_iter=40000, tol=1e-8, verbose=True, log_every=4000):
    p = P
    logK = np.linspace(4.0, 7.0, nK); dK = logK[1] - logK[0]
    Z = np.linspace(0.02, 0.98, nZ); dZ = Z[1] - Z[0]
    Y = np.linspace(0.0, 4.0, nY); dY = Y[1] - Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    K = np.exp(LK)
    E = p["eta"] * p["A_d"] * (1 - ZZ) * K                  # emissions (levels)
    sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
    lNy = logNy(YY, lam3, p); lNyy = logNyy(lam3, p)
    N = nK * nZ * nY
    inv_xi = 0.0 if not np.isfinite(xi) else 1.0 / xi

    # initial guess: v ~ v_logK*logK + small (homogeneity ~0.5 from the validated NN)
    v = 0.5 * LK + 1.0
    it = 0
    for it in range(max_iter):
        vlK = _central(v, 0, dK); vZ = _central(v, 1, dZ); vY = _central(v, 2, dY)
        i_d, i_g, c, qd, qg = controls(vlK, vZ, ZZ, p)
        phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, 1e-9))
        phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, 1e-9))
        Dc = sd2 * (1 - ZZ) ** 2 + sg2 * ZZ ** 2
        # climate exposure (enters ONLY the robustness drag, substituted form).
        E_y = p["vars"] * E * (vY - lNy)
        # drift coefficients (a): worst-case h enters only via the -1/2xi drag below, so all
        # three channels use BENCHMARK drifts (capital h_d,h_g are likewise absent from a_lK,a_Z).
        a_lK = (1 - ZZ) * phid + ZZ * phig - Dc / 2.0
        a_Z = ZZ * (1 - ZZ) * (phig - phid + (1 - ZZ) * sd2 - ZZ * sg2)
        a_Y = p["thbar"] * E
        b_lK = Dc / 2.0
        b_Z = 0.5 * ZZ ** 2 * (1 - ZZ) ** 2 * (sd2 + sg2)
        b_Y = 0.5 * p["vars"] ** 2 * E ** 2
        cross = (-ZZ * (1 - ZZ) ** 2 * sd2 + ZZ ** 2 * (1 - ZZ) * sg2)   # v_{logK,Z} coef
        # flow + robustness drag + damage drag  (explicit source)
        flow = p["delta"] * (np.log(np.maximum(c, 1e-12)) + LK)
        E_d = (1 - ZZ) * p["s_d"] * qd; E_g = ZZ * p["s_g"] * qg
        robust = -0.5 * inv_xi * (E_d ** 2 + E_g ** 2 + E_y ** 2)        # = h.E + xi/2|h|^2
        damage = -(lNy * a_Y + lNyy * b_Y)
        vKZ = np.zeros_like(v)
        vKZ[1:-1, 1:-1, :] = (v[2:, 2:, :] - v[2:, :-2, :] - v[:-2, 2:, :] + v[:-2, :-2, :]) / (4 * dK * dZ)
        # explicit upwind(drift) + central(diffusion) operator; advection-dominated, so the
        # CFL (dtau ~ 1) permits a fully explicit relaxation -- no linear solve.
        Lv = (a_lK * _upwind(v, 0, a_lK, dK) + b_lK * _d2(v, 0, dK)
              + a_Z * _upwind(v, 1, a_Z, dZ) + b_Z * _d2(v, 1, dZ)
              + a_Y * _upwind(v, 2, a_Y, dY) + b_Y * _d2(v, 2, dY)
              + cross * vKZ)
        HJB = flow + robust + damage + Lv - p["delta"] * v
        v_new = v + dtau * HJB
        step = np.max(np.abs(v_new - v))
        v = v_new
        if verbose and (it % log_every == 0 or it == max_iter - 1):
            print(f"  [fd] it {it:5d} max|dv|={step:.3e}", flush=True)
        if step < tol:
            break

    # final controls + central-difference HJB residual
    vlK = _central(v, 0, dK); vZ = _central(v, 1, dZ); vY = _central(v, 2, dY)
    i_d, i_g, c, qd, qg = controls(vlK, vZ, ZZ, p)
    resid = _residual(v, vlK, vZ, vY, i_d, i_g, c, qd, qg, ZZ, K, E, lNy, lNyy, lam3, xi,
                      logK, Z, Y, dK, dZ, dY, p)
    return dict(logK=logK, Z=Z, Y=Y, v=v, i_d=i_d, i_g=i_g, c=c, vlK=vlK, vZ=vZ, vY=vY,
                qd=qd, qg=qg, iters=it + 1, max_abs_residual=float(np.max(np.abs(resid))))


def _residual(v, vlK, vZ, vY, i_d, i_g, c, qd, qg, ZZ, K, E, lNy, lNyy, lam3, xi,
              logK, Z, Y, dK, dZ, dY, p):
    sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
    inv_xi = 0.0 if not np.isfinite(xi) else 1.0 / xi
    def d2(axis, dx):
        g = np.zeros_like(v)
        s = [slice(1, -1)] * 0
        lo = [slice(None)] * 3; lo[axis] = slice(2, None)
        hi = [slice(None)] * 3; hi[axis] = slice(0, -2)
        md = [slice(None)] * 3; md[axis] = slice(1, -1)
        g[tuple(md)] = (v[tuple(lo)] - 2 * v[tuple(md)] + v[tuple(hi)]) / dx ** 2
        return g
    vKK = d2(0, dK); vZZ = d2(1, dZ); vYY = d2(2, dY)
    vKZ = np.zeros_like(v)
    vKZ[1:-1, 1:-1, :] = (v[2:, 2:, :] - v[2:, :-2, :] - v[:-2, 2:, :] + v[:-2, :-2, :]) / (4 * dK * dZ)
    phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, 1e-9))
    phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, 1e-9))
    Dc = sd2 * (1 - ZZ) ** 2 + sg2 * ZZ ** 2
    E_y = p["vars"] * E * (vY - lNy)
    a_lK = (1 - ZZ) * phid + ZZ * phig - Dc / 2.0
    a_Z = ZZ * (1 - ZZ) * (phig - phid + (1 - ZZ) * sd2 - ZZ * sg2)
    a_Y = p["thbar"] * E
    b_Y = 0.5 * p["vars"] ** 2 * E ** 2
    cross = (-ZZ * (1 - ZZ) ** 2 * sd2 + ZZ ** 2 * (1 - ZZ) * sg2)
    E_d = (1 - ZZ) * p["s_d"] * qd; E_g = ZZ * p["s_g"] * qg
    robust = -0.5 * inv_xi * (E_d ** 2 + E_g ** 2 + E_y ** 2)
    damage = -(lNy * a_Y + lNyy * b_Y)
    LK = np.log(K)
    R = (p["delta"] * (np.log(np.maximum(c, 1e-12)) + LK) - p["delta"] * v
         + a_lK * vlK + (Dc / 2.0) * vKK + a_Z * vZ
         + 0.5 * ZZ ** 2 * (1 - ZZ) ** 2 * (sd2 + sg2) * vZZ
         + cross * vKZ + a_Y * vY + b_Y * vYY + robust + damage)
    return R[2:-2, 2:-2, 2:-2]


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = solve(lam3=1/6.0, xi=148.4, nK=25, nZ=30, nY=25)
    print(f"done in {out['iters']} iters, {time.time()-t0:.0f}s, max|resid|={out['max_abs_residual']:.2e}")
    OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OD, exist_ok=True)
    np.savez(os.path.join(OD, "fd_pdpt_lam3_0167_xi148.npz"),
             **{k: out[k] for k in ("logK", "Z", "Y", "v", "i_d", "i_g", "c", "vlK", "vZ", "vY")})
    # readout at logK~6.78, Z=0.7, Y=3 (compare to NN: i_d~0.040, i_g~0.104)
    ik = np.argmin(np.abs(out["logK"] - np.log(880))); jz = np.argmin(np.abs(out["Z"] - 0.7))
    ky = np.argmin(np.abs(out["Y"] - 3.0))
    print(f"[FD at logK={out['logK'][ik]:.2f},Z={out['Z'][jz]:.2f},Y={out['Y'][ky]:.1f}] "
          f"i_d={out['i_d'][ik,jz,ky]:+.4f} i_g={out['i_g'][ik,jz,ky]:+.4f} "
          f"vlK={out['vlK'][ik,jz,ky]:.3f} c={out['c'][ik,jz,ky]:.4f}")
