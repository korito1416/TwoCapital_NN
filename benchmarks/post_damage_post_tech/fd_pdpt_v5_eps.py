"""
fd_pdpt_v5_eps -- vanishing-viscosity (anisotropic eps*v_ZZ) layered onto fd_pdpt_v5's PIBYS
policy-evaluation, via a Lie/Strang-split implicit Z-diffusion screened solve that PRESERVES the
PIBYS edge-clamp BC.

EVALUATE(eps):
  STEP A (advection, full, UNCHANGED):  v_half = FD.simulate_v(...)
  STEP B (implicit Z-diffusion, screened, edge-clamp BC): along every (logK,Y) line,
            (delta*I - eps*D_ZZ) v_new = delta * v_half
         with EDGE-CLAMP face rows v_new[face]=v_half[face] (identity rows -- NOT zero-flux Neumann).
  At eps=0 -> v_new == v_half EXACTLY.  Same solve_banded path for all eps (no eps==0 early return).

Outer loop = fd_pdpt_v5_stable.solve_stable VERBATIM; only EVALUATE replaced. controls/_grad/_drift/
_residual/QFLOOR reused from FD; qfloor exposed as a param.
"""
import os
import time
import numpy as np
from scipy.linalg import solve_banded

import fd_pdpt_v5 as FD

P = FD.P
OD = FD.OD


def _controls_qf(qd, qg, Z, p=P, qfloor=FD.QFLOOR):
    qd = np.maximum(qd, qfloor)
    qg = np.maximum(qg, qfloor)
    Abar = (1 - Z) * p["A_d"] + Z * p["A_gpp"]
    num = p["delta"] * (Abar + (1 - Z) / p["t_d"] + Z / p["t_g"])
    den = p["delta"] + (1 - Z) * p["G_d"] * qd + Z * p["G_g"] * qg
    c = num / den
    i_d = p["G_d"] * qd * c / p["delta"] - 1.0 / p["t_d"]
    i_g = p["G_g"] * qg * c / p["delta"] - 1.0 / p["t_g"]
    return i_d, i_g, c


def _screened_zdiff(v_half, eps, dZ, delta):
    nK, nZ, nY = v_half.shape
    r = eps / dZ ** 2
    ab = np.zeros((3, nZ))
    ab[1, 1:nZ - 1] = delta + 2.0 * r
    ab[0, 2:nZ] = -r
    ab[2, 0:nZ - 2] = -r
    ab[1, 0] = 1.0; ab[1, nZ - 1] = 1.0
    rhs = (delta * v_half).copy()
    rhs[:, 0, :] = v_half[:, 0, :]
    rhs[:, -1, :] = v_half[:, -1, :]
    B = np.moveaxis(rhs, 1, 0).reshape(nZ, nK * nY)
    X = solve_banded((1, 1), ab, B)
    v_new = np.moveaxis(X.reshape(nZ, nK, nY), 0, 1)
    return v_new


def evaluate_eps(logK, Z, Y, i_d, i_g, lam3, eps, T, dt, p, y_cap):
    dZ = Z[1] - Z[0]
    v_half = FD.simulate_v(logK, Z, Y, i_d, i_g, lam3, T=T, dt=dt, p=p, y_cap=y_cap)
    v_new = _screened_zdiff(v_half, eps, dZ, p["delta"])
    return v_new, v_half


def _interior_mask(logK, Z, Y, lk_box=(4.3, 6.7), z_box=(0.3, 0.95), y_box=(0.5, 4.0)):
    lki = (logK >= lk_box[0]) & (logK <= lk_box[1])
    zi = (Z >= z_box[0]) & (Z <= z_box[1])
    yi = (Y >= y_box[0]) & (Y <= y_box[1])
    return np.ix_(lki, zi, yi)


def solve(eps=0.0, lam3=1 / 6.0, xi=148.4, nK=31, nZ=61, nY=31, T=1200.0, dt=2.5,
          howard_max=200, tol_pocket=1e-6, relax=0.3, tail_avg=6, warm=True,
          verbose=True, y_max=4.0, y_cap=None, qfloor=FD.QFLOOR,
          init_controls=None, ref=(np.log(880), 0.7, 3.0)):
    p = P
    yc = FD.Y_CAP if y_cap is None else y_cap
    logK = np.linspace(4.0, 7.0, nK); dK = logK[1] - logK[0]
    Z = np.linspace(0.02, 0.98, nZ); dZ = Z[1] - Z[0]
    Y = np.linspace(0.0, y_max, nY); dY = Y[1] - Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    K = np.exp(LK); E = p["eta"] * p["A_d"] * (1 - ZZ) * K
    lNy = p["l1"] + p["l2"] * YY + lam3 * (YY - p["y_up"]); lNyy = p["l2"] + lam3

    if init_controls is not None:
        i_d, i_g = init_controls[0].copy(), init_controls[1].copy()
    else:
        init = FD._init_from_v3(logK, Z, Y) if warm else None
        if init is not None:
            i_d, i_g = init
        else:
            i_d = np.zeros_like(LK); i_g = np.full_like(LK, 0.05)

    box = _interior_mask(logK, Z, Y)
    ik = np.argmin(np.abs(logK - ref[0])); jz = np.argmin(np.abs(Z - ref[1])); ky = np.argmin(np.abs(Y - ref[2]))
    t0 = time.time()
    hist_id = []; hist_ig = []
    di_int = np.inf; di_full = np.inf
    it = 0
    for it in range(howard_max):
        v, _ = evaluate_eps(logK, Z, Y, i_d, i_g, lam3, eps, T, dt, p, yc)
        vlK = FD._grad(v, 0, dK); vZ = FD._grad(v, 1, dZ)
        qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
        i_d_new, i_g_new, c = _controls_qf(qd, qg, ZZ, p, qfloor)
        di_full = max(np.max(np.abs(i_d_new - i_d)), np.max(np.abs(i_g_new - i_g)))
        di_int = max(np.max(np.abs((i_d_new - i_d)[box])), np.max(np.abs((i_g_new - i_g)[box])))
        i_d = (1 - relax) * i_d + relax * i_d_new
        i_g = (1 - relax) * i_g + relax * i_g_new
        hist_id.append(i_d.copy()); hist_ig.append(i_g.copy())
        if len(hist_id) > tail_avg:
            hist_id.pop(0); hist_ig.pop(0)
        if verbose and (it % 5 == 0 or it < 5):
            print(f"  [eps={eps:.1e} howard {it:3d}] di_full={di_full:.2e} di_int={di_int:.2e} | "
                  f"id_ref={i_d[ik,jz,ky]:+.5f} ig_ref={i_g[ik,jz,ky]:+.5f} vZ={vZ[ik,jz,ky]:.4f}",
                  flush=True)
        if di_int < tol_pocket and it > tail_avg:
            break

    i_d = np.mean(hist_id, axis=0); i_g = np.mean(hist_ig, axis=0)
    v, _ = evaluate_eps(logK, Z, Y, i_d, i_g, lam3, eps, T, dt, p, yc)
    vlK = FD._grad(v, 0, dK); vZ = FD._grad(v, 1, dZ); vY = FD._grad(v, 2, dY)
    qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
    i_d_gate, i_g_gate, c = _controls_qf(qd, qg, ZZ, p, qfloor)
    gate_int = max(np.max(np.abs((i_d_gate - i_d)[box])), np.max(np.abs((i_g_gate - i_g)[box])))
    i_d, i_g = i_d_gate, i_g_gate
    R = FD._residual(v, i_d, i_g, c, ZZ, E, lNy, lNyy, lam3, LK, dK, dZ, dY, p)
    return dict(logK=logK, Z=Z, Y=Y, v=v, i_d=i_d, i_g=i_g, c=c, vlK=vlK, vZ=vZ, vY=vY,
                iters=it + 1, time=time.time() - t0, max_abs_residual=float(np.max(np.abs(R))),
                di_int_final=float(di_int), gate_int=float(gate_int), eps=eps, qfloor=qfloor)


if __name__ == "__main__":
    out = solve(eps=0.0, nK=21, nZ=31, nY=21, T=1200.0, dt=2.5, howard_max=20, verbose=True)
    print(f"\ndone eps=0 {out['iters']} iters {out['time']:.0f}s gate={out['gate_int']:.2e}")
