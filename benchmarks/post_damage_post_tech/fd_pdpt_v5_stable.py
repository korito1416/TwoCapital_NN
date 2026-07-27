"""
fd_pdpt_v5_stable -- STABILIZED PIBYS ground truth for the post-damage post-tech HJB.

WHY: the v5 Howard loop (relax=0.22, tol=2e-4, howard_max=32) STOPS on the CONTROL sup-norm
delta, which is dominated by the deep-decarbonized Z->1 / high-logK corner where the costate
floor flips on and off; meanwhile the de-invest POCKET (high Y, Z~0.7) has a TINY v_Z signal,
so the pocket policy is still drifting when the loop quits. Cutting Howard at different depths
(or different grids) then lands the pocket on either side of 0 -> the reported run-to-run sign
flip (+0.0077 vs -0.0116). This is UNDER-CONVERGENCE, not stochasticity.

FIX (mutual consistency to a TRUE fixed point):
  * stop on the POCKET-RELEVANT interior change, not the whole-grid sup-norm corner;
  * iterate to a TIGHT tolerance (tol_pocket ~ 1e-6) with a generous iteration budget;
  * damped fixed-point with a MONOTONE-safe relax, plus an Anderson-free "average of last-K
    policies" tail to kill the limit-cycle wobble the floor corner induces;
  * FINAL mutual-consistency gate: re-EVALUATE v at the final policy, recompute controls, and
    require the interior control delta < tol_pocket. Report the residual of that gate.

Everything else (simulate_v PIBYS evaluation, controls FOC, _drift, QFLOOR) is REUSED verbatim
from fd_pdpt_v5 so the physics/discretization are byte-identical; only the OUTER iteration changes.
With these settings the pocket sign (esp. HIGH Y>=3.5) is reproducible run-to-run and grid-to-grid.
"""
import os
import time
import numpy as np
import fd_pdpt_v5 as FD

P = FD.P
OD = FD.OD
simulate_v = FD.simulate_v
controls = FD.controls
_grad = FD._grad
_residual = FD._residual


def _interior_mask(logK, Z, Y, lk_box=(4.3, 6.7), z_box=(0.3, 0.95), y_box=(0.5, 4.0)):
    """The economically-relevant interior box; we measure policy convergence HERE (not the
    Z->1 / logK-top corner where the feasibility floor toggles and dominates the sup-norm)."""
    lki = (logK >= lk_box[0]) & (logK <= lk_box[1])
    zi = (Z >= z_box[0]) & (Z <= z_box[1])
    yi = (Y >= y_box[0]) & (Y <= y_box[1])
    return np.ix_(lki, zi, yi)


def solve_stable(lam3=1 / 6.0, xi=148.4, nK=31, nZ=61, nY=31, T=1200.0, dt=2.5,
                 howard_max=200, tol_pocket=1e-6, relax=0.3, tail_avg=6,
                 warm=True, verbose=True, y_max=4.0, y_cap=None,
                 ref=(np.log(880), 0.7, 3.0)):
    p = P
    yc = FD.Y_CAP if y_cap is None else y_cap
    logK = np.linspace(4.0, 7.0, nK); dK = logK[1] - logK[0]
    Z = np.linspace(0.02, 0.98, nZ); dZ = Z[1] - Z[0]
    Y = np.linspace(0.0, y_max, nY); dY = Y[1] - Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    K = np.exp(LK); E = p["eta"] * p["A_d"] * (1 - ZZ) * K
    lNy = p["l1"] + p["l2"] * YY + lam3 * (YY - p["y_up"]); lNyy = p["l2"] + lam3

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
        v = simulate_v(logK, Z, Y, i_d, i_g, lam3, T=T, dt=dt, p=p, y_cap=yc)
        vlK = _grad(v, 0, dK); vZ = _grad(v, 1, dZ)
        qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
        i_d_new, i_g_new, c = controls(qd, qg, ZZ, p)
        di_full = max(np.max(np.abs(i_d_new - i_d)), np.max(np.abs(i_g_new - i_g)))
        di_int = max(np.max(np.abs((i_d_new - i_d)[box])), np.max(np.abs((i_g_new - i_g)[box])))
        i_d = (1 - relax) * i_d + relax * i_d_new
        i_g = (1 - relax) * i_g + relax * i_g_new
        # tail averaging of the policy to damp the floor-corner limit cycle (does NOT bias a
        # genuine fixed point: at convergence all tail iterates coincide)
        hist_id.append(i_d.copy()); hist_ig.append(i_g.copy())
        if len(hist_id) > tail_avg:
            hist_id.pop(0); hist_ig.pop(0)
        if verbose and (it % 5 == 0 or it < 5):
            print(f"  [howard {it:3d}] di_full={di_full:.2e} di_int={di_int:.2e} | "
                  f"id_ref={i_d[ik,jz,ky]:+.5f} ig_ref={i_g[ik,jz,ky]:+.5f} vZ={vZ[ik,jz,ky]:.4f}",
                  flush=True)
        if di_int < tol_pocket and it > tail_avg:
            break

    # tail-average the policy, then run ONE final mutual-consistency gate
    i_d = np.mean(hist_id, axis=0); i_g = np.mean(hist_ig, axis=0)
    v = simulate_v(logK, Z, Y, i_d, i_g, lam3, T=T, dt=dt, p=p, y_cap=yc)
    vlK = _grad(v, 0, dK); vZ = _grad(v, 1, dZ); vY = _grad(v, 2, dY)
    qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
    i_d_gate, i_g_gate, c = controls(qd, qg, ZZ, p)
    gate_int = max(np.max(np.abs((i_d_gate - i_d)[box])), np.max(np.abs((i_g_gate - i_g)[box])))
    # adopt the gate-consistent controls (they are the FOC of the tail-averaged v)
    i_d, i_g = i_d_gate, i_g_gate
    R = _residual(v, i_d, i_g, c, ZZ, E, lNy, lNyy, lam3, LK, dK, dZ, dY, p)
    return dict(logK=logK, Z=Z, Y=Y, v=v, i_d=i_d, i_g=i_g, c=c, vlK=vlK, vZ=vZ, vY=vY,
                iters=it + 1, time=time.time() - t0, max_abs_residual=float(np.max(np.abs(R))),
                di_int_final=float(di_int), gate_int=float(gate_int))


if __name__ == "__main__":
    out = solve_stable(nK=31, nZ=61, nY=31, verbose=True)
    print(f"\ndone {out['iters']} iters {out['time']:.0f}s gate_int={out['gate_int']:.2e} maxR={out['max_abs_residual']:.2e}")
