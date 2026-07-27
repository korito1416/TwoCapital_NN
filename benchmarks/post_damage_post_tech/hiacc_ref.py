"""
hiacc_ref.py -- build the HIGH-ACCURACY FD reference for the post-damage-post-tech controls/costate.

ESTABLISHED THIS SESSION (policy-frozen probes at the 31x61x31 stable npz; T=1500):
  * HORIZON T converged: T=1200 already ~2e-6 vs T=2400 -> T=1500 is safe.
  * DOMINANT error is the TIME STEP dt (RK2 + C0 trilinear control interp along trajectories):
    clean self-convergence order p ~ 1.57. dt=2.5 (old stable npz) is ~2.6e-3 on i_d / 1.8e-2 on vZ.
    dt-RICHARDSON is VALID and VALIDATED: R(0.25,0.125) vs R(0.125,0.0625) agree to i_d 5.0e-5,
    i_g 6.6e-5, vZ 3.4e-4 -> the (0.125,0.0625) extrapolant is the dt->0 limit to ~5e-5 (controls).
  * SPATIAL error: costates from a central stencil; v_logK moved ~5% under nK/nY -> spatial
    refinement also needed; we use a 4TH-ORDER stencil + spatial Richardson across 3 grids.

EFFICIENT DESIGN (validated by policy_dt_indep.py -- the converged POLICY is dt-robust):
  Per spatial grid g (warm-started from the previous coarser converged policy to keep Howard short):
    (A) run the PIBYS Howard fixed point at a CHEAP dt (=2.5) to a tight policy tol -> the POLICY.
    (B) FINAL READOUT: freeze that policy, evaluate v at dt=0.125 AND dt=0.0625, dt-Richardson the
        4th-order costates/controls to the dt->0 limit. (dt error ~5e-5 on controls.)
  Then spatial-Richardson the dt-extrapolated fields across the 3 grids.

REFERENCE ACCURACY = max(spatial_resid, dt_resid) per field over the apples-to-apples interior box,
reported and saved. Target << 1e-3 (toward 1e-4) on the CONTROLS (i_d,i_g) and the costate (vZ).

Run on a compute node (fine grids heavy): srun ... python hiacc_ref.py
"""
import os, time, numpy as np, sys
sys.path.insert(0, '.')
from scipy.interpolate import RegularGridInterpolator as RGI
import fd_pdpt_v5 as FD
from stable_fd_eval import _box_mask

P = FD.P
OD = FD.OD
LAM3 = 1 / 6.0
T_HORIZON = 1500.0
P_DT = 1.57                       # measured dt-convergence order
DT_LOOP = 2.5                     # cheap dt for the Howard policy loop (policy is dt-robust)
DT_READ = (0.125, 0.0625)        # fine dt pair for the frozen-policy readout (dt error ~5e-5)


def grad4(v, axis, dx):
    """4th-order central interior, 2nd-order on the one-in rims (fields are on nodes)."""
    g = np.zeros_like(v); sl = [slice(None)] * 3
    p2 = sl.copy(); p2[axis] = slice(4, None)
    p1 = sl.copy(); p1[axis] = slice(3, -1)
    m1 = sl.copy(); m1[axis] = slice(1, -3)
    m2 = sl.copy(); m2[axis] = slice(0, -4)
    mid = sl.copy(); mid[axis] = slice(2, -2)
    g[tuple(mid)] = (-v[tuple(p2)] + 8 * v[tuple(p1)] - 8 * v[tuple(m1)] + v[tuple(m2)]) / (12 * dx)
    g2 = FD._grad(v, axis, dx)
    edge = np.zeros(v.shape, bool)
    for idx in (0, 1, -2, -1):
        e = sl.copy(); e[axis] = idx; edge[tuple(e)] = True
    return np.where(edge, g2, g)


def warm_from(policy_src, logK, Z, Y):
    sK, sZ, sY, sid, sig = policy_src
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    q = np.stack([np.clip(LK, sK[0], sK[-1]).ravel(),
                  np.clip(ZZ, sZ[0], sZ[-1]).ravel(),
                  np.clip(YY, sY[0], sY[-1]).ravel()], axis=1)
    gid = RGI((sK, sZ, sY), sid, bounds_error=False, fill_value=None)
    gig = RGI((sK, sZ, sY), sig, bounds_error=False, fill_value=None)
    return gid(q).reshape(LK.shape), gig(q).reshape(LK.shape)


def readout(logK, Z, Y, i_d, i_g, dt):
    """Frozen-policy value eval at dt; return 4th-order costates + FOC controls."""
    dK = logK[1] - logK[0]; dZ = Z[1] - Z[0]; dY = Y[1] - Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    v = FD.simulate_v(logK, Z, Y, i_d, i_g, LAM3, T=T_HORIZON, dt=dt, p=P, y_cap=FD.Y_CAP)
    vlK = grad4(v, 0, dK); vZ = grad4(v, 1, dZ); vY = grad4(v, 2, dY)
    qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
    id_, ig_, c = FD.controls(qd, qg, ZZ, P)
    return dict(logK=logK, Z=Z, Y=Y, v=v, i_d=id_, i_g=ig_, c=c, vlK=vlK, vZ=vZ, vY=vY)


def dt_extrap(coarse, fine, dtc, dtf, p=P_DT):
    r = (dtc / dtf) ** p
    out = dict(logK=fine['logK'], Z=fine['Z'], Y=fine['Y'])
    for k in ('i_d', 'i_g', 'vZ', 'vlK', 'vY', 'v', 'c'):
        out[k] = fine[k] + (fine[k] - coarse[k]) / (r - 1)
    out['raw_fine'] = fine
    return out


def solve_grid(nK, nZ, nY, warm, howard_max=120, tol_pocket=2e-7, relax=0.3,
               tail_avg=6, verbose=True):
    logK = np.linspace(4.0, 7.0, nK); dK = logK[1] - logK[0]
    Z = np.linspace(0.02, 0.98, nZ); dZ = Z[1] - Z[0]
    Y = np.linspace(0.0, 4.0, nY)
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    box = _box_mask(logK, Z, Y)
    i_d, i_g = warm_from(warm, logK, Z, Y)
    t0 = time.time(); hist_id = []; hist_ig = []; di_int = np.inf; it = 0
    for it in range(howard_max):
        v = FD.simulate_v(logK, Z, Y, i_d, i_g, LAM3, T=T_HORIZON, dt=DT_LOOP, p=P, y_cap=FD.Y_CAP)
        vlK = grad4(v, 0, dK); vZ = grad4(v, 1, dZ)
        qd = vlK - ZZ * vZ; qg = vlK + (1 - ZZ) * vZ
        i_d_new, i_g_new, c = FD.controls(qd, qg, ZZ, P)
        di_int = max(np.max(np.abs((i_d_new - i_d)[box])), np.max(np.abs((i_g_new - i_g)[box])))
        i_d = (1 - relax) * i_d + relax * i_d_new
        i_g = (1 - relax) * i_g + relax * i_g_new
        hist_id.append(i_d.copy()); hist_ig.append(i_g.copy())
        if len(hist_id) > tail_avg:
            hist_id.pop(0); hist_ig.pop(0)
        if verbose and (it % 10 == 0 or it < 3):
            print(f"    [{nK}x{nZ}x{nY} loop {it:3d}] di_int={di_int:.2e}", flush=True)
        if di_int < tol_pocket and it > tail_avg:
            break
    i_d = np.mean(hist_id, axis=0); i_g = np.mean(hist_ig, axis=0)
    loop_t = time.time() - t0
    if verbose:
        print(f"    [{nK}x{nZ}x{nY}] policy loop done: {it+1} iters {loop_t:.0f}s di_int={di_int:.1e}; readout dt={DT_READ}", flush=True)
    # frozen-policy dt-Richardson readout
    rc = readout(logK, Z, Y, i_d, i_g, DT_READ[0])
    rf = readout(logK, Z, Y, i_d, i_g, DT_READ[1])
    ext = dt_extrap(rc, rf, DT_READ[0], DT_READ[1])
    ext['grid'] = (nK, nZ, nY); ext['time'] = time.time() - t0; ext['iters'] = it + 1
    ext['di_int'] = float(di_int)
    return ext


def interp_fields(src, tgt_axes, keys=('i_d', 'i_g', 'vZ', 'vlK')):
    LK, ZZ, YY = np.meshgrid(*tgt_axes, indexing="ij")
    q = np.stack([LK.ravel(), ZZ.ravel(), YY.ravel()], axis=1)
    out = {}
    for k in keys:
        f = RGI((src['logK'], src['Z'], src['Y']), src[k], bounds_error=False, fill_value=None)
        out[k] = f(q).reshape(LK.shape)
    return out


if __name__ == "__main__":
    print("=" * 92, flush=True)
    print("HIGH-ACCURACY FD REFERENCE  (cheap-dt Howard policy + frozen-policy dt-Richardson readout", flush=True)
    print(" x spatial Richardson, 4th-order costates, T=1500)", flush=True)
    print("=" * 92, flush=True)
    d0 = np.load('outputs/fd_pdpt_v5_stable_lam3_0167_xi148.npz')
    seed = (d0['logK'], d0['Z'], d0['Y'], d0['i_d'], d0['i_g'])

    GRIDS = [(31, 61, 31), (41, 81, 41), (51, 101, 51)]

    sols = []; warm = seed
    for g in GRIDS:
        print(f"\n--- spatial grid {g[0]}x{g[1]}x{g[2]} ---", flush=True)
        s = solve_grid(*g, warm=warm, verbose=True)
        print(f"  grid {g} total {s['time']:.0f}s", flush=True)
        sols.append(s)
        warm = (s['logK'], s['Z'], s['Y'], s['i_d'], s['i_g'])

    # spatial self-convergence (finer interpolated to coarser axes, box max)
    print("\n" + "=" * 92, flush=True)
    print("SPATIAL self-convergence of dt-extrapolated fields (box max):", flush=True)
    for i in range(len(sols) - 1):
        a, b = sols[i], sols[i + 1]
        axes = (a['logK'], a['Z'], a['Y']); box = _box_mask(*axes)
        bi = interp_fields(b, axes)
        for k in ('i_d', 'i_g', 'vZ'):
            pass
        print("  {} vs {}: i_d={:.3e} i_g={:.3e} vZ={:.3e}".format(
            a['grid'], b['grid'],
            float(np.max(np.abs(a['i_d'] - bi['i_d'])[box])),
            float(np.max(np.abs(a['i_g'] - bi['i_g'])[box])),
            float(np.max(np.abs(a['vZ'] - bi['vZ'])[box]))), flush=True)

    # spatial order estimate on coarsest common axes
    coarse, mid, fine = sols[0], sols[1], sols[2]
    axes_c = (coarse['logK'], coarse['Z'], coarse['Y']); box_c = _box_mask(*axes_c)
    m_on_c = interp_fields(mid, axes_c); f_on_c = interp_fields(fine, axes_c)
    print("\nSPATIAL order estimate q (three-grid):", flush=True)
    for k in ('i_d', 'i_g', 'vZ'):
        e1 = float(np.max(np.abs(coarse[k] - m_on_c[k])[box_c]))
        e2 = float(np.max(np.abs(m_on_c[k] - f_on_c[k])[box_c]))
        print(f"    {k}: e(g0,g1)={e1:.3e} e(g1,g2)={e2:.3e} ratio={e1/e2 if e2>0 else np.inf:.2f}", flush=True)

    # spatial Richardson of the two finest (q=2 conservative; policy resolution limits to ~2nd order)
    axes_f = (fine['logK'], fine['Z'], fine['Y']); box_f = _box_mask(*axes_f)
    mid_on_f = interp_fields(mid, axes_f, keys=('i_d', 'i_g', 'vZ', 'vlK'))
    rK = (mid['logK'][1] - mid['logK'][0]) / (fine['logK'][1] - fine['logK'][0])
    q_sp = 2.0; rr = rK ** q_sp
    ref = dict(logK=fine['logK'], Z=fine['Z'], Y=fine['Y'])
    for k in ('i_d', 'i_g', 'vZ', 'vlK'):
        ref[k] = fine[k] + (fine[k] - mid_on_f[k]) / (rr - 1)
    for k in ('vY', 'v', 'c'):
        ref[k] = fine[k]

    print("\nESTIMATED REFERENCE ACCURACY (box max -- bound on the reference's own control/costate error):", flush=True)
    acc = {}
    for k in ('i_d', 'i_g', 'vZ'):
        sp_resid = float(np.max(np.abs(fine[k] - ref[k])[box_f]))
        dt_resid = float(np.max(np.abs(fine['raw_fine'][k] - fine[k])[box_f]))
        acc[k] = max(sp_resid, dt_resid)
        print(f"  {k}: spatial_resid={sp_resid:.3e}  dt_resid={dt_resid:.3e}  -> ACCURACY ~ {acc[k]:.3e}", flush=True)

    out_path = os.path.join(OD, 'fd_pdpt_hiacc_ref_lam3_0167_xi148.npz')
    np.savez_compressed(out_path,
                        logK=ref['logK'], Z=ref['Z'], Y=ref['Y'],
                        v=ref['v'], i_d=ref['i_d'], i_g=ref['i_g'], c=ref['c'],
                        vlK=ref['vlK'], vZ=ref['vZ'], vY=ref['vY'],
                        acc_i_d=acc['i_d'], acc_i_g=acc['i_g'], acc_vZ=acc['vZ'],
                        grids=str(GRIDS), dt_read=str(DT_READ), dt_loop=DT_LOOP,
                        p_dt=P_DT, q_sp=q_sp, T=T_HORIZON)
    print(f"\nSAVED -> {out_path}", flush=True)
    print(f"ACCURACY (box max): i_d~{acc['i_d']:.2e} i_g~{acc['i_g']:.2e} vZ~{acc['vZ']:.2e}", flush=True)
