"""
Tighten the only remaining caveat in fd_pdpt_v5: the v_logK READOUT moved ~5% under nK/nY refinement.
Two suspects: (a) nearest-NODE readout aliasing (logK=6.78 is not a grid node, so the nearest node jumps
as nK changes), (b) O(dK^2) error in the central-difference gradient. We fix (a) by reading v_logK at the
EXACT point via interpolation, fix (b) with a 4th-order gradient, and Richardson-extrapolate the sequence.
If v_logK converges to a stable value, "just adjust the grid" is confirmed and the FD is fully grid-converged.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
import fd_pdpt_v5 as FD

REF = (np.log(880), 0.7, 3.0)


def grad(v, axis, dx, order):
    g = np.zeros_like(v)
    sl = [slice(None)] * 3
    if order == 2:
        a = sl.copy(); a[axis] = slice(2, None); b = sl.copy(); b[axis] = slice(0, -2)
        m = sl.copy(); m[axis] = slice(1, -1)
        g[tuple(m)] = (v[tuple(a)] - v[tuple(b)]) / (2 * dx)
    else:  # 4th-order central interior: (-f[i+2]+8f[i+1]-8f[i-1]+f[i-2])/(12 dx)
        p2 = sl.copy(); p2[axis] = slice(4, None); p1 = sl.copy(); p1[axis] = slice(3, -1)
        m1 = sl.copy(); m1[axis] = slice(1, -3); m2 = sl.copy(); m2[axis] = slice(0, -4)
        mid = sl.copy(); mid[axis] = slice(2, -2)
        g[tuple(mid)] = (-v[tuple(p2)] + 8 * v[tuple(p1)] - 8 * v[tuple(m1)] + v[tuple(m2)]) / (12 * dx)
        # fall back to 2nd order on the one-in rims so the field is filled for interpolation
        g2 = grad(v, axis, dx, 2)
        edge = np.zeros(v.shape, bool)
        e = sl.copy(); e[axis] = 1; edge[tuple(e)] = True
        e = sl.copy(); e[axis] = -2; edge[tuple(e)] = True
        g = np.where(edge, g2, g)
    return g


def read_exact(out, order):
    """v_logK and v_Z at the EXACT reference point via interpolation (no nearest-node aliasing)."""
    lk, Z, Y = out["logK"], out["Z"], out["Y"]
    dK = lk[1] - lk[0]; dZ = Z[1] - Z[0]
    vlK = grad(out["v"], 0, dK, order); vZ = grad(out["v"], 1, dZ, order)
    pt = np.array([[min(max(REF[0], lk[0]), lk[-1]), REF[1], REF[2]]])
    g_vlK = RGI((lk, Z, Y), vlK, bounds_error=False, fill_value=None)(pt)[0]
    g_vZ = RGI((lk, Z, Y), vZ, bounds_error=False, fill_value=None)(pt)[0]
    g_id = RGI((lk, Z, Y), out["i_d"], bounds_error=False, fill_value=None)(pt)[0]
    return g_vlK, g_vZ, g_id, dK


def richardson(vals, dks, order):
    """Extrapolate vals(dk) -> dk=0 assuming leading error O(dk^order), from the two finest grids."""
    (v1, d1), (v2, d2) = (vals[-2], dks[-2]), (vals[-1], dks[-1])
    r = (d1 / d2) ** order
    return v2 + (v2 - v1) / (r - 1)


def sweep(name, grids):
    print(f"\n--- {name} ---")
    print(f"  {'grid':>14} {'dK':>7} {'vlK(2nd)':>9} {'vlK(4th)':>9} {'vZ(4th)':>9} {'i_d':>9}")
    rows = []
    for (nK, nZ, nY) in grids:
        out = FD.solve(nK=nK, nZ=nZ, nY=nY, T=1200.0, dt=2.5, howard_max=32, verbose=False)
        v2, _, _, dK = read_exact(out, 2)
        v4, vZ4, idr, _ = read_exact(out, 4)
        rows.append((nK, nZ, nY, dK, v2, v4, vZ4, idr))
        print(f"  {nK:2d}x{nZ:2d}x{nY:2d}".rjust(14) + f" {dK:7.4f} {v2:9.4f} {v4:9.4f} {vZ4:9.4f} {idr:+9.4f}", flush=True)
    dks = [r[3] for r in rows]
    v2s = [r[4] for r in rows]; v4s = [r[5] for r in rows]
    print(f"  Richardson v_logK*: from 2nd-order seq = {richardson(v2s, dks, 2):.4f}  | "
          f"from 4th-order seq = {richardson(v4s, dks, 4):.4f}")
    return rows


def main():
    print("=" * 92)
    print("fd_pdpt_v5 v_logK grid convergence at EXACT reference (logK=6.78, Z=0.7, Y=3.0)")
    print("interpolated readout (no nearest-node aliasing) + 4th-order gradient + Richardson")
    print("=" * 92)
    sweep("nK refinement (nZ=41, nY=25 fixed)", [(17, 41, 25), (25, 41, 25), (33, 41, 25), (41, 41, 25)])
    sweep("nY refinement (nK=33, nZ=41 fixed)", [(33, 41, 17), (33, 41, 25), (33, 41, 33)])
    print("\nVERDICT: if vlK(4th) is stable across grids and the two Richardson estimates agree,")
    print("the readout was the issue and the FD is fully grid-converged -> 'adjust the grid' = YES.")


if __name__ == "__main__":
    main()
