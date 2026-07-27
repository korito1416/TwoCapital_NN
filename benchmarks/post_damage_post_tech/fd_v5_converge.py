"""
Grid-convergence (Richardson) study for the PIBYS solver fd_pdpt_v5 (honest high-Y damage, de-invest).
The pathology was in the advection-dominated Z direction, so we refine nZ aggressively and check that the
reference-point readout (i_d, v_logK, v_Z) STOPS drifting -- the thing upwind v3 failed. We also run a COLD
start to prove the de-invest answer is not a warm-start artifact, and report successive-change magnitudes.
Saves the finest solution for fd_vs_nn.py / policy_eval.py.
"""
import os
import numpy as np
import fd_pdpt_v5 as FD

OD = FD.OD


def readout(out):
    ik = np.argmin(np.abs(out["logK"] - np.log(880)))
    jz = np.argmin(np.abs(out["Z"] - 0.7)); ky = np.argmin(np.abs(out["Y"] - 3.0))
    return dict(i_d=float(out["i_d"][ik, jz, ky]), i_g=float(out["i_g"][ik, jz, ky]),
               vlK=float(out["vlK"][ik, jz, ky]), vZ=float(out["vZ"][ik, jz, ky]),
               c=float(out["c"][ik, jz, ky]), maxR=out["max_abs_residual"], iters=out["iters"], t=out["time"])


def line(tag, r, prev=None):
    d = ""
    if prev is not None:
        d = f"   d(i_d)={r['i_d']-prev['i_d']:+.4f} d(vlK)={r['vlK']-prev['vlK']:+.4f} d(vZ)={r['vZ']-prev['vZ']:+.4f}"
    print(f"  {tag:20s} i_d={r['i_d']:+.4f}  i_g={r['i_g']:+.4f}  vlK={r['vlK']:.4f}  v_Z={r['vZ']:.4f}  "
          f"c={r['c']:.4f}  maxR={r['maxR']:.1e} ({r['t']:.0f}s){d}", flush=True)


def main():
    print("=" * 104)
    print("PIBYS grid-convergence (honest damage): reference logK=6.78, Z=0.7, Y=3.0  (lam3=1/6, xi=148.4)")
    print("=" * 104)

    print("\n--- Z-REFINEMENT (nK=21, nY=15 fixed): the advection-dominated direction; deltas must SHRINK ---")
    prev = None
    for nZ in (21, 41, 81, 121):
        r = readout(FD.solve(nK=21, nZ=nZ, nY=15, T=1200.0, dt=2.5, howard_max=32, verbose=False))
        line(f"21 x {nZ:3d} x 15", r, prev); prev = r

    print("\n--- FULL refinement (all directions) ---")
    finest = None; prev = None
    for (nK, nZ, nY) in ((21, 31, 21), (31, 61, 31)):
        out = FD.solve(nK=nK, nZ=nZ, nY=nY, T=1200.0, dt=2.5, howard_max=32, verbose=False)
        r = readout(out); line(f"{nK} x {nZ} x {nY}", r, prev); prev = r; finest = out

    print("\n--- INIT INDEPENDENCE: cold vs warm at 25 x 61 x 21 ---")
    line("25 x 61 x 21 COLD", readout(FD.solve(nK=25, nZ=61, nY=21, howard_max=40, warm=False, verbose=False)))
    line("25 x 61 x 21 warm", readout(FD.solve(nK=25, nZ=61, nY=21, howard_max=32, warm=True, verbose=False)))

    np.savez(os.path.join(OD, "fd_pdpt_v5_lam3_0167_xi148.npz"),
             **{k: finest[k] for k in ("logK", "Z", "Y", "v", "i_d", "i_g", "c", "vlK", "vZ", "vY")})
    print(f"\n[saved] outputs/fd_pdpt_v5_lam3_0167_xi148.npz (finest = 31x61x31)")
    print("\nPASS if: (1) Z-refinement deltas SHRINK (grid-converged, unlike v3); (2) i_d<0 (de-invest);")
    print("         (3) COLD ~ warm (init-independent).")


if __name__ == "__main__":
    main()
