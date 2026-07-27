"""Check time-step (dt) and horizon (T) sensitivity of the controls/costate at the BASE grid.

If refining dt and T moves i_d/i_g/vZ by >>1e-4, then SPATIAL refinement alone cannot make a
1e-4 reference -- we must also tighten dt and T. We freeze the policy at the stable-npz controls
(the converged coarse fixed point) and only re-EVALUATE v under different (dt,T), then read the
FOC controls/costate. This isolates the time-integration/truncation error in the value functional
from the outer Howard convergence.
"""
import numpy as np, sys
sys.path.insert(0, '.')
import fd_pdpt_v5 as FD

P = FD.P
LAM3 = 1 / 6.0
d = np.load('outputs/fd_pdpt_v5_stable_lam3_0167_xi148.npz')
logK, Z, Y = d['logK'], d['Z'], d['Y']
dK = logK[1]-logK[0]; dZ = Z[1]-Z[0]; dY = Y[1]-Y[0]
LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
i_d_fix, i_g_fix = d['i_d'], d['i_g']

# economically-relevant interior box for measuring change
from stable_fd_eval import _box_mask
box = _box_mask(logK, Z, Y)

def eval_controls(T, dt):
    v = FD.simulate_v(logK, Z, Y, i_d_fix, i_g_fix, LAM3, T=T, dt=dt, p=P, y_cap=FD.Y_CAP)
    vlK = FD._grad(v, 0, dK); vZ = FD._grad(v, 1, dZ)
    qd = vlK - ZZ*vZ; qg = vlK + (1-ZZ)*vZ
    i_d, i_g, c = FD.controls(qd, qg, ZZ, P)
    return i_d, i_g, vZ

print("=== dt sensitivity (T=1200 fixed) ===", flush=True)
ref = eval_controls(1200.0, 1.0)
for dt in [4.0, 2.5, 1.5, 1.0]:
    i_d, i_g, vZ = eval_controls(1200.0, dt)
    eid = np.max(np.abs(i_d - ref[0])[box]); eig = np.max(np.abs(i_g - ref[1])[box]); evz = np.max(np.abs(vZ - ref[2])[box])
    print(f"  dt={dt:4.1f}: box dmax i_d={eid:.3e} i_g={eig:.3e} vZ={evz:.3e}  (vs dt=1.0)", flush=True)

print("\n=== T sensitivity (dt=1.5 fixed) ===", flush=True)
refT = eval_controls(2400.0, 1.5)
for T in [800.0, 1200.0, 1800.0, 2400.0]:
    i_d, i_g, vZ = eval_controls(T, 1.5)
    eid = np.max(np.abs(i_d - refT[0])[box]); eig = np.max(np.abs(i_g - refT[1])[box]); evz = np.max(np.abs(vZ - refT[2])[box])
    print(f"  T={T:6.0f}: box dmax i_d={eid:.3e} i_g={eig:.3e} vZ={evz:.3e}  (vs T=2400)", flush=True)
