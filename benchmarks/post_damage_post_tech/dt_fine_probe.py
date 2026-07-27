"""Push dt finer to map the dt-convergence ORDER and find where controls stop moving < 1e-4.
Policy frozen at stable npz; T=1500 (safely converged). Self-convergence: compare each dt to the
finest. Also report successive-ratio to estimate the order p (RK2 nominal p=2, but C0 control
interpolation may degrade it toward p=1)."""
import numpy as np, sys
sys.path.insert(0, '.')
import fd_pdpt_v5 as FD
from stable_fd_eval import _box_mask

P = FD.P; LAM3 = 1/6.0
d = np.load('outputs/fd_pdpt_v5_stable_lam3_0167_xi148.npz')
logK, Z, Y = d['logK'], d['Z'], d['Y']
dK = logK[1]-logK[0]; dZ = Z[1]-Z[0]
LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
i_d_fix, i_g_fix = d['i_d'], d['i_g']
box = _box_mask(logK, Z, Y)

def eval_controls(T, dt):
    v = FD.simulate_v(logK, Z, Y, i_d_fix, i_g_fix, LAM3, T=T, dt=dt, p=P, y_cap=FD.Y_CAP)
    vlK = FD._grad(v, 0, dK); vZ = FD._grad(v, 1, dZ)
    qd = vlK - ZZ*vZ; qg = vlK + (1-ZZ)*vZ
    i_d, i_g, c = FD.controls(qd, qg, ZZ, P)
    return i_d, i_g, vZ

T = 1500.0
dts = [2.0, 1.0, 0.5, 0.25, 0.125]
res = {}
for dt in dts:
    res[dt] = eval_controls(T, dt)
    print(f"  computed dt={dt}", flush=True)

fin = res[dts[-1]]
print("\n=== dt convergence vs finest dt={} (box max) ===".format(dts[-1]), flush=True)
prev = None
errs = []
for dt in dts[:-1]:
    eid = np.max(np.abs(res[dt][0]-fin[0])[box])
    eig = np.max(np.abs(res[dt][1]-fin[1])[box])
    evz = np.max(np.abs(res[dt][2]-fin[2])[box])
    errs.append((dt, eid, eig, evz))
    print(f"  dt={dt:6.3f}: i_d={eid:.3e} i_g={eig:.3e} vZ={evz:.3e}", flush=True)

# successive-grid order estimate on i_d (compare consecutive pairs, halving dt)
print("\n=== order estimate p (consecutive dt halving on i_d) ===", flush=True)
for i in range(len(dts)-2):
    a,b,c = res[dts[i]][0], res[dts[i+1]][0], res[dts[i+2]][0]
    e1 = np.max(np.abs(a-c)[box]); e2 = np.max(np.abs(b-c)[box])
    if e2>0:
        p = np.log2(e1/e2)
        print(f"  dt {dts[i]}->{dts[i+1]}->{dts[i+2]}: ratio={e1/e2:.2f} -> p~{p:.2f}", flush=True)
