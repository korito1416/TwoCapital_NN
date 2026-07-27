"""VALIDATE dt-Richardson: extrapolate the controls/costate in dt (order p~1.57) and check the
extrapolated field against an independent finer dt. If |Richardson(0.5,0.25) - true(0.0625)| << 1e-4
and |Richardson(0.25,0.125) - Richardson(0.125,0.0625)| << 1e-4, dt-Richardson is a valid 1e-4 knob.

Policy frozen at stable npz; T=1500. Field-wise box-max errors.
"""
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

def eval_fields(dt, T=1500.0):
    v = FD.simulate_v(logK, Z, Y, i_d_fix, i_g_fix, LAM3, T=T, dt=dt, p=P, y_cap=FD.Y_CAP)
    vlK = FD._grad(v, 0, dK); vZ = FD._grad(v, 1, dZ)
    qd = vlK - ZZ*vZ; qg = vlK + (1-ZZ)*vZ
    i_d, i_g, c = FD.controls(qd, qg, ZZ, P)
    return dict(i_d=i_d, i_g=i_g, vZ=vZ)

dts = [0.5, 0.25, 0.125, 0.0625]
F = {}
for dt in dts:
    F[dt] = eval_fields(dt); print(f"  computed dt={dt}", flush=True)

P_ORDER = 1.57
def rich(coarse, fine, dtc, dtf, p=P_ORDER):
    r = (dtc/dtf)**p
    return {k: fine[k] + (fine[k]-coarse[k])/(r-1) for k in fine}

R1 = rich(F[0.5],   F[0.25],   0.5,   0.25)    # Richardson from (0.5,0.25)
R2 = rich(F[0.25],  F[0.125],  0.25,  0.125)   # from (0.25,0.125)
R3 = rich(F[0.125], F[0.0625], 0.125, 0.0625)  # from (0.125,0.0625) -- the BEST

def bmax(a,b,k): return float(np.max(np.abs(a[k]-b[k])[box]))

print("\n=== Richardson self-consistency (box max) ===", flush=True)
for k in ('i_d','i_g','vZ'):
    print(f"  {k}: |R1-R3|={bmax(R1,R3,k):.3e}  |R2-R3|={bmax(R2,R3,k):.3e}  "
          f"|fine0.0625-R3|={bmax(F[0.0625],R3,k):.3e}", flush=True)

print("\n=== raw fine(0.0625) vs R3 (the extrapolated reference) -- the residual dt error in R3 ~ |R2-R3| ===", flush=True)
for k in ('i_d','i_g','vZ'):
    print(f"  {k}: fine0.0625 - R3 = {bmax(F[0.0625],R3,k):.3e}   (R3 removes this)", flush=True)
