"""Does the CONVERGED Howard POLICY depend on the dt used during the sweeps?

If we can run the (expensive) Howard fixed-point loop at a CHEAP dt and then do dt-Richardson only
at the FINAL frozen-policy value readout, the fine-grid reference becomes affordable. Test: run the
stable Howard loop at dt=2.5 vs dt=0.5 on the base grid; compare the converged control fields. If
they agree to << 1e-4 (box max), the policy is dt-robust and the cheap-loop + fine-readout design
is valid."""
import numpy as np, sys
sys.path.insert(0, '.')
import fd_pdpt_v5_stable as ST
from stable_fd_eval import _box_mask

outs = {}
for dt in (2.5, 0.5):
    o = ST.solve_stable(nK=31, nZ=61, nY=31, T=1500.0, dt=dt, howard_max=120,
                        tol_pocket=2e-7, relax=0.3, tail_avg=6, warm=True, verbose=False)
    outs[dt] = o
    print(f"  dt={dt}: iters={o['iters']} gate={o['gate_int']:.1e} time={o['time']:.0f}s", flush=True)

box = _box_mask(outs[2.5]['logK'], outs[2.5]['Z'], outs[2.5]['Y'])
for k in ('i_d', 'i_g', 'vZ'):
    dif = float(np.max(np.abs(outs[2.5][k] - outs[0.5][k])[box]))
    print(f"  converged-policy |dt2.5 - dt0.5| {k}: {dif:.3e}", flush=True)
