"""Time one simulate_v eval and one stable-solve across grids to calibrate the fine-grid run."""
import time, numpy as np, sys
sys.path.insert(0, '.')
import fd_pdpt_v5 as FD
import fd_pdpt_v5_stable as ST

P = FD.P
for (nK, nZ, nY) in [(31, 61, 31), (41, 81, 41), (61, 121, 61)]:
    logK = np.linspace(4.0, 7.0, nK)
    Z = np.linspace(0.02, 0.98, nZ)
    Y = np.linspace(0.0, 4.0, nY)
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    i_d = np.zeros_like(LK); i_g = np.full_like(LK, 0.05)
    t0 = time.time()
    v = FD.simulate_v(logK, Z, Y, i_d, i_g, 1/6.0, T=1200.0, dt=2.5)
    dt_eval = time.time() - t0
    print(f"{nK}x{nZ}x{nY}: one simulate_v(T=1200,dt=2.5) = {dt_eval:.1f}s", flush=True)
