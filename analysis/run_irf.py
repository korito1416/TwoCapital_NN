"""Run the CRN stochastic IRF on the coupled worst-case simulator for several xi; save one npz."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_jump_sim_vec import VecRobustJumpSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-folder", required=True)
    ap.add_argument("--xis", default="0.05,0.1,148.6")
    ap.add_argument("--n-paths", type=int, default=512)
    ap.add_argument("--years", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=1.0 / 12.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--y0", type=float, default=1.2)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xis = [float(x) for x in a.xis.split(",")]
    sim = VecRobustJumpSim(a.export_folder, xis[0], batch_size=a.n_paths * 5)
    out = {"xis": np.array(xis)}
    for xi in xis:
        sim.xi = np.float32(xi); sim.lx = np.float32(np.log(xi))
        t0 = time.time()
        d = sim.simulate_irf(a.n_paths, a.years, a.dt, a.seed, a.y0, robust=not a.reference)
        key = f"{xi:g}"
        out[f"{key}::t"] = d["t"]
        for k, v in d.items():
            if k.startswith("irf_") or k.startswith("base_"):
                out[f"{key}::{k}"] = v
        out["shocks"] = d["shocks"]
        # quick sanity: baseline post-tech share end + a couple of IRF end-values
        print(f"xi={xi:g} in {time.time()-t0:.0f}s  base_posttech_end={d['base_posttech'][-1]:.3f}  "
              f"IRF Temp->Y end={d['irf_Temperature_Y'][-1]:+.4f}  Knowledge->A_g end={d['irf_Knowledge_A_g'][-1]:+.5f}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
