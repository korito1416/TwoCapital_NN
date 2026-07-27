"""Run the vectorised jump simulator for several xi under BOTH measures (worst-case + reference)
and save path time-series (mean + 10/90 bands per quantity) for the Haoyang-style xi-overlay figures.

One trained network covers every xi (log-xi is a network INPUT), so we load once and just reset
sim.xi / sim.lx between runs -- no reloading per xi/measure."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_jump_sim_vec import VecRobustJumpSim


def run_one(sim, xi, args, robust):
    sim.xi = np.float32(xi); sim.lx = np.float32(np.log(xi))
    t0 = time.time()
    tables, rf, series = sim.simulate(args.n_paths, args.years, args.dt, args.seed, args.y0, robust=robust)
    print(f"  {'ROBUST' if robust else 'REF   '} xi={xi:g} in {time.time()-t0:.0f}s  "
          f"post-tech share={rf[2]+rf[3]:.3f}", flush=True)
    return tables, rf, series


def collect(sim, xis, args, robust, out):
    for xi in xis:
        tables, rf, series = run_one(sim, xi, args, robust)
        key = f"{xi:g}"
        for nm, arr in series.items():
            out[f"{key}::{nm}"] = arr
        out[f"{key}::regime_frac"] = rf
        for tk, tv in tables.items():                 # Lars's pooled tables (allocation + marginal values)
            out[f"{key}::pooled::{tk}"] = np.array(tv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-folder", required=True)
    ap.add_argument("--xis", default="0.05,0.1,148.6")
    ap.add_argument("--n-paths", type=int, default=1024)
    ap.add_argument("--years", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=1.0 / 12.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--y0", type=float, default=1.2)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    xis = [float(x) for x in args.xis.split(",")]

    sim = VecRobustJumpSim(args.export_folder, xis[0], batch_size=args.n_paths)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    for robust, tag in [(True, "robust"), (False, "reference")]:
        print(f"=== {tag} ===", flush=True)
        out = {"xis": np.array(xis), "reference": np.array([0 if robust else 1])}
        collect(sim, xis, args, robust, out)
        path = outdir / f"paths_{tag}.npz"
        np.savez_compressed(path, **out)
        print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
