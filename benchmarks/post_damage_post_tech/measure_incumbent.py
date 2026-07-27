"""
measure_incumbent.py -- Phase 1 of the leaderboard re-test.

Trains the INCUMBENT (baseline-autodiff-ctrlfit = egm_howard_ctrlfit_ab.run_egm_howard, ctrlfit=True)
multi-seed and grades EVERY seed with the SHARED stable-FD eval module (stable_fd_eval.grade), so the
incumbent is on the identical leaderboard as the other 5 arms. NEVER uses the old under-converged npz.
"""
import os, sys, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import egm_howard_ctrlfit_ab as EGM
from stable_fd_eval import grade, load_stable_fd, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--nK", type=int, default=21)
    ap.add_argument("--nZ", type=int, default=31)
    ap.add_argument("--nY", type=int, default=21)
    ap.add_argument("--howard", type=int, default=12)
    ap.add_argument("--fit", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=1500)
    args = ap.parse_args()

    d = load_stable_fd()
    print(f"=== INCUMBENT (baseline-autodiff-ctrlfit) vs STABLE FD ===", flush=True)
    print(f"grid {args.nK}x{args.nZ}x{args.nY}, {args.howard} Howard sweeps, seeds {args.seeds}", flush=True)
    print(f"FD de-invest targets: min i_d={np.min(d['i_d'][(np.meshgrid(d['logK'],d['Z'],d['Y'],indexing='ij')[1]>=0.9)&(np.meshgrid(d['logK'],d['Z'],d['Y'],indexing='ij')[2]>=3.5)]):+.5f}", flush=True)

    per_seed = []
    for seed in args.seeds:
        t0 = time.time()
        out = EGM.run_egm_howard(seed, nK=args.nK, nZ=args.nZ, nY=args.nY,
                                 n_howard=args.howard, fit_steps=args.fit,
                                 warm_steps=args.warm, ctrlfit=True, verbose=False)
        m = grade(out, d)
        per_seed.append(m)
        print(f"  [seed {seed}] {time.time()-t0:.0f}s  "
              f"box_i_d={m['box_err_i_d']:.3e} box_i_g={m['box_err_i_g']:.3e} | "
              f"di_i_d={m['di_err_i_d']:.3e} di_vZ={m['di_err_vZ']:.3e} "
              f"di_min_i_d={m['di_min_i_d']:+.4f} (FD {m['di_FD_min_i_d']:+.4f}) "
              f"depth_gap={m['di_match_depth']:.3e}", flush=True)

    agg = summarize(per_seed, label="INCUMBENT baseline-autodiff-ctrlfit")
    np.savez(os.path.join(HERE, "outputs", "leaderboard_incumbent.npz"),
             **{f"{k}_{stat}": v for k, sv in agg.items() for stat, v in sv.items()})
    print("\nsaved -> outputs/leaderboard_incumbent.npz", flush=True)


if __name__ == "__main__":
    main()
