"""Merge per-xi robust_paths npz (from the parallel sbatch jobs) into the two combined npz the
plotter/report expect: paths_robust.npz and paths_reference.npz (all xi keys)."""
import argparse, glob
from pathlib import Path
import numpy as np


def merge(pattern, out):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    merged = {}
    xis = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        xis.extend([float(x) for x in d["xis"]])
        for k in d.files:
            if k in ("xis", "reference"):
                continue
            merged[k] = d[k]
        merged["reference"] = d["reference"]
    merged["xis"] = np.array(sorted(set(xis)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)
    print(f"merged {len(files)} files -> {out}  (xis={merged['xis']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="dir containing xi_*/paths_robust.npz etc.")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    merge(f"{a.base_dir}/xi_*/paths_robust.npz", f"{a.out_dir}/paths_robust.npz")
    merge(f"{a.base_dir}/xi_*/paths_reference.npz", f"{a.out_dir}/paths_reference.npz")


if __name__ == "__main__":
    main()
