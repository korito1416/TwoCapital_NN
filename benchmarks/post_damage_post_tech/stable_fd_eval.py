"""
stable_fd_eval.py -- THE SINGLE SHARED, STABLE-FD GRADING MODULE for the de-invest-costate leaderboard.

Every arm of the 6-method leaderboard (baseline-autodiff-ctrlfit, fosls-flux-primary, siren,
vanishing-viscosity-corrected, qd-direct, inoue-upwind-derivative) imports `grade()` from HERE and
is graded IDENTICALLY against the ONE stabilized FD ground truth:

    outputs/fd_pdpt_v5_stable_lam3_0167_xi148.npz   (tol 1e-6, howard_max=200, tail-averaged,
                                                      gate ~1e-6, reproducible ~1e-8)

The OLD fd_pdpt_v5_lam3_0167_xi148.npz was UNDER-CONVERGED and is NEVER used here.

UNIFIED METRICS (vs the stable FD):
  (1) BOX error over the economically-relevant interior box (drop boundary layers / corners):
        box_err_i_d = max|i_d - FD|,  box_err_i_g = max|i_g - FD|.
  (2) DE-INVEST-REGION error over the GENUINE de-invest region (Z>=0.9, Y>=3.5, where the
      stable FD i_d goes negative; the real signal is MILD, global min ~ -0.0248):
        di_err_i_d = max|i_d - FD|,  di_mean_i_d = mean(i_d) (NN's own depth in the region),
        di_err_vZ  = max|vZ - FD|    (costate accuracy where it matters),
        di_FD_mean_i_d, di_FD_min_i_d (the FD targets), and
        di_match_depth = |NN min i_d - FD min i_d| (does it match the mild ~-0.025 depth?).

GRADING IS ALWAYS ON CONTROLS/COSTATE vs the stable FD -- NEVER on v or loss (the FD v is
self-inconsistent up to the gate tolerance, so v is not a valid leaderboard target).

USAGE:
    from stable_fd_eval import grade, load_stable_fd, STABLE_NPZ
    metrics = grade(out)   # out = dict with logK,Z,Y (1-D axes) + i_d,i_g,vZ (3-D on that grid)
                           # arm's grid may be coarser than FD: fields are interpolated to FD grid.

The interior box and de-invest region are defined ONCE here so all arms are apples-to-apples.
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

HERE = os.path.dirname(os.path.abspath(__file__))
STABLE_NPZ = os.path.join(HERE, "outputs", "fd_pdpt_v5_stable_lam3_0167_xi148.npz")

# ---- the ONE interior box (apples-to-apples; drops boundary layers and the Z->1/logK-top corner
#      feasibility-floor toggling region that dominates a raw sup-norm) ----
BOX_LK = (4.3, 6.7)
BOX_Z  = (0.30, 0.95)
BOX_Y  = (0.5, 4.0)

# ---- the GENUINE de-invest region (where the stable FD i_d<0; mild, high-Z corner) ----
DEINVEST_Z_MIN = 0.90
DEINVEST_Y_MIN = 3.5


def load_stable_fd():
    """Load the stable FD ground truth as a plain dict of numpy arrays."""
    d = np.load(STABLE_NPZ)
    return {k: d[k] for k in d.files}


def _box_mask(logK, Z, Y, lk=BOX_LK, z=BOX_Z, y=BOX_Y):
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    return ((LK >= lk[0]) & (LK <= lk[1]) &
            (ZZ >= z[0]) & (ZZ <= z[1]) &
            (YY >= y[0]) & (YY <= y[1]))


def _deinvest_mask(logK, Z, Y):
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    return (ZZ >= DEINVEST_Z_MIN) & (YY >= DEINVEST_Y_MIN)


def _interp_to_fd(out, key, d):
    """Interpolate an arm's field (on its own grid) to the FD grid. If the arm is already on the
    FD grid this is the identity (RGI is exact at nodes)."""
    a = np.asarray(out[key], dtype=float)
    if (len(out["logK"]) == len(d["logK"]) and len(out["Z"]) == len(d["Z"])
            and len(out["Y"]) == len(d["Y"])
            and np.allclose(out["logK"], d["logK"]) and np.allclose(out["Z"], d["Z"])
            and np.allclose(out["Y"], d["Y"])):
        return a
    f = RGI((np.asarray(out["logK"], float), np.asarray(out["Z"], float),
             np.asarray(out["Y"], float)), a, bounds_error=False, fill_value=None)
    LK, ZZ, YY = np.meshgrid(d["logK"], d["Z"], d["Y"], indexing="ij")
    q = np.stack([LK.ravel(), ZZ.ravel(), YY.ravel()], axis=1)
    return f(q).reshape(LK.shape)


def grade(out, d=None):
    """Grade one arm's output against the stable FD. `out` must contain 1-D axes logK,Z,Y and the
    3-D fields i_d, i_g, vZ on that grid. Returns the UNIFIED metrics dict.

    All fields are interpolated to the FD grid; masks are defined on the FD grid so every arm is
    measured on the identical set of points.
    """
    if d is None:
        d = load_stable_fd()
    logK, Z, Y = d["logK"], d["Z"], d["Y"]
    box = _box_mask(logK, Z, Y)
    dim = _deinvest_mask(logK, Z, Y)

    i_d = _interp_to_fd(out, "i_d", d)
    i_g = _interp_to_fd(out, "i_g", d)
    vZ  = _interp_to_fd(out, "vZ", d)

    m = {}
    # (1) BOX error
    m["box_err_i_d"] = float(np.max(np.abs(i_d - d["i_d"])[box]))
    m["box_err_i_g"] = float(np.max(np.abs(i_g - d["i_g"])[box]))
    # (2) DE-INVEST-REGION error (Z>=0.9, Y>=3.5)
    m["di_err_i_d"]  = float(np.max(np.abs(i_d - d["i_d"])[dim]))
    m["di_mean_i_d"] = float(np.mean(i_d[dim]))
    m["di_min_i_d"]  = float(np.min(i_d[dim]))
    m["di_err_vZ"]   = float(np.max(np.abs(vZ - d["vZ"])[dim]))
    # FD targets in the region (so the table is self-documenting)
    m["di_FD_mean_i_d"] = float(np.mean(d["i_d"][dim]))
    m["di_FD_min_i_d"]  = float(np.min(d["i_d"][dim]))
    m["di_FD_frac_neg"] = float(np.mean(d["i_d"][dim] < 0))
    # (3) does it match the mild ~-0.025 de-invest DEPTH?
    m["di_match_depth"] = float(abs(m["di_min_i_d"] - m["di_FD_min_i_d"]))
    m["di_frac_neg"]    = float(np.mean(i_d[dim] < 0))
    return m


def summarize(per_seed, label=""):
    """Given a list of per-seed metric dicts, print median [min,max] for each metric and return
    the aggregated dict (for the leaderboard). Multi-seed (>=3) is the report convention."""
    keys = list(per_seed[0].keys())
    agg = {}
    print(f"=== {label} SUMMARY (median [min, max] over {len(per_seed)} seeds) ===", flush=True)
    for k in keys:
        a = np.array([s[k] for s in per_seed], dtype=float)
        agg[k] = dict(median=float(np.median(a)), min=float(a.min()), max=float(a.max()))
        print(f"    {k:18s}: median={np.median(a):+.4e}  [{a.min():+.4e}, {a.max():+.4e}]", flush=True)
    return agg


if __name__ == "__main__":
    # self-test: grade the stable FD against ITSELF -> all errors must be ~0 (consistency check).
    d = load_stable_fd()
    self_out = {k: d[k] for k in ("logK", "Z", "Y", "i_d", "i_g", "vZ")}
    m = grade(self_out, d)
    print("SELF-TEST (FD vs FD -- errors must be 0):")
    for k, v in m.items():
        print(f"  {k:18s} = {v:+.6e}")
    assert m["box_err_i_d"] < 1e-12 and m["di_err_vZ"] < 1e-12, "self-test failed!"
    print("\nSELF-TEST PASSED. FD targets in de-invest region:")
    print(f"  FD min i_d = {m['di_FD_min_i_d']:+.5f}  (mild ~-0.025 expected)")
    print(f"  FD mean i_d = {m['di_FD_mean_i_d']:+.5f}, frac neg = {m['di_FD_frac_neg']:.3f}")
