"""
Sensitivity of the optimal dirty-investment sign (invest i_d>0 vs de-invest i_d<0) at the reference
point (logK=6.78, Z=0.7, Y=3.0) to the two climate-damage knobs, using the PIBYS solver fd_pdpt_v5:
  Sweep 1: Y_max  -- the temperature-damage HORIZON (top of the Y domain; original model = 4.0).
           y_cap = y_max, so damage is counted up to the domain top and saturates beyond (model-faithful).
  Sweep 2: lambda3 -- the HIGH-TEMPERATURE damage curvature/penalty ((logN)_Y = l1+l2 Y+lambda3 (Y-2.5)),
           model values {0, 1/12, 1/6, 1/4, 1/3}; run at the original domain Y_max=4 AND extended Y_max=16.
Coarse-ish grid (we want the trend + the invest->de-invest threshold, not 4-digit precision).
Saves outputs/fd_sensitivity.png and prints the crossing thresholds.
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fd_pdpt_v5 as FD

REF = (np.log(880), 0.7, 3.0)
DYT = 0.20                 # target dY (keeps [0,4] resolution ~ fixed as the domain grows)
OD = FD.OD
SOLVE = dict(nK=15, nZ=25, dt=3.5, T=1050.0, howard_max=28, relax=0.25, warm=True, verbose=False)


def read_ref(out, key):
    lk, Z, Y = out["logK"], out["Z"], out["Y"]
    pt = np.array([[min(max(REF[0], lk[0]), lk[-1]), REF[1], REF[2]]])
    return float(RGI((lk, Z, Y), out[key], bounds_error=False, fill_value=None)(pt)[0])


def solve_report(lam3, y_max, label):
    nY = int(round(y_max / DYT)) + 1
    out = FD.solve(lam3=lam3, nY=nY, y_max=y_max, y_cap=y_max, **SOLVE)
    r = {k: read_ref(out, k) for k in ("i_d", "i_g", "vlK", "vZ", "c")}
    r["maxR"] = out["max_abs_residual"]; r["t"] = out["time"]
    sign = "DE-INVEST" if r["i_d"] < 0 else "invest"
    print(f"  {label:24s} i_d={r['i_d']:+.4f} ({sign:9s}) i_g={r['i_g']:+.4f} vlK={r['vlK']:.3f} "
          f"vZ={r['vZ']:.3f} c={r['c']:.4f}  (maxR={r['maxR']:.1e}, {r['t']:.0f}s)", flush=True)
    return r


def crossing(xs, ys):
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:
            return xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    return None


def main():
    print("=" * 98)
    print("SENSITIVITY of i_d at ref (logK=6.78, Z=0.7, Y=3.0) -- PIBYS, y_cap=y_max (truncated damage horizon)")
    print("=" * 98)

    print("\n--- Sweep 1: Y_max (temperature-damage horizon),  lambda3 = 1/6 ---")
    ymaxs = [4, 5, 6, 8, 10, 12, 16, 20]
    s1 = [solve_report(1 / 6.0, ym, f"Y_max={ym}") for ym in ymaxs]
    id1 = [r["i_d"] for r in s1]
    thr_ym = crossing(ymaxs, id1)
    print(f"  => invest -> de-invest threshold:  Y_max* ~ {thr_ym:.1f}" if thr_ym
          else "  => no sign change across the Y_max range")

    lam3s = [0.0, 1 / 12.0, 1 / 6.0, 1 / 4.0, 1 / 3.0]
    print("\n--- Sweep 2a: lambda3 (high-T penalty),  Y_max = 4  (ORIGINAL model domain) ---")
    s2a = [solve_report(l, 4.0, f"lambda3={l:.3f}") for l in lam3s]
    id2a = [r["i_d"] for r in s2a]; thr_l4 = crossing(lam3s, id2a)
    print(f"  => threshold lambda3* ~ {thr_l4:.3f}" if thr_l4
          else "  => NO crossing: within Y_max=4, i_d keeps the same sign for ALL lambda3 (horizon dominates)")

    print("\n--- Sweep 2b: lambda3 (high-T penalty),  Y_max = 16  (EXTENDED domain) ---")
    s2b = [solve_report(l, 16.0, f"lambda3={l:.3f}") for l in lam3s]
    id2b = [r["i_d"] for r in s2b]; thr_l16 = crossing(lam3s, id2b)
    print(f"  => threshold lambda3* ~ {thr_l16:.3f}" if thr_l16 else "  => no crossing across lambda3 at Y_max=16")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax[0].plot(ymaxs, id1, "b-o", ms=5)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].axvline(4, color="g", ls="--", lw=1.2, label="original $Y_{max}=4$")
    if thr_ym: ax[0].axvline(thr_ym, color="r", ls=":", lw=1.4, label=f"threshold $\\approx{thr_ym:.1f}$")
    ax[0].fill_between(ymaxs, 0, np.maximum(id1, 0), color="orange", alpha=0.12)
    ax[0].fill_between(ymaxs, np.minimum(id1, 0), 0, color="green", alpha=0.12)
    ax[0].set_xlabel("$Y_{max}$  (temperature-damage horizon, $^\\circ$C)"); ax[0].set_ylabel("$i_d$ at reference")
    ax[0].set_title("Sweep 1: $i_d$ vs $Y_{max}$   ($\\lambda_3=1/6$)"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(lam3s, id2a, "b-o", ms=5, label="$Y_{max}=4$ (orig. model)")
    ax[1].plot(lam3s, id2b, "m-s", ms=5, label="$Y_{max}=16$ (extended)")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].axvline(1 / 6.0, color="g", ls="--", lw=1.2, label="calibrated $\\lambda_3=1/6$")
    ax[1].set_xlabel("$\\lambda_3$  (high-temperature damage penalty)"); ax[1].set_ylabel("$i_d$ at reference")
    ax[1].set_title("Sweep 2: $i_d$ vs $\\lambda_3$"); ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle("Post-damage post-tech: sensitivity of the dirty-investment sign to the temperature-damage "
                 "horizon $Y_{max}$ and curvature $\\lambda_3$  (PIBYS solver)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OD, "fd_sensitivity.png"); fig.savefig(p, dpi=150); print("\nsaved", p, flush=True)
    np.savez(os.path.join(OD, "fd_sensitivity_data.npz"),
             ymaxs=ymaxs, id_ymax=id1, lam3s=lam3s, id_lam3_y4=id2a, id_lam3_y16=id2b)


if __name__ == "__main__":
    main()
