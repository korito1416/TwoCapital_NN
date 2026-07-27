"""
What drives the strong GREEN effect (large i_g, Z->1 transition) in the post-damage post-tech regime,
and WHICH PARAMETER matters most? Green and dirty share identical adjustment-cost tech (alpha, Gamma,
theta, sigma); they differ only in (i) productivity A_g'' vs A_d and (ii) emissions eta (dirty only).

Decomposition (the clean answer to "what brings green's effect"):
  - eta = 0           : kill the climate/emissions channel -> if green stays strong, it's PRODUCTIVITY.
  - A_g'' = A_d       : kill the productivity gap          -> if green collapses to dirty, it's PRODUCTIVITY.
Then rank parameters by how much they move green investment i_g (and the transition speed a_Z) at the ref.

Uses the PIBYS solver; overrides parameters by mutating fd_pdpt_v5.P in place. numpy only.
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fd_pdpt_v5 as FD

REF = (np.log(880), 0.7, 3.0)
OD = FD.OD
BASE = dict(FD.P)
SOLVE = dict(nK=15, nZ=25, nY=21, y_max=4.0, y_cap=4.0, T=900.0, dt=4.0, howard_max=26, relax=0.25, warm=True, verbose=False)


def readout(out):
    pt = np.array([[REF[0], REF[1], REF[2]]])
    g = lambda k: float(RGI((out["logK"], out["Z"], out["Y"]), out[k], bounds_error=False, fill_value=None)(pt)[0])
    i_d, i_g = g("i_d"), g("i_g"); Z = REF[1]; p = FD.P
    phid = p["a_d"] + p["G_d"] * np.log(max(1 + p["t_d"] * i_d, 1e-9))
    phig = p["a_g"] + p["G_g"] * np.log(max(1 + p["t_g"] * i_g, 1e-9))
    a_Z = Z * (1 - Z) * (phig - phid)                       # transition speed (dZ/dt), sigma^2 terms negligible
    return dict(i_d=i_d, i_g=i_g, gap=i_g - i_d, a_Z=a_Z, vZ=g("vZ"))


def solve_over(ov, lam3=1 / 6.0):
    FD.P.clear(); FD.P.update(BASE); FD.P.update(ov)
    out = FD.solve(lam3=lam3, **SOLVE)
    r = readout(out)
    FD.P.clear(); FD.P.update(BASE)
    return r


def main():
    print("=" * 92)
    print("GREEN drivers at ref (logK=6.78, Z=0.7, Y=3.0).  base: A_d=0.1303, A_g''=0.1567, eta=0.291, lam3=1/6")
    print("=" * 92)
    base = solve_over({})
    print(f"\nBASELINE:  i_g={base['i_g']:+.4f}  i_d={base['i_d']:+.4f}  gap(i_g-i_d)={base['gap']:+.4f}  "
          f"a_Z(transition)={base['a_Z']:+.4f}  v_Z={base['vZ']:.3f}")

    print("\n--- DECOMPOSITION: turn off one channel at a time ---")
    no_clim = solve_over({"eta": 0.0})
    no_prod = solve_over({"A_gpp": BASE["A_d"]})
    print(f"  eta=0 (no emissions/climate): i_g={no_clim['i_g']:+.4f} gap={no_clim['gap']:+.4f} a_Z={no_clim['a_Z']:+.4f}"
          f"   -> green effect {'SURVIVES (=> productivity)' if no_clim['gap']>0.5*base['gap'] else 'collapses (=> climate)'}")
    print(f"  A_g''=A_d (no prod. gap):     i_g={no_prod['i_g']:+.4f} gap={no_prod['gap']:+.4f} a_Z={no_prod['a_Z']:+.4f}"
          f"   -> green effect {'COLLAPSES (=> productivity)' if no_prod['gap']<0.5*base['gap'] else 'survives (=> climate)'}")

    print("\n--- PARAMETER SWEEPS (effect on green investment i_g and transition a_Z) ---")
    sweeps = {
        "A_g'' (green prod.)": ("A_gpp", [0.1303, 0.1380, 0.1470, 0.1567, 0.1680, 0.1800], None),
        "A_d (dirty prod.)":   ("A_d",   [0.1100, 0.1303, 0.1450, 0.1600], None),
        "eta (emissions)":     ("eta",   [0.0, 0.10, 0.291, 0.45, 0.60], None),
        "lambda3 (damage)":    ("lam3",  [0.0, 1 / 12, 1 / 6, 1 / 4, 1 / 3], "lam3"),
    }
    results = {}
    for name, (key, vals, is_lam) in sweeps.items():
        row = []
        for v in vals:
            r = solve_over({}, lam3=v) if is_lam else solve_over({key: v})
            row.append((v, r["i_g"], r["i_d"], r["a_Z"]))
            print(f"  {name:20s} {key}={v:.4f}: i_g={r['i_g']:+.4f} i_d={r['i_d']:+.4f} a_Z={r['a_Z']:+.4f}", flush=True)
        results[name] = row

    print("\n--- RANKING: which parameter moves green investment i_g the most over its swept range ---")
    rng = {name: (max(r[1] for r in row) - min(r[1] for r in row)) for name, row in results.items()}
    for name in sorted(rng, key=lambda n: -rng[n]):
        print(f"  {name:22s} range of i_g = {rng[name]:.4f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for name, row in results.items():
        xs = [r[0] / row[len(row)//2][0] for r in row]      # normalize x to its base value (~1.0 at base)
        ax[0].plot(xs, [r[1] for r in row], "-o", ms=4, label=name)
    ax[0].axhline(base["i_g"], color="k", ls="--", lw=0.8, alpha=0.5, label="baseline $i_g$")
    ax[0].set_xlabel("parameter / its baseline value"); ax[0].set_ylabel("green investment $i_g$ at ref")
    ax[0].set_title("Sensitivity of green investment $i_g$"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    names = list(rng.keys()); order = sorted(range(len(names)), key=lambda i: rng[names[i]])
    ax[1].barh([names[i] for i in order], [rng[names[i]] for i in order], color="seagreen")
    ax[1].set_xlabel("range of $i_g$ over swept range  (bigger = more influential)")
    ax[1].set_title("Which parameter drives green most?")
    ax[1].grid(alpha=0.3, axis="x")
    fig.suptitle("Drivers of the green transition (post-damage post-tech, PIBYS)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OD, "fd_green_drivers.png"); fig.savefig(p, dpi=150); print("\nsaved", p, flush=True)


if __name__ == "__main__":
    main()
