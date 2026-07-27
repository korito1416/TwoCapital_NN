"""PHYSRISK economy map -- physical-risk green-Solow, differential capital
destruction (FD-3D lift; the MINIMAL PAIR with JONES).

Consumed by models_terminal_anchor/make_map_anchor.py (MAP MODULE INTERFACE):
exports MAP_NAME, PROVENANCE, fields(reg, lk, Z, Y, lr, l3, lx).

VALUE TRANSFORM: PHYSRISK deletes the utility damage (-delta logN) -- temperature
destroys capital in the DRIFTS instead -- so the economy's true value at
production states is exactly
    V(x) = logK + W_reg(Z, Y, s; lambda3, xi),        s = logR - logK.
Production nets learn the net-space object v = V + logN(Y; lambda3) (repo
convention, zoo_lift_common.v_from_V), so the fitted target is
    v(x) = logK + W_reg(Z, Y, s; lambda3, xi) + logN(Y; lambda3, regime).
W comes from the FD ladder (solvers/physrisk_ladder.sbatch -> outputs/physrisk/):
trilinear in (Z, Y, s), linear in lambda3 across the 5 post-damage slices, linear
in logxi across the 3 xi solves (0.05, 0.1, 148.4) -- zoo_lift_common
stack_interp_l3_lx, flat/clamped extrapolation (grid covers the box image
s in [-6, 2], so no clipping is needed for in-box inputs).

Policies (production rate conventions -- the economy's rates ARE production's):
    i_d, i_g = FD fields, same interpolation stack;
    i_r      = FD field in BOTH pre-tech regimes, floored at 1e-8
               (nontrivial in logK, logR, Z, Y through the e^{-s/2} W_s c FOC,
               production's exact functional form); None post-tech (no i_r net).

Distinct imprint vs JONES (same geometry class): i_d carries the high-(Y,Z)
de-investment region, W has growth-damage (drift-multiplied) Y-shaping instead
of utility-level shaping, and the self-decarbonizing Z-drift tilts W_Z.

FD source directory: $PHYSRISK_FD_DIR if set, else ../outputs/physrisk
(relative to this file). The full ladder must have been run; a coarse-validation
directory (outputs/physrisk_coarse, xi tag 'inf' absent there for 0.1/148.4)
does NOT satisfy the loader -- all 3 xi rungs x 12 solves are required.

Run this file directly for the smoke test (set PHYSRISK_FD_DIR to test against
a partial/coarse set is NOT supported by design -- honesty over convenience).
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "solvers")))
import zoo_lift_common as ZC                     # noqa: E402
import physrisk_callbacks as PC                  # noqa: E402

MAP_NAME = "PHYSRISK_physical_risk_green_solow_capital_destruction_fd3d"

XI_GRID = np.array(PC.XI_FAMILY)                 # (0.05, 0.1, 148.4)
LX_GRID = np.log(XI_GRID)
L3_GRID = np.array(PC.LAM3_GRID)

PROVENANCE = dict(
    economy="PHYSRISK -- physical-risk green-Solow: JONES's scale-free skeleton "
            "with the utility damage deleted and temperature destroying capital "
            "(Lambda_j(Y) = gamma_j D(Y), gamma_d=1.0 > gamma_g=0.25)",
    design="benchmarks/economy_zoo/design_physrisk.json (PORTFOLIO.json build 5; "
           "minimal pair with JONES)",
    solver="benchmarks/economy_zoo/solvers/physrisk_callbacks.py on the shared "
           "harness solvers/fd_reduced.py (PIBYS, step_exact semi-implicit jump "
           "sinks, robust drift feedback, exp-clip 35); ladder "
           "solvers/physrisk_ladder.sbatch (36 solves: 4 regimes x lambda3 x xi)",
    verification="coarse gates in outputs/physrisk_coarse/"
                 "physrisk_validation_PROVENANCE.json: V1 linear-jump limit, "
                 "V2 step_exact at J*dt=21.8 (hazard identical to JONES), "
                 "V3 xi=0.05 stability, FK level Monte Carlo",
    value_transform="v = logK + W_reg(Z,Y,s; lambda3, xi) + logN(Y; lambda3) "
                    "(repo net-space convention; PHYSRISK itself has NO utility "
                    "damage -- logN is added only to match the production net "
                    "convention v = V + logN)",
    interpolation="trilinear (Z,Y,s) x linear lambda3 x linear logxi "
                  "(zoo_lift_common.stack_interp_l3_lx, clamped extrapolation)",
    key_numbers=dict(gamma_d=PC.GAM_D, gamma_g=PC.GAM_G,
                     hazard_at_s0=float(np.exp(-4.364) * PC.K0 / PC.VARRHO),
                     E0=float(PC.ETA * 0.1303 * 0.3 * PC.K0),
                     s_grid=[PC.S_LO, PC.S_HI]),
)

_FD_DIR = os.environ.get(
    "PHYSRISK_FD_DIR",
    os.path.abspath(os.path.join(_HERE, "..", "outputs", "physrisk")))

_STACKS = None       # lazy: {regime: {field: interp(Z,Y,s,l3,lx)}}


def _build_stacks():
    """Load the 36 ladder npz files and build the 5-D interpolation stacks."""
    stacks = {}
    for reg in PC.REGIMES:
        pre_t = reg.endswith("PreTech")
        post_d = reg.startswith("PostDamage")
        ils = list(range(5)) if post_d else [None]
        fields = ("i_d", "i_g", "i_r", "W") if pre_t else ("i_d", "i_g", "W")
        per_xi = []
        grids = None
        for xi in XI_GRID:
            per_l3 = []
            for il in ils:
                fp = os.path.join(_FD_DIR, PC.solve_name(reg, il, xi) + ".npz")
                if not os.path.exists(fp):
                    raise FileNotFoundError(
                        "PHYSRISK map needs the FULL FD ladder output %s -- run "
                        "solvers/submit_physrisk_ladder.sh (all 3 xi rungs) "
                        "first, or point PHYSRISK_FD_DIR at a complete set." % fp)
                d = np.load(fp)
                grids = (np.array(d["Z"]), np.array(d["Y"]), np.array(d["S"]))
                per_l3.append({k: np.array(d[k]) for k in fields})
            per_xi.append(per_l3)
        g_l3 = L3_GRID if post_d else np.array([0.0])
        st = {}
        for k in fields:
            # value axes: (Z, Y, s, l3, lx)
            arr = np.stack(
                [np.stack([per_xi[ix][jl][k] for jl in range(len(ils))], axis=-1)
                 for ix in range(len(XI_GRID))], axis=-1)
            st[k] = ZC.stack_interp_l3_lx(grids[0], grids[1], grids[2],
                                          g_l3, LX_GRID, arr)
        stacks[reg] = st
    return stacks


def _stacks():
    global _STACKS
    if _STACKS is None:
        _STACKS = _build_stacks()
    return _STACKS


def fields(reg, lk, Z, Y, lr, l3, lx):
    """Production-state targets. Inputs (n,1) float arrays."""
    lk, Z, Y, lr, l3, lx = [np.asarray(a, float).ravel()
                            for a in (lk, Z, Y, lr, l3, lx)]
    n = lk.shape[0]
    st = _stacks()[reg]
    pre_t = reg.endswith("PreTech")
    post_d = reg.startswith("PostDamage")
    s = ZC.s_of(lk, lr)              # clamped by the interp stack ([-6,2] grid)
    l3q = l3 if post_d else np.zeros(n)
    W = st["W"](Z, Y, s, l3q, lx)
    V = lk + W                                        # true PHYSRISK value
    v = ZC.v_from_V(V, Y, l3, post_damage=post_d)     # net-space target
    out = dict(v=v.reshape(n, 1),
               i_d=st["i_d"](Z, Y, s, l3q, lx).reshape(n, 1),
               i_g=st["i_g"](Z, Y, s, l3q, lx).reshape(n, 1))
    if pre_t:
        out["i_r"] = np.maximum(st["i_r"](Z, Y, s, l3q, lx), 1e-8).reshape(n, 1)
    else:
        out["i_r"] = None
    return out


# ------------------------------------------------------------------ smoke test
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n = 4096
    ok = True
    for reg in ("PreDamagePreTech", "PreDamagePostTech",
                "PostDamagePreTech", "PostDamagePostTech"):
        ylo = 2.5 if reg.startswith("PostDamage") else 0.0
        lk = rng.uniform(4, 7, (n, 1))
        Z = rng.uniform(0.01, 0.99, (n, 1))
        Y = rng.uniform(ylo, 4, (n, 1))
        lr = rng.uniform(1, 6, (n, 1))
        l3 = rng.uniform(0, 1 / 3, (n, 1))
        lx = rng.uniform(np.log(0.05), np.log(148.4), (n, 1))
        F = fields(reg, lk, Z, Y, lr, l3, lx)
        for k in ("v", "i_d", "i_g"):
            assert F[k].shape == (n, 1) and np.all(np.isfinite(F[k])), (reg, k)
        if reg.endswith("PreTech"):
            assert F["i_r"] is not None and np.all(F["i_r"] >= 1e-8)
            # i_r must have genuine logR structure (the zoo's V_logR axis)
            F2 = fields(reg, lk, Z, Y, lr + 0.8, l3, lx)
            dev = float(np.max(np.abs(F2["i_r"] - F["i_r"])))
            print("  %s: i_r logR-sensitivity max|di_r|=%.2e (must be > 0)"
                  % (reg, dev), flush=True)
            ok &= dev > 1e-6
        else:
            assert F["i_r"] is None
        # post-tech regimes must be flat in logR (s enters only via pre-tech W)
        if reg.endswith("PostTech"):
            F2 = fields(reg, lk, Z, Y, lr + 1.7, l3, lx)
            dev = float(np.max(np.abs(F2["v"] - F["v"])))
            print("  %s: logR-flatness max|dv|=%.2e (must be ~0)" % (reg, dev),
                  flush=True)
            ok &= dev < 1e-8
        print("%-22s v[%8.3f, %8.3f]  i_d[%+.4f, %+.4f]  i_g[%.4f, %.4f]  i_r=%s"
              % (reg, F["v"].min(), F["v"].max(), F["i_d"].min(), F["i_d"].max(),
                 F["i_g"].min(), F["i_g"].max(),
                 "None" if F["i_r"] is None else
                 "[%.2e, %.4f]" % (F["i_r"].min(), F["i_r"].max())), flush=True)
    # value-transform spot check: v - logN - lk must equal the raw W stack
    one = np.ones((2, 1))
    F = fields("PreDamagePreTech", 6.78 * one, 0.7 * one, 1.1 * one,
               2.416 * one, 0.0 * one, np.log(0.1) * one)
    W = _stacks()["PreDamagePreTech"]["W"](
        np.array([0.7]), np.array([1.1]), np.array([2.416 - 6.78]),
        np.array([0.0]), np.array([np.log(0.1)]))
    manual = 6.78 + float(W[0]) + float(ZC.logN(1.1, 0.0, False))
    ok &= abs(F["v"][0, 0] - manual) < 1e-10
    print("v transform spot check: map %.6f manual %.6f" % (F["v"][0, 0], manual),
          flush=True)
    print("SMOKE TEST:", "PASS" if ok else "FAIL", flush=True)
