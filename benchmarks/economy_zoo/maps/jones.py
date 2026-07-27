"""JONES economy warm-start map (economy zoo, FD-3D flagship).

Exports MAP_NAME, PROVENANCE, fields(reg, lk, Z, Y, lr, l3, lx) per the
models_terminal_anchor/make_map_anchor.py interface. Consumes the FULL JONES
FD ladder written by solvers/jones_callbacks.py (via jones_ladder.sbatch /
submit_jones_ladder.sh) into solvers/outputs/jones/: 4 regimes x lambda3
slices x xi in {148.4, 0.1, 0.05}. FAILS LOUDLY at import if any ladder npz
is missing.

VALUE OBJECT
------------
The FD ladder solves the reduced NET-SPACE value W:  v = V + logN(Y;lambda3)
= logK + W(Z, Y, s), s = logR - logK (exact reduction, sympy-verified in the
design; the two scale-free surgical changes make every HJB coefficient a
function of (Z, Y, s) only). Therefore

    fields()['v'] = logK + W_reg(Z, Y, s; lambda3, logxi)

IS ALREADY the repo net-space object make_map_anchor.py fits v_nn to -- the
per-regime +/-logN transform is inside W by construction (the solver flow
carries the v5 form -(lNy*a_Y + lNyy*b_Y) with the regime-correct slope), so
NO logN is added here.

Interpolation stack (zoo_lift_common.LinearInterpND): trilinear in (Z, Y, s)
x linear across the 5 lambda3 slices (post-damage regimes; size-1 axis for
pre-damage) x linear in logxi across the 3 xi solves. Post-tech regimes are
s-independent (V_logK = 1 exactly): W enters as a constant s-axis and the
policies depend on (Z, Y) only -- matching the production net layout (no i_r
net post-tech). s = lr - lk needs NO clip: the grid [-6, 2] covers the full
production box image (logK in [4,7] x logR in [1,6]).

Policies: i_d, i_g are the solved FD fields (feasibility 1 + theta*i > 0 and
head cap 0.95 enforced); i_r (pre-tech only) is the FD field of the exact
production FOC  i_r = [psi0*0.5*e^{-s/2}*W_s*(C/K)/delta]^2  -- nontrivial in
logK, logR, Z AND Y through (s, W_s, C/K); clipped to [1e-8, 0.95] (the
make_map_anchor i_r net fits -log(i_r)).

HOMOGENEITY ASSERT: the lift checks V_logK = 1 - W_s at every requested point
(the unit-slope member of the verified FD solution family, vs the trained
attractor's ~0.37) and RAISES if any point leaves (0.5, 1.5); percentile
stats are recorded in fields.last_guard_stats.
"""
import os
import numpy as np

import zoo_lift_common as ZL

_HERE = os.path.dirname(os.path.abspath(__file__))
LADDER_DIR = os.environ.get(
    "JONES_LADDER_DIR",
    os.path.abspath(os.path.join(_HERE, "..", "solvers", "outputs", "jones")))

NL = 5
LAM3_GRID = np.array([0.0, 1.0 / 12, 1.0 / 6, 1.0 / 4, 1.0 / 3])
XI_ASC = [0.05, 0.1, 148.4]                  # ascending for the logxi axis
LX_GRID = np.log(np.array(XI_ASC))
REGIMES = ("PreDamagePreTech", "PreDamagePostTech",
           "PostDamagePreTech", "PostDamagePostTech")

MAP_NAME = "jones_fd3d_v1"

_XT = {0.05: "xi0p05", 0.1: "xi0p1", 148.4: "xi148p4"}


def _fname(regime, l3idx, xi):
    lt = f"l3{l3idx}" if l3idx is not None else "l3x"
    return f"jones_{regime}_{lt}_{_XT[xi]}.npz"


def _expected_files():
    out = []
    for reg in REGIMES:
        l3s = range(NL) if reg.startswith("PostDamage") else [None]
        for k in l3s:
            for xi in XI_ASC:
                out.append((reg, k, xi, os.path.join(LADDER_DIR,
                                                     _fname(reg, k, xi))))
    return out


# ---------------------------------------------------------------- load (loud)
_missing = [p for (_r, _k, _x, p) in _expected_files() if not os.path.exists(p)]
if _missing:
    raise FileNotFoundError(
        "JONES ladder outputs missing (%d of %d): the map cannot be built.\n"
        "First missing: %s\nRun benchmarks/economy_zoo/solvers/"
        "submit_jones_ladder.sh (dependency-ordered Slurm DAG) and wait for "
        "completion, or point JONES_LADDER_DIR at a complete ladder directory."
        % (len(_missing), len(_expected_files()), _missing[0]))


def _build_stacks():
    stacks = {}
    for reg in REGIMES:
        pretech = reg.endswith("PreTech")
        postdam = reg.startswith("PostDamage")
        l3s = list(range(NL)) if postdam else [None]
        l3ax = LAM3_GRID if postdam else np.array([0.0])
        first = np.load(os.path.join(LADDER_DIR, _fname(reg, l3s[0], XI_ASC[0])))
        Zg, Yg, Sg = first["Z"], first["Y"], first["S"]
        names = ["W", "i_d", "i_g"] + (["i_r", "W_S"] if pretech else [])
        cube = {n: np.empty(Zg.size * Yg.size * Sg.size * len(l3s) * len(XI_ASC))
                .reshape(Zg.size, Yg.size, Sg.size, len(l3s), len(XI_ASC))
                for n in names}
        for a, k in enumerate(l3s):
            for b, xi in enumerate(XI_ASC):
                d = np.load(os.path.join(LADDER_DIR, _fname(reg, k, xi)))
                assert (np.array_equal(d["Z"], Zg) and np.array_equal(d["Y"], Yg)
                        and np.array_equal(d["S"], Sg)), \
                    f"grid mismatch in {reg} l3={k} xi={xi}"
                for n in names:
                    cube[n][:, :, :, a, b] = d[n]
        stacks[reg] = {
            "interp": {n: ZL.stack_interp_l3_lx(Zg, Yg, Sg, l3ax, LX_GRID,
                                                cube[n]) for n in names},
            "pretech": pretech, "grids": (Zg, Yg, Sg, l3ax, LX_GRID)}
    return stacks


_STACKS = _build_stacks()

PROVENANCE = dict(
    economy="JONES: scale-free semi-endogenous intensity economy -- production "
            "model with exactly two scale-effect removals (tech hazard "
            "e^{s}/rho_s, s=logR-logK, rho_s=746.67/880; intensity emissions "
            "E(Z)=eta*A_d*(1-Z)*880), full nonlinear logN(Y;lambda3), damage "
            "jump J_n(Y), full robustness (h incl. h_y, jump distortions g)",
    design="benchmarks/economy_zoo/design_jones.json",
    solver="benchmarks/economy_zoo/solvers/jones_callbacks.py on "
           "solvers/fd_reduced.py (PIBYS harness, degenerate-validated vs "
           "fd_pdpt_v5_stable, W-RMS 6.6e-8)",
    ladder_dir=LADDER_DIR,
    value_object="net-space v = V + logN = logK + W(Z,Y,s); NO logN added in "
                 "the map (already inside W); exact reduction V_logK = 1 - W_s",
    stack="trilinear (Z,Y,s) x linear lambda3 (post-damage) x linear logxi "
          "over xi in {0.05, 0.1, 148.4}",
    homogeneity="V_logK = 1 - W_s asserted in (0.5, 1.5) at every lift point",
)

_HEADMAX = 0.95


def fields(reg, lk, Z, Y, lr, l3, lx):
    """Production states -> dict(v, i_d, i_g, i_r) of (n,1) float arrays."""
    if reg not in REGIMES:
        raise ValueError(f"unknown regime {reg!r}")
    st = _STACKS[reg]
    lk = np.asarray(lk, float).reshape(-1, 1)
    Z = np.clip(np.asarray(Z, float).reshape(-1, 1), 1e-6, 1 - 1e-6)
    Y = np.asarray(Y, float).reshape(-1, 1)
    lr = np.asarray(lr, float).reshape(-1, 1)
    l3 = np.clip(np.asarray(l3, float).reshape(-1, 1), 0.0, 1.0 / 3)
    lx = np.clip(np.asarray(lx, float).reshape(-1, 1), LX_GRID[0], LX_GRID[-1])
    s = lr - lk                      # no clip: grid [-6,2] covers the box image

    itp = st["interp"]
    W = itp["W"](Z, Y, s, l3, lx)
    v = lk + W
    i_d = ZL.feasible_invest(itp["i_d"](Z, Y, s, l3, lx), ZL.THETA_D,
                             i_max=_HEADMAX)
    i_g = ZL.feasible_invest(itp["i_g"](Z, Y, s, l3, lx), ZL.THETA_G,
                             i_max=_HEADMAX)
    if st["pretech"]:
        i_r = np.clip(itp["i_r"](Z, Y, s, l3, lx), 1e-8, _HEADMAX)
        W_s = itp["W_S"](Z, Y, s, l3, lx)
        vlk = 1.0 - W_s
    else:
        i_r = None
        vlk = np.ones_like(W)        # exact: post-tech W is s-independent

    stats = dict(regime=reg, n=int(lk.size),
                 v_logK_min=float(np.min(vlk)), v_logK_max=float(np.max(vlk)),
                 v_logK_p05=float(np.percentile(vlk, 5)),
                 v_logK_p50=float(np.percentile(vlk, 50)),
                 v_logK_p95=float(np.percentile(vlk, 95)),
                 W_min=float(np.min(W)), W_max=float(np.max(W)))
    fields.last_guard_stats = stats
    if not (stats["v_logK_min"] > 0.5 and stats["v_logK_max"] < 1.5):
        raise AssertionError(
            f"JONES homogeneity violated in {reg}: v_logK = 1 - W_s in "
            f"[{stats['v_logK_min']:.3f}, {stats['v_logK_max']:.3f}] "
            f"leaves (0.5, 1.5) -- ladder solve suspect, refusing to lift")
    return dict(v=v, i_d=i_d, i_g=i_g, i_r=i_r)


if __name__ == "__main__":
    # smoke: LHS sample per regime; shapes, finiteness, guard stats, x0 check
    rng = np.random.RandomState(7)
    N = 20000
    print("map:", MAP_NAME, "| ladder:", LADDER_DIR)
    for reg in REGIMES:
        ylo = 2.5 if reg.startswith("PostDamage") else 0.0
        lk = rng.uniform(4, 7, (N, 1)); Zs = rng.uniform(0.01, 0.99, (N, 1))
        Ys = rng.uniform(ylo, 4, (N, 1)); lrs = rng.uniform(1, 6, (N, 1))
        l3s = rng.uniform(0, 1 / 3, (N, 1))
        lxs = rng.uniform(np.log(0.05), np.log(148.6), (N, 1))
        F = fields(reg, lk, Zs, Ys, lrs, l3s, lxs)
        gs = fields.last_guard_stats
        assert F["v"].shape == (N, 1)
        for kk in ("v", "i_d", "i_g"):
            assert np.all(np.isfinite(F[kk])), (reg, kk)
        if reg.endswith("PostTech"):
            assert F["i_r"] is None
        else:
            assert np.all(np.isfinite(F["i_r"])) and np.all(F["i_r"] > 0)
        print("%-20s v[%8.3f,%8.3f] i_d[%+.4f,%.4f] i_g[%+.4f,%.4f] i_r[%s] "
              "v_logK p5/p50/p95 = %.3f/%.3f/%.3f" %
              (reg, F["v"].min(), F["v"].max(), F["i_d"].min(), F["i_d"].max(),
               F["i_g"].min(), F["i_g"].max(),
               ("%.5f,%.4f" % (F["i_r"].min(), F["i_r"].max()))
               if F["i_r"] is not None else "-",
               gs["v_logK_p05"], gs["v_logK_p50"], gs["v_logK_p95"]))
    x0 = [np.array([[np.log(880.0)]]), np.array([[0.7]]), np.array([[1.1]]),
          np.array([[np.log(11.2)]]), np.array([[0.0]]),
          np.array([[np.log(148.4)]])]
    F = fields("PreDamagePreTech", *x0)
    print("x0 (s0=%.3f): v=%.4f i_d=%.5f i_g=%.5f i_r=%.5f v_logK=%.4f" %
          (float(np.log(11.2) - np.log(880.0)), F["v"][0, 0], F["i_d"][0, 0],
           F["i_g"][0, 0], F["i_r"][0, 0],
           fields.last_guard_stats["v_logK_p50"]))
    print("SMOKE TEST OK")
