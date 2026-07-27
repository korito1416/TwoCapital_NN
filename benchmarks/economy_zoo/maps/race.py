"""RACE economy map — Aghion–Howitt patent race with growth damages (analytic lift).

Consumed by models_terminal_anchor/make_map_anchor.py (MAP MODULE INTERFACE):
exports MAP_NAME, PROVENANCE, fields(reg, lk, Z, Y, lr, l3, lx).

VALUE TRANSFORM (the object the verifier checks): the v target is the RACE
economy's value function expressed at production states,
    v(x) = logK + beta_d log(1-Z) + beta_g log(Z) + q1(reg, l3) * Y + V0^reg(xi, kappa_l)
with (beta_d, beta_g) = (0.3, 0.7), xi = exp(lx), and the damage/lambda3 dependence
entering ANALYTICALLY through the growth-damage coefficient — NOT through a logN
level-damage transform (RACE has Dell–Jones–Olken GROWTH damages, no utility
damage): kappa = 0.001 in PreDamage regimes, kappa_l = 0.001(1 + 3*lambda3) in
PostDamage regimes, slope q1 = -kappa_l/delta (so -0.100 pre-damage, down to
-0.200 at lambda3 = 1/3) and V0^reg(xi, kappa_l) from the tech-jump fixed point
Delta*(xi, kappa_l) (PreTech regimes use V0^pre, PostTech regimes V0^post).

logR is COLLAPSED — v and all policies are exactly flat in lr (deliberate,
hypothesis-bearing: initializes V_logR = 0, the no-knowledge-stock pole on the
axis the loss-decomposition study flagged as the largest cross-run divergence).

Policies (production rate conventions):
    i_d = 0.103131 everywhere;
    i_g = i_g*(xi, kappa_l) pre-tech (0.0791 -> 0.0380 across xi at kappa=0.001),
          0.125760 post-tech;
    i_r = Z * i_r*(xi, kappa_l) in PreTech regimes, floored at 1e-8 (RACE's rate is
          I_r/K_g; production wants I_r/(K_d+K_g) = Z * i_r^econ), and i_r*=0 at the
          corner xi < xibar = 0.04356 (just below the production box lx >= log 0.05);
          None in PostTech regimes (R&D inactive).

Evaluation: EXACT vectorized 200-step bisection of the scalar fixed point at the
query (xi, kappa) via solvers/race.py — the synthesis' 60-pt-logxi spline table is
still produced in outputs/race.npz (max spline-vs-exact dev 5.0e-3 on V0, 1.3e-4
on i_r inside the production box); exact evaluation is used here to avoid any
interpolation error near the corner kink. exp arguments clipped at 35 inside the
solver (repo convention).

Run this file directly for the smoke test (shapes, finiteness, logR-flatness,
design-table spot values).
"""
import os, importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "race_solver", os.path.join(_HERE, "..", "solvers", "race.py"))
_race = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_race)

MAP_NAME = "RACE_aghion_howitt_patent_race_growth_damages_analytic"

PROVENANCE = dict(
    economy="RACE — Aghion–Howitt patent race with growth damages (fully analytic)",
    design="benchmarks/economy_zoo/design_race.json (PORTFOLIO.json closed-form lens "
           "Economy B; critique: no math errors; relabeled calibrated AH spec)",
    solver="benchmarks/economy_zoo/solvers/race.py (exact vectorized bisection of the "
           "scalar fixed point Phi(Delta)=V0^post-V0^pre(Delta)-Delta per (xi,kappa))",
    verification="G1 sympy + 1e4-state HJB residual max 4.9e-17 "
                 "(solvers/verify_race_sympy.py); G2 bracket + root |Phi|<4e-15; "
                 "G3 S1-corner/neutral-limit/collapse anchors; G4 Feynman-Kac MC level "
                 "certificate (neutral + distorted-measure xi=0.05) — see "
                 "outputs/race_PROVENANCE.json",
    outputs="benchmarks/economy_zoo/outputs/race.npz",
    value_transform="v = lk + 0.3 log(1-Z) + 0.7 log Z + q1(reg,l3) Y + V0^reg(xi,kappa_l); "
                    "q1=-kappa_l/delta; kappa_l=0.001 (PreDamage) or 0.001(1+3 l3) "
                    "(PostDamage); GROWTH damages, no logN transform; flat in logR",
    key_numbers=dict(i_d_star=0.103131, i_g_post_tech=0.125760,
                     i_r_range_predamage=(0.00629, 0.0542), xibar=0.04356,
                     q1_predamage=-0.1),
)

_BD, _BG = _race.PAR["beta_d"], _race.PAR["beta_g"]


def _kappa(reg, l3):
    if reg.startswith("PostDamage"):
        return _race.kappa_of(l3)
    return np.full_like(np.asarray(l3, float), _race.PAR["kappa0"])


def fields(reg, lk, Z, Y, lr, l3, lx):
    """Production-state targets. Inputs (n,1) arrays; lr is ignored (flat in logR)."""
    lk, Z, Y, l3, lx = [np.asarray(a, float) for a in (lk, Z, Y, l3, lx)]
    n = lk.shape[0]
    xi = np.exp(lx.ravel())
    kap = _kappa(reg, l3.ravel())
    sol = _race.solve(xi, kap)
    pre_tech = reg.endswith("PreTech")
    V0 = sol["V0pre"] if pre_tech else sol["V0post"]
    q1 = sol["q1"]
    v = (lk.ravel() + _BD * np.log(1.0 - Z.ravel()) + _BG * np.log(Z.ravel())
         + q1 * Y.ravel() + V0)
    i_d = np.full(n, float(_race.I_D_STAR))
    i_g = sol["i_g"] if pre_tech else np.full(n, float(_race.I_G_PP))
    out = dict(v=v.reshape(n, 1), i_d=i_d.reshape(n, 1), i_g=i_g.reshape(n, 1))
    if pre_tech:
        # RACE rate is I_r/K_g; production convention is I_r/K_total = Z * i_r^econ
        out["i_r"] = np.maximum(Z.ravel() * sol["i_r"], 1e-8).reshape(n, 1)
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
        lx = rng.uniform(np.log(0.05), np.log(148.6), (n, 1))
        F = fields(reg, lk, Z, Y, lr, l3, lx)
        F2 = fields(reg, lk, Z, Y, lr + 1.7, l3, lx)     # logR flatness
        for k in ("v", "i_d", "i_g"):
            assert F[k].shape == (n, 1) and np.all(np.isfinite(F[k])), (reg, k)
            assert np.max(np.abs(F[k] - F2[k])) == 0.0, (reg, k, "not flat in logR")
        if reg.endswith("PreTech"):
            assert F["i_r"].shape == (n, 1) and np.all(np.isfinite(F["i_r"]))
            assert np.max(np.abs(F["i_r"] - F2["i_r"])) == 0.0
            assert np.all(F["i_r"] >= 1e-8) and np.all(F["i_r"] < 0.06)
        else:
            assert F["i_r"] is None
        print("%-22s v[%8.3f, %8.3f]  i_g[%.4f, %.4f]  i_d=%.6f  i_r=%s"
              % (reg, F["v"].min(), F["v"].max(), F["i_g"].min(), F["i_g"].max(),
                 F["i_d"][0, 0],
                 "None" if F["i_r"] is None else
                 "[%.2e, %.4f]" % (F["i_r"].min(), F["i_r"].max())), flush=True)

    # design-table spot checks at the query states (pre-damage, Z chosen so Z*i_r is clean)
    one = np.ones((3, 1))
    lxq = np.log(np.array([[0.05], [0.1], [148.4]]))
    F = fields("PreDamagePreTech", 6.1 * one, 0.7 * one, 1.1 * one, 3.0 * one,
               0.0 * one, lxq)
    ir_econ = F["i_r"].ravel() / 0.7
    exp_ir = np.array([0.0062885, 0.042634, 0.054176])
    exp_ig = np.array([0.079074, 0.047909, 0.037976])
    dev_ir = np.max(np.abs(ir_econ - exp_ir))
    dev_ig = np.max(np.abs(F["i_g"].ravel() - exp_ig))
    print("spot-check i_r^econ at xi=(0.05,0.1,148.4):", np.round(ir_econ, 6),
          "max dev vs solver table %.1e" % dev_ir, flush=True)
    print("spot-check i_g pre-tech:", np.round(F["i_g"].ravel(), 6),
          "max dev %.1e" % dev_ig, flush=True)
    ok &= dev_ir < 1e-4 and dev_ig < 1e-4
    # v formula spot check (manual recomputation)
    s = _race.solve(np.array([0.05]), np.array([0.001]))
    v_manual = (6.1 + 0.3 * np.log(0.3) + 0.7 * np.log(0.7) - 0.1 * 1.1
                + float(s["V0pre"][0]))
    ok &= abs(F["v"][0, 0] - v_manual) < 1e-12
    print("v spot check (xi=0.05): map %.6f manual %.6f" % (F["v"][0, 0], v_manual),
          flush=True)
    # monotone i_r in xi within box
    lxg = np.linspace(np.log(0.05), np.log(148.6), 200).reshape(-1, 1)
    o = np.ones_like(lxg)
    Fm = fields("PreDamagePreTech", 6 * o, 0.5 * o, 1 * o, 3 * o, 0 * o, lxg)
    ok &= bool(np.all(np.diff(Fm["i_r"].ravel()) > 0))
    print("i_r strictly increasing in xi across production box:",
          bool(np.all(np.diff(Fm["i_r"].ravel()) > 0)), flush=True)
    print("SMOKE TEST:", "PASS" if ok else "FAIL", flush=True)
