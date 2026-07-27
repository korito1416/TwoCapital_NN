"""G1 gate for RACE — symbolic algebra (sympy) + 1e4-random-state HJB residuals.

Symbolic checks (exact, sympy):
  S1  i_d* = (Gamma th A - delta)/(th(delta+Gamma)) solves the investment FOC.
  S2  Budget closure: c_g formula satisfies A_g - i_g(c_g) - i_r(c_g,P) - c_g = 0.
  S3  Brownian minimization: min_h {s Vx h + xi h^2/2} = -(1/2xi) s^2 Vx^2 at
      h* = -(1/xi) s Vx.
  S4  Jump minimization: g* = exp(-Delta/xi) is stationary and the minimized
      jump contribution J g Delta + xi J(1-g+g log g) equals xi J(1-e^{-Delta/xi}) = J P.
  S5  HJB state-coefficient matching: with V = beta_d lkd + beta_g lkg + q1 Y + V0,
      beta_d + beta_g = 1, the residual is INDEPENDENT of (lkd, lkg, Y) iff
      q1 = -kappa/delta; and the Y-coefficient equation yields exactly that q1.
  S6  The constant part of the pre-tech residual solved for V0pre reproduces the
      implemented closed form delta*V0pre = common + delta beta_g log c_g
      + beta_g phi(i_g) + q1 muY_pre + Lambda(i_r) P.

Numeric check (float64): N = 10,000 random states across
  - both tech regimes (pre-tech with jump/option terms, post-tech without),
  - both damage regimes (kappa_pre = 0.001 vs kappa_l = 0.001(1+3 lambda3)),
  - xi log-uniform in [0.01, 150] plus 1000 points at xi = inf,
  - lambda3 uniform in [0, 1/3], lkd/lkg in [2, 8], Y in [0, 4] ([2.5, 4] post-damage).
Full HJB residual (with closed-form minimizers and the bisection Delta*) and all
investment FOC residuals must satisfy max|.| <= 1e-12; at corner points the i_r
deviation inequality P rho_b th_r c_g0/(delta beta_g) - 1 <= 0 must hold.

Writes outputs/race_G1_verify.json (picked up by solvers/race.py into provenance).
"""
import os, json, importlib.util
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("race_solver", os.path.join(HERE, "race.py"))
race = importlib.util.module_from_spec(spec)
spec.loader.exec_module(race)
PAR = race.PAR

TOL = 1e-12
report = {"symbolic": {}, "numeric": {}}


# ============================================================== symbolic
def check(name, ok, detail=""):
    report["symbolic"][name] = dict(passed=bool(ok), detail=detail)
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return ok


print("G1 symbolic (sympy):", flush=True)
delta, Gamma, th, thr, rb, bd, bg = sp.symbols(
    "delta Gamma theta theta_r rho_b beta_d beta_g", positive=True)
A, Ag, P, xi, kappa, s = sp.symbols("A A_g P xi kappa s", positive=True)
i, hh, Vx, Dl, gg = sp.symbols("i h V_x Delta g", real=True)

# S1: investment FOC closed form
istar = (Gamma * th * A - delta) / (th * (delta + Gamma))
foc = -delta / (A - i) + Gamma * th / (1 + th * i)
ok1 = sp.simplify(foc.subs(i, istar)) == 0
check("S1_i_d_star_solves_FOC", ok1)

# S2: budget closure of c_g(P)
cg = (Ag + 1 / th + 1 / thr) / (1 + Gamma / delta + P * rb / (delta * bg))
ig = (Gamma * th * cg / delta - 1) / th
ir = (P * rb * thr * cg / (delta * bg) - 1) / thr
ok2 = sp.simplify(Ag - ig - ir - cg) == 0
check("S2_budget_closure", ok2)

# S3: Brownian minimization identity
expr_h = s * Vx * hh + xi * hh ** 2 / 2
hstar = sp.solve(sp.diff(expr_h, hh), hh)[0]
ok3 = (sp.simplify(hstar + s * Vx / xi) == 0
       and sp.simplify(expr_h.subs(hh, hstar) + s ** 2 * Vx ** 2 / (2 * xi)) == 0)
check("S3_h_star_and_minimized_value", ok3)

# S4: jump minimization identity
J = sp.symbols("J", positive=True)
expr_g = J * gg * Dl + xi * J * (1 - gg + gg * sp.log(gg))
gstar = sp.exp(-Dl / xi)
ok4a = sp.simplify(sp.diff(expr_g, gg).subs(gg, gstar)) == 0
ok4b = sp.simplify(expr_g.subs(gg, gstar) - xi * J * (1 - sp.exp(-Dl / xi))) == 0
check("S4_g_star_stationary_and_value_JP", ok4a and ok4b)

# S5: HJB state-coefficient matching + q1
lkd, lkg, Y, V0, q1s = sp.symbols("lkd lkg Y V0 q1", real=True)
sd, sg, sbar, muY, id_, ig_, ir_, LamS, Ps, drag_ = sp.symbols(
    "sigma_d sigma_g sbar muY i_d i_g i_r Lambda P_s drag", real=True)
V = bd * lkd + bg * lkg + q1s * Y + V0
res = (-delta * V
       + delta * bd * (sp.log(A - id_) + lkd) + delta * bg * (sp.log(Ag - ig_ - ir_) + lkg)
       + bd * ((sp.Symbol("phi_d")) - kappa * Y - sd ** 2 / 2)
       + bg * ((sp.Symbol("phi_g")) - kappa * Y - sg ** 2 / 2)
       + q1s * muY - drag_ + LamS * Ps)
res = res.subs(bd, 1 - bg)
c_lkd = sp.simplify(sp.diff(res, lkd))
c_lkg = sp.simplify(sp.diff(res, lkg))
c_Y = sp.simplify(sp.diff(res, Y))
q1_sol = sp.solve(c_Y, q1s)[0]
ok5 = (c_lkd == 0 and c_lkg == 0 and sp.simplify(q1_sol + kappa / delta) == 0
       and sp.simplify(c_Y.subs(q1s, -kappa / delta)) == 0)
check("S5_state_coefficients_vanish_q1_eq_minus_kappa_over_delta", ok5,
      "coeff(lkd)=%s coeff(lkg)=%s q1=%s" % (c_lkd, c_lkg, q1_sol))

# S6: constant part reproduces the implemented delta*V0pre closed form
const_part = res.subs([(lkd, 0), (lkg, 0), (Y, 0), (q1s, -kappa / delta)])
V0_sol = sp.solve(const_part, V0)[0]
formula = ((1 - bg) * delta * sp.log(A - id_) + bg * delta * sp.log(Ag - ig_ - ir_)
           + (1 - bg) * sp.Symbol("phi_d") + bg * sp.Symbol("phi_g")
           - ((1 - bg) * sd ** 2 + bg * sg ** 2) / 2
           + (-kappa / delta) * muY - drag_ + LamS * Ps) / delta
ok6 = sp.simplify(V0_sol - formula) == 0
check("S6_constant_part_matches_dV0_formula", ok6)

sym_ok = ok1 and ok2 and ok3 and ok4a and ok4b and ok5 and ok6


# ============================================================== numeric, 1e4 states
print("G1 numeric (1e4 random states, xi x lambda3, pre/post tech x pre/post damage):",
      flush=True)
rng = np.random.RandomState(20260719)
N = 10000
p = PAR
d = p["delta"]

lx = rng.uniform(np.log(0.01), np.log(150.0), N)
xiv = np.exp(lx)
xiv[rng.choice(N, 1000, replace=False)] = np.inf
l3 = rng.uniform(0.0, 1.0 / 3.0, N)
postdmg = rng.random_sample(N) < 0.5
kap = np.where(postdmg, race.kappa_of(l3), p["kappa0"])
lkd = rng.uniform(2.0, 8.0, N)
lkg = rng.uniform(2.0, 8.0, N)
Y = np.where(postdmg, rng.uniform(2.5, 4.0, N), rng.uniform(0.0, 4.0, N))

sol = race.solve(xiv, kap)
q1 = sol["q1"]
dragv = race.drag(xiv, q1)
muY_post = p["muY_pre"] * p["muY_post_factor"]

# pre-tech HJB residual (minimizers substituted; jump term = Lambda * P)
V_pre = p["beta_d"] * lkd + p["beta_g"] * lkg + q1 * Y + sol["V0pre"]
res_pre = (-d * V_pre
           + d * p["beta_d"] * (np.log(p["A_d"] - race.I_D_STAR) + lkd)
           + d * p["beta_g"] * (np.log(sol["c_g"]) + lkg)
           + p["beta_d"] * (race.phi(race.I_D_STAR) - kap * Y - 0.5 * p["sigma_d"] ** 2)
           + p["beta_g"] * (race.phi(sol["i_g"]) - kap * Y - 0.5 * p["sigma_g"] ** 2)
           + q1 * p["muY_pre"] - dragv + sol["Lambda"] * sol["P"])

# post-tech HJB residual
V_post = p["beta_d"] * lkd + p["beta_g"] * lkg + q1 * Y + sol["V0post"]
res_post = (-d * V_post
            + d * p["beta_d"] * (np.log(p["A_d"] - race.I_D_STAR) + lkd)
            + d * p["beta_g"] * (np.log(race.C_G_PP) + lkg)
            + p["beta_d"] * (race.phi(race.I_D_STAR) - kap * Y - 0.5 * p["sigma_d"] ** 2)
            + p["beta_g"] * (race.phi(race.I_G_PP) - kap * Y - 0.5 * p["sigma_g"] ** 2)
            + q1 * muY_post - dragv)

# FOC residuals at the solved policies
foc_id = -d / (p["A_d"] - race.I_D_STAR) + p["Gamma"] * p["theta"] / (1 + p["theta"] * race.I_D_STAR)
foc_ig = -d / sol["c_g"] + p["Gamma"] * p["theta"] / (1 + p["theta"] * sol["i_g"])
foc_ig_pp = -d / race.C_G_PP + p["Gamma"] * p["theta"] / (1 + p["theta"] * race.I_G_PP)
interior = sol["i_r"] > 0
foc_ir = np.zeros(N)
foc_ir[interior] = (-d * p["beta_g"] / sol["c_g"][interior]
                    + sol["P"][interior] * p["rho_b"] * p["theta_r"]
                    / (1 + p["theta_r"] * sol["i_r"][interior]))
# corner optimality: no profitable i_r deviation at i_r = 0
corner_dev = (sol["P"][~interior] * p["rho_b"] * p["theta_r"] * sol["c_g"][~interior]
              / (d * p["beta_g"]) - 1.0)

stats = dict(
    max_abs_res_pre=float(np.max(np.abs(res_pre))),
    max_abs_res_post=float(np.max(np.abs(res_post))),
    max_abs_foc_id=float(abs(foc_id)),
    max_abs_foc_ig=float(np.max(np.abs(foc_ig))),
    max_abs_foc_ig_pp=float(abs(foc_ig_pp)),
    max_abs_foc_ir_interior=float(np.max(np.abs(foc_ir))) if interior.any() else 0.0,
    max_corner_deviation=float(np.max(corner_dev)) if (~interior).any() else -1.0,
    n_states=N, n_xi_inf=1000, n_corner=int((~interior).sum()),
    max_abs_Phi_at_root=float(np.max(np.abs(sol["Phi_at_root"]))))
num_ok = (stats["max_abs_res_pre"] <= TOL and stats["max_abs_res_post"] <= TOL
          and stats["max_abs_foc_id"] <= TOL and stats["max_abs_foc_ig"] <= TOL
          and stats["max_abs_foc_ig_pp"] <= TOL
          and stats["max_abs_foc_ir_interior"] <= TOL
          and stats["max_corner_deviation"] <= TOL)
for k, v in stats.items():
    print("  %-28s %.3e" % (k, v) if isinstance(v, float) else "  %-28s %s" % (k, v),
          flush=True)
report["numeric"] = dict(**stats, tol=TOL, passed=bool(num_ok))
print("  numeric gate (<=1e-12): %s" % ("PASS" if num_ok else "FAIL"), flush=True)

report["passed"] = bool(sym_ok and num_ok)
out = os.path.join(race.OUT_DIR, "race_G1_verify.json")
os.makedirs(race.OUT_DIR, exist_ok=True)
with open(out, "w") as f:
    json.dump(report, f, indent=1, default=float)
print("G1 OVERALL: %s  (wrote %s)" % ("PASS" if report["passed"] else "FAIL", out), flush=True)
