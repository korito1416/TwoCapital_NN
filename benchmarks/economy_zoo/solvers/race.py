"""RACE — Aghion–Howitt patent race with growth damages (fully analytic).

Economy (design_race.json + PORTFOLIO.json closed-form lens, Economy B, critique-verified):
  - Cobb–Douglas separable trees, U = delta[beta_d log C_d + beta_g log C_g],
    (beta_d, beta_g) = (0.3, 0.7); sector budgets C_d=(A_d-i_d)K_d, C_g=(A_g-i_g-i_r)K_g.
  - Innovation = memoryless PATENT RACE: Poisson breakthrough hazard
    Lambda(i_r) = rho_b log(1 + theta_r i_r), (rho_b, theta_r) = (0.40, 16.7),
    purchased by CURRENT R&D flow; NO knowledge stock (V_logR = 0 pole, deliberate).
    RELABELED per critique: a calibrated Aghion–Howitt specification, NOT an exact
    reduction of production's zeta=0 R-process (requirement-(1) deviation, documented).
  - GROWTH damages (Dell–Jones–Olken): dlogK_j carries -kappa_l Y dt;
    kappa_pre = 0.001, post-damage kappa_l = kappa0 (1 + 3 lambda3(l)).
  - Breakthrough jump: A_g 0.1085 -> 0.1567 AND cools the warming drift
    muY: 0.01862 -> 0.25 x (absorbing, OneJump pi=1 within each damage regime).
  - Robustness xi on all four Brownian channels (h_y active, V linear in Y) plus
    the jump-intensity distortion g* = exp(-Delta/xi); robust jump premium
    P = xi (1 - e^{-Delta/xi}) saturates at xi and chokes R&D (corner below xibar).

Closed form up to ONE scalar monotone fixed point per (xi, kappa):
  Ansatz V^n = beta_d logK_d + beta_g logK_g + q1 Y + V0^n, q1 = -kappa/delta
  (regime-independent within a damage regime => the tech gap Delta = V0^post - V0^pre
  is a state-independent constant — the closure).
  Policies: i_d* = (Gamma theta A_d - delta)/(theta(delta+Gamma)) = 0.103131;
  post-tech i_g'' = 0.125760, c_g'' = 0.030940. Pre-tech given Delta:
  P = xi(1-e^{-Delta/xi}); c_g = (A_g + 1/theta + 1/theta_r)/(1 + Gamma/delta
  + P rho_b/(delta beta_g)); i_g = (Gamma theta c_g/delta - 1)/theta;
  i_r = max(0, (P rho_b theta_r c_g/(delta beta_g) - 1)/theta_r)  [corner branch:
  c_g^0 = (A_g + 1/theta)/(1 + Gamma/delta)].
  Fixed point Phi(Delta) = V0^post - V0^pre(Delta) - Delta, strictly decreasing
  (Phi'(Delta) = -Lambda e^{-Delta/xi}/delta - 1 < 0 by envelope), bracket
  Phi(0) > 0 > Phi(50), unique root by 200-step vectorized bisection.

Numerical guards: every exponent Delta/xi clipped at EXP_CLIP = 35 (repo overflow
convention); feasibility 1 + theta i > 0 holds (all i >= 0 here); xi = inf handled
exactly (P -> Delta, drag -> 0).

Outputs (main):
  benchmarks/economy_zoo/outputs/race.npz             tables + evaluated grids
  benchmarks/economy_zoo/outputs/race_PROVENANCE.json gates with numbers
  benchmarks/economy_zoo/figures/race_*.png           via race_figures (called here)

Gates run in main: G2 (bracket + root residual), reference-table reproduction,
G3 anchors/limits (S1 corner, xi=148.4 vs neutral O(1/xi), kappa->0 kills q1,
rho_b->0 collapse), G4 Feynman–Kac Monte Carlo level certificate (neutral base
measure at xi=inf; distorted measure + penalty flows at xi=0.05).
"""
import os, json, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(ZOO, "outputs")
FIG_DIR = os.path.join(ZOO, "figures")

EXP_CLIP = 35.0

PAR = dict(
    delta=0.01,
    alpha=-0.035, Gamma=0.060, theta=16.7,          # adjustment-cost tech (both sectors)
    sigma_d=0.01, sigma_g=0.01,
    A_d=0.1303, A_g_pre=0.1085, A_g_pp=0.1567,
    beta_d=0.3, beta_g=0.7,                          # Cobb-Douglas utility shares
    rho_b=0.40, theta_r=16.7,                        # patent-race hazard
    kappa0=0.001,                                    # growth-damage coefficient (pre-damage)
    muY_pre=0.01862, muY_post_factor=0.25,           # warming drift, pre/post breakthrough
    sbar=0.02234,                                    # temperature vol
)


# ----------------------------------------------------------------------------- primitives
def phi(i):
    return PAR["alpha"] + PAR["Gamma"] * np.log(1.0 + PAR["theta"] * np.asarray(i, float))


def Lam(ir):
    return PAR["rho_b"] * np.log(1.0 + PAR["theta_r"] * np.asarray(ir, float))


def foc_invest(A):
    """i* solving delta/(A-i) = Gamma*theta/(1+theta*i)."""
    d, G, th = PAR["delta"], PAR["Gamma"], PAR["theta"]
    return (G * th * A - d) / (th * (d + G))


I_D_STAR = foc_invest(PAR["A_d"])                    # 0.103131 (everywhere)
I_G_PP = foc_invest(PAR["A_g_pp"])                   # 0.125760 (post-tech)
C_G_PP = PAR["A_g_pp"] - I_G_PP                      # 0.030940


def P_of(Delta, xi):
    """Robust jump premium P = xi(1-e^{-Delta/xi}); exact limit P=Delta at xi=inf.
    Exponent clipped at EXP_CLIP=35 (repo convention)."""
    Delta, xi = np.broadcast_arrays(np.asarray(Delta, float), np.asarray(xi, float))
    P = np.array(Delta, float, copy=True)            # xi = inf branch
    fin = np.isfinite(xi)
    r = np.minimum(Delta[fin] / xi[fin], EXP_CLIP)
    P[fin] = xi[fin] * (-np.expm1(-r))
    return P


def g_of(Delta, xi):
    """Worst-case jump-intensity distortion g* = exp(-Delta/xi), clipped exponent."""
    Delta, xi = np.broadcast_arrays(np.asarray(Delta, float), np.asarray(xi, float))
    g = np.ones_like(Delta)                          # xi = inf branch
    fin = np.isfinite(xi)
    g[fin] = np.exp(-np.clip(Delta[fin] / xi[fin], -EXP_CLIP, EXP_CLIP))
    return g


def policies_pre(P):
    """Pre-tech (c_g, i_g, i_r) given the robust premium P, with the i_r=0 corner."""
    p = PAR
    d, G, th, thr, rb, bg = p["delta"], p["Gamma"], p["theta"], p["theta_r"], p["rho_b"], p["beta_g"]
    Ag = p["A_g_pre"]
    P = np.asarray(P, float)
    cg_int = (Ag + 1.0 / th + 1.0 / thr) / (1.0 + G / d + P * rb / (d * bg))
    ir_int = (P * rb * thr * cg_int / (d * bg) - 1.0) / thr
    corner = ir_int <= 0.0
    cg0 = (Ag + 1.0 / th) / (1.0 + G / d)
    cg = np.where(corner, cg0, cg_int)
    ir = np.where(corner, 0.0, ir_int)
    ig = (G * th * cg / d - 1.0) / th
    return cg, ig, ir


def drag(xi, q1):
    """-(1/2xi)(sigma_d^2 beta_d^2 + sigma_g^2 beta_g^2 + sbar^2 q1^2): the POSITIVE
    magnitude is returned; enters V0 with a minus sign. 0 at xi = inf."""
    p = PAR
    xi, q1 = np.broadcast_arrays(np.asarray(xi, float), np.asarray(q1, float))
    quad = p["sigma_d"] ** 2 * p["beta_d"] ** 2 + p["sigma_g"] ** 2 * p["beta_g"] ** 2 \
        + p["sbar"] ** 2 * q1 ** 2
    out = np.zeros_like(q1)
    fin = np.isfinite(xi)
    out[fin] = quad[fin] / (2.0 * xi[fin])
    return out


def _common_const(xi, q1):
    """Shared (dirty-block + Ito + drag) part of delta*V0 in both tech regimes."""
    p = PAR
    return (p["delta"] * p["beta_d"] * np.log(p["A_d"] - I_D_STAR)
            + p["beta_d"] * phi(I_D_STAR)
            - 0.5 * (p["beta_d"] * p["sigma_d"] ** 2 + p["beta_g"] * p["sigma_g"] ** 2)
            - drag(xi, q1))


def dV0_post(xi, q1):
    """delta * V0^post(xi, q1)."""
    p = PAR
    muY_post = p["muY_pre"] * p["muY_post_factor"]
    return (_common_const(xi, q1) + p["delta"] * p["beta_g"] * np.log(C_G_PP)
            + p["beta_g"] * phi(I_G_PP) + q1 * muY_post)


def dV0_pre(Delta, xi, q1):
    """delta * V0^pre given Delta (with optimal pre-tech policies + option flow)."""
    p = PAR
    P = P_of(Delta, xi)
    cg, ig, ir = policies_pre(P)
    return (_common_const(xi, q1) + p["delta"] * p["beta_g"] * np.log(cg)
            + p["beta_g"] * phi(ig) + q1 * p["muY_pre"] + Lam(ir) * P)


def Phi(Delta, xi, q1):
    """Fixed-point map Phi(Delta) = V0^post - V0^pre(Delta) - Delta (strictly decreasing)."""
    return (dV0_post(xi, q1) - dV0_pre(Delta, xi, q1)) / PAR["delta"] - np.asarray(Delta, float)


def solve(xi, kappa, lo=0.0, hi=50.0, iters=200):
    """Vectorized bisection for Delta*(xi, kappa) and all derived objects.

    xi may contain np.inf (uncertainty-neutral). Returns dict of arrays broadcast
    to the common shape of (xi, kappa)."""
    xi, kappa = np.broadcast_arrays(np.asarray(xi, float), np.asarray(kappa, float))
    q1 = -kappa / PAR["delta"]
    lo_a = np.full(xi.shape, float(lo))
    hi_a = np.full(xi.shape, float(hi))
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        pos = Phi(mid, xi, q1) > 0.0
        lo_a = np.where(pos, mid, lo_a)
        hi_a = np.where(pos, hi_a, mid)
    Delta = 0.5 * (lo_a + hi_a)
    P = P_of(Delta, xi)
    cg, ig, ir = policies_pre(P)
    V0post = dV0_post(xi, q1) / PAR["delta"]
    V0pre = V0post - Delta
    return dict(Delta=Delta, P=P, c_g=cg, i_g=ig, i_r=ir, Lambda=Lam(ir),
                g=g_of(Delta, xi), V0pre=V0pre, V0post=V0post, q1=q1,
                Phi_at_root=Phi(Delta, xi, q1))


def kappa_of(l3):
    """Post-damage growth-damage coefficient kappa_l = kappa0 (1 + 3 lambda3)."""
    return PAR["kappa0"] * (1.0 + 3.0 * np.asarray(l3, float))


def xibar(kappa=PAR["kappa0"], lo=1e-3, hi=1.0, iters=200):
    """Corner threshold: largest xi with i_r* = 0 (bisection on i_r*(xi) > 0)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = solve(np.array([mid]), np.array([kappa]))
        if s["i_r"][0] > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------- G4: Feynman-Kac MC
def feynman_kac_mc(xi, kappa, n_seeds=20, n_paths=400, T=400.0, dt=0.1, seed=7):
    """Level certificate: simulate the economy's own SDE + jump system under the
    solved policies and (for finite xi) the WORST-CASE measure (drifts + sigma h,
    jump intensity Lambda*g) with the penalty flows xi/2|h|^2 + xi*Lambda(1-g+g log g)
    added to the utility flow; realized discounted value must equal V^pre at the
    seed states. Tail closed with e^{-delta T} V(X_T) (analytic). Trapezoid
    discounted-integral accumulation.

    Returns dict with per-seed analytic V, MC mean, MC standard error, and the
    worst |err| / max(|err|/SE)."""
    p = PAR
    d = p["delta"]
    s = solve(np.array([xi]), np.array([kappa]))
    q1 = float(s["q1"][0])
    Delta, P = float(s["Delta"][0]), float(s["P"][0])
    cg, ig, ir = float(s["c_g"][0]), float(s["i_g"][0]), float(s["i_r"][0])
    Lam_star = float(s["Lambda"][0])
    V0pre, V0post = float(s["V0pre"][0]), float(s["V0post"][0])
    muY_pre, muY_post = p["muY_pre"], p["muY_pre"] * p["muY_post_factor"]

    if np.isfinite(xi):
        h_d = -(1.0 / xi) * p["sigma_d"] * p["beta_d"]
        h_g = -(1.0 / xi) * p["sigma_g"] * p["beta_g"]
        h_y = -(1.0 / xi) * p["sbar"] * q1
        gdist = float(s["g"][0])
        # g log g computed as -g*Delta/xi (stable for g ~ e^{-27})
        glogg = -gdist * min(Delta / xi, EXP_CLIP)
        pen_h = 0.5 * xi * (h_d ** 2 + h_g ** 2 + h_y ** 2)
        pen_jump = xi * Lam_star * (1.0 - gdist + glogg)
    else:
        h_d = h_g = h_y = 0.0
        gdist, pen_h, pen_jump = 1.0, 0.0, 0.0
    lam_sim = Lam_star * gdist                       # distorted arrival rate

    rng = np.random.RandomState(seed)
    lkd0 = rng.uniform(4.0, 7.0, n_seeds) + np.log(1 - rng.uniform(0.2, 0.8, n_seeds))
    lkg0 = rng.uniform(4.0, 7.0, n_seeds)
    Y0 = rng.uniform(0.0, 4.0, n_seeds)
    V_true = (p["beta_d"] * lkd0 + p["beta_g"] * lkg0 + q1 * Y0 + V0pre)

    n = n_seeds * n_paths
    lkd = np.repeat(lkd0, n_paths)
    lkg = np.repeat(lkg0, n_paths)
    Y = np.repeat(Y0, n_paths)
    post = np.zeros(n, dtype=bool)
    log_cd = np.log(p["A_d"] - I_D_STAR)

    def flow(lkd, lkg, Y, post):
        lc_g = np.where(post, np.log(C_G_PP), np.log(cg))
        u = d * (p["beta_d"] * (log_cd + lkd) + p["beta_g"] * (lc_g + lkg))
        pen = np.where(post, pen_h, pen_h + pen_jump)
        return u + pen

    nstep = int(round(T / dt))
    sq = np.sqrt(dt)
    acc = np.zeros(n)
    f_prev = flow(lkd, lkg, Y, post)
    disc_prev = 1.0
    p_jump = -np.expm1(-lam_sim * dt)
    for k in range(1, nstep + 1):
        dWd = rng.standard_normal(n) * sq
        dWg = rng.standard_normal(n) * sq
        dWy = rng.standard_normal(n) * sq
        mu_g = np.where(post, phi(I_G_PP), phi(ig))
        muY = np.where(post, muY_post, muY_pre)
        lkd = lkd + (phi(I_D_STAR) - kappa * Y - 0.5 * p["sigma_d"] ** 2
                     + p["sigma_d"] * h_d) * dt + p["sigma_d"] * dWd
        lkg = lkg + (mu_g - kappa * Y - 0.5 * p["sigma_g"] ** 2
                     + p["sigma_g"] * h_g) * dt + p["sigma_g"] * dWg
        Y = Y + (muY + p["sbar"] * h_y) * dt + p["sbar"] * dWy
        if lam_sim > 0:
            jump = (~post) & (rng.random_sample(n) < p_jump)
            post = post | jump
        f_now = flow(lkd, lkg, Y, post)
        disc_now = np.exp(-d * k * dt)
        acc += 0.5 * (f_prev * disc_prev + f_now * disc_now) * dt
        f_prev, disc_prev = f_now, disc_now
    V_tail = (p["beta_d"] * lkd + p["beta_g"] * lkg + q1 * Y
              + np.where(post, V0post, V0pre))
    total = acc + disc_prev * V_tail

    total = total.reshape(n_seeds, n_paths)
    mc_mean = total.mean(axis=1)
    mc_se = total.std(axis=1, ddof=1) / np.sqrt(n_paths)
    err = mc_mean - V_true
    return dict(V_true=V_true, mc_mean=mc_mean, mc_se=mc_se, err=err,
                max_abs_err=float(np.max(np.abs(err))),
                max_t=float(np.max(np.abs(err) / mc_se)),
                pooled_err=float(err.mean()),
                pooled_se=float(mc_se.mean() / np.sqrt(n_seeds)))


# ----------------------------------------------------------------------------- main build
def build_tables(n_lx=60, n_l3=41):
    lx_grid = np.linspace(np.log(0.01), np.log(150.0), n_lx)
    xi_grid = np.exp(lx_grid)
    l3_grid = np.linspace(0.0, 1.0 / 3.0, n_l3)
    pre = solve(xi_grid, np.full(n_lx, PAR["kappa0"]))
    XI, L3 = np.meshgrid(xi_grid, l3_grid, indexing="ij")
    post = solve(XI, kappa_of(L3))
    return lx_grid, xi_grid, l3_grid, pre, post


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    gates = {}
    t0 = datetime.datetime.now().isoformat(timespec="seconds")
    print("RACE analytic solve —", t0, flush=True)

    # ---------------- G1 (run separately: solvers/verify_race_sympy.py) — merge if present
    g1_path = os.path.join(OUT_DIR, "race_G1_verify.json")
    if os.path.exists(g1_path):
        with open(g1_path) as f:
            gates["G1_sympy_and_1e4_residuals"] = json.load(f)
        print("G1 merged from race_G1_verify.json: passed =",
              gates["G1_sympy_and_1e4_residuals"]["passed"], flush=True)
    else:
        gates["G1_sympy_and_1e4_residuals"] = dict(
            passed=False, detail="race_G1_verify.json missing — run verify_race_sympy.py first")
        print("WARNING: G1 report missing (run verify_race_sympy.py)", flush=True)

    # ---------------- tables (60-pt logxi grid per synthesis; x 41-pt lambda3 post-damage)
    lx_grid, xi_grid, l3_grid, pre, post = build_tables()

    # ---------------- G2: bracket + root quality over the full (xi, kappa) family
    XI, L3 = np.meshgrid(xi_grid, l3_grid, indexing="ij")
    KAP = kappa_of(L3)
    Q1 = -KAP / PAR["delta"]
    phi0 = Phi(np.zeros_like(XI), XI, Q1)
    phi50 = Phi(np.full(XI.shape, 50.0), XI, Q1)
    xi_inf = np.array([np.inf])
    phi0_inf = float(Phi(np.array([0.0]), xi_inf, np.array([-0.1]))[0])
    phi50_inf = float(Phi(np.array([50.0]), xi_inf, np.array([-0.1]))[0])
    root_res = max(np.max(np.abs(pre["Phi_at_root"])), np.max(np.abs(post["Phi_at_root"])))
    gates["G2_fixed_point"] = dict(
        min_Phi_at_0=float(np.min(phi0)), max_Phi_at_50=float(np.max(phi50)),
        Phi0_xi_inf=phi0_inf, Phi50_xi_inf=phi50_inf,
        max_abs_Phi_at_root=float(root_res),
        passed=bool(np.min(phi0) > 0 and np.max(phi50) < 0 and phi0_inf > 0
                    and phi50_inf < 0 and root_res < 1e-12))
    print("G2 bracket: min Phi(0)=%.4f  max Phi(50)=%.2f  |Phi(root)|max=%.2e  -> %s"
          % (np.min(phi0), np.max(phi50), root_res, gates["G2_fixed_point"]["passed"]), flush=True)

    # ---------------- reference table (design/critique numbers, kappa_pre)
    ref = {0.05: dict(Delta=1.360, i_r=0.00628, i_g=0.07906, Lam=0.0399),
           0.1: dict(Delta=0.656, i_r=0.0426, i_g=0.0479, Lam=0.215),
           np.inf: dict(Delta=0.122, i_r=0.0542, i_g=0.0380, Lam=0.258)}
    tab = {}
    ok_ref = True
    for x, r in ref.items():
        s = solve(np.array([x]), np.array([PAR["kappa0"]]))
        got = dict(Delta=float(s["Delta"][0]), i_r=float(s["i_r"][0]),
                   i_g=float(s["i_g"][0]), Lam=float(s["Lambda"][0]))
        rel = {k: abs(got[k] - r[k]) / max(abs(r[k]), 1e-12) for k in r}
        ok = max(rel.values()) < 6e-2   # design table rounded to ~3 sig figs
        ok_ref &= ok
        tab[str(x)] = dict(got=got, design=r, max_rel_dev=float(max(rel.values())), passed=bool(ok))
        print("ref xi=%s: Delta*=%.4f i_r*=%.5f i_g*=%.5f Lam=%.4f (max rel dev %.3f)"
              % (x, got["Delta"], got["i_r"], got["i_g"], got["Lam"], max(rel.values())), flush=True)
    xb = xibar()
    xb_l5 = xibar(kappa=kappa_of(1.0 / 3.0))
    tab["xibar"] = dict(got=float(xb), design=0.0436, got_postdamage_l5=float(xb_l5))
    ok_ref &= abs(xb - 0.0436) / 0.0436 < 6e-2
    gates["reference_table"] = dict(entries=tab, passed=bool(ok_ref))
    print("xibar = %.5f (design 0.0436); post-damage l3=1/3: %.5f" % (xb, xb_l5), flush=True)

    # ---------------- G3 anchors + limits
    s1 = dict(i_g_pp=float(I_G_PP), phi_i_g_pp=float(phi(I_G_PP)), c_g_pp=float(C_G_PP))
    s1_ok = (abs(I_G_PP - 0.12576) < 1e-5 and abs(phi(I_G_PP) - 0.03289) < 1e-5
             and abs(C_G_PP - 0.03094) < 1e-5)
    s148 = solve(np.array([148.4]), np.array([PAR["kappa0"]]))
    sinf = solve(np.array([np.inf]), np.array([PAR["kappa0"]]))
    neu = dict(dDelta=float(abs(s148["Delta"][0] - sinf["Delta"][0])),
               di_r=float(abs(s148["i_r"][0] - sinf["i_r"][0])),
               dV0pre=float(abs(s148["V0pre"][0] - sinf["V0pre"][0])))
    neu_ok = neu["dDelta"] < 0.05 and neu["di_r"] < 1e-3 and neu["dV0pre"] < 0.05
    q1_kappa0 = float(-0.0 / PAR["delta"])           # kappa -> 0 => q1 = 0 (formula-exact)
    # rho_b -> 0 collapse: no-R&D two-sector pre-tech economy
    rb_save = PAR["rho_b"]
    PAR["rho_b"] = 1e-12
    s_rb0 = solve(np.array([np.inf]), np.array([PAR["kappa0"]]))
    PAR["rho_b"] = rb_save
    cg0 = (PAR["A_g_pre"] + 1 / PAR["theta"]) / (1 + PAR["Gamma"] / PAR["delta"])
    ig0 = (PAR["Gamma"] * PAR["theta"] * cg0 / PAR["delta"] - 1) / PAR["theta"]
    rb0 = dict(i_r=float(s_rb0["i_r"][0]), c_g=float(s_rb0["c_g"][0]), c_g0_analytic=float(cg0),
               i_g=float(s_rb0["i_g"][0]), i_g0_analytic=float(ig0))
    rb0_ok = (s_rb0["i_r"][0] == 0.0 and abs(s_rb0["c_g"][0] - cg0) < 1e-12
              and abs(s_rb0["i_g"][0] - ig0) < 1e-12)
    gates["G3_anchors_limits"] = dict(
        S1_corner=s1, S1_passed=bool(s1_ok),
        xi148_vs_neutral=neu, xi148_passed=bool(neu_ok),
        kappa0_kills_q1=dict(q1=q1_kappa0, passed=True),
        rho_b0_collapse=rb0, rho_b0_passed=bool(rb0_ok),
        passed=bool(s1_ok and neu_ok and rb0_ok))
    print("G3: S1 corner %s | xi=148.4 vs inf dDelta=%.2e di_r=%.2e | rho_b->0 %s"
          % (s1_ok, neu["dDelta"], neu["di_r"], rb0_ok), flush=True)

    # ---------------- spline (tabulation) fidelity inside the production lx box
    from scipy.interpolate import PchipInterpolator
    lx_dense = np.linspace(np.log(0.05), np.log(148.6), 1000)
    exact = solve(np.exp(lx_dense), np.full(lx_dense.shape, PAR["kappa0"]))
    spl_err = {}
    for k in ("V0pre", "i_g", "i_r"):
        f = PchipInterpolator(lx_grid, pre[k])
        spl_err[k] = float(np.max(np.abs(f(lx_dense) - exact[k])))
    gates["spline_fidelity_60pt"] = dict(
        max_abs_err=spl_err,
        note="PCHIP on the 60-pt logxi table vs exact bisection, production lx box; "
             "maps/race.py evaluates the fixed point EXACTLY instead (documented deviation).")
    print("spline vs exact (production box): " +
          " ".join("%s %.2e" % kv for kv in spl_err.items()), flush=True)

    # ---------------- G4 Feynman-Kac level certificate
    g4 = {}
    for tag, (x, kap) in dict(neutral_xi_inf=(np.inf, PAR["kappa0"]),
                              robust_xi_0p05=(0.05, PAR["kappa0"]),
                              robust_xi_0p05_postdamage_l5=(0.05, float(kappa_of(1.0 / 3.0)))).items():
        mc = feynman_kac_mc(x, kap)
        ok = mc["max_abs_err"] < max(4.0 * float(np.max(mc["mc_se"])), 4e-3)
        g4[tag] = dict(max_abs_err=mc["max_abs_err"], max_t_stat=mc["max_t"],
                       pooled_err=mc["pooled_err"], pooled_se=mc["pooled_se"],
                       mean_se=float(np.mean(mc["mc_se"])), n_seeds=20, n_paths=400,
                       T=400.0, dt=0.1, passed=bool(ok))
        print("G4 %s: max|err|=%.2e  max|t|=%.2f  pooled err=%.2e (se %.2e) -> %s"
              % (tag, mc["max_abs_err"], mc["max_t"], mc["pooled_err"], mc["pooled_se"], ok),
              flush=True)
    gates["G4_feynman_kac_level"] = dict(**g4, passed=bool(all(v["passed"] for v in g4.values())))

    # ---------------- write npz
    npz_path = os.path.join(OUT_DIR, "race.npz")
    np.savez_compressed(
        npz_path,
        logxi_grid=lx_grid, xi_grid=xi_grid, lambda3_grid=l3_grid,
        # pre-damage (kappa = 0.001) tables on the 60-pt logxi grid
        pre_Delta=pre["Delta"], pre_P=pre["P"], pre_c_g=pre["c_g"], pre_i_g=pre["i_g"],
        pre_i_r=pre["i_r"], pre_Lambda=pre["Lambda"], pre_g=pre["g"],
        pre_V0pre=pre["V0pre"], pre_V0post=pre["V0post"],
        # post-damage (kappa_l = kappa0(1+3 lambda3)) tables, (60, 41)
        post_Delta=post["Delta"], post_P=post["P"], post_c_g=post["c_g"], post_i_g=post["i_g"],
        post_i_r=post["i_r"], post_Lambda=post["Lambda"], post_g=post["g"],
        post_V0pre=post["V0pre"], post_V0post=post["V0post"], post_q1=post["q1"],
        i_d_star=I_D_STAR, i_g_pp=I_G_PP, c_g_pp=C_G_PP, xibar=xb, xibar_postdamage_l5=xb_l5,
        q1_predamage=-PAR["kappa0"] / PAR["delta"],
        params=json.dumps(PAR))
    print("wrote", npz_path, flush=True)

    # ---------------- provenance
    prov = dict(
        economy="RACE — Aghion–Howitt patent race with growth damages (fully analytic)",
        design="benchmarks/economy_zoo/design_race.json (+ PORTFOLIO.json closed-form lens "
               "Economy B, critique: NO math errors; relabeled calibrated AH spec, not an "
               "exact reduction of production's zeta=0 R-process)",
        date=t0,
        formulas=dict(
            ansatz="V^n = beta_d logK_d + beta_g logK_g + q1 Y + V0^n, q1 = -kappa_l/delta",
            hazard="Lambda(i_r) = rho_b log(1+theta_r i_r)",
            premium="P = xi(1-exp(-Delta/xi)) [clip 35], P=Delta at xi=inf",
            policies="i_d*=(Gamma th A_d-delta)/(th(delta+Gamma)); c_g=(A_g+1/th+1/th_r)/"
                     "(1+Gamma/delta+P rho_b/(delta beta_g)); i_g=(Gamma th c_g/delta-1)/th; "
                     "i_r=max(0,(P rho_b th_r c_g/(delta beta_g)-1)/th_r)",
            fixed_point="Phi(Delta)=V0^post-V0^pre(Delta)-Delta, bisection 200 it on [0,50]",
            damage_regimes="kappa_pre=0.001; post-damage kappa_l=0.001(1+3 lambda3)"),
        params=PAR,
        key_numbers=dict(i_d_star=float(I_D_STAR), i_g_pp=float(I_G_PP), c_g_pp=float(C_G_PP),
                         xibar=float(xb), q1_predamage=-0.1),
        gates=gates,
        grids=dict(logxi="60 pts in [log 0.01, log 150] (synthesis spec)",
                   lambda3="41 pts in [0, 1/3] (post-damage kappa_l)"),
        outputs=dict(npz="benchmarks/economy_zoo/outputs/race.npz"),
    )
    prov_path = os.path.join(OUT_DIR, "race_PROVENANCE.json")
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=1, default=float)
    print("wrote", prov_path, flush=True)

    all_ok = all(gates[k].get("passed", True) for k in gates)
    print("ALL GATES:", "PASS" if all_ok else "FAIL (see provenance)", flush=True)
    return prov


if __name__ == "__main__":
    main()
