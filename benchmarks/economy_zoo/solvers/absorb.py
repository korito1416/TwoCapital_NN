"""ABSORB economy solver (Nelson-Phelps directed-absorption catch-up).

Design: benchmarks/economy_zoo/design_absorb.json (synthesis corrections applied:
lambda0 = (lambda3/2)*yhat^2 in the post-damage climate constant; ladder tau = 1.25).

Value separates exactly (xi = infinity member):
    V = beta*logK_d + b_d + (1-beta)*logK_g + F(s) + w_reg(Y),   s = logR - logK_g.

Blocks
  (i)  dirty: algebraic          i_d* = (Gamma*theta*A_d - delta)/(theta*(delta+Gamma))
  (ii) green/knowledge: 1-D elliptic ODE in s on [-8, 7] (covers the production box
       image s in [-6.0, 6.61]), Howard iteration, first-order upwind drift + centered
       second derivative, Neumann (mirror) left BC, OUTFLOW right BC (backward drift
       difference + zero-curvature extrapolation ghost). NOTE deviation from the
       design's "Dirichlet anchor at closed-form F(+inf)": at the far right the drift
       is mu ~ -0.032 < 0 (knowledge dilution: s drifts LEFT), so the right boundary
       needs no condition; the true solution at finite s is F_static + C*e^{(delta/mu)s}
       and imposing F(+inf) at s_max=7 injects an O(0.18) error confined to a spurious
       boundary layer (observed F' -> 0.82 there). Instead the closed-form F(+inf) is
       used as a FAR-FIELD LIMIT CHECK: the measured tail decay rate of F - F_static
       must match the predicted delta/|mu_tail|. Controls are CLOSED FORM given F'
       (stable citardauq root of the c_g quadratic) -- no inner Newton needed; the
       F' < 1-beta = 0.66 feasibility gate is enforced by clip during iteration and
       VERIFIED unclipped at convergence.
  (iii) climate: post-damage closed-form quadratics w^l(Y) = -(a_l Y^2 + b_l Y + c_l);
       pre-damage linear 1-D BVP on Y in [0, 8] (extended past the production box so
       the huge jump intensity at Y=8 gives an algebraic Dirichlet anchor), direct
       sparse solve.

Gates (numbers reported in absorb_PROVENANCE.json):
  G2 F(s):   Howard convergence < 1e-12, discrete residual < 1e-10, max F' < 0.66,
             Richardson 401->801->1601, far-field anchor consistency.
  G3 w(Y):   discrete residual < 1e-10, Richardson 1601->3201,
             bounds w^(l=5) <= w_pre <= w^(l=1) (= no-jump quadratic) on [0, 4].
  G4 global: assembled full 4-state HJB residual at 1e4 random states (independent
             re-derivation of utility/drifts/jump sum), grid-consistent derivatives,
             <= 1e-10; off-grid cubic-spline residual reported as info.

Writes: ../outputs/absorb.npz, ../outputs/absorb_PROVENANCE.json
Run:    python absorb.py
"""
import json
import os
import datetime
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------- parameters
P = dict(
    delta=0.01, alpha=-0.035, Gamma=0.06, theta=16.7,
    sigma_d=0.01, sigma_g=0.01, sigma_r=0.0078,
    A_d=0.1303, A_g0=0.1085, A_gpp=0.1567,
    psi0=0.10583, psi1=0.5, zeta=0.0,
    beta=0.34,
    # ladder (corrected tau)
    tau=1.25, s0=float(np.log(11.2 / 616.0)),        # -4.0073 = log(R0/K_g0)
    # climate
    lam1=0.00017675, lam2=0.0044, yhat=2.5,
    r1=1.5, r2=0.36, y_lower=1.5, L=5,
    eta=0.291, Z0=0.7, K0=880.0, thetabar=1.86e-3, varsig=1.2 * 1.86e-3,
)
P["iota"] = P["eta"] * P["A_d"] * (1 - P["Z0"]) * P["K0"]         # 10.0102
P["s_m"] = P["s0"] + P["tau"] * np.log(19.0)                      # -0.3267
dl, Gm, th, bt = P["delta"], P["Gamma"], P["theta"], P["beta"]
sg, sr, ps0 = P["sigma_g"], P["sigma_r"], P["psi0"]
io, tb, vs = P["iota"], P["thetabar"], P["varsig"]
l1, l2, yh = P["lam1"], P["lam2"], P["yhat"]
ONE_B = 1.0 - bt
NU = 0.5 * (sg**2 + sr**2)                                        # s-diffusion coef

GATES = {}


def clip_exp(x):
    return np.exp(np.clip(x, -35.0, 35.0))


def phi(i):
    return P["alpha"] + Gm * np.log(1.0 + th * i)


def A_g(s):
    sig0 = 1.0 / 20.0                                             # logistic((s0-s_m)/tau)
    lo = 1.0 / (1.0 + clip_exp(-(np.asarray(s) - P["s_m"]) / P["tau"]))
    return P["A_g0"] + (P["A_gpp"] - P["A_g0"]) * (lo - sig0) / (1.0 - sig0)


# ------------------------------------------------------------- (i) dirty block
i_d_star = (Gm * th * P["A_d"] - dl) / (th * (dl + Gm))           # 0.103131
c_d = P["A_d"] - i_d_star
phi_d_star = phi(i_d_star)
b_d = bt * (np.log(c_d) + (phi_d_star - 0.5 * P["sigma_d"]**2) / dl)


def static_green(Ag):
    """Myopic green corner (F'=0): post-tech style closed forms at productivity Ag."""
    ig = (Gm * th * Ag - dl) / (th * (dl + Gm))
    c = Ag - ig
    return ig, c, ONE_B * (np.log(c) + (phi(ig) - 0.5 * sg**2) / dl)


i_g_pp, c_g_pp, F_inf = static_green(P["A_gpp"])                  # 0.12576, 0.03094


# ----------------------------------------------- (ii) green/knowledge F(s) ODE
def controls_of(s, p):
    """Closed-form (c_g, i_g, i_r) given clipped p = F'(s) >= 0. Stable root."""
    Ag = A_g(s)
    B = ps0 * clip_exp(-0.5 * s) * p / (2.0 * dl * ONE_B)
    M = 1.0 + Gm * (ONE_B - p) / (dl * ONE_B)
    A1 = Ag + 1.0 / th
    c = 2.0 * A1 / (M + np.sqrt(M**2 + 4.0 * B**2 * A1))          # citardauq
    i_g = Gm * (ONE_B - p) * c / (dl * ONE_B) - 1.0 / th
    i_r = (B * c) ** 2
    return c, i_g, i_r


def centered_grad(F, h):
    p = np.empty_like(F)
    p[1:-1] = (F[2:] - F[:-2]) / (2 * h)
    p[0] = (F[1] - F[0]) / h
    p[-1] = (F[-1] - F[-2]) / h
    return p


def solve_F(N, smin=-8.0, smax=7.0, tol=1e-12, itmax=300, pclip=0.6599):
    s = np.linspace(smin, smax, N)
    h = s[1] - s[0]
    _, _, F = static_green(A_g(s))                                # init: myopic value
    conv = np.inf
    for it in range(itmax):
        p_raw = centered_grad(F, h)
        p = np.clip(p_raw, 0.0, pclip)
        c, ig, ir = controls_of(s, p)
        u = dl * ONE_B * np.log(c) + ONE_B * (phi(ig) - 0.5 * sg**2)
        mu = ps0 * np.sqrt(ir) * clip_exp(-0.5 * s) - 0.5 * sr**2 - phi(ig) + 0.5 * sg**2
        mup, mum = np.maximum(mu, 0.0), np.maximum(-mu, 0.0)
        lo = mum / h + NU / h**2                                  # coef of F_{i-1}
        up = mup / h + NU / h**2                                  # coef of F_{i+1}
        di = dl + mup / h + mum / h + 2.0 * NU / h**2
        rhs = u.copy()
        main, upper, lower = di.copy(), up.copy(), lo.copy()
        # left mirror ghost F_{-1} = F_1: fold lower coef into upper
        upper[0] = up[0] + lo[0]
        # right OUTFLOW: backward drift diff + zero-curvature ghost (no diffusion row)
        main[N - 1] = dl + (mup[N - 1] + mum[N - 1]) / h
        lower[N - 1] = (mup[N - 1] + mum[N - 1]) / h              # F' ~ (F_N-1 - F_N-2)/h
        upper[N - 1] = 0.0
        Amat = diags([-lower[1:], main, -upper[:-1]], [-1, 0, 1], format="csc")
        Fn = spsolve(Amat, rhs)
        conv = np.max(np.abs(Fn - F))
        F = Fn
        if conv < tol:
            break
    # final diagnostics with converged controls
    p_raw = centered_grad(F, h)
    p = np.clip(p_raw, 0.0, pclip)
    c, ig, ir = controls_of(s, p)
    u = dl * ONE_B * np.log(c) + ONE_B * (phi(ig) - 0.5 * sg**2)
    mu = ps0 * np.sqrt(ir) * clip_exp(-0.5 * s) - 0.5 * sr**2 - phi(ig) + 0.5 * sg**2
    res = discrete_residual_F(s, F, u, mu)
    return dict(s=s, h=h, F=F, Fp=p_raw, c=c, i_g=ig, i_r=ir, u=u, mu=mu,
                conv=conv, iters=it + 1, resid=res)


def discrete_residual_F(s, F, u, mu):
    """delta*F - u - mu*F'_upwind - nu*F'' with mirror-left/outflow-right BCs."""
    N = len(s)
    h = s[1] - s[0]
    Fm1 = np.r_[F[1], F[:-1]]                                     # mirror ghost left
    Fp1 = np.r_[F[1:], 2 * F[-1] - F[-2]]                         # extrapolation ghost
    mup, mum = np.maximum(mu, 0.0), np.maximum(-mu, 0.0)
    res = (dl * F - u - mup * (Fp1 - F) / h + mum * (F - Fm1) / h
           - NU * (Fp1 - 2 * F + Fm1) / h**2)
    # last row (outflow, requires mu<0 there): backward difference, no diffusion
    assert mu[-1] < 0, "right boundary is not outflow (mu >= 0) - BC invalid"
    res[-1] = dl * F[-1] - u[-1] - mu[-1] * (F[-1] - F[-2]) / h
    return res


print("=" * 78)
print("(ii) F(s) Howard solve, Richardson 401 -> 801 -> 1601")
sols = {N: solve_F(N) for N in (401, 801, 1601)}
for N, S in sols.items():
    print(f"  N={N:5d}: iters={S['iters']:3d} conv={S['conv']:.2e} "
          f"max|resid|={np.abs(S['resid'][:-1]).max():.2e} "
          f"maxF'={S['Fp'].max():.4f} F(s0)={np.interp(P['s0'], S['s'], S['F']):.6f}")
S = sols[1601]
rich_1 = np.abs(np.interp(sols[801]["s"], sols[401]["s"], sols[401]["F"]) - sols[801]["F"]).max()
rich_2 = np.abs(np.interp(sols[1601]["s"], sols[801]["s"], sols[801]["F"]) - sols[1601]["F"]).max()
maxFp = float(S["Fp"].max())
clip_active = bool((S["Fp"] > 0.6599).any() or (S["Fp"] < -1e-10).any())
# far-field limit check: F - F_static must decay toward F(+inf) at rate delta/mu_tail
tail = (S["s"] >= 5.5) & (S["s"] <= 6.5)
D = np.array([static_green(A_g(x))[2] for x in S["s"][tail]]) - S["F"][tail]
slope_meas = float(np.polyfit(S["s"][tail], np.log(D), 1)[0])
slope_pred = float(dl / S["mu"][tail].mean())                     # mu<0 -> negative slope
tail_rel_err = abs(slope_meas - slope_pred) / abs(slope_pred)
GATES["G2_F_ode"] = dict(
    howard_conv=float(S["conv"]), discrete_residual_max=float(np.abs(S["resid"]).max()),
    maxFprime=maxFp, Fprime_gate_066=bool(maxFp < 0.66), clip_active_at_convergence=clip_active,
    richardson_401_801=float(rich_1), richardson_801_1601=float(rich_2),
    richardson_rate=float(rich_1 / rich_2),
    tail_decay_slope_measured=slope_meas, tail_decay_slope_predicted=slope_pred,
    tail_decay_rel_err=float(tail_rel_err),
    F_smax_minus_F_inf=float(S["F"][-1] - F_inf),                 # exponential mode, info
    i_r_nonneg=bool((S["i_r"] >= 0).all()), c_g_positive=bool((S["c"] > 0).all()),
    feas_1_plus_theta_ig=bool((1 + th * S["i_g"] > 0).all()),
)
g2_pass = (S["conv"] < 1e-12 and np.abs(S["resid"]).max() < 1e-10
           and maxFp < 0.66 and not clip_active and tail_rel_err < 0.15)
GATES["G2_F_ode"]["PASS"] = bool(g2_pass)
print(f"  G2: conv={S['conv']:.2e} resid={np.abs(S['resid']).max():.2e} "
      f"maxF'={maxFp:.4f}<0.66 rich(801->1601)={rich_2:.2e} rate={rich_1/rich_2:.2f}")
print(f"      tail decay slope meas={slope_meas:.4f} vs pred delta/mu={slope_pred:.4f} "
      f"(rel err {tail_rel_err:.3f}) -> {'PASS' if g2_pass else 'FAIL'}")

# ------------------------------------------------ (iii) climate: post-damage CF
def w_post_coef(lam3):
    lam3 = np.asarray(lam3, dtype=float)
    a = (l2 + lam3) / 2.0
    b = (l1 - lam3 * yh) + 2.0 * a * io * tb / dl
    c = lam3 / 2.0 * yh**2 + (b * io * tb + a * io**2 * vs**2) / dl
    return a, b, c


def w_post(Y, lam3):
    a, b, c = w_post_coef(lam3)
    return -(a * np.asarray(Y)**2 + b * np.asarray(Y) + c)


LAM3_L = np.array([(1.0 / 3.0) * (l - 1) / 4.0 for l in range(1, 6)])  # l = 1..5


def J_n(Y):
    Y = np.asarray(Y, dtype=float)
    return np.where(Y >= P["y_lower"],
                    P["r1"] * (clip_exp(P["r2"] / 2 * (Y - P["y_lower"])**2) - 1.0), 0.0)


# --------------------------------------------- (iii) climate: pre-damage 1-D BVP
def solve_w_pre(N, ymin=0.0, ymax=8.0):
    Y = np.linspace(ymin, ymax, N)
    h = Y[1] - Y[0]
    drift = io * tb                                               # > 0 constant
    nu = 0.5 * io**2 * vs**2
    Jtot = J_n(Y)                                                 # sum_l (1/L) J = J_n
    wl = np.array([w_post(Y, lam3) for lam3 in LAM3_L])           # (5, N)
    jump_src = (Jtot / P["L"]) * wl.sum(axis=0)
    f = -dl * (l1 * Y + 0.5 * l2 * Y**2) + jump_src
    # right Dirichlet: algebraic balance at ymax (J ~ 3e3/yr dominates derivatives)
    w_right = (-dl * (l1 * ymax + 0.5 * l2 * ymax**2)
               + (J_n(ymax) / P["L"]) * sum(w_post(ymax, lam3) for lam3 in LAM3_L)
               ) / (dl + J_n(ymax))
    lo = np.full(N, nu / h**2)                                    # drift>0: forward upwind
    up = drift / h + nu / h**2
    di = dl + Jtot + drift / h + 2.0 * nu / h**2
    rhs = f.copy()
    main, upper, lower = di.copy(), np.full(N, up), lo.copy()
    # left BC: linear-extrapolation ghost w_{-1} = 2w_0 - w_1  (w''(0)=0 -> drop diffusion)
    main[0] = dl + Jtot[0] + drift / h + 2.0 * nu / h**2 - 2.0 * nu / h**2
    upper[0] = drift / h + nu / h**2 - nu / h**2
    rhs[N - 2] += up * w_right
    Amat = diags([-lower[1:N - 1], main[:N - 1], -upper[:N - 2]], [-1, 0, 1], format="csc")
    w = np.empty(N)
    w[:N - 1] = spsolve(Amat, rhs[:N - 1])
    w[N - 1] = w_right
    # discrete residual
    wm1 = np.r_[2 * w[0] - w[1], w[:-1]]
    wp1 = np.r_[w[1:], np.nan]
    res = (dl * w + Jtot * w - f - drift * (wp1 - w) / h - nu * (wp1 - 2 * w + wm1) / h**2)
    res[-1] = 0.0
    return dict(Y=Y, h=h, w=w, resid=res, w_right=w_right,
                wp=centered_grad(w, h), Jtot=Jtot)


print("(iii) pre-damage climate BVP, Richardson 1601 -> 3201 on Y in [0,8]")
wsols = {N: solve_w_pre(N) for N in (1601, 3201)}
W = wsols[3201]
rich_w = np.abs(np.interp(W["Y"], wsols[1601]["Y"], wsols[1601]["w"]) - W["w"]).max()
mask4 = W["Y"] <= 4.0 + 1e-12
w1 = w_post(W["Y"], LAM3_L[0])                                    # = no-jump quadratic
w5 = w_post(W["Y"], LAM3_L[4])
bounds_ok = bool((W["w"][mask4] <= w1[mask4] + 1e-10).all()
                 and (W["w"][mask4] >= w5[mask4] - 1e-10).all())
res_w = float(np.abs(W["resid"][:-1]).max())
GATES["G3_w_pre"] = dict(
    discrete_residual_max=res_w, richardson_1601_3201=float(rich_w),
    bounds_w5_le_wpre_le_w1_on_0_4=bounds_ok,
    w_pre_at_Y0=float(np.interp(0.0, W["Y"], W["w"])),
    w_pre_at_1p1=float(np.interp(1.1, W["Y"], W["w"])),
    w_right_anchor=float(W["w_right"]),
)
g3_pass = res_w < 1e-10 and bounds_ok
GATES["G3_w_pre"]["PASS"] = bool(g3_pass)
print(f"  G3: resid={res_w:.2e} rich={rich_w:.2e} bounds={bounds_ok} "
      f"w_pre(1.1)={GATES['G3_w_pre']['w_pre_at_1p1']:.6f} -> {'PASS' if g3_pass else 'FAIL'}")

# --------------------------------------------------------- (iv) G4 global check
print("(iv) G4 global assembled 4-state HJB residual (independent re-derivation)")
rng = np.random.RandomState(7)
n4 = 10000
# random states; s and Y snapped to grid nodes so discrete derivatives are exact
lkd = rng.uniform(2, 8, n4)
lkg = rng.uniform(3, 7, n4)
si = rng.randint(1, len(S["s"]) - 1, n4)                          # interior s nodes
yi = rng.randint(1, int(4.0 / W["h"]), n4)                        # Y nodes in (0,4)
s_pt, Y_pt = S["s"][si], W["Y"][yi]
lr = s_pt + lkg

# value and derivatives (grid-consistent: upwind F', discrete second differences)
h_s, h_y = S["h"], W["h"]
F_, u_, mu_ = S["F"], S["u"], S["mu"]
Fm1 = np.r_[F_[1], F_[:-1]]
Fp1 = np.r_[F_[1:], np.nan]
Fpp_d = (Fp1 - 2 * F_ + Fm1) / h_s**2
Fp_up = np.where(mu_ >= 0, (Fp1 - F_) / h_s, (F_ - Fm1) / h_s)
w_ = W["w"]
wm1 = np.r_[2 * w_[0] - w_[1], w_[:-1]]
wp1 = np.r_[w_[1:], np.nan]
wpp_d = (wp1 - 2 * w_ + wm1) / h_y**2
wp_up = (wp1 - w_) / h_y                                          # drift>0 forward

V_pt = bt * lkd + b_d + ONE_B * lkg + F_[si] + w_[yi]
cg_pt, ig_pt, ir_pt = S["c"][si], S["i_g"][si], S["i_r"][si]
util = (dl * bt * (np.log(P["A_d"] - i_d_star) + lkd)
        + dl * ONE_B * (np.log(cg_pt) + lkg)
        - dl * (l1 * Y_pt + 0.5 * l2 * Y_pt**2))
drift_terms = (bt * (phi(i_d_star) - 0.5 * P["sigma_d"]**2)                       # V_lkd
               + (ONE_B - Fp_up[si]) * 0.0                                        # (see below)
               )
# logK_g drift enters via V_lkg = (1-beta) - F' ; logR drift via V_lr = F'
mu_kg = phi(ig_pt) - 0.5 * sg**2
mu_r = ps0 * np.sqrt(ir_pt) * clip_exp(-0.5 * s_pt) - 0.5 * sr**2
drift_terms = (bt * (phi(i_d_star) - 0.5 * P["sigma_d"]**2)
               + ONE_B * mu_kg
               + Fp_up[si] * (mu_r - mu_kg)                       # = F'*(ds drift)
               + wp_up[yi] * io * tb)
second_terms = (0.5 * sg**2 * 0.0 + NU * Fpp_d[si]                # F'' from s-diffusion
                + 0.5 * io**2 * vs**2 * wpp_d[yi])
jump_term = (J_n(Y_pt) / P["L"]) * sum(w_post(Y_pt, lam3) - w_[yi] for lam3 in LAM3_L)
res_glob = -dl * V_pt + util + drift_terms + second_terms + jump_term
res_glob_max = float(np.abs(res_glob).max())

# post-damage regimes: climate part analytic (residual 0 exactly), green block at nodes
l3r = rng.uniform(0, 1 / 3, n4)
Ypd = rng.uniform(2.5, 6, n4)
a_, b_, c_ = w_post_coef(l3r)
w_pd = -(a_ * Ypd**2 + b_ * Ypd + c_)
lt0, lt1, lt2 = l3r / 2 * yh**2, l1 - l3r * yh, l2 + l3r
res_pd_climate = (-dl * w_pd - dl * (lt0 + lt1 * Ypd + lt2 / 2 * Ypd**2)
                  + io * tb * (-(2 * a_ * Ypd + b_)) + 0.5 * io**2 * vs**2 * (-2 * a_))
V_pd = bt * lkd + b_d + ONE_B * lkg + F_[si] + w_pd
res_pd = (-dl * V_pd
          + dl * bt * (np.log(P["A_d"] - i_d_star) + lkd)
          + dl * ONE_B * (np.log(cg_pt) + lkg)
          - dl * (lt0 + lt1 * Ypd + lt2 / 2 * Ypd**2)
          + bt * (phi(i_d_star) - 0.5 * P["sigma_d"]**2)
          + ONE_B * mu_kg + Fp_up[si] * (mu_r - mu_kg)
          + io * tb * (-(2 * a_ * Ypd + b_)) + NU * Fpp_d[si]
          + 0.5 * io**2 * vs**2 * (-2 * a_))
res_pd_max = float(np.abs(res_pd).max())

# off-grid spline residual (info only): interpolation-limited, honest number
FS, wS = CubicSpline(S["s"], S["F"]), CubicSpline(W["Y"], W["w"])
s_r = rng.uniform(-7.5, 6.5, n4)
Y_r = rng.uniform(0.05, 3.95, n4)
pr = np.clip(FS(s_r, 1), 0.0, 0.6599)
cr, igr, irr = controls_of(s_r, pr)
mu_kg_r = phi(igr) - 0.5 * sg**2
mu_r_r = ps0 * np.sqrt(irr) * clip_exp(-0.5 * s_r) - 0.5 * sr**2
res_off = (-dl * (bt * lkd + b_d + ONE_B * lkg + FS(s_r) + wS(Y_r))
           + dl * bt * (np.log(P["A_d"] - i_d_star) + lkd)
           + dl * ONE_B * (np.log(cr) + lkg)
           - dl * (l1 * Y_r + 0.5 * l2 * Y_r**2)
           + bt * (phi(i_d_star) - 0.5 * P["sigma_d"]**2)
           + ONE_B * mu_kg_r + FS(s_r, 1) * (mu_r_r - mu_kg_r)
           + wS(Y_r, 1) * io * tb
           + NU * FS(s_r, 2) + 0.5 * io**2 * vs**2 * wS(Y_r, 2)
           + (J_n(Y_r) / P["L"]) * sum(w_post(Y_r, lam3) - wS(Y_r) for lam3 in LAM3_L))
GATES["G4_global"] = dict(
    predamage_assembled_max=res_glob_max,
    postdamage_assembled_max=res_pd_max,
    postdamage_climate_analytic_max=float(np.abs(res_pd_climate).max()),
    offgrid_spline_median=float(np.median(np.abs(res_off))),
    offgrid_spline_max=float(np.abs(res_off).max()),
    n_points=n4,
)
g4_pass = res_glob_max <= 1e-10 and res_pd_max <= 1e-10
GATES["G4_global"]["PASS"] = bool(g4_pass)
print(f"  G4: pre-damage assembled max={res_glob_max:.2e}, post-damage max={res_pd_max:.2e}"
      f" (climate-analytic {np.abs(res_pd_climate).max():.2e}) -> {'PASS' if g4_pass else 'FAIL'}")
print(f"      off-grid spline residual (info): median={np.median(np.abs(res_off)):.2e} "
      f"max={np.abs(res_off).max():.2e}")

# ------------------------------------------------------------- reference point
s0 = P["s0"]
F_s0 = float(np.interp(s0, S["s"], S["F"]))
Fp_s0 = float(np.interp(s0, S["s"], S["Fp"]))
c_s0, ig_s0, ir_s0 = (float(np.interp(s0, S["s"], S[k])) for k in ("c", "i_g", "i_r"))
w_pre_11 = float(np.interp(1.1, W["Y"], W["w"]))
lk0, Z0v, lr0 = np.log(880.0), 0.7, np.log(11.2)
v_x0 = (lk0 + bt * np.log(1 - Z0v) + ONE_B * np.log(Z0v) + b_d + F_s0 + w_pre_11)
KEY = dict(i_d_star=float(i_d_star), c_d=float(c_d), b_d=float(b_d),
           i_g_posttech=float(i_g_pp), F_inf=float(F_inf),
           s0=float(s0), F_s0=F_s0, Fprime_s0=Fp_s0,
           i_g_econ_s0=ig_s0, i_r_econ_s0=ir_s0, c_g_s0=c_s0,
           i_r_prod_s0_Z07=float(0.7 * ir_s0),
           w_pre_Y1p1=w_pre_11, v_at_production_x0=float(v_x0),
           maxFprime=maxFp)
print("reference point (production x0: K=880, Z=0.7, Y=1.1, R=11.2 -> s0=-4.007):")
for k, v in KEY.items():
    print(f"    {k:22s} = {v:.6f}")

# ---------------------------------------------------------------------- outputs
npz_path = os.path.join(OUT, "absorb.npz")
np.savez_compressed(
    npz_path,
    s_grid=S["s"], F=S["F"], Fp=S["Fp"], i_g_s=S["i_g"], i_r_s=S["i_r"],
    c_g_s=S["c"], A_g_s=A_g(S["s"]), mu_s=S["mu"],
    Y_grid=W["Y"], w_pre=W["w"], w_pre_p=W["wp"],
    lam3_levels=LAM3_L,
    w_post_a=w_post_coef(LAM3_L)[0], w_post_b=w_post_coef(LAM3_L)[1],
    w_post_c=w_post_coef(LAM3_L)[2],
    scalars=np.array([bt, float(b_d), float(i_d_star), float(F_inf), float(i_g_pp),
                      float(c_d)]),
    scalar_names=np.array(["beta", "b_d", "i_d_star", "F_inf", "i_g_posttech", "c_d"]),
)
prov = dict(
    economy="ABSORB — Nelson-Phelps directed-absorption catch-up (design_absorb.json)",
    date=datetime.date.today().isoformat(),
    value_decomposition="V = beta*logK_d + b_d + (1-beta)*logK_g + F(s) + w_reg(Y), "
                        "s = logR - logK_g, xi = infinity (neutral member)",
    corrections_applied=["lambda0 = (lambda3/2)*yhat^2 in post-damage constant c_l",
                         "ladder tau = 1.25 (s_m = s0 + tau*ln 19 = -0.3267)",
                         "F grid widened to s in [-8, 7] (production box image)",
                         "stable citardauq root for the c_g quadratic"],
    params=P, key_numbers=KEY, gates=GATES,
    method=dict(F_ode="Howard, upwind-1st drift + centered 2nd, mirror-Neumann left, "
                      "OUTFLOW right BC (mu<0 there; design's Dirichlet F(+inf) anchor "
                      "replaced by a tail-decay-rate limit check, see gates); "
                      "closed-form controls given F' (no inner Newton: c_g quadratic "
                      "solved exactly)",
                w_pre="linear tridiagonal BVP, Y in [0,8], forward-upwind drift, "
                      "Dirichlet right = algebraic jump balance at Y=8",
                grids=dict(F_nodes=1601, w_nodes=3201)),
    files=dict(npz=os.path.abspath(npz_path)),
)
with open(os.path.join(OUT, "absorb_PROVENANCE.json"), "w") as fjson:
    json.dump(prov, fjson, indent=1, default=float)
print(f"wrote {npz_path}")
print(f"wrote {os.path.join(OUT, 'absorb_PROVENANCE.json')}")
allpass = g2_pass and g3_pass and g4_pass
print(f"OVERALL: G2 {'PASS' if g2_pass else 'FAIL'} | G3 {'PASS' if g3_pass else 'FAIL'} "
      f"| G4 {'PASS' if g4_pass else 'FAIL'}")
if not allpass:
    raise SystemExit(1)
