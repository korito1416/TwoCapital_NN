#!/usr/bin/env python
"""GHKM economy solver (economy zoo, build 3).

Economy: fossil-flow emissions, dirty-productivity damages, knowledge-in-production
(design_ghkm.json; PORTFOLIO.json climate-coupling lens, economy 2; the synthesis
ADOPTED the f(Y)-ODE robust damage-jump route and REJECTED the affine-hazard variant).

Pipeline (all closed-form or 1-D):
  1. linear coefficient recursion (a_d, a_g, a_r, m, p^l, shares)  [G1-verified]
  2. calibration: gamma0 = 0, kappa = 0.25, F0 = pre-damage xi=oo fossil choice,
     chi_F = K_d0/F0, chi = K_g0/R0, alpha0 matches production phi at i = 0.09
  3. xi = oo linear chain (p_1 = p_2, all four c-constants)  -> Newton inits
  4. per xi on a 25-node log-xi grid in [0.05, 148.6]:
       Delta_c fixed point (brentq), post-damage constants c^l(xi),
       PreDamagePostTech u_2(Y; xi) then PreDamagePreTech u_1(Y; xi)
       via damped-Newton collocation on 401 Y-nodes in [0, 4]
  5. gates: Newton residual <= 1e-10, 401 -> 1601 refinement, xi = 148.6 vs
     neutral O(1/xi), interiority + budget, comparative statics, exp-audit,
     Feynman-Kac Monte Carlo level certificate (xi = oo, terminal + full chain)
  6. writes outputs/ghkm.npz + outputs/ghkm_PROVENANCE.json

Value convention: u_reg(Y; xi) = f_reg(Y; xi) + c_reg(xi) is the entire
Y-plus-constant part of V; f(0) = 0 normalization is implicit (c_reg = u_reg(0)).

Run:  python ghkm.py
"""
import json
import os
# pin BLAS to one thread BEFORE importing numpy: threaded MKL busy-spins on the
# shared login node and turns a 401x401 solve into ~10 s (measured 2026-07-19)
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import time
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline, PchipInterpolator

T0 = time.time()
def tick(msg):
    print('[%7.1fs] %s' % (time.time() - T0, msg), flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

# ============================================================ parameters
dl = 0.01          # delta
td, tg = 0.34, 0.66          # theta_d, theta_g (Cobb-Douglas utility weights)
be, nn = 0.3, 0.3            # beta (knowledge share), nu (fossil share)
G = 0.06                     # Gamma_d = Gamma_g = Gamma_r
thc = 16.7                   # theta_c (log-log adjustment scale = production theta)
sgd, sgg, sgr = 0.01, 0.01, 0.0078
Ad, Ag, Agpp = 0.1303, 0.1085, 0.1567
gamma0, gamma1, kappa = 0.0, 0.15, 0.25
thetabar = 1.86e-3
varsigma = 1.2*1.86e-3
eta0 = 0.291
b = thetabar*eta0            # Y drift loading on F
w = varsigma*eta0            # Y vol loading on F
K0, Z0, Y0, R0 = 880.0, 0.7, 1.1, 11.2
varrho = 746.67
r1, r2, ylow = 1.5, 0.36, 1.5
varpi_n = r1*(np.exp(0.5*r2*(2.0-ylow)**2) - 1.0)     # damage hazard J_n(2.0)
varpi_g = R0/varrho                                    # tech hazard
lam3 = np.array([0., 1/12., 1/6., 1/4., 1/3.])
gam1l = gamma1 + kappa*lam3
# alpha0: log-log drift matches production phi = -0.035 + 0.06 log(1+16.7 i) at i=0.09
alpha0 = -0.035 + 0.06*(np.log(1 + thc*0.09) - np.log(thc*0.09))

# ============================================================ derived coefficients (G1)
aD = dl*td*(1-nn)/(dl + G*nn)
Dden = dl + G*be + G*(1-be)
aG = tg*(1-be)*(dl+G)/Dden
aR = tg*be*(dl+G)/Dden
m = td*(G+dl)/(dl + G*nn)
sd = G*(1-nn)/(G+dl)
sg = G*(1-be)/(G+dl)
sr = G*be/(G+dl)
p_l = -gam1l*m
gapstar = np.log(sg/sr)      # post-tech logK_g - logR attractor (= log((1-be)/be))
tick('coefficients: a_d=%.6f a_g=%.6f a_r=%.6f (V_logK=%.4f) m=%.6f' %
     (aD, aG, aR, aD+aG, m))
tick('shares: s_d=%.6f s_g=%.6f s_r=%.6f (1-s_g-s_r=%.6f), gap*=%.6f' %
     (sd, sg, sr, 1-sg-sr, gapstar))

# ============================================================ calibration
pbar = float(np.mean(p_l))
p2inf = (-dl*m*gamma1 + varpi_n*pbar)/(dl + varpi_n)
p1inf = (-dl*m*gamma1 + varpi_n*pbar + varpi_g*p2inf)/(dl + varpi_n + varpi_g)
assert abs(p1inf - p2inf) < 1e-15
F0 = nn*dl*m/(-p2inf*b)      # pre-damage xi=oo fossil choice = calibration anchor
chi = K0*Z0/R0               # = 55.0
chiF = K0*(1-Z0)/F0
Fl_inf = nn*dl/(gam1l*thetabar*eta0)      # post-damage xi=oo fossil per l
tick('calibration: p_pre(inf)=%.6f F0=%.4f chi=%.3f chi_F=%.4f alpha0=%.6f' %
     (p1inf, F0, chi, chiF, alpha0))
tick('post-damage F^l(inf) = %s  (mu_Y^l = %s)' %
     (np.array2string(Fl_inf, precision=3), np.array2string(b*Fl_inf, precision=5)))

# ============================================================ constant blocks
lthc = np.log(thc)
B_d = (dl*td*(np.log(1-sd) + np.log(Ad) - gamma0)
       + aD*(alpha0 + G*(lthc + np.log(sd) + np.log(Ad) - gamma0)))

def Gg_const(A):
    out = dl*tg*(np.log(1-sg-sr) + np.log(A) + be*np.log(chi))
    out += aG*(alpha0 + G*(lthc + np.log(sg) + np.log(A) + be*np.log(chi)))
    out += aR*(alpha0 + G*(lthc + np.log(sr) + np.log(A) + be*np.log(chi)))
    return out

GgA, GgApp = Gg_const(Ag), Gg_const(Agpp)
DGg = (dl*tg + aG*G + aR*G)*np.log(Agpp/Ag)
assert abs((GgApp - GgA) - DGg) < 1e-14

def Omega(xi):
    """Brownian misspecification drag (0 at xi = inf; scalar or array xi)."""
    return -(aD**2*sgd**2 + aG**2*sgg**2 + aR**2*sgr**2)/(2.0*np.asarray(xi, float))

def Fpos(q, xi):
    """Positive root of the robust fossil FOC quadratic (stable as xi -> oo;
    scalar or array xi, xi = inf handled by Aq -> 0)."""
    q = np.asarray(q, dtype=float)
    Bq = -q*b
    Cq = -nn*dl*m
    Aq = q**2*w**2/np.asarray(xi, float)
    return -2.0*Cq/(Bq + np.sqrt(Bq**2 - 4.0*Aq*Cq))

def Gd_const(g1, xi):
    p = -np.asarray(g1)*m
    F = Fpos(p, xi)
    return (B_d + nn*dl*m*np.log(chiF*F) + p*b*F
            - (w**2*F**2/(2.0*np.asarray(xi, float)))*p**2)

def Delta_c(xi):
    if np.isinf(xi):
        return DGg/(dl + varpi_g)
    f = lambda D: dl*D - DGg + xi*varpi_g*(1.0 - np.exp(-min(D/xi, 35.0)))
    return brentq(f, 0.0, DGg/dl + 1.0, xtol=1e-15, rtol=8.9e-16)

def c_pp(g1, xi):            # PostDamagePostTech constant
    return (Gd_const(g1, xi) + GgApp + Omega(xi))/dl

def c_pre(g1, xi):           # PostDamagePreTech constant
    return c_pp(g1, xi) - Delta_c(xi)

# xi = oo linear chain constants (Newton inits + FK targets)
Dc_inf = Delta_c(np.inf)
c_pp_inf = c_pp(gam1l, np.inf)
c_pre_inf = c_pp_inf - Dc_inf
F2inf = F1inf = Fpos(p2inf, np.inf)
c2_inf = (B_d + GgApp + nn*dl*m*(np.log(chiF*F2inf) - 1.0)
          + varpi_n*np.mean(c_pp_inf))/(dl + varpi_n)
c1_inf = (B_d + GgA + nn*dl*m*(np.log(chiF*F1inf) - 1.0)
          + varpi_n*np.mean(c_pre_inf) + varpi_g*c2_inf)/(dl + varpi_n + varpi_g)
tick('xi=oo chain: Dc=%.6f c2=%.6f c1=%.6f c_pp=%s' %
     (Dc_inf, c2_inf, c1_inf, np.array2string(c_pp_inf, precision=4)))

# ============================================================ pre-damage ODE machinery
def derivs(u, h):
    q = np.empty_like(u)
    q[1:-1] = (u[2:] - u[:-2])/(2*h)
    q[0] = (-3*u[0] + 4*u[1] - u[2])/(2*h)
    q[-1] = (3*u[-1] - 4*u[-2] + u[-3])/(2*h)
    upp = np.empty_like(u)
    upp[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/h**2
    upp[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3])/h**2
    upp[-1] = (2*u[-1] - 5*u[-2] + 4*u[-3] - u[-4])/h**2
    return q, upp

EXPCLIP = 35.0
def ode_residual(u, Y, h, xi, Kc, cl, u2=None, track=None):
    """Residual of the pre-damage HJB ODE for u(Y) = f(Y) + c."""
    q, upp = derivs(u, h)
    qe = np.minimum(q, -1e-6)                     # slope guard (must be inactive)
    F = Fpos(qe, xi)
    R = (-dl*u - dl*m*gamma1*Y + Kc + nn*dl*m*np.log(chiF*F)
         + qe*b*F + 0.5*w**2*F**2*upp)
    if np.isinf(xi):
        gap = p_l[None, :]*Y[:, None] + cl[None, :] - u[:, None]
        R += varpi_n*np.mean(gap, axis=1)
        if u2 is not None:
            R += varpi_g*(u2 - u)
    else:
        R -= (w**2*F**2/(2.0*xi))*qe**2
        ex = (u[:, None] - (p_l[None, :]*Y[:, None] + cl[None, :]))/xi   # -(V^l-V)/xi
        if track is not None:
            track['max_exp'] = max(track.get('max_exp', 0.0), float(np.max(np.abs(ex))))
            track['guard'] = track.get('guard', False) or bool(np.any(q > -1e-6))
        R += xi*varpi_n*np.mean(1.0 - np.exp(np.clip(ex, -EXPCLIP, EXPCLIP)), axis=1)
        if u2 is not None:
            ex2 = (u - u2)/xi
            if track is not None:
                track['max_exp'] = max(track['max_exp'], float(np.max(np.abs(ex2))))
            R += xi*varpi_g*(1.0 - np.exp(np.clip(ex2, -EXPCLIP, EXPCLIP)))
    return R

def diff_matrices(Y):
    """Dense first/second-derivative collocation matrices (2nd-order interior,
    one-sided at the two boundary rows)."""
    n = len(Y); h = Y[1] - Y[0]
    Dq = np.zeros((n, n)); Dp = np.zeros((n, n))
    for i in range(1, n-1):
        Dq[i, i-1], Dq[i, i+1] = -1/(2*h), 1/(2*h)
        Dp[i, i-1], Dp[i, i], Dp[i, i+1] = 1/h**2, -2/h**2, 1/h**2
    Dq[0, 0:3] = np.array([-3., 4., -1.])/(2*h)
    Dq[-1, -3:] = np.array([1., -4., 3.])/(2*h)
    Dp[0, 0:4] = np.array([2., -5., 4., -1.])/h**2
    Dp[-1, -4:] = np.array([-1., 4., -5., 2.])/h**2
    return Dq, Dp

def ode_jacobian(u, Y, Dq, Dp, xi, cl, u2=None):
    """Exact Jacobian of ode_residual. Envelope: the fossil-FOC bracket vanishes
    at F = F(q), leaving dT/dq = b F - (w^2F^2 q/xi) + w^2 F F'(q) upp."""
    h = Y[1] - Y[0]
    q, upp = derivs(u, h)
    qe = np.minimum(q, -1e-6)
    F = Fpos(qe, xi)
    mask = (q < -1e-6).astype(float)              # fossil terms flat where guard binds
    if np.isinf(xi):
        Fp = b*F**2/(nn*dl*m)
        A = (b*F + w**2*F*Fp*upp)*mask
        C = -dl - varpi_n - (varpi_g if u2 is not None else 0.0)
        C = np.full_like(u, C)
    else:
        Fp = (b - 2*qe*w**2*F/xi)/(nn*dl*m/F**2 + qe**2*w**2/xi)
        A = (b*F - (w**2*F**2*qe)/xi + w**2*F*Fp*upp)*mask
        ex = (u[:, None] - (p_l[None, :]*Y[:, None] + cl[None, :]))/xi
        live = (np.abs(ex) < EXPCLIP)
        C = -dl - varpi_n*np.mean(np.exp(np.clip(ex, -EXPCLIP, EXPCLIP))*live, axis=1)
        if u2 is not None:
            ex2 = (u - u2)/xi
            C -= varpi_g*np.exp(np.clip(ex2, -EXPCLIP, EXPCLIP))*(np.abs(ex2) < EXPCLIP)
    B = w**2*F**2/2.0
    return A[:, None]*Dq + B[:, None]*Dp + np.diag(C)

def newton_ode(u0, Y, xi, Kc, cl, u2=None, tol=1e-12, maxit=80):
    h = Y[1] - Y[0]
    Dq, Dp = diff_matrices(Y)
    u = u0.copy()
    R = ode_residual(u, Y, h, xi, Kc, cl, u2)
    nrm = np.max(np.abs(R))
    for it in range(maxit):
        if nrm <= tol:
            break
        J = ode_jacobian(u, Y, Dq, Dp, xi, cl, u2)
        du = np.linalg.solve(J, -R)
        lam = 1.0
        while True:
            ut = u + lam*du
            Rt = ode_residual(ut, Y, h, xi, Kc, cl, u2)
            nt = np.max(np.abs(Rt))
            if nt < nrm or nt <= tol or lam < 1e-8:
                break
            lam *= 0.5
        u, R, nrm = ut, Rt, nt
    return u, nrm, it

# ============================================================ solve family over log-xi grid
NY, NXI = 401, 25
Ygrid = np.linspace(0.0, 4.0, NY)
hY = Ygrid[1] - Ygrid[0]
logxigrid = np.linspace(np.log(0.05), np.log(148.6), NXI)
xigrid = np.exp(logxigrid)

U1 = np.empty((NXI, NY)); U2 = np.empty((NXI, NY))
Q1 = np.empty((NXI, NY)); Q2 = np.empty((NXI, NY))
F1t = np.empty((NXI, NY)); F2t = np.empty((NXI, NY))
DC = np.empty(NXI)
c_pp_tab = np.empty((NXI, 5)); c_pre_tab = np.empty((NXI, 5))
newton_res = np.zeros(NXI)
max_exp_seen = 0.0
guard_hit = False

u2_prev = p2inf*Ygrid + c2_inf
u1_prev = p1inf*Ygrid + c1_inf
for k in range(NXI - 1, -1, -1):             # continuation: large xi -> small xi
    xi = xigrid[k]
    DC[k] = Delta_c(xi)
    c_pp_tab[k] = c_pp(gam1l, xi)
    c_pre_tab[k] = c_pp_tab[k] - DC[k]
    Kc2 = B_d + GgApp + Omega(xi)
    Kc1 = B_d + GgA + Omega(xi)
    u2, r2n, it2 = newton_ode(u2_prev, Ygrid, xi, Kc2, c_pp_tab[k])
    u1, r1n, it1 = newton_ode(u1_prev, Ygrid, xi, Kc1, c_pre_tab[k], u2=u2)
    trk = {}
    ode_residual(u2, Ygrid, hY, xi, Kc2, c_pp_tab[k], track=trk)
    ode_residual(u1, Ygrid, hY, xi, Kc1, c_pre_tab[k], u2=u2, track=trk)
    max_exp_seen = max(max_exp_seen, trk.get('max_exp', 0.0))
    guard_hit = guard_hit or trk.get('guard', False)
    U2[k], U1[k] = u2, u1
    Q2[k], _ = derivs(u2, hY); Q1[k], _ = derivs(u1, hY)
    F2t[k] = Fpos(np.minimum(Q2[k], -1e-6), xi)
    F1t[k] = Fpos(np.minimum(Q1[k], -1e-6), xi)
    newton_res[k] = max(r2n, r1n)
    u2_prev, u1_prev = u2, u1
    if k in (NXI-1, NXI//2, 0):
        tick('xi=%9.4f: newton_res=%.2e iters=(%d,%d) u1(0)=%.4f u1(4)=%.4f '
             'F1 in [%.2f,%.2f]' % (xi, newton_res[k], it2, it1, u1[0], u1[-1],
                                    F1t[k].min(), F1t[k].max()))
gate_newton = float(np.max(newton_res))
tick('GATE newton residual: max over 50 solves = %.3e (target <= 1e-10)' % gate_newton)
tick('GATE exp audit: max |exponent| = %.2f (clip 35);  slope guard hit = %s'
     % (max_exp_seen, guard_hit))

# ============================================================ refinement gate 401 -> 1601
refine = {}
for k in [0, NXI//2, NXI-1]:
    xi = xigrid[k]
    Yf = np.linspace(0.0, 4.0, 1601)
    Kc2 = B_d + GgApp + Omega(xi)
    Kc1 = B_d + GgA + Omega(xi)
    u2f, r2f, _ = newton_ode(np.interp(Yf, Ygrid, U2[k]), Yf, xi, Kc2, c_pp_tab[k])
    u1f, r1f, _ = newton_ode(np.interp(Yf, Ygrid, U1[k]), Yf, xi, Kc1, c_pre_tab[k],
                             u2=u2f)
    d2 = float(np.max(np.abs(u2f[::4] - U2[k])))
    d1 = float(np.max(np.abs(u1f[::4] - U1[k])))
    refine['xi=%.4g' % xi] = dict(du2=d2, du1=d1, res_fine=float(max(r2f, r1f)))
    tick('GATE refinement xi=%.4g: max|u2_1601-u2_401|=%.3e max|u1..|=%.3e '
         'fine_res=%.2e' % (xi, d2, d1, max(r2f, r1f)))

# ============================================================ xi=148.6 vs neutral O(1/xi)
u2_neu, rn2, _ = newton_ode(p2inf*Ygrid + c2_inf, Ygrid, np.inf, B_d + GgApp,
                            c_pp_inf)
u1_neu, rn1, _ = newton_ode(p1inf*Ygrid + c1_inf, Ygrid, np.inf, B_d + GgA,
                            c_pre_inf, u2=u2_neu)
lin_dev = float(max(np.max(np.abs(u2_neu - (p2inf*Ygrid + c2_inf))),
                    np.max(np.abs(u1_neu - (p1inf*Ygrid + c1_inf)))))
neutral_gap = float(max(np.max(np.abs(U2[-1] - u2_neu)), np.max(np.abs(U1[-1] - u1_neu))))
tick('GATE xi=oo ODE reproduces linear chain: max dev = %.3e (analytic cross-check)'
     % lin_dev)
tick('GATE xi=148.6 vs neutral: max|u - u_inf| = %.4f  (x xi = %.2f, O(1) => O(1/xi))'
     % (neutral_gap, neutral_gap*xigrid[-1]))

# ============================================================ interiority / statics gates
inter = dict(
    shares_interior=bool(0 < sd < 1 and 0 < sg < 1 and 0 < sr < 1 and sg + sr < 1),
    F_positive=bool(np.all(F1t > 0) and np.all(F2t > 0) and np.all(Fl_inf > 0)),
    p_decreasing_in_gamma1=bool(np.all(np.diff(p_l) < 0)),
    Fl_decreasing_in_l=bool(np.all(np.diff(Fl_inf) < 0)),
    slope_guard_inactive=bool(not guard_hit),
)
# fossil revelation table (xi = oo): F^l vs pre-damage F0
revelation = Fl_inf/F0
tick('GATE interiority: %s' % inter)
tick('STATIC fossil at damage revelation F^l/F_pre = %s  (cut for l>=%d)' %
     (np.array2string(revelation, precision=3),
      1 + int(np.argmax(revelation < 1.0))))

# ============================================================ production-box feasibility audit
rng = np.random.RandomState(11)
NS = 100000
lk_s = rng.uniform(4, 7, NS); Z_s = rng.uniform(0.01, 0.99, NS)
lr_s = rng.uniform(1, 6, NS); l3_s = rng.uniform(0, 1/3., NS)
lx_s = rng.uniform(np.log(0.05), np.log(148.6), NS)
xi_s = np.exp(lx_s)
spl_q1 = RectBivariateSpline(logxigrid, Ygrid, Q1, kx=3, ky=3)
spl_q2 = RectBivariateSpline(logxigrid, Ygrid, Q2, kx=3, ky=3)
audit = {}
for reg, ylo, Aprod, pretech in [('PreDamagePreTech', 0.0, Ag, True),
                                 ('PreDamagePostTech', 0.0, Agpp, False),
                                 ('PostDamagePreTech', 2.5, Ag, True),
                                 ('PostDamagePostTech', 2.5, Agpp, False)]:
    Y_s = rng.uniform(ylo, 4.0, NS)
    lkd_s = lk_s + np.log(1 - Z_s)
    lkg_s = lk_s + np.log(Z_s)
    if reg.startswith('Post'):
        g1s = gamma1 + kappa*l3_s
        Fs = Fpos(-g1s*m, xi_s)
        gam_eff = g1s
    else:
        q = (spl_q1 if pretech else spl_q2).ev(lx_s, Y_s)
        Fs = Fpos(np.minimum(q, -1e-6), xi_s)
        gam_eff = gamma1
    i_d = sd*Ad*np.exp(np.clip(-gamma0 - gam_eff*Y_s + nn*(np.log(chiF*Fs) - lkd_s),
                               -EXPCLIP, EXPCLIP))
    lr_eff = lr_s if pretech else (lkg_s - gapstar)
    Aecon = Ag if pretech else Agpp
    i_g = sg*Aecon*np.exp(np.clip(be*(np.log(chi) + lr_eff - lkg_s), -EXPCLIP, EXPCLIP))
    i_r = (sr*Aecon*Z_s**(1-be)*np.exp(np.clip(be*(np.log(chi) + lr_s - lk_s),
                                               -EXPCLIP, EXPCLIP)) if pretech else 0.0)
    CK = (Ad - i_d)*(1 - Z_s) + (Aprod - i_g)*Z_s - i_r
    audit[reg] = dict(
        frac_ig_gt_head=float(np.mean(i_g > 0.95)),
        frac_id_gt_head=float(np.mean(i_d > 0.95)),
        frac_CK_lt_floor=float(np.mean(CK < 1e-3)),
        i_d_range=[float(i_d.min()), float(i_d.max())],
        i_g_range=[float(i_g.min()), float(i_g.max())],
        i_r_range=([float(np.min(i_r)), float(np.max(i_r))] if pretech else None),
    )
    tick('AUDIT %s: P(i_g>0.95)=%.4f P(i_d>0.95)=%.4f P(C/K<1e-3)=%.4f' %
         (reg, audit[reg]['frac_ig_gt_head'], audit[reg]['frac_id_gt_head'],
          audit[reg]['frac_CK_lt_floor']))

# ============================================================ Feynman-Kac level certificate
def V_ansatz(lkd, lkg, lr, Y, damaged, teched, lidx):
    """xi = oo value at the given regime (vectorized over paths)."""
    base = aD*lkd + aG*lkg + aR*lr
    yc = np.where(damaged, p_l[lidx]*Y, np.where(teched, p2inf*Y, p1inf*Y))
    cc = np.where(damaged & teched, c_pp_inf[lidx],
                  np.where(damaged & ~teched, c_pre_inf[lidx],
                           np.where(teched, c2_inf, c1_inf)))
    return base + yc + cc

def fk_chain(seed_state, npaths=8000, T=100.0, dt=1/24., rngseed=0,
             start_damaged=False, start_teched=False, lfix=None):
    """Simulate the economy under its solved xi=oo policies; return
    (Vhat, SE, V_ansatz(seed))."""
    rs = np.random.RandomState(rngseed)
    lkd = np.full(npaths, seed_state[0]); lkg = np.full(npaths, seed_state[1])
    lr = np.full(npaths, seed_state[2]); Y = np.full(npaths, seed_state[3])
    nst = int(round(T/dt))
    if start_damaged:
        tau_n = np.zeros(npaths)
        lidx = np.full(npaths, lfix if lfix is not None else 0)
    else:
        tau_n = rs.exponential(1.0/varpi_n, npaths)
        lidx = rs.randint(0, 5, npaths)
    tau_g = (np.zeros(npaths) if start_teched
             else rs.exponential(1.0/varpi_g, npaths))
    disc_util = np.zeros(npaths)
    sqdt = np.sqrt(dt)
    lchi = np.log(chi)
    for i in range(nst):
        t = i*dt
        dm = tau_n <= t + dt/2
        tc = tau_g <= t + dt/2
        g1p = np.where(dm, gam1l[lidx], gamma1)
        Fp = np.where(dm, Fl_inf[lidx], F1inf)
        Ap = np.where(tc, Agpp, Ag)
        logYd = np.log(Ad) - gamma0 - g1p*Y + (1-nn)*lkd + nn*np.log(chiF*Fp)
        logYg = np.log(Ap) + be*lchi + (1-be)*lkg + be*lr
        U = dl*(td*(np.log(1-sd) + logYd) + tg*(np.log(1-sg-sr) + logYg))
        disc_util += np.exp(-dl*(t + dt/2))*U*dt
        mud = alpha0 + G*(lthc + np.log(sd) + logYd - lkd)
        mug = alpha0 + G*(lthc + np.log(sg) + logYg - lkg)
        mur = alpha0 + G*(lthc + np.log(sr) + logYg - lr)
        lkd += mud*dt + sgd*sqdt*rs.randn(npaths)
        lkg += mug*dt + sgg*sqdt*rs.randn(npaths)
        lr += mur*dt + sgr*sqdt*rs.randn(npaths)
        Y += b*Fp*dt + w*Fp*sqdt*rs.randn(npaths)
    dmT = tau_n <= T; tcT = tau_g <= T
    VT = V_ansatz(lkd, lkg, lr, Y, dmT, tcT, lidx)
    M = disc_util + np.exp(-dl*T)*VT
    V0 = float(V_ansatz(np.array([seed_state[0]]), np.array([seed_state[1]]),
                        np.array([seed_state[2]]), np.array([seed_state[3]]),
                        np.array([start_damaged]), np.array([start_teched]),
                        np.array([lfix if lfix is not None else 0]))[0])
    return float(np.mean(M)), float(np.std(M)/np.sqrt(npaths)), V0

x0 = (np.log(K0*(1-Z0)), np.log(K0*Z0), np.log(R0), Y0)
seeds = [x0,
         (x0[0]+0.5, x0[1]-0.5, x0[2]+0.5, 0.5),
         (x0[0]-0.5, x0[1]+0.5, x0[2]-0.5, 2.0),
         (5.0, 5.0, 3.0, 1.0),
         (6.0, 4.5, 2.0, 3.0)]
fk = {'terminal': [], 'chain': []}
for si, s in enumerate(seeds):
    vh, se, v0 = fk_chain(s, rngseed=100+si, start_damaged=True, start_teched=True,
                          lfix=2)
    fk['terminal'].append(dict(seed=list(s), Vhat=vh, SE=se, V=v0, diff=vh-v0))
    tick('FK terminal (l=3) seed %d: Vhat=%.5f V=%.5f diff=%+.5f (SE=%.5f)' %
         (si, vh, v0, vh-v0, se))
for si, s in enumerate(seeds):
    vh, se, v0 = fk_chain(s, rngseed=200+si)
    fk['chain'].append(dict(seed=list(s), Vhat=vh, SE=se, V=v0, diff=vh-v0))
    tick('FK full chain    seed %d: Vhat=%.5f V=%.5f diff=%+.5f (SE=%.5f)' %
         (si, vh, v0, vh-v0, se))
# dt-scaling study on the worst seeds: if |diff| shrinks ~linearly in dt the
# discrepancy is Euler/jump-switching discretization bias, not a level error
fk['dt_study'] = []
for tag, s, kwargs in [('terminal_seed3', seeds[3],
                        dict(start_damaged=True, start_teched=True, lfix=2)),
                       ('chain_seed0', seeds[0], dict())]:
    row = dict(case=tag)
    for dt_ in (1/12., 1/24., 1/48.):
        vh, se, v0 = fk_chain(s, npaths=16000, dt=dt_, rngseed=999, **kwargs)
        row['dt=%g' % dt_] = dict(diff=vh - v0, SE=se)
    fk['dt_study'].append(row)
    tick('FK dt-study %s: ' % tag + '  '.join(
        'dt=1/%d: %+.5f(%.5f)' % (round(1/dt_), row['dt=%g' % dt_]['diff'],
                                  row['dt=%g' % dt_]['SE'])
        for dt_ in (1/12., 1/24., 1/48.)))
fk_worst = max(max(abs(r['diff']) for r in fk['terminal']),
               max(abs(r['diff']) for r in fk['chain']))
fk_worst_se = max(max(abs(r['diff'])/max(r['SE'], 1e-12) for r in fk['terminal']),
                  max(abs(r['diff'])/max(r['SE'], 1e-12) for r in fk['chain']))
tick('GATE Feynman-Kac: worst |Vhat-V| = %.5f (worst |diff|/SE = %.2f)' %
     (fk_worst, fk_worst_se))

# ============================================================ save npz + provenance
npz_path = os.path.join(OUT, 'ghkm.npz')
np.savez_compressed(
    npz_path,
    # parameters
    delta=dl, theta_d=td, theta_g=tg, beta=be, nu=nn, Gamma=G, theta_c=thc,
    sigma_d=sgd, sigma_g=sgg, sigma_r=sgr, A_d=Ad, A_g=Ag, A_gpp=Agpp,
    gamma0=gamma0, gamma1=gamma1, kappa=kappa, thetabar=thetabar,
    varsigma=varsigma, eta0=eta0, b=b, w=w, alpha0=alpha0,
    varpi_n=varpi_n, varpi_g=varpi_g, lam3=lam3, gam1l=gam1l,
    K0=K0, Z0=Z0, Y0=Y0, R0=R0,
    # derived coefficients / policies
    a_d=aD, a_g=aG, a_r=aR, m=m, s_d=sd, s_g=sg, s_r=sr,
    p_l=p_l, gapstar=gapstar, chi=chi, chi_F=chiF, F0=F0,
    B_d=B_d, Gg_A=GgA, Gg_App=GgApp, DGg=DGg,
    # xi = oo chain
    p1inf=p1inf, p2inf=p2inf, Dc_inf=Dc_inf, c_pp_inf=c_pp_inf,
    c_pre_inf=c_pre_inf, c1_inf=c1_inf, c2_inf=c2_inf, Fl_inf=Fl_inf, F1inf=F1inf,
    # tabulated pre-damage family
    Ygrid=Ygrid, logxigrid=logxigrid, U1=U1, U2=U2, Q1=Q1, Q2=Q2,
    F1=F1t, F2=F2t, DC=DC, c_pp_tab=c_pp_tab, c_pre_tab=c_pre_tab,
    newton_res=newton_res)
tick('wrote %s' % npz_path)

prov = dict(
    economy='GHKM (fossil-flow emissions, dirty-productivity damages, '
            'knowledge-in-production)',
    design='benchmarks/economy_zoo/design_ghkm.json (f(Y)-ODE route ADOPTED, '
           'affine-hazard variant REJECTED per synthesis)',
    solver='solvers/ghkm.py',
    date=time.strftime('%Y-%m-%d %H:%M'),
    formulas=dict(
        ansatz='V = a_d logK_d + a_g logK_g + a_r logR + {p^l Y + c^l | u_reg(Y;xi)}',
        a_d='delta th_d (1-nu)/(delta+G nu)', a_g='th_g(1-beta)(delta+G)/D',
        a_r='th_g beta (delta+G)/D', m='th_d(G+delta)/(delta+G nu)',
        shares='s_d=G(1-nu)/(G+delta), s_g=G(1-beta)/(G+delta), s_r=G beta/(G+delta)',
        fossil_FOC='nu delta m/F + q b - (q^2 w^2/xi) F = 0, q = V_Y',
        Delta_c='delta Dc = DGg - xi varpi_g (1-exp(-Dc/xi))',
        ode='0 = -delta u - delta m gamma1 Y + Kc + nu delta m log(chiF F) + u_Y b F '
            '+ (w^2F^2/2)u_YY - (w^2F^2/2xi)u_Y^2 + xi varpi_n mean_l(1-exp((u-p^lY-c^l)/xi)) '
            '[+ xi varpi_g(1-exp((u1-u2)/xi))]'),
    calibration=dict(
        gamma0='0 (design leaves it free; i_d(x0)=0.066 lands in the target '
               '0.065-0.08 band)',
        kappa='0.25 = literal absolute-increment reading of "post-damage slope '
              'matches production (logN)_y at Y=2.75": kappa*l3 = l3*(Y-yhat)|_{2.75}. '
              'Relative-increment alternative (kappa=3.05) documented and rejected '
              'as the less literal reading.',
        F0='pre-damage pre-tech xi=oo fossil choice (fixes chi_F=K_d0/F0 so the '
           'fossil intensity ratio is 1 at t=0)',
        alpha0='matches production phi(0.09) under log-log adjustment; same alpha '
               'for K_d, K_g, R',
        values=dict(alpha0=alpha0, F0=F0, chi=chi, chi_F=chiF, gapstar=gapstar)),
    coefficients=dict(a_d=aD, a_g=aG, a_r=aR, V_logK=aD+aG, m=m,
                      s_d=sd, s_g=sg, s_r=sr, p_l=list(p_l),
                      p_pre_inf=p1inf, Dc_inf=Dc_inf,
                      c1_inf=c1_inf, c2_inf=c2_inf,
                      c_pp_inf=list(c_pp_inf), Fl_inf=list(Fl_inf)),
    gates=dict(
        G1_sympy='PASS (solvers/verify_ghkm_sympy.py: A1-A7 symbolic, numeric '
                 'residual 4.8e-17 / 1.2e-16 <= 1e-12 at 1e4 states)',
        G2_newton_residual=dict(value=gate_newton, target='<=1e-10',
                                passed=bool(gate_newton <= 1e-10)),
        G2_refinement_401_to_1601=refine,
        G2_exp_audit=dict(max_abs_exponent=max_exp_seen, clip=35.0,
                          slope_guard_hit=bool(guard_hit)),
        G3_neutral_limit=dict(max_dev_xi148=neutral_gap,
                              times_xi=neutral_gap*float(xigrid[-1]),
                              xi_inf_ode_vs_linear=lin_dev),
        G3_interiority=inter,
        G3_comparative_statics=dict(
            p_l=list(p_l), F_l_over_F_pre=list(revelation),
            note='fossil is cut at revelation for above-median severities '
                 '(l >= 3); mild revelations (l = 1, 2) RAISE fossil use because '
                 'the pre-damage choice prices the expected jump - the flat-SCC '
                 'logic holds per-regime (mu_Y^l = nu delta / gamma1^l)'),
        G4_feynman_kac=fk,
        G5_box_audit_preguard=audit),
    notes=['post-tech regimes: economy still invests s_r in knowledge; the lift '
           'collapses logR at the attractor gap* = log(s_g/s_r) and drops the '
           'i_r net (production has none) - documented theory feature',
           'fossil control F has no production-net counterpart; embodied in v/i_d'])
prov_path = os.path.join(OUT, 'ghkm_PROVENANCE.json')
with open(prov_path, 'w') as f:
    json.dump(prov, f, indent=1)
tick('wrote %s' % prov_path)
tick('DONE')
