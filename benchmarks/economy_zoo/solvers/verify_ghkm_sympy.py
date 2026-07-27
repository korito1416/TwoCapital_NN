#!/usr/bin/env python
"""G1 gate for the GHKM economy (fossil-flow emissions, productivity damages,
knowledge-in-production).

Symbolically verifies the linear-coefficient recursion of the design
(design_ghkm.json / PORTFOLIO.json climate-coupling lens, economy 2):

  ansatz  V = a_d logK_d + a_g logK_g + a_r logR + p Y + c   (post-damage regimes)

against the full robust HJB

  0 = -delta V + U + a.mu + p muY - (1/2xi)(a_d^2 sg_d^2 + a_g^2 sg_g^2 + a_r^2 sg_r^2)
      - (1/2xi) p^2 w^2 F^2  [+ xi varpi_g (1 - exp(-Dc/xi)) tech jump, pre-tech]

with U = delta[th_d log C_d + th_g log C_g], log-log adjustment drifts, robust
fossil FOC.  The Hamiltonian is LINEAR in the state atoms (logK_d, logK_g, logR, Y),
so {four coefficient identities + the constant equation defining c} is a COMPLETE
symbolic proof that the residual vanishes identically.  Checks (general Gammas):

  A1  coefficient matching  =>  a_d, a_g, a_r formulas + identity a_g + a_r = th_g
  A2  share FOCs  =>  s_d*, s_g*, s_r* (general Gam_g != Gam_r) + SOC
  A3  Y matching  =>  p^l = -gamma1^l m,  m = th_d (Gam_d+delta)/(delta+Gam_d nu)
  A4  robust fossil FOC: dH/dF = nu delta m/F + p b - (p^2 w^2/xi) F; the stated
      positive root zeroes it; xi=oo limit F* = -nu delta m/(p b)
  A5  the constant equation reproduces c = (G_d + G_g + Omega)/delta with
      G_d = B_d + nu delta m log(chiF F) + p b F - (w^2 F^2/2 xi)p^2  => with A1-A4,
      the FULL post-damage-post-tech residual == 0 identically (linearity argument)
  A6  post-damage-PRE-tech residual reduces EXACTLY to the Delta_c fixed-point
      identity  delta Dc - DGg + xi varpi_g (1 - exp(-Dc/xi)) = 0
  A7  xi=oo pre-damage linear recursion: p_2 = (-delta m gamma1 + varpi_n pbar)/
      (delta+varpi_n) and p_1 = p_2
  N1  numeric spot check: post-damage residual at 1e4 random states
      (both post-damage regimes, random l, random xi) <= 1e-12

Run:  python verify_ghkm_sympy.py
"""
import time
import numpy as np
import sympy as sp

T0 = time.time()
def tick(msg):
    print('[%6.1fs] %s' % (time.time() - T0, msg), flush=True)

# ----------------------------------------------------------------------------- symbols
(delta, thd, thg, beta, nu) = sp.symbols('delta theta_d theta_g beta nu', positive=True)
(Gd, Gg, Gr) = sp.symbols('Gamma_d Gamma_g Gamma_r', positive=True)
(ald, alg, alr, thc) = sp.symbols('alpha_d alpha_g alpha_r theta_c')
(Ad, Ag, chi, chiF) = sp.symbols('A_d A_g chi chi_F', positive=True)
(gam0, gam1) = sp.symbols('gamma_0 gamma_1')
(b, w) = sp.symbols('b w', positive=True)            # b = thetabar*eta0, w = varsigma*eta0
(sgd, sgg, sgr) = sp.symbols('sigma_d sigma_g sigma_r', positive=True)
xi = sp.Symbol('xi', positive=True)
(varpi_g, varpi_n) = sp.symbols('varpi_g varpi_n', positive=True)
(lkd, lkg, lr, Y) = sp.symbols('logK_d logK_g logR Y', real=True)
(sd, sg, sr, F) = sp.symbols('s_d s_g s_r F', positive=True)
(a_d, a_g, a_r, p, c) = sp.symbols('a_d a_g a_r p c', real=True)
Dc = sp.Symbol('Delta_c', real=True)
log = sp.log

def is_zero(e):
    """Exact zero test. Fully atomize logs/exps (expand_log force), replace each
    transcendental atom by a fresh symbol, and decide zero of the resulting
    RATIONAL function by cancel(). Exact because after atomization the log/exp
    atoms are algebraically independent generators."""
    e = sp.expand(sp.expand_log(sp.expand(e), force=True))
    if e == 0:
        return True
    atoms = list(e.atoms(sp.log)) + [a for a in e.atoms(sp.exp)]
    rep = {a: sp.Symbol('ATOM_%d' % i) for i, a in enumerate(atoms)}
    e2 = e.xreplace(rep)
    if sp.cancel(sp.together(sp.expand(e2))) == 0:
        return True
    return sp.simplify(e) == 0

fails = []
def check(name, expr_zero):
    ok = is_zero(expr_zero)
    tick(('PASS' if ok else 'FAIL') + '  ' + name)
    if not ok:
        fails.append(name)

def hamiltonian(a_dv, a_gv, a_rv, pv, cv, Agv, gam1v, jump_const=0):
    """Full robust post-damage HJB rhs (V linear => V_YY = 0), controls symbolic."""
    V = a_dv*lkd + a_gv*lkg + a_rv*lr + pv*Y + cv
    logYd = log(Ad) - gam0 - gam1v*Y + (1-nu)*lkd + nu*log(chiF*F)
    logYg = log(Agv) + beta*log(chi) + (1-beta)*lkg + beta*lr
    U = delta*(thd*(log(1-sd) + logYd) + thg*(log(1-sg-sr) + logYg))
    mud = ald + Gd*(log(thc) + log(sd) + logYd - lkd)
    mug = alg + Gg*(log(thc) + log(sg) + logYg - lkg)
    mur = alr + Gr*(log(thc) + log(sr) + logYg - lr)
    muY = b*F
    Omega = -(a_dv**2*sgd**2 + a_gv**2*sgg**2 + a_rv**2*sgr**2)/(2*xi)   # h_d,h_g,h_r drag
    fossil_drag = -(pv**2)*(w**2)*(F**2)/(2*xi)                          # h_y drag
    return (-delta*V + U + a_dv*mud + a_gv*mug + a_rv*mur + pv*muY
            + Omega + fossil_drag + jump_const)

H = hamiltonian(a_d, a_g, a_r, p, c, Ag, gam1)
tick('Hamiltonian built')

# --------------------------------------------------------------- A1 coefficient matching
Hp = sp.expand(H)
coef_lkd = Hp.coeff(lkd); coef_lkg = Hp.coeff(lkg); coef_lr = Hp.coeff(lr)
Dsym = delta + Gg*beta + Gr*(1-beta)
sol = sp.solve([coef_lkd, coef_lkg, coef_lr], [a_d, a_g, a_r], dict=True)[0]
check('A1 a_d = delta*th_d*(1-nu)/(delta+Gam_d*nu)',
      sol[a_d] - delta*thd*(1-nu)/(delta+Gd*nu))
check('A1 a_g = th_g*(1-beta)*(delta+Gam_r)/D', sol[a_g] - thg*(1-beta)*(delta+Gr)/Dsym)
check('A1 a_r = th_g*beta*(delta+Gam_g)/D', sol[a_r] - thg*beta*(delta+Gg)/Dsym)
check('A1 identity a_g + a_r = th_g', sol[a_g] + sol[a_r] - thg)
aD, aG, aR = sol[a_d], sol[a_g], sol[a_r]

# --------------------------------------------------------------- A2 share FOCs (general)
H1 = H.subs([(a_d, aD), (a_g, aG), (a_r, aR)])
sd_star = Gd*(1-nu)/(Gd+delta)
sg_star = Gg*(1-beta)/(Gg+delta)
sr_star = Gr*beta/(Gr+delta)
check('A2 dH/ds_d = 0 at s_d* (general Gammas)', sp.diff(H1, sd).subs(sd, sd_star))
check('A2 dH/ds_g = 0 at s_g* (general Gammas)',
      sp.diff(H1, sg).subs([(sg, sg_star), (sr, sr_star)]))
check('A2 dH/ds_r = 0 at s_r* (general Gammas)',
      sp.diff(H1, sr).subs([(sg, sg_star), (sr, sr_star)]))
soc_d = sp.simplify(sp.diff(H1, sd, 2))
tick('INFO  A2 SOC d2H/dsd2 = %s  (< 0 on s_d in (0,1): interior max)' % soc_d)

# --------------------------------------------------------------- A3 Y matching
m_sym = thd*(Gd+delta)/(delta+Gd*nu)
H1s = sp.expand(H1.subs([(sd, sd_star), (sg, sg_star), (sr, sr_star)]))
p_l = sp.solve(H1s.coeff(Y), p)[0]
check('A3 p^l = -gamma1^l*m, m = th_d(Gam_d+delta)/(delta+Gam_d*nu)', p_l - (-gam1*m_sym))

# --------------------------------------------------------------- A4 fossil FOC
dHdF = sp.together(sp.diff(H1, F))
check('A4 dH/dF = nu*delta*m/F + p*b - (p^2 w^2/xi) F',
      dHdF - (nu*delta*m_sym/F + p*b - p**2*w**2*F/xi))
Aq = p**2*w**2/xi; Bq = -p*b; Cq = -nu*delta*m_sym
F_rob = (-Bq + sp.sqrt(Bq**2 - 4*Aq*Cq))/(2*Aq)
# dH/dF * F = -(Aq F^2 + Bq F + Cq); verify quadratic vanishes at the positive root
check('A4 robust positive root zeroes the fossil quadratic',
      (Aq*F_rob**2 + Bq*F_rob + Cq))
check('A4 xi=oo limit: F* = -nu*delta*m/(p b) zeroes the neutral FOC',
      (nu*delta*m_sym/F + p*b).subs(F, -nu*delta*m_sym/(p*b)))

# --------------------------------------------------------------- A5 constant equation -> c
H2 = sp.expand(H1s.subs(p, p_l))
# linearity in state atoms: residual = coef.states + const; coefs are zero by A1/A3:
check('A5 lkd coefficient == 0 after substitution', H2.coeff(lkd))
check('A5 lkg coefficient == 0 after substitution', H2.coeff(lkg))
check('A5 lr  coefficient == 0 after substitution', H2.coeff(lr))
check('A5 Y   coefficient == 0 after substitution', H2.coeff(Y))
const_part = H2.subs([(lkd, 0), (lkg, 0), (lr, 0), (Y, 0)])
# const_part is linear in c with coefficient -delta: extract c* directly
c_star = sp.expand(const_part.subs(c, 0)/delta)
# design formula: c = (B_d + nu*delta*m*log(chiF F) + p b F - w^2F^2 p^2/(2xi) + G_g + Omega)/delta
B_d_sym = (delta*thd*(log(1-sd_star) + log(Ad) - gam0)
           + aD*(ald + Gd*(log(thc) + log(sd_star) + log(Ad) - gam0)))
G_g_sym = (delta*thg*(log(1-sg_star-sr_star) + log(Ag) + beta*log(chi))
           + aG*(alg + Gg*(log(thc) + log(sg_star) + log(Ag) + beta*log(chi)))
           + aR*(alr + Gr*(log(thc) + log(sr_star) + log(Ag) + beta*log(chi))))
Omega_sym = -(aD**2*sgd**2 + aG**2*sgg**2 + aR**2*sgr**2)/(2*xi)
G_d_sym = (B_d_sym + nu*delta*m_sym*log(chiF*F) + p_l*b*F - (w**2*F**2/(2*xi))*p_l**2)
c_formula = (G_d_sym + G_g_sym + Omega_sym)/delta
check('A5 c = (G_d + G_g + Omega)/delta reproduces the constant equation '
      '(=> FULL post-damage-post-tech residual == 0 identically, by linearity)',
      c_star - c_formula)

# --------------------------------------------------------------- A6 pre-tech + Dc fixed point
Agpp = sp.Symbol('A_g_pp', positive=True)
jump = xi*varpi_g*(1 - sp.exp(-Dc/xi))
c_post_pp = c_star.subs(Ag, Agpp)          # post-tech continuation constant uses Ag''
H_pre = hamiltonian(aD, aG, aR, p_l, c_post_pp - Dc, Ag, gam1, jump_const=jump)
H_pre = sp.expand(H_pre.subs([(sd, sd_star), (sg, sg_star), (sr, sr_star)]))
check('A6 pre-tech lkd coefficient == 0', H_pre.coeff(lkd))
check('A6 pre-tech lkg coefficient == 0', H_pre.coeff(lkg))
check('A6 pre-tech lr  coefficient == 0', H_pre.coeff(lr))
check('A6 pre-tech Y   coefficient == 0', H_pre.coeff(Y))
DGg = (delta*thg + aG*Gg + aR*Gr)*(log(Agpp) - log(Ag))
check('A6 pre-tech residual == delta*Dc - DGg + xi*varpi_g(1-e^{-Dc/xi}) '
      '(the Delta_c fixed-point identity; == 0 at the fixed point)',
      H_pre.subs([(lkd, 0), (lkg, 0), (lr, 0), (Y, 0)]) - (delta*Dc - DGg + jump))

# --------------------------------------------------------------- A7 xi=oo pre-damage p recursion
pbar, p1, p2 = sp.symbols('pbar p_1 p_2', real=True)
eq2 = -delta*p2 - delta*m_sym*gam1 + varpi_n*(pbar - p2)
p2_sol = sp.solve(eq2, p2)[0]
check('A7 p_2 = (-delta*m*gamma1 + varpi_n*pbar)/(delta+varpi_n)',
      p2_sol - (-delta*m_sym*gam1 + varpi_n*pbar)/(delta + varpi_n))
eq1 = -delta*p1 - delta*m_sym*gam1 + varpi_n*(pbar - p1) + varpi_g*(p2_sol - p1)
p1_sol = sp.solve(eq1, p1)[0]
check('A7 p_1 = p_2 (tech jump does not move the Y slope)', p1_sol - p2_sol)

# =========================================================== N1 numeric residual 1e4 states
tick('--- N1 numeric spot check (both post-damage regimes) ---')
import numpy.random as npr
from scipy.optimize import brentq

PAR = dict(delta=0.01, thd=0.34, thg=0.66, beta=0.3, nu=0.3, G=0.06, thc=16.7,
           sgd=0.01, sgg=0.01, sgr=0.0078, Ad=0.1303, Agv=0.1085, Agpp=0.1567,
           gam0=0.0, gam1=0.15, kappa=0.25,
           b=1.86e-3*0.291, w=1.2*1.86e-3*0.291,
           varpi_n=1.5*(np.exp(0.18*0.25) - 1.0), varpi_g=11.2/746.67)
alpha0 = -0.035 + 0.06*(np.log(1 + 16.7*0.09) - np.log(16.7*0.09))
lam3 = np.array([0., 1/12., 1/6., 1/4., 1/3.])
dl, td, tg, be, nn_, G = (PAR['delta'], PAR['thd'], PAR['thg'], PAR['beta'],
                          PAR['nu'], PAR['G'])
bn, wn = PAR['b'], PAR['w']
aDn = dl*td*(1-nn_)/(dl+G*nn_)
Dn = dl + G*be + G*(1-be)
aGn = tg*(1-be)*(dl+G)/Dn
aRn = tg*be*(dl+G)/Dn
mn = td*(G+dl)/(dl+G*nn_)
sdn = G*(1-nn_)/(G+dl); sgn = G*(1-be)/(G+dl); srn = G*be/(G+dl)

def Fpos(q, xiv):
    Aq_ = q**2*wn**2/xiv; Bq_ = -q*bn; Cq_ = -nn_*dl*mn
    return -2*Cq_/(Bq_ + np.sqrt(Bq_**2 - 4*Aq_*Cq_))

gam1l_n = PAR['gam1'] + PAR['kappa']*lam3
pbar_n = -mn*np.mean(gam1l_n)
p2inf = (-dl*mn*PAR['gam1'] + PAR['varpi_n']*pbar_n)/(dl + PAR['varpi_n'])
F0 = nn_*dl*mn/(-p2inf*bn)
chiFn = 880.0*0.3/F0
chin = 880.0*0.7/11.2

def Gg_const(Aval):
    lc = np.log(PAR['thc'])
    out = dl*tg*(np.log(1-sgn-srn) + np.log(Aval) + be*np.log(chin))
    out += aGn*(alpha0 + G*(lc + np.log(sgn) + np.log(Aval) + be*np.log(chin)))
    out += aRn*(alpha0 + G*(lc + np.log(srn) + np.log(Aval) + be*np.log(chin)))
    return out

B_dn = (dl*td*(np.log(1-sdn) + np.log(PAR['Ad']) - PAR['gam0'])
        + aDn*(alpha0 + G*(np.log(PAR['thc']) + np.log(sdn) + np.log(PAR['Ad'])
                           - PAR['gam0'])))

def Omega_n(xiv):
    return -(aDn**2*PAR['sgd']**2 + aGn**2*PAR['sgg']**2 + aRn**2*PAR['sgr']**2)/(2*xiv)

def Gd_const(g1l, xiv):
    pl = -g1l*mn
    Fl = Fpos(pl, xiv)
    return (B_dn + nn_*dl*mn*np.log(chiFn*Fl) + pl*bn*Fl - (wn**2*Fl**2/(2*xiv))*pl**2)

DGg_n = (dl*tg + aGn*G + aRn*G)*np.log(PAR['Agpp']/PAR['Agv'])

def Delta_c_n(xiv):
    f = lambda D_: dl*D_ - DGg_n + xiv*PAR['varpi_g']*(1 - np.exp(-min(D_/xiv, 35.)))
    return brentq(f, 0.0, DGg_n/dl + 1.0, xtol=1e-15, rtol=8.9e-16)

subs_common = [(delta, dl), (thd, td), (thg, tg), (beta, be), (nu, nn_),
               (Gd, G), (Gg, G), (Gr, G), (thc, PAR['thc']),
               (ald, alpha0), (alg, alpha0), (alr, alpha0),
               (Ad, PAR['Ad']), (chi, chin), (chiF, chiFn),
               (gam0, PAR['gam0']), (b, bn), (w, wn),
               (sgd, PAR['sgd']), (sgg, PAR['sgg']), (sgr, PAR['sgr']),
               (varpi_g, PAR['varpi_g']), (varpi_n, PAR['varpi_n'])]
H_num_expr = hamiltonian(a_d, a_g, a_r, p, c, Ag, gam1).subs(subs_common)
f_H = sp.lambdify((lkd, lkg, lr, Y, sd, sg, sr, F, a_d, a_g, a_r, p, c, Ag, gam1, xi),
                  H_num_expr, 'numpy')

rng = npr.RandomState(7)
N = 10000
lkd_s = rng.uniform(-1.0, 7.0, N); lkg_s = rng.uniform(-1.0, 7.0, N)
lr_s = rng.uniform(1.0, 6.0, N); Y_s = rng.uniform(0.0, 4.0, N)
xi_s = np.exp(rng.uniform(np.log(0.05), np.log(148.6), N))
g1_s = PAR['gam1'] + PAR['kappa']*lam3[rng.randint(0, 5, N)]
p_s = -g1_s*mn
F_s = Fpos(p_s, xi_s)
c_pp = (Gd_const(g1_s, xi_s) + Gg_const(PAR['Agpp']) + Omega_n(xi_s))/dl

res_pp = f_H(lkd_s, lkg_s, lr_s, Y_s, sdn, sgn, srn, F_s,
             aDn, aGn, aRn, p_s, c_pp, PAR['Agpp'], g1_s, xi_s)
max_pp = np.max(np.abs(res_pp))
tick('post-damage-post-tech  max|residual| = %.3e' % max_pp)

Dc_s = np.array([Delta_c_n(x) for x in xi_s])
c_pre = c_pp - Dc_s
jump_s = xi_s*PAR['varpi_g']*(1 - np.exp(-np.clip(Dc_s/xi_s, -35., 35.)))
res_pt = f_H(lkd_s, lkg_s, lr_s, Y_s, sdn, sgn, srn, F_s,
             aDn, aGn, aRn, p_s, c_pre, PAR['Agv'], g1_s, xi_s) + jump_s
max_pt = np.max(np.abs(res_pt))
tick('post-damage-pre-tech   max|residual| = %.3e' % max_pt)

for name, mx in [('N1 post-damage-post-tech residual <= 1e-12 at 1e4 states', max_pp),
                 ('N1 post-damage-pre-tech  residual <= 1e-12 at 1e4 states', max_pt)]:
    ok = mx <= 1e-12
    tick(('PASS' if ok else 'FAIL') + '  ' + name)
    if not ok:
        fails.append(name)

print()
print('G1 GATE: ALL CHECKS PASS' if not fails else 'G1 GATE: FAILURES: %s' % fails)
raise SystemExit(0 if not fails else 1)
