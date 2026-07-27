"""GHKM economy warm-start map (economy zoo build 3).

Exports MAP_NAME, PROVENANCE, fields(reg, lk, Z, Y, lr, l3, lx) per the
make_map_anchor.py interface. Consumes solvers/ghkm.py output (outputs/ghkm.npz).

VALUE OBJECT (the exact transform the verifier should check)
------------------------------------------------------------
fields()['v'] is the GHKM ECONOMY'S OWN VALUE FUNCTION evaluated at the
production state via the bijection

    logK_d = logK + log(1-Z),   logK_g = logK + log(Z),   (logR, Y shared)

    pre-damage regimes :  v = a_d logK_d + a_g logK_g + a_r logR* + u_reg(Y; xi)
    post-damage regimes:  v = a_d logK_d + a_g logK_g + a_r logR* + p^l Y + c_reg(l3, xi)

with u_reg(Y; xi) = f_reg(Y; xi) + c_reg(xi) the solved 1-D robust ODE value
(bicubic spline in (log xi, Y) of the 25 x 401 table), p^l = -(gamma1 + kappa
lambda3) m, and c_reg(l3, xi) evaluated in closed form (analytic in lambda3
and xi; Delta_c(xi) by vectorized Newton on its fixed point).

NO production logN(Y) is added: this economy has PRODUCTIVITY damages inside
dirty output (gamma1^l Y in log Y_d), not a separate utility damage multiplier,
so the lambda3/damage dependence enters v analytically through gamma1^l in
(p, c, F, i_d) exactly as the design specifies. The production trainer will
reinterpret the fitted v under its own v = V + logN convention; the zoo measures
basin selection by the map's GEOMETRY (V_logK = a_d + a_g = 0.547 exactly,
zero Y-curvature post-damage, log-share structure in Z), which is unaffected
by that regime-constant-slope reinterpretation.

logR handling: production post-tech nets have NO logR input, so for post-tech
regimes logR* = collapse at the mean-reverting attractor of logK_g - logR:
logR* = logK + logZ - gap*, gap* = log(s_g/s_r) = log((1-beta)/beta). The
economy's still-active post-tech s_r is embedded in v, not lifted as a net.

POLICIES
--------
i_d = I_d/K_d with I_d = s_d Y_d (the fossil choice F* has NO production-net
counterpart -- it is a flow input, chosen from the robust fossil FOC
nu delta m/F + V_Y b - (V_Y^2 w^2/xi) F = 0 -- so its effect is EMBODIED in
v and i_d):
    i_d = s_d A_d exp(-gamma0 - gamma1^reg Y) (chi_F F* / K_d)^nu
with gamma1^reg = gamma1 (pre-damage; V_Y = u'(Y;xi) from the spline) or
gamma1 + kappa*l3 (post-damage; V_Y = p^l).
    i_g = s_g A_g^reg (chi R*/K_g)^beta      (R* collapsed post-tech as above)
    i_r = I_r/(K_d+K_g) = s_r A_g^reg Z^(1-beta) chi^beta exp(beta(logR-logK))
i_r is the PRODUCTION rate on total capital (economy funds R&D from green
output: I_r = s_r Y_g); returned for pre-tech regimes only, None post-tech.

GUARDS (recorded per-call in fields.last_guard_stats)
-----------------------------------------------------
  exp-clip +-35 on every exponent; slope guard q <= -1e-6 in the fossil FOC;
  head clip i_d, i_g <= 0.95 (production tanh head range is (-1/theta, 1)),
  i_r in [1e-8, 0.95]; budget guard: where production-lens consumption
  C/K = (A_d - i_d)(1-Z) + (A_g^prod - i_g)Z - i_r < 1e-3, all rates are
  scaled proportionally so C/K = 1e-3 (continuous at the boundary). This is
  the documented simplification of the design's FOC-consistent fallback: the
  economy's decreasing-returns sectors out-produce the production AK budget
  over roughly half the box, so the guard region is LARGE (~55%); within it
  the i_d : i_g : i_r proportions of the economy are preserved.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from scipy.interpolate import RectBivariateSpline

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPZ = os.path.abspath(os.path.join(_HERE, '..', 'outputs', 'ghkm.npz'))
_d = np.load(_NPZ)

_f = lambda k: float(_d[k])
dl, td, tg, be, nn = (_f('delta'), _f('theta_d'), _f('theta_g'), _f('beta'),
                      _f('nu'))
G, b, w = _f('Gamma'), _f('b'), _f('w')
aD, aG, aR, m = _f('a_d'), _f('a_g'), _f('a_r'), _f('m')
sd, sg, sr = _f('s_d'), _f('s_g'), _f('s_r')
Ad, Ag, Agpp = _f('A_d'), _f('A_g'), _f('A_gpp')
gamma0, gamma1, kappa = _f('gamma0'), _f('gamma1'), _f('kappa')
chi, chiF, gapstar = _f('chi'), _f('chi_F'), _f('gapstar')
B_d, GgApp, DGg = _f('B_d'), _f('Gg_App'), _f('DGg')
varpi_g = _f('varpi_g')
_sig2 = (aD**2*_f('sigma_d')**2 + aG**2*_f('sigma_g')**2 + aR**2*_f('sigma_r')**2)
_Ygrid, _lxgrid = _d['Ygrid'], _d['logxigrid']
_spl = {'u1': RectBivariateSpline(_lxgrid, _Ygrid, _d['U1'], kx=3, ky=3),
        'u2': RectBivariateSpline(_lxgrid, _Ygrid, _d['U2'], kx=3, ky=3),
        'q1': RectBivariateSpline(_lxgrid, _Ygrid, _d['Q1'], kx=3, ky=3),
        'q2': RectBivariateSpline(_lxgrid, _Ygrid, _d['Q2'], kx=3, ky=3)}

EXPCLIP = 35.0
HEADMAX = 0.95
CK_FLOOR = 1e-3

MAP_NAME = 'ghkm_closed_form_v1'
PROVENANCE = dict(
    economy='GHKM: fossil-flow emissions, dirty-productivity damages, '
            'knowledge-in-production (Cobb-Douglas utility over sector goods, '
            'log-log adjustment, constant jump hazards, robust f(Y)-ODE '
            'damage-jump channel)',
    design='benchmarks/economy_zoo/design_ghkm.json',
    solver='benchmarks/economy_zoo/solvers/ghkm.py -> outputs/ghkm.npz',
    verification='G1 sympy gate PASS (verify_ghkm_sympy.py); Newton residual '
                 '3e-13; 401->1601 refinement 3.2e-8; FK level MC within '
                 '~1.5 SE at dt=1/48; see outputs/ghkm_PROVENANCE.json',
    value_object='economy value V at production states via logK_d=logK+log(1-Z), '
                 'logK_g=logK+logZ; NO production logN added (productivity '
                 'damages); post-tech logR collapsed at gap*=log(s_g/s_r)',
    key_numbers=dict(V_logK=aD+aG, a_d=aD, a_g=aG, a_r=aR,
                     s_d=sd, s_g=sg, s_r=sr, gapstar=gapstar),
)

def _clipexp(x):
    return np.exp(np.clip(x, -EXPCLIP, EXPCLIP))

def _Fpos(q, xi):
    Bq = -q*b
    Cq = -nn*dl*m
    Aq = q**2*w**2/xi
    return -2.0*Cq/(Bq + np.sqrt(Bq**2 - 4.0*Aq*Cq))

def _Delta_c(xi):
    """Vectorized Newton on delta*Dc - DGg + xi*varpi_g*(1-exp(-Dc/xi)) = 0."""
    Dc = np.full_like(xi, DGg/(dl + varpi_g))
    for _ in range(40):
        e = _clipexp(-Dc/xi)
        psi = dl*Dc - DGg + xi*varpi_g*(1.0 - e)
        Dc = Dc - psi/(dl + varpi_g*e)
    return Dc

def _c_pp(g1, xi):
    """PostDamagePostTech constant, analytic in (gamma1^l, xi)."""
    p = -g1*m
    F = _Fpos(p, xi)
    Gd = B_d + nn*dl*m*np.log(chiF*F) + p*b*F - (w**2*F**2/(2.0*xi))*p**2
    return (Gd + GgApp - _sig2/(2.0*xi))/dl

def fields(reg, lk, Z, Y, lr, l3, lx):
    """Map production states -> dict(v, i_d, i_g, i_r) of (n,1) arrays."""
    lk = np.asarray(lk, float).reshape(-1, 1)
    Z = np.clip(np.asarray(Z, float).reshape(-1, 1), 1e-6, 1 - 1e-6)
    Y = np.asarray(Y, float).reshape(-1, 1)
    lr = np.asarray(lr, float).reshape(-1, 1)
    l3 = np.clip(np.asarray(l3, float).reshape(-1, 1), 0.0, 1/3.)
    lx = np.clip(np.asarray(lx, float).reshape(-1, 1),
                 _lxgrid[0], _lxgrid[-1])
    xi = np.exp(lx)
    lkd = lk + np.log(1 - Z)
    lkg = lk + np.log(Z)
    pretech = reg in ('PreDamagePreTech', 'PostDamagePreTech')
    predam = reg in ('PreDamagePreTech', 'PreDamagePostTech')
    if reg not in ('PreDamagePreTech', 'PreDamagePostTech',
                   'PostDamagePreTech', 'PostDamagePostTech'):
        raise ValueError('unknown regime %r' % reg)

    # ---- logR entering value/policies: collapsed for post-tech (no net input)
    lr_star = lr if pretech else (lkg - gapstar)
    Aecon = Ag if pretech else Agpp          # economy green productivity
    Aprod = Ag if pretech else Agpp          # production A_g of the regime

    # ---- value + climate slope (V_Y) + fossil
    base = aD*lkd + aG*lkg + aR*lr_star
    if predam:
        Yc = np.clip(Y, _Ygrid[0], _Ygrid[-1])
        key = '1' if pretech else '2'
        u = _spl['u' + key].ev(lx.ravel(), Yc.ravel()).reshape(-1, 1)
        q = _spl['q' + key].ev(lx.ravel(), Yc.ravel()).reshape(-1, 1)
        v = base + u
        g1 = gamma1
        F = _Fpos(np.minimum(q, -1e-6), xi)
    else:
        g1 = gamma1 + kappa*l3
        p = -g1*m
        cpp = _c_pp(g1, xi)
        cc = cpp if not pretech else (cpp - _Delta_c(xi))
        v = base + p*Y + cc
        F = _Fpos(p, xi)

    # ---- policies (raw economy rates at production states)
    i_d = sd*Ad*_clipexp(-gamma0 - g1*Y + nn*(np.log(chiF*F) - lkd))
    i_g = sg*Aecon*_clipexp(be*(np.log(chi) + lr_star - lkg))
    if pretech:
        i_r = sr*Aecon*Z**(1-be)*_clipexp(be*(np.log(chi) + lr - lk))
    else:
        i_r = None

    # ---- guards: head clip, then production-budget proportional rescale
    n_headclip = int(np.sum(i_d > HEADMAX) + np.sum(i_g > HEADMAX)
                     + (np.sum(i_r > HEADMAX) if i_r is not None else 0))
    i_d = np.minimum(i_d, HEADMAX)
    i_g = np.minimum(i_g, HEADMAX)
    if i_r is not None:
        i_r = np.clip(i_r, 1e-8, HEADMAX)
    ir_ = i_r if i_r is not None else 0.0
    gross = Ad*(1 - Z) + Aprod*Z
    spend = i_d*(1 - Z) + i_g*Z + ir_
    CK = gross - spend
    bad = CK < CK_FLOOR
    scale = np.where(bad, (gross - CK_FLOOR)/np.maximum(spend, 1e-12), 1.0)
    i_d = i_d*scale
    i_g = i_g*scale
    if i_r is not None:
        i_r = np.maximum(i_r*scale, 1e-8)
    fields.last_guard_stats = dict(
        regime=reg, n=int(len(lk)), n_headclip=n_headclip,
        frac_budget_rescaled=float(np.mean(bad)),
        CK_min_postguard=float(np.min(gross - (i_d*(1-Z) + i_g*Z
                                               + (i_r if i_r is not None else 0.0)))))
    return dict(v=v, i_d=i_d, i_g=i_g, i_r=i_r)


if __name__ == '__main__':
    # smoke test: LHS sample per regime; shapes, finiteness, guard stats,
    # spline-node round trip, x0 values
    rng = np.random.RandomState(5)
    N = 20000
    print('map:', MAP_NAME, '| npz:', _NPZ)
    for reg in ('PreDamagePreTech', 'PreDamagePostTech',
                'PostDamagePreTech', 'PostDamagePostTech'):
        ylo = 2.5 if reg.startswith('Post') else 0.0
        lk = rng.uniform(4, 7, (N, 1)); Zs = rng.uniform(0.01, 0.99, (N, 1))
        Ys = rng.uniform(ylo, 4, (N, 1)); lrs = rng.uniform(1, 6, (N, 1))
        l3s = rng.uniform(0, 1/3., (N, 1))
        lxs = rng.uniform(np.log(0.05), np.log(148.6), (N, 1))
        F = fields(reg, lk, Zs, Ys, lrs, l3s, lxs)
        gs = fields.last_guard_stats
        assert F['v'].shape == (N, 1)
        for kk in ('v', 'i_d', 'i_g'):
            assert np.all(np.isfinite(F[kk])), (reg, kk)
        if reg.endswith('PostTech'):
            assert F['i_r'] is None
        else:
            assert F['i_r'].shape == (N, 1) and np.all(np.isfinite(F['i_r']))
            assert np.all(F['i_r'] > 0)
        print('%-20s v[%7.3f,%7.3f]  i_d[%.4f,%.4f]  i_g[%.4f,%.4f]  i_r[%s]  '
              'headclip=%d budget_rescaled=%.3f CKmin=%.2e' %
              (reg, F['v'].min(), F['v'].max(), F['i_d'].min(), F['i_d'].max(),
               F['i_g'].min(), F['i_g'].max(),
               ('%.5f,%.4f' % (F['i_r'].min(), F['i_r'].max()))
               if F['i_r'] is not None else '-',
               gs['n_headclip'], gs['frac_budget_rescaled'],
               gs['CK_min_postguard']))
    # x0 sanity (t = 0 state, xi = 148.6): expect i_d ~ 0.066, i_g ~ 0.065,
    # i_r ~ 0.0195 per the design
    x0 = [np.array([[np.log(880.)]]), np.array([[0.7]]), np.array([[1.1]]),
          np.array([[np.log(11.2)]]), np.array([[0.]]),
          np.array([[np.log(148.6)]])]
    F = fields('PreDamagePreTech', *x0)
    print('x0 check: v=%.4f i_d=%.5f i_g=%.5f i_r=%.5f' %
          (F['v'][0, 0], F['i_d'][0, 0], F['i_g'][0, 0], F['i_r'][0, 0]))
    # spline node round-trip at a grid point
    iy, ix = 100, 12
    xtest = [np.array([[5.0]]), np.array([[0.5]]),
             np.array([[_Ygrid[iy]]]), np.array([[3.0]]), np.array([[0.1]]),
             np.array([[_lxgrid[ix]]])]
    Ft = fields('PreDamagePreTech', *xtest)
    u_direct = _d['U1'][ix, iy]
    v_reconstructed = (aD*(5.0 + np.log(0.5)) + aG*(5.0 + np.log(0.5))
                       + aR*3.0 + u_direct)
    print('spline node round-trip: |v_map - v_table| = %.2e' %
          abs(Ft['v'][0, 0] - v_reconstructed))
    print('SMOKE TEST OK')
