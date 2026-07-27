#!/usr/bin/env python
"""GHKM economy figures (economy zoo build 3). Reads outputs/ghkm.npz, writes
figures/ghkm_{comparative_statics,value_policies,fields,paths}.png per the
design's plots spec: value + policies (s_d, s_r shares -> i_d, i_r rates; F* fuel
choice), lifted fields, and 100-year simulated paths with jumps at hazard.

Run:  python plot_ghkm.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
d = np.load(os.path.join(HERE, '..', 'outputs', 'ghkm.npz'))

g = lambda k: float(d[k])
dl, td, tg, be, nn, G = (g('delta'), g('theta_d'), g('theta_g'), g('beta'),
                         g('nu'), g('Gamma'))
b, w = g('b'), g('w')
aD, aG, aR, m = g('a_d'), g('a_g'), g('a_r'), g('m')
sd, sg, sr = g('s_d'), g('s_g'), g('s_r')
Ad, Ag, Agpp = g('A_d'), g('A_g'), g('A_gpp')
gamma0, gamma1, kappa = g('gamma0'), g('gamma1'), g('kappa')
chi, chiF, F0, gapstar = g('chi'), g('chi_F'), g('F0'), g('gapstar')
alpha0, thc = g('alpha0'), g('theta_c')
varpi_n, varpi_g = g('varpi_n'), g('varpi_g')
K0, Z0, Y0, R0 = g('K0'), g('Z0'), g('Y0'), g('R0')
lam3, gam1l, p_l, Fl_inf = d['lam3'], d['gam1l'], d['p_l'], d['Fl_inf']
Ygrid, logxigrid = d['Ygrid'], d['logxigrid']
U1, U2, Q1, Q2, F1t = d['U1'], d['U2'], d['Q1'], d['Q2'], d['F1']
p1inf, c1_inf, c2_inf = g('p1inf'), g('c1_inf'), g('c2_inf')
c_pp_inf, c_pre_inf = d['c_pp_inf'], d['c_pre_inf']
sgd, sgg, sgr = g('sigma_d'), g('sigma_g'), g('sigma_r')

def Fpos(q, xi):
    q = np.asarray(q, float)
    Bq = -q*b; Cq = -nn*dl*m; Aq = q**2*w**2/np.asarray(xi, float)
    return -2.0*Cq/(Bq + np.sqrt(Bq**2 - 4.0*Aq*Cq))

spl_u1 = RectBivariateSpline(logxigrid, Ygrid, U1, kx=3, ky=3)
spl_q1 = RectBivariateSpline(logxigrid, Ygrid, Q1, kx=3, ky=3)
C = plt.rcParams['axes.prop_cycle'].by_key()['color']

# =================================================== 1. comparative statics
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
bes = np.linspace(0.05, 0.6, 100)
Dv = dl + G*bes + G*(1-bes)
ax[0, 0].plot(bes, tg*(1-bes)*(dl+G)/Dv, label='$a_g$', color=C[2])
ax[0, 0].plot(bes, tg*bes*(dl+G)/Dv, label='$a_r$', color=C[0])
ax[0, 0].axhline(aD, color=C[3], ls='--', lw=1, label='$a_d$ (indep. of beta)')
ax[0, 0].axvline(be, color='k', lw=0.6, ls=':')
ax[0, 0].set_xlabel('knowledge share beta'); ax[0, 0].set_ylabel('value coefficient')
ax[0, 0].legend(); ax[0, 0].set_title('coefficients vs beta ($a_g+a_r=theta_g$)')
ax[0, 1].plot(bes, G*(1-bes)/(G+dl), label='$s_g$', color=C[2])
ax[0, 1].plot(bes, G*bes/(G+dl), label='$s_r$', color=C[0])
ax[0, 1].axhline(sd, color=C[3], ls='--', lw=1, label='$s_d$')
ax[0, 1].axvline(be, color='k', lw=0.6, ls=':')
ax[0, 1].set_xlabel('beta'); ax[0, 1].set_ylabel('investment share')
ax[0, 1].legend(); ax[0, 1].set_title('constant policy shares vs beta')
g1s = np.linspace(0.10, 0.30, 100)
ax[1, 0].plot(g1s, -g1s*m, color=C[3], label='post-damage $p^{l}=-gamma_1^l m$')
ax[1, 0].plot(gam1l, p_l, 'o', color=C[3])
ax[1, 0].axhline(p1inf, color=C[0], ls='--', lw=1,
                 label='pre-damage $p$ (jump-expectation)')
ax[1, 0].set_xlabel('$gamma_1^{l}$'); ax[1, 0].set_ylabel('$V_Y$')
ax[1, 0].legend(); ax[1, 0].set_title('marginal climate value vs damage slope')
xs = np.arange(6)
F_005 = Fpos(p_l, 0.05)
ax[1, 1].bar(xs - 0.18, np.r_[F0, Fl_inf], 0.36, label='$xi=infty$', color=C[0])
ax[1, 1].bar(xs + 0.18, np.r_[Fpos(spl_q1.ev(np.log(0.05), Y0), 0.05), F_005],
             0.36, label='$xi=0.05$', color=C[1])
ax[1, 1].set_xticks(xs)
ax[1, 1].set_xticklabels(['pre\n(at $x_0$)'] + ['$l=%d$' % (i+1) for i in range(5)])
ax[1, 1].set_ylabel('fossil flow $F^*$')
ax[1, 1].legend(); ax[1, 1].set_title('fossil choice: discrete cut at damage revelation')
fig.suptitle('GHKM comparative statics (flat-SCC logic: $mu_Y^l = nu delta/gamma_1^l$)')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'ghkm_comparative_statics.png'), dpi=140)
plt.close(fig)

# =================================================== 2. value + policies
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
xis_show = [0.05, 0.15, 1.0, 10.0, 148.6]
for i, xv in enumerate(xis_show):
    ax[0, 0].plot(Ygrid, spl_u1.ev(np.full_like(Ygrid, np.log(xv)), Ygrid),
                  color=plt.cm.viridis(i/(len(xis_show)-1)), label='$xi$=%.2f' % xv)
ax[0, 0].plot(Ygrid, p1inf*Ygrid + c1_inf, 'k--', lw=1, label='$xi=infty$ (linear)')
ax[0, 0].set_xlabel('Y'); ax[0, 0].set_ylabel('$u_1(Y;xi)=f_1+c_1$')
ax[0, 0].legend(fontsize=8)
ax[0, 0].set_title('pre-damage pre-tech value: restored log-xi geometry')
lks = np.linspace(4, 7, 50)
x0 = dict(lk=np.log(K0), Z=Z0, lr=np.log(R0))
for lab, pY, cc, col in [('PreD-PreT', p1inf*Y0, c1_inf, C[0]),
                         ('PreD-PostT', p1inf*Y0, c2_inf, C[2]),
                         ('PostD-PreT (l=3)', p_l[2]*2.75, c_pre_inf[2], C[1]),
                         ('PostD-PostT (l=3)', p_l[2]*2.75, c_pp_inf[2], C[3])]:
    v = (aD*(lks + np.log(1-Z0)) + aG*(lks + np.log(Z0)) + aR*np.log(R0) + pY + cc)
    ax[0, 1].plot(lks, v, color=col, label=lab)
ax[0, 1].set_xlabel('logK'); ax[0, 1].set_ylabel('V')
ax[0, 1].legend(fontsize=8)
ax[0, 1].set_title('value in logK: exact slope $V_{logK}=a_d+a_g=%.3f$' % (aD+aG))
lkd0 = np.log(K0*(1-Z0))
for i, (lab, g1v, Fv, col) in enumerate(
        [('pre-damage (xi=0.05)', gamma1,
          Fpos(np.minimum(spl_q1.ev(np.log(0.05)*np.ones_like(Ygrid), Ygrid), -1e-6),
               0.05), C[0])] +
        [('post-damage l=%d' % (l+1), gam1l[l], Fl_inf[l]*np.ones_like(Ygrid),
          plt.cm.autumn(l/4.)) for l in (0, 2, 4)]):
    i_d = sd*Ad*np.exp(-gamma0 - g1v*Ygrid + nn*(np.log(chiF*Fv) - lkd0))
    ax[1, 0].plot(Ygrid, i_d, color=col, label=lab)
ax[1, 0].set_xlabel('Y'); ax[1, 0].set_ylabel('$i_d = I_d/K_d$')
ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title('dirty investment at $x_0$: damage strangulation in Y')
for i, xv in enumerate(xis_show):
    q = spl_q1.ev(np.full_like(Ygrid, np.log(xv)), Ygrid)
    ax[1, 1].plot(Ygrid, Fpos(np.minimum(q, -1e-6), xv),
                  color=plt.cm.viridis(i/(len(xis_show)-1)), label='$xi$=%.2f' % xv)
ax[1, 1].axhline(F0, color='k', ls='--', lw=1, label='$xi=infty$')
ax[1, 1].set_xlabel('Y'); ax[1, 1].set_ylabel('$F(Y;xi)$ pre-damage')
ax[1, 1].legend(fontsize=8); ax[1, 1].set_title('robust fossil choice')
fig.suptitle('GHKM value and policies')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'ghkm_value_policies.png'), dpi=140)
plt.close(fig)

# =================================================== 3. lifted production-space fields
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
lkv = np.linspace(4, 7, 80); Yv = np.linspace(0, 4, 80)
LK, YY = np.meshgrid(lkv, Yv)
q = spl_q1.ev(np.log(0.05)*np.ones(YY.size), YY.ravel()).reshape(YY.shape)
Fv = Fpos(np.minimum(q, -1e-6), 0.05)
ID = sd*Ad*np.exp(-gamma0 - gamma1*YY + nn*(np.log(chiF*Fv) - (LK + np.log(1-Z0))))
pc = ax[0].pcolormesh(LK, YY, ID, shading='auto', cmap='viridis')
fig.colorbar(pc, ax=ax[0], label='$i_d$')
ax[0].set_xlabel('logK'); ax[0].set_ylabel('Y')
ax[0].set_title('$i_d$ over (logK, Y) at Z=0.7 (PreD-PreT, xi=0.05)')
Zv = np.linspace(0.01, 0.99, 80); gapv = np.linspace(-5, 2, 80)
ZZ, GP = np.meshgrid(Zv, gapv)          # GP = logR - logK
IR = sr*Ag*ZZ**(1-be)*np.exp(be*(np.log(chi) + GP))
pc = ax[1].pcolormesh(ZZ, GP, IR, shading='auto', cmap='viridis')
fig.colorbar(pc, ax=ax[1], label='$i_r = I_r/K$')
ax[1].set_xlabel('Z'); ax[1].set_ylabel('logR - logK')
ax[1].set_title('$i_r$ over (Z, logR-logK) (pre-tech; rate on total K)')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'ghkm_fields.png'), dpi=140)
plt.close(fig)

# =================================================== 4. 100-year simulated paths
rng = np.random.RandomState(3)
npaths, T, dt = 3000, 100.0, 1/12.
nst = int(T/dt)
lkd = np.full(npaths, np.log(K0*(1-Z0))); lkg = np.full(npaths, np.log(K0*Z0))
lr = np.full(npaths, np.log(R0)); Y = np.full(npaths, Y0)
tau_n = rng.exponential(1/varpi_n, npaths); tau_g = rng.exponential(1/varpi_g, npaths)
lidx = rng.randint(0, 5, npaths)
traj = {k: np.empty((nst+1, npaths)) for k in ('lkd', 'lkg', 'lr', 'Y', 'F', 'Z')}
lthc = np.log(thc); lchi = np.log(chi)
F1inf = float(d['F1inf'])
for i in range(nst+1):
    t = i*dt
    dm = tau_n <= t; tc = tau_g <= t
    g1p = np.where(dm, gam1l[lidx], gamma1)
    Fp = np.where(dm, Fl_inf[lidx], F1inf)
    Ap = np.where(tc, Agpp, Ag)
    traj['lkd'][i], traj['lkg'][i], traj['lr'][i], traj['Y'][i] = lkd, lkg, lr, Y
    traj['F'][i] = Fp
    traj['Z'][i] = 1/(1 + np.exp(lkd - lkg))
    if i == nst:
        break
    logYd = np.log(Ad) - gamma0 - g1p*Y + (1-nn)*lkd + nn*np.log(chiF*Fp)
    logYg = np.log(Ap) + be*lchi + (1-be)*lkg + be*lr
    lkd = lkd + (alpha0 + G*(lthc + np.log(sd) + logYd - lkd))*dt \
        + sgd*np.sqrt(dt)*rng.randn(npaths)
    lkg = lkg + (alpha0 + G*(lthc + np.log(sg) + logYg - lkg))*dt \
        + sgg*np.sqrt(dt)*rng.randn(npaths)
    lr = lr + (alpha0 + G*(lthc + np.log(sr) + logYg - lr))*dt \
        + sgr*np.sqrt(dt)*rng.randn(npaths)
    Y = Y + b*Fp*dt + w*Fp*np.sqrt(dt)*rng.randn(npaths)
tv = np.arange(nst+1)*dt
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
panels = [('lkd', '$logK_d$'), ('lkg', '$logK_g$'), ('lr', '$logR$'),
          ('Z', 'green share $Z_t$'), ('Y', 'temperature $Y_t$'),
          ('F', 'fossil $F_t$')]
for k, (key, lab) in enumerate(panels):
    a = ax[k//3, k % 3]
    for j in range(12):
        a.plot(tv, traj[key][:, j], color='grey', lw=0.5, alpha=0.6)
    a.plot(tv, traj[key].mean(axis=1), color='crimson', lw=2, label='mean (3000 paths)')
    a.set_xlabel('years'); a.set_title(lab)
    if k == 0:
        a.legend(fontsize=8)
fig.suptitle('GHKM 100-year simulated paths, xi=infty policies, jumps at hazards '
             '(varpi_n=%.3f, varpi_g=%.3f); grey = sample paths' % (varpi_n, varpi_g))
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'ghkm_paths.png'), dpi=140)
plt.close(fig)
print('wrote 4 figures to', FIG)
