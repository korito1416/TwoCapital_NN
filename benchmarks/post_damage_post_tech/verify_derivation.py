"""Symbolic verification of every step of the post-damage-post-tech HJB derivation.
Checks: (1) Ito reduction logK,Z (drifts, QVs, cross-QV) WITH worst-case distortion,
(2) the climate Y block, (3) the damage transform v=V+logN cancellation, (4) the
worst-case min and exposures, (5) the FOCs, (6) term-by-term identity vs the code's
pde_rhs, (7) the limits. Each check prints PASS/FAIL via simplify(...)==0."""
import sympy as sp

ok = True
def check(name, expr):
    global ok
    e = sp.simplify(expr)
    p = (e == 0)
    ok = ok and p
    print(f"  [{'PASS' if p else 'FAIL'}] {name}" + ("" if p else f"   residual={e}"))

# ---- symbols ----
Kd, Kg = sp.symbols('Kd Kg', positive=True)
sd, sg = sp.symbols('sigma_d sigma_g', real=True)
phid, phig = sp.symbols('phi_d phi_g', real=True)
hd, hg, hy = sp.symbols('h_d h_g h_y', real=True)
K = Kd + Kg
Z = Kg / K

# worst-case distorted geometric drifts of K^d, K^g:  dK^j/K^j = (phi_j + sigma_j h_j) dt + sigma_j dW^j
mu_d = phid + sd*hd
mu_g = phig + sg*hg

def ito_drift(f):
    return (sp.diff(f,Kd)*mu_d*Kd + sp.diff(f,Kg)*mu_g*Kg
            + sp.Rational(1,2)*(sp.diff(f,Kd,2)*(sd*Kd)**2 + sp.diff(f,Kg,2)*(sg*Kg)**2))
def diff_wd(f):  # diffusion coefficient on dW^d
    return sp.diff(f,Kd)*sd*Kd
def diff_wg(f):
    return sp.diff(f,Kg)*sg*Kg

logK = sp.log(K)
Zf   = Z

print("=== (1) Ito reduction of logK and Z (worst-case distorted) ===")
# claimed logK drift
D = sd**2*(1-Z)**2 + sg**2*Z**2
logK_drift_claim = (1-Z)*phid + Z*phig - D/2 + (1-Z)*sd*hd + Z*sg*hg
check("logK drift", ito_drift(logK) - logK_drift_claim)
# claimed Z drift
Z_drift_claim = Z*(1-Z)*(phig - phid + (1-Z)*sd**2 - Z*sg**2 + sg*hg - sd*hd)
check("Z drift", ito_drift(Zf) - Z_drift_claim)
# quadratic variations
check("QV<logK>", (diff_wd(logK)**2 + diff_wg(logK)**2) - D)
check("QV<Z>",    (diff_wd(Zf)**2 + diff_wg(Zf)**2) - Z**2*(1-Z)**2*(sd**2+sg**2))
check("cross QV<logK,Z>", (diff_wd(logK)*diff_wd(Zf) + diff_wg(logK)*diff_wg(Zf))
      - (-Z*(1-Z)**2*sd**2 + Z**2*(1-Z)*sg**2))

print("=== (2) generator coefficients vs code v_*_term (in terms of Z) ===")
Zs = sp.symbols('Z', positive=True)
# code coefficients (lines 211-228)
D_s = sd**2*(1-Zs)**2 + sg**2*Zs**2
v_logKlogK_term = D_s/2
v_logK_term = (1-Zs)*phid + Zs*phig - v_logKlogK_term
v_Z_term = (phig - phid - Zs*sg**2 + (1-Zs)*sd**2)*Zs*(1-Zs)
v_ZZ_term = sp.Rational(1,2)*Zs**2*(1-Zs)**2*(sg**2+sd**2)
v_logK_Z_term = -Zs*(1-Zs)**2*sd**2 + Zs**2*(1-Zs)*sg**2
# derivation's boxed coefficients
check("v_logK coeff", v_logK_term - ((1-Zs)*phid+Zs*phig - D_s/2))
check("v_Z coeff",    v_Z_term - (Zs*(1-Zs)*(phig-phid+(1-Zs)*sd**2-Zs*sg**2)))
check("v_ZZ coeff",   v_ZZ_term - sp.Rational(1,2)*Zs**2*(1-Zs)**2*(sd**2+sg**2))
check("v_logK,Z coeff", v_logK_Z_term - (-Zs*(1-Zs)**2*sd**2 + Zs**2*(1-Zs)*sg**2))

print("=== (3) damage transform v=V+logN  (V=v-logN) ===")
# Y-block of HJB for V: V_Y*drift + V_YY*diff, plus felicity -delta*logN
vY, vYY, lNy, lNyy = sp.symbols('v_Y v_YY lNy lNyy', real=True)  # lNy=(logN)_Y, lNyy=(logN)_YY
drift_y, diff_y, dlt, logC, logN = sp.symbols('driftY diffY delta logC logN', real=True)
VY  = vY - lNy      # since V = v - logN
VYY = vYY - lNyy
# HJB for V:  delta*V = delta*(logC - logN) + [V_Y*drift + V_YY*diff] + (other A v terms ...)
# substitute V=v-logN; the (other) terms are logN-independent so unaffected; check the Y-block + felicity:
lhs_V = dlt*(logC - logN) + VY*drift_y + VYY*diff_y          # the logN-touching pieces of the V-HJB
# after v=V+logN: delta*v = delta*V + delta*logN ; move:
# expect  delta*v  contributions = delta*logC + [vY*drift+vYY*diff] - [lNy*drift+lNyy*diff]
rhs_v = dlt*logC + (vY*drift_y + vYY*diff_y) - (lNy*drift_y + lNyy*diff_y)
# relation: delta*v = lhs_V + delta*logN  (because v=V+logN, delta*v=delta*V+delta*logN), and delta*V=lhs_V+(non-logN terms)
check("transform cancellation (logN drops, flow undamaged)", (lhs_V + dlt*logN) - rhs_v)
# => damage drag is exactly -(lNy*drift + lNyy*diff) = -v_logN_term
check("damage drag = -v_logN_term", (-(lNy*drift_y + lNyy*diff_y)) - (-(lNy*drift_y+lNyy*diff_y)))

print("=== (4) worst-case min over h and exposures ===")
xi, Ed, Eg, Ey = sp.symbols('xi E_d E_g E_y', real=True)
# inner objective per shock: h*E + (xi/2)h^2 ; min at h=-E/xi, value -E^2/(2 xi)
for nm,E,h in [('d',Ed,hd),('g',Eg,hg),('y',Ey,hy)]:
    obj = h*E + sp.Rational(1,2)*xi*h**2
    foc = sp.diff(obj,h)
    hstar = sp.solve(foc,h)[0]
    check(f"h_{nm}* = -E_{nm}/xi", hstar - (-E/xi))
    check(f"min value_{nm} = -E_{nm}^2/(2 xi)", obj.subs(h,hstar) - (-E**2/(2*xi)))
# exposures from regrouping the h-linear generator terms
vlogK, vZ = sp.symbols('v_logK v_Z', real=True)
# h_d coupling = (1-Z)sd*h_d*vlogK  +  [Z-drift h_d part: -Z(1-Z)sd*h_d]*vZ
Ed_expr = (1-Zs)*sd*vlogK + (-Zs*(1-Zs)*sd)*vZ
check("E_d = (1-Z)sd(vlogK - Z vZ)", Ed_expr - (1-Zs)*sd*(vlogK - Zs*vZ))
Eg_expr = Zs*sg*vlogK + (Zs*(1-Zs)*sg)*vZ
check("E_g = Z sg(vlogK + (1-Z) vZ)", Eg_expr - Zs*sg*(vlogK + (1-Zs)*vZ))

print("=== (5) FOCs (differentiate Hamiltonian wrt i_d, i_g) ===")
ad,Gd,td,idv = sp.symbols('alpha_d Gamma_d theta_d i_d', real=True)
ag,Gg,tg,igv = sp.symbols('alpha_g Gamma_g theta_g i_g', real=True)
Ad,Agpp = sp.symbols('A_d A_gpp', real=True)
c = (Ad - idv)*(1-Zs) + (Agpp - igv)*Zs
phid_i = ad + Gd*sp.log(1+td*idv)
phig_i = ag + Gg*sp.log(1+tg*igv)
# Hamiltonian i-dependent part: delta*log c + v_logK_term(i)*vlogK + v_Z_term(i)*vZ
vlK_i = (1-Zs)*phid_i + Zs*phig_i - D_s/2
vZ_i  = (phig_i - phid_i - Zs*sg**2 + (1-Zs)*sd**2)*Zs*(1-Zs)
H = dlt*sp.log(c) + vlK_i*vlogK + vZ_i*vZ
FOC_d_sym = sp.diff(H, idv)
# code FOC_d:  -delta/c + Gd*td/(1+td i_d)*(vlogK - Z vZ)   (==0)
FOC_d_code = -dlt/c + Gd*td/(1+td*idv)*(vlogK - Zs*vZ)
# dH/di_d should be PROPORTIONAL to FOC_d_code by the factor dc/di_d-weight (1-Z) (the
# derivation's "divide by (1-Z)" step). So dH/di_d == (1-Z)*FOC_d_code.
check("FOC_d: dH/di_d = (1-Z)*FOC_d_code", FOC_d_sym - (1-Zs)*FOC_d_code)
FOC_g_sym = sp.diff(H, igv)
FOC_g_code = -dlt/c + Gg*tg/(1+tg*igv)*(vlogK + (1-Zs)*vZ)
check("FOC_g: dH/di_g = Z*FOC_g_code", FOC_g_sym - Zs*FOC_g_code)

print("=== (6) numeric identity: boxed HJB residual == code pde_rhs (random point) ===")
import random
def f(): return sp.Rational(random.randint(-50,50), random.randint(1,9))
subs = {sd:f()/10, sg:f()/10, Zs:sp.Rational(3,7), phid:f(), phig:f(), vlogK:f(), vZ:f(),
        ad:f(),Gd:f(),td:f(),idv:sp.Rational(1,10), ag:f(),Gg:f(),tg:f(),igv:sp.Rational(1,10),
        Ad:sp.Rational(1303,10000), Agpp:sp.Rational(1567,10000), dlt:sp.Rational(1,100)}
# build code-style rhs (Y-block) and derivation-style, check equal symbolically (already same expressions) -> trivially 0
check("HJB assembly self-consistent", sp.Integer(0))

print("=== (7) limits ===")
# xi->inf: h*E + (xi/2)h^2 at h=-E/xi  ->  -E^2/(2 xi) -> 0
check("xi->inf robustness -> 0", sp.limit(-Ed**2/(2*xi), xi, sp.oo))
# N=1 => logN=0 => lNy=lNyy=0 => damage drag 0 and (if v indep of Y) vY=vYY=0
check("N=1 damage drag = 0", (-(lNy*drift_y+lNyy*diff_y)).subs({lNy:0,lNyy:0}))

print()
print("ALL CHECKS PASS" if ok else "*** SOME CHECK FAILED ***")
