"""G1 verification for the ABSORB economy (Nelson-Phelps directed-absorption catch-up).

Checks the CLOSED-FORM blocks of the ABSORB design (design_absorb.json, with the
synthesis corrections lambda0 = (lambda3/2)*yhat^2 and tau = 1.25):

  (A) Dirty block (algebraic): i_d* = (Gamma*theta*A_d - delta)/(theta*(delta+Gamma)),
      b_d = beta*[log(A_d - i_d*) + (phi_d(i_d*) - sigma_d^2/2)/delta].
      - sympy SYMBOLIC: FOC(i_d*) == 0 exactly (rational arithmetic), and the
        dirty HJB block residual is identically 0.
      - second-order: Hamiltonian is strictly concave in i_d (phi concave + log concave).
  (B) Post-damage climate quadratics w^l(Y) = -(a_l Y^2 + b_l Y + c_l) with
        a_l = (lambda2+lambda3)/2,
        b_l = (lambda1 - lambda3*yhat) + 2*a_l*iota*thetabar/delta,
        c_l = (lambda3/2)*yhat^2 + (b_l*iota*thetabar + a_l*iota^2*varsigma^2)/delta.
      - sympy SYMBOLIC: the ODE residual
          -delta*w - delta*logN_post(Y) + iota*thetabar*w' + (iota^2 varsigma^2/2) w''
        is identically 0 in (Y, lambda3).
      - sympy SYMBOLIC: value AND slope continuity of logN at yhat=2.5
        (this is exactly the lambda0-correction check).
  (C) Green-block closed-form control map: given (s, F', A_g(s)), the quadratic root
        B^2 c^2 + M c - (A_g + 1/theta) = 0,  B = psi0 e^{-s/2} F' /(2 delta (1-beta)),
        M = 1 + Gamma((1-beta)-F')/(delta(1-beta)),
      with i_g = Gamma((1-beta)-F') c/(delta(1-beta)) - 1/theta, i_r = B^2 c^2
      satisfies BOTH FOCs and the budget c+i_g+i_r = A_g(s).
  (D) NUMERIC: 1e4 random states -> max |residual| <= 1e-12 for (A),(B),(C).

Run:  python verify_absorb_sympy.py
"""
import numpy as np
import sympy as sp

# ----------------------------------------------------------------------------- params
delta, Gamma, theta = sp.Rational(1, 100), sp.Rational(6, 100), sp.Rational(167, 10)
A_d = sp.Rational(1303, 10000)
sigma = sp.Rational(1, 100)                      # sigma_d = sigma_g
beta = sp.Rational(34, 100)
lam1 = sp.Rational(17675, 100000000)             # 0.00017675
lam2 = sp.Rational(44, 10000)                    # 0.0044
yhat = sp.Rational(5, 2)
# iota = eta*A_d*(1-Z0)*K0 (exact rationals: eta=0.291, Z0=0.7, K0=880)
iota = sp.Rational(291, 1000) * A_d * sp.Rational(3, 10) * 880
thetabar = sp.Rational(186, 100000)              # 1.86e-3
varsig = sp.Rational(12, 10) * thetabar

FAILS = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not ok:
        FAILS.append(name)


print("=" * 78)
print("(A) dirty block — symbolic")
i = sp.symbols("i", positive=True)
phi = -sp.Rational(35, 1000) + Gamma * sp.log(1 + theta * i)
i_d_star = (Gamma * theta * A_d - delta) / (theta * (delta + Gamma))
FOC = delta * beta / (A_d - i) - beta * Gamma * theta / (1 + theta * i)
check("i_d* solves FOC exactly", sp.simplify(FOC.subs(i, i_d_star)) == 0,
      f"(i_d*={float(i_d_star):.6f})")
# b_d makes the dirty HJB block identically zero (logK_d terms cancel by delta*beta)
c_d = A_d - i_d_star
b_d = beta * (sp.log(c_d) + (phi.subs(i, i_d_star) - sigma**2 / 2) / delta)
resid_dirty = (-delta * b_d + delta * beta * sp.log(A_d - i) + beta * (phi - sigma**2 / 2)
               ).subs(i, i_d_star)
check("dirty HJB block residual == 0", sp.simplify(resid_dirty) == 0)
# strict concavity of the dirty Hamiltonian in i_d
H_d = delta * beta * sp.log(A_d - i) + beta * phi
H_d2 = sp.diff(H_d, i, 2)
check("dirty Hamiltonian strictly concave (H''(i_d*)<0)",
      float(H_d2.subs(i, i_d_star)) < 0, f"(H''={float(H_d2.subs(i, i_d_star)):.3f})")

print("(B) post-damage climate quadratics — symbolic (identity in Y and lambda3)")
Y, lam3 = sp.symbols("Y lambda3", real=True)
lt2 = lam2 + lam3                                 # lambda_tilde_2
lt1 = lam1 - lam3 * yhat                          # lambda_tilde_1  (ybar=yhat=2.5)
lt0 = lam3 / 2 * yhat**2                          # lambda_tilde_0  (CORRECTED lambda0)
a_l = lt2 / 2
b_l = lt1 + 2 * a_l * iota * thetabar / delta
c_l = lt0 + (b_l * iota * thetabar + a_l * iota**2 * varsig**2) / delta
w = -(a_l * Y**2 + b_l * Y + c_l)
logN_post = lt0 + lt1 * Y + lt2 / 2 * Y**2
resid_post = (-delta * w - delta * logN_post + iota * thetabar * sp.diff(w, Y)
              + iota**2 * varsig**2 / 2 * sp.diff(w, Y, 2))
check("post-damage ODE residual == 0 identically", sp.expand(resid_post) == 0)
logN_pre = lam1 * Y + lam2 / 2 * Y**2
check("logN value continuity at yhat (lambda0 correction)",
      sp.simplify((logN_post - logN_pre).subs(Y, yhat)) == 0)
check("logN slope continuity at yhat",
      sp.simplify(sp.diff(logN_post - logN_pre, Y).subs(Y, yhat)) == 0)
# spot values for l=5 (lambda3 = 1/3)
a5, b5 = float(a_l.subs(lam3, sp.Rational(1, 3))), float(b_l.subs(lam3, sp.Rational(1, 3)))
check("l=5 coefficients match design (a=0.16887, b=-0.20436)",
      abs(a5 - 0.1688667) < 1e-6 and abs(b5 + 0.204361) < 1e-4, f"(a={a5:.6f}, b={b5:.6f})")

print("(C) green-block closed-form control map — symbolic")
s, Fp, cg = sp.symbols("s Fprime c_g", positive=True)
psi0 = sp.Rational(10583, 100000)
Ag = sp.symbols("A_g", positive=True)
B = psi0 * sp.exp(-s / 2) * Fp / (2 * delta * (1 - beta))
M = 1 + Gamma * ((1 - beta) - Fp) / (delta * (1 - beta))
c_root = (-M + sp.sqrt(M**2 + 4 * B**2 * (Ag + 1 / theta))) / (2 * B**2)
i_g_of = Gamma * ((1 - beta) - Fp) * cg / (delta * (1 - beta)) - 1 / theta
i_r_of = B**2 * cg**2
budget = cg + i_g_of + i_r_of - Ag
check("budget identity at quadratic root == 0",
      sp.simplify(budget.subs(cg, c_root)) == 0)
foc_g = delta * (1 - beta) / cg - Gamma * theta / (1 + theta * i_g_of) * ((1 - beta) - Fp)
foc_r = delta * (1 - beta) / cg - psi0 / 2 * i_r_of ** sp.Rational(-1, 2) * sp.exp(-s / 2) * Fp * sp.sqrt(i_r_of) / sp.sqrt(i_r_of)
# i_r FOC: delta(1-beta)/c = (psi0/2) i_r^(-1/2) e^(-s/2) F'
foc_r = delta * (1 - beta) / cg - psi0 / 2 * i_r_of ** sp.Rational(-1, 2) * sp.exp(-s / 2) * Fp
check("i_g FOC identity == 0 (any c_g)", sp.simplify(foc_g) == 0)
check("i_r FOC identity == 0 (any c_g>0)", sp.simplify(foc_r) == 0)

# ----------------------------------------------------------------------------- numeric
print("(D) numeric residuals at 1e4 random states")
rng = np.random.RandomState(0)
n = 10000
dl, Gm, th, Adf, sig, bt = 0.01, 0.06, 16.7, 0.1303, 0.01, 0.34
l1, l2, yh = 0.00017675, 0.0044, 2.5
io, tb, vs = float(iota), 1.86e-3, 1.2 * 1.86e-3
ps0 = 0.10583

# (A) numeric: dirty FOC + block residual (constants, but evaluate at random logK_d too)
idn = (Gm * th * Adf - dl) / (th * (dl + Gm))
phin = lambda x: -0.035 + Gm * np.log(1 + th * x)
bdn = bt * (np.log(Adf - idn) + (phin(idn) - sig**2 / 2) / dl)
lkd = rng.uniform(2, 8, n)
resA = (-dl * (bt * lkd + bdn) + dl * bt * (np.log(Adf - idn) + lkd)
        + bt * (phin(idn) - sig**2 / 2))
rA = np.abs(resA).max()
check("dirty block residual (with logK_d cancellation)", rA <= 1e-12, f"max={rA:.2e}")

# (B) numeric: post-damage quadratic ODE residual at random (Y, lambda3)
Yr = rng.uniform(0, 6, n)
l3r = rng.uniform(0, 1 / 3, n)
lt2n = l2 + l3r
lt1n = l1 - l3r * yh
lt0n = l3r / 2 * yh**2
an = lt2n / 2
bn = lt1n + 2 * an * io * tb / dl
cn = lt0n + (bn * io * tb + an * io**2 * vs**2) / dl
wn = -(an * Yr**2 + bn * Yr + cn)
wp = -(2 * an * Yr + bn)
wpp = -2 * an
resB = (-dl * wn - dl * (lt0n + lt1n * Yr + lt2n / 2 * Yr**2) + io * tb * wp
        + io**2 * vs**2 / 2 * wpp)
rB = np.abs(resB).max()
check("post-damage climate quadratic residual", rB <= 1e-12, f"max={rB:.2e}")

# (C) numeric: control map satisfies FOCs + budget at random (s, F', A_g)
sr = rng.uniform(-8, 7, n)
fpr = rng.uniform(1e-4, 0.659, n)
agr = rng.uniform(0.106, 0.1567, n)
Bn = ps0 * np.exp(-sr / 2) * fpr / (2 * dl * (1 - bt))
Mn = 1 + Gm * ((1 - bt) - fpr) / (dl * (1 - bt))
# numerically stable (citardauq) form of the positive quadratic root
cg_ = 2 * (agr + 1 / th) / (Mn + np.sqrt(Mn**2 + 4 * Bn**2 * (agr + 1 / th)))
ig_ = Gm * ((1 - bt) - fpr) * cg_ / (dl * (1 - bt)) - 1 / th
ir_ = Bn**2 * cg_**2
res_budget = cg_ + ig_ + ir_ - agr
res_focg = dl * (1 - bt) / cg_ - Gm * th / (1 + th * ig_) * ((1 - bt) - fpr)
res_focr = dl * (1 - bt) / cg_ - ps0 / 2 * ir_**-0.5 * np.exp(-sr / 2) * fpr
# scale-free residuals (the FOC magnitudes ~ delta/c ~ 0.3)
rC = max(np.abs(res_budget).max(),
         np.abs(res_focg * cg_ / (dl * (1 - bt))).max(),
         np.abs(res_focr * cg_ / (dl * (1 - bt))).max())
check("green control-map FOC+budget residual (relative)", rC <= 1e-12, f"max={rC:.2e}")
check("green controls feasible (c_g>0, i_r>=0, 1+theta*i_g>0)",
      bool((cg_ > 0).all() and (ir_ >= 0).all() and (1 + th * ig_ > 0).all()))

print("=" * 78)
if FAILS:
    print(f"G1 FAIL: {FAILS}")
    raise SystemExit(1)
print("G1 PASS: all symbolic identities exact; numeric max residual "
      f"{max(rA, rB, rC):.2e} <= 1e-12")
