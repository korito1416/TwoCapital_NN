"""ABSORB economy -> production warm-start map (economy-zoo pipeline).

Exports MAP_NAME, PROVENANCE, fields(reg, lk, Z, Y, lr, l3, lx) per the
models_terminal_anchor/make_map_anchor.py interface.

VALUE TRANSFORM (the exact object fit into v_nn; verifier: check THIS formula):
    v(x) = logK + beta*log(1-Z) + (1-beta)*log(Z) + b_d + F_reg(s) + w_reg(Y; lambda3)
with beta = 0.34, s = logR - logK - log(Z) (= logR - logK_g under the exact change of
variables logK_d = logK + log(1-Z), logK_g = logK + log(Z)), and
  F_reg = F(s)          solved 1-D green/knowledge ODE   (pre-tech regimes)
        = F(+inf)       closed-form saturation constant  (post-tech regimes)
  w_reg = w_pre(Y)      solved 1-D damage-jump BVP       (pre-damage regimes;
                        embeds the jump anticipation over the 5 equal-weight
                        lambda3 levels, so it does NOT depend on the l3 input)
        = w^l(Y;lambda3) = -(a*Y^2 + b*Y + c)            (post-damage regimes;
                        ANALYTIC in the continuous l3 input:
                        a = (lam2+l3)/2, b = (lam1 - l3*yhat) + 2*a*iota*thetabar/delta,
                        c = (l3/2)*yhat^2 + (b*iota*thetabar + a*iota^2*varsig^2)/delta)
The damage dependence enters ONLY through w_reg exactly as design_absorb.json
specifies -- no -log(N) de-trending is applied on top. The logxi input enters the VALUE
LEVEL analytically (structured robustness); policies are xi-invariant by the
economy's log-separability theorem (documented, not an omission).

POLICIES:
Pre-tech regimes (PreDamagePreTech, PostDamagePreTech): FOC-CONSISTENT mode
(design correction (iii), the lift-budget fix). The production 3-FOC + budget
system is solved at the LIFTED value gradients
    a_K = v_logK - Z*v_Z       = beta/(1-Z)
    a_G = v_logK + (1-Z)*v_Z   = ((1-beta) - F'(s))/Z
    a_R = v_logR               = F'(s)
under production A_g = 0.1085 and production R&D technology
psi_r'(i_r) = (psi0/2)*i_r^(-1/2)*exp((logK-logR)/2), i_r = I_r/K_total.
The system is CLOSED FORM: c = C/K solves the quadratic
    (q/delta)^2 c^2 + (1 + Gamma*(1-F')/delta) c - Abar = 0        (positive root)
    q = (psi0/2)*exp((lk-lr)/2)*F'(s),  Abar = (1-Z)*A_d + Z*A_g + 1/theta
then
    i_d = Gamma*beta*c/(delta*(1-Z)) - 1/theta
    i_g = Gamma*((1-beta)-F'(s))*c/(delta*Z) - 1/theta
    i_r = (q*c/delta)^2        [ALREADY the production rate I_r/(K_d+K_g)]
which guarantees C/K = c > 0 box-wide and 1+theta*i > 0 (given F' < 1-beta,
gate-verified). i_r varies in logR, logK AND Z. Near the Z-box edges the exact
FOC targets for i_d (Z->1) / i_g (Z->0) exceed the production net's bounded
investment-rate head range (-1/theta, 1); targets are clipped to
[-0.0598, 0.98] there (documented; affects only Z >~ 0.95 / Z <~ 0.05).
Post-tech regimes: i_d = 0.103131, i_g = 0.125760 constants (feasible:
C/K = 0.02717*(1-Z) + 0.03094*Z > 0), i_r = None.

Provenance: solved surfaces from benchmarks/economy_zoo/outputs/absorb.npz
(gates in absorb_PROVENANCE.json); F and w_pre interpolated with cubic splines,
F' taken from the F spline derivative, clipped to [0, 0.6599].
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPZ = os.path.abspath(os.path.join(_HERE, "..", "outputs", "absorb.npz"))
_D = np.load(_NPZ)

# parameters (must match solvers/absorb.py)
delta, Gamma, theta = 0.01, 0.06, 16.7
psi0 = 0.10583
A_d, A_g_pre = 0.1303, 0.1085
beta = float(_D["scalars"][0])
b_d = float(_D["scalars"][1])
i_d_star = float(_D["scalars"][2])           # 0.103131 (post-tech constant)
F_inf = float(_D["scalars"][3])
i_g_posttech = float(_D["scalars"][4])       # 0.125760
lam1, lam2, yhat = 0.00017675, 0.0044, 2.5
iota = 0.291 * 0.1303 * 0.3 * 880.0
thetabar, varsig = 1.86e-3, 1.2 * 1.86e-3
ONE_B = 1.0 - beta
_S_MIN, _S_MAX = float(_D["s_grid"][0]), float(_D["s_grid"][-1])
_FS = CubicSpline(_D["s_grid"], _D["F"])
_FPS = _FS.derivative()
_WS = CubicSpline(_D["Y_grid"], _D["w_pre"])

MAP_NAME = "absorb (Nelson-Phelps directed-absorption catch-up; structured-robustness xi in the LEVEL, policies xi-flat by theorem)"
PROVENANCE = {
    "economy": "ABSORB - two-tree Cobb-Douglas complements, deterministic embodied "
               "green ladder A_g(s), s = logR - logK_g; design_absorb.json with "
               "synthesis corrections (lambda0=(l3/2)yhat^2, tau=1.25)",
    "solution": "closed-form dirty block + 1-D Howard F(s) on [-8,7] + closed-form "
                "post-damage climate quadratics + 1-D pre-damage damage-jump BVP",
    "policy_mode": "pre-tech: production-FOC-consistent at lifted gradients "
                   "(closed-form quadratic in C/K); post-tech: constants",
    "files": {"npz": _NPZ,
              "provenance": _NPZ.replace("absorb.npz", "absorb_PROVENANCE.json"),
              "solver": "benchmarks/economy_zoo/solvers/absorb.py",
              "verify": "benchmarks/economy_zoo/solvers/verify_absorb_sympy.py"},
    "xi": "structured robustness on the 3 endogenous channels: level term -(1/2 xi delta)(sd^2 b^2 + sg^2(1-b)^2 + sr^2 F'(s)^2); policies xi-invariant (log-separability theorem)",
}

_IR_CLIP_LO, _IR_CLIP_HI = -0.0598, 0.98     # bounded head (-1/theta, 1) representable


def _col(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def _w_post(Y, l3):
    a = (lam2 + l3) / 2.0
    b = (lam1 - l3 * yhat) + 2.0 * a * iota * thetabar / delta
    c = l3 / 2.0 * yhat**2 + (b * iota * thetabar + a * iota**2 * varsig**2) / delta
    return -(a * Y**2 + b * Y + c)


def _pretech_policies(lk, Z, lr, Fp):
    """Production-FOC-consistent (i_d, i_g, i_r) at the lifted gradients."""
    q = 0.5 * psi0 * np.exp(np.clip(0.5 * (lk - lr), -35.0, 35.0)) * Fp
    Abar = (1.0 - Z) * A_d + Z * A_g_pre + 1.0 / theta
    M = 1.0 + Gamma * (1.0 - Fp) / delta
    qd2 = (q / delta) ** 2
    c = 2.0 * Abar / (M + np.sqrt(M**2 + 4.0 * qd2 * Abar))       # citardauq root
    i_d = Gamma * beta * c / (delta * (1.0 - Z)) - 1.0 / theta
    i_g = Gamma * (ONE_B - Fp) * c / (delta * Z) - 1.0 / theta
    i_r = (q * c / delta) ** 2
    return (np.clip(i_d, _IR_CLIP_LO, _IR_CLIP_HI),
            np.clip(i_g, _IR_CLIP_LO, _IR_CLIP_HI), i_r)


def fields(reg, lk, Z, Y, lr, l3, lx):
    lk, Z, Y, lr, l3 = map(_col, (lk, Z, Y, lr, l3))
    n = len(Y)
    base = lk + beta * np.log(1.0 - Z) + ONE_B * np.log(Z) + b_d
    pretech = reg.endswith("PreTech")
    predam = reg.startswith("PreDamage")
    w = _WS(np.clip(Y, 0.0, 8.0)) if predam else _w_post(Y, l3)
    if pretech:
        s = np.clip(lr - lk - np.log(Z), _S_MIN, _S_MAX)
        F = _FS(s)
        Fp = np.clip(_FPS(s), 0.0, 0.6599)
        i_d, i_g, i_r = _pretech_policies(lk, Z, lr, Fp)
    else:
        F = np.full((n, 1), F_inf)
        i_d = np.full((n, 1), i_d_star)
        i_g = np.full((n, 1), i_g_posttech)
        i_r = None
    # structured-robustness xi LEVEL term (economy theorem: with log-separable
    # structure, uncertainty aversion moves the value LEVEL, not investment):
    #   dv(xi) = -(1/(2 xi delta)) [ sd^2 b^2 + sg^2 (1-b)^2 + sr^2 F'(s)^2 ]
    xi = np.exp(np.clip(_col(lx), np.log(0.05), np.log(148.6)))
    load2 = (0.01 ** 2) * beta ** 2 + (0.01 ** 2) * ONE_B ** 2
    if pretech:
        load2 = load2 + (0.0078 ** 2) * Fp ** 2
    v_xi = -load2 / (2.0 * xi * 0.01)
    return {"v": base + F + w + v_xi, "i_d": i_d, "i_g": i_g, "i_r": i_r}
