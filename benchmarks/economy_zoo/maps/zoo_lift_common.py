"""zoo_lift_common.py — SHARED lift utilities for economy-zoo map modules.

Every maps/<econ>.py imports from here. Scope (per the zoo build spec):

  1. State transforms between production states (lk=logK, Z, Y, lr=logR) and the
     zoo economies' own coordinates:
        logK_d = lk + log(1-Z),  logK_g = lk + log(Z),
        s   = lr - lk            (JONES / PHYSRISK reduction),
        s_g = lr - lk - log(Z)   (ABSORB / GHKM green-scaled variant),
     with guarded logs and an exact inverse (lk, Z) <- (logK_d, logK_g).
  2. A multilinear interpolation stack over reduced grids: `LinearInterpND`
     (flat/clamped extrapolation, exact at nodes, exact for multilinear
     functions) + a `trilinear` convenience — covers the designs'
     "trilinear (Z,Y,s) + lambda3-linear + logxi-linear" stack as a 5-D case.
  3. The repo +/- logN value transform.  Production nets learn
        v = V + logN(Y; lambda3)          (equivalently V = v - logN),
     verified against models/PreDamagePreTech.py (h_y uses
     dV/dY = dv_dY - dlogN_dY with dlogN_dY = lambda1 + lambda2*Y) and
     models/PostDamagePostTech.py (dlogN_dY = lambda1 + lambda2*Y
     + lambda3*(Y - yhat), yhat = y_upper = 2.5).  The level logN is the
     value+slope-continuous integral of that production slope
     (PAPER_HJB_REFERENCE.md flags the paper's printed matching constants as
     typos and mandates value+slope continuity at yhat):
        pre-damage : logN = lambda1*Y + (lambda2/2)*Y^2
        post-damage: logN = lambda1*Y + (lambda2/2)*Y^2
                            + (lambda3/2)*max(Y - yhat, 0)^2
  4. Numerical guards: `clip_exp` (exponent clipped to +/-35) and
     `feasible_invest` (floor on 1 + theta*i).
  5. FOC-consistent policy extraction: given lifted v-costates
     (v_logK, v_Z, and optionally v_logR), solve the production FOC + budget
     system in CLOSED FORM (log utility makes it linear/quadratic in C/K):
        delta/c = Gamma_j*theta_j/(1+theta_j*i_j) * mu_j,
            mu_d = v_logK - Z*v_Z,  mu_g = v_logK + (1-Z)*v_Z,
        delta/c = psi0*psi1*i_r^(psi1-1)*exp(psi1*(lk-lr)) * v_logR (psi1=1/2),
        c = (A_d - i_d)(1-Z) + (A_g - i_g)Z - i_r.

Parameters are the production values (models_warmstart/params.py /
PAPER_HJB_REFERENCE.md), hard-coded here so map modules stay import-light.

Self-test: `python zoo_lift_common.py` runs analytic unit tests of EVERY
helper (transform round-trips, interp node/linear exactness + clamping,
logN continuity + production-slope match, guard behavior, FOC exact
recovery) and prints the residuals.
"""

import numpy as np

# ---------------------------------------------------------------- parameters
# Production calibration (models_warmstart/params.py; PAPER_HJB_REFERENCE.md)
DELTA = 0.01
ALPHA_D, GAMMA_D, THETA_D, SIGMA_D = -0.035, 0.060, 16.7, 0.01
ALPHA_G, GAMMA_G, THETA_G, SIGMA_G = -0.035, 0.060, 16.7, 0.01
A_D = 0.1303
A_G_PRE, A_G_INTERM, A_G_POST = 0.1085, 0.1303, 0.1567
PSI_0, PSI_1, ZETA, SIGMA_R = 0.10583, 0.5, 0.0, 0.0078
LAMBDA_1, LAMBDA_2 = 0.00017675, 2 * 0.0022      # lambda2 = 0.0044
Y_HAT = 2.5                                      # y_upper: damage-jump anchor
EXP_CLIP = 35.0

REGIMES = ("PreDamagePreTech", "PreDamagePostTech",
           "PostDamagePreTech", "PostDamagePostTech")


def is_post_damage(reg):
    return reg.startswith("PostDamage")


def is_post_tech(reg):
    return reg in ("PreDamagePostTech", "PostDamagePostTech")


def a_g_of_regime(reg):
    """Green productivity in the regime as production defines it
    (post-tech regimes in the 4-regime anchor layout = breakthrough A_g'')."""
    return A_G_POST if reg in ("PreDamagePostTech", "PostDamagePostTech") \
        else A_G_PRE


# ---------------------------------------------------------------- guards
def clip_exp(x, clip=EXP_CLIP):
    """exp with the exponent clipped to [-clip, +clip] (float32-overflow guard,
    same +/-35 rule as the production jump-distortion fix)."""
    return np.exp(np.clip(np.asarray(x, dtype=float), -clip, clip))


def glog(x, floor=1e-12):
    """Guarded log: log(max(x, floor)). Never returns nan/-inf."""
    return np.log(np.maximum(np.asarray(x, dtype=float), floor))


def feasible_invest(i, theta, floor=1e-8, i_max=None):
    """Enforce production feasibility 1 + theta*i >= floor (> 0), i.e.
    i >= (floor - 1)/theta; optional upper cap i <= i_max."""
    i = np.asarray(i, dtype=float)
    lo = (floor - 1.0) / theta
    out = np.maximum(i, lo)
    if i_max is not None:
        out = np.minimum(out, i_max)
    return out


def as_col(x):
    """Return x as an (n,1) float array."""
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1)


# ---------------------------------------------------------------- transforms
def logKd_of(lk, Z, floor=1e-12):
    """logK_d = logK + log(1-Z)   (guarded at Z -> 1)."""
    return np.asarray(lk, dtype=float) + glog(1.0 - np.asarray(Z, dtype=float),
                                              floor)


def logKg_of(lk, Z, floor=1e-12):
    """logK_g = logK + log(Z)     (guarded at Z -> 0)."""
    return np.asarray(lk, dtype=float) + glog(Z, floor)


def lk_Z_of_sectors(lkd, lkg):
    """Exact inverse: lk = log(K_d + K_g) via logaddexp, Z = exp(lkg - lk)."""
    lkd = np.asarray(lkd, dtype=float)
    lkg = np.asarray(lkg, dtype=float)
    lk = np.logaddexp(lkd, lkg)
    Z = np.exp(lkg - lk)
    return lk, Z


def s_of(lk, lr):
    """s = logR - logK (JONES/PHYSRISK knowledge-intensity coordinate)."""
    return np.asarray(lr, dtype=float) - np.asarray(lk, dtype=float)


def s_g_of(lk, Z, lr, floor=1e-12):
    """s_g = logR - logK_g = lr - lk - log(Z) (green-scaled variant)."""
    return np.asarray(lr, dtype=float) - logKg_of(lk, Z, floor)


# ---------------------------------------------------------------- logN stack
def logN(Y, l3=0.0, post_damage=False):
    """Repo damage level logN(Y; lambda3).  Pre-damage: lambda1*Y +
    (lambda2/2)Y^2.  Post-damage adds (lambda3/2)*max(Y - yhat, 0)^2 —
    the value+slope-continuous integral of the production slope."""
    Y = np.asarray(Y, dtype=float)
    base = LAMBDA_1 * Y + 0.5 * LAMBDA_2 * Y ** 2
    if not post_damage:
        return base
    l3 = np.asarray(l3, dtype=float)
    return base + 0.5 * l3 * np.maximum(Y - Y_HAT, 0.0) ** 2


def dlogN_dY(Y, l3=0.0, post_damage=False):
    """Production damage slope: pre = lambda1 + lambda2*Y; post adds
    lambda3*(Y - yhat) for Y >= yhat (models/PostDamage*.py h_y formula)."""
    Y = np.asarray(Y, dtype=float)
    base = LAMBDA_1 + LAMBDA_2 * Y
    if not post_damage:
        return base
    l3 = np.asarray(l3, dtype=float)
    return base + l3 * np.maximum(Y - Y_HAT, 0.0)


def v_from_V(V, Y, l3=0.0, post_damage=False):
    """Net-space target from the true value: v = V + logN(Y; lambda3).
    This is the object make_map_anchor.py fits the v_nn to."""
    return np.asarray(V, dtype=float) + logN(Y, l3, post_damage)


def V_from_v(v, Y, l3=0.0, post_damage=False):
    """True value from net space: V = v - logN(Y; lambda3)."""
    return np.asarray(v, dtype=float) - logN(Y, l3, post_damage)


# ---------------------------------------------------------------- interp
class LinearInterpND:
    """Multilinear interpolation on a rectilinear grid with FLAT (clamped)
    extrapolation outside the box.

    grids : sequence of d strictly-increasing 1-D arrays (size-1 axes allowed:
            the value is constant along them).
    values: array of shape (len(g_0), ..., len(g_{d-1})).

    __call__(x_0, ..., x_{d-1}) with broadcastable arrays -> array shaped like
    the broadcast input.  Exact at nodes; exact for multilinear functions.
    """

    def __init__(self, grids, values):
        self.grids = [np.asarray(g, dtype=float).ravel() for g in grids]
        self.values = np.asarray(values, dtype=float)
        if self.values.shape != tuple(len(g) for g in self.grids):
            raise ValueError(f"values shape {self.values.shape} != grid shape "
                             f"{tuple(len(g) for g in self.grids)}")
        for g in self.grids:
            if len(g) > 1 and np.any(np.diff(g) <= 0):
                raise ValueError("grids must be strictly increasing")

    def __call__(self, *coords):
        if len(coords) != len(self.grids):
            raise ValueError(f"expected {len(self.grids)} coords, "
                             f"got {len(coords)}")
        coords = np.broadcast_arrays(*[np.asarray(c, dtype=float)
                                       for c in coords])
        shape = coords[0].shape
        flat = [c.ravel() for c in coords]
        idx_lo, wts = [], []
        for g, x in zip(self.grids, flat):
            if len(g) == 1:                      # constant axis
                idx_lo.append(np.zeros(len(x), dtype=int))
                wts.append(np.zeros(len(x)))
                continue
            xc = np.clip(x, g[0], g[-1])         # flat extrapolation
            i = np.clip(np.searchsorted(g, xc, side="right") - 1,
                        0, len(g) - 2)
            w = (xc - g[i]) / (g[i + 1] - g[i])
            idx_lo.append(i)
            wts.append(w)
        d = len(self.grids)
        out = np.zeros(len(flat[0]))
        for corner in range(2 ** d):
            w_corner = np.ones(len(flat[0]))
            idx = []
            for k in range(d):
                hi = (corner >> k) & 1
                if len(self.grids[k]) == 1:
                    if hi:                        # size-1 axis has no hi corner
                        w_corner = None
                        break
                    idx.append(idx_lo[k])
                    continue
                idx.append(idx_lo[k] + hi)
                w_corner = w_corner * (wts[k] if hi else 1.0 - wts[k])
            if w_corner is None:
                continue
            out += w_corner * self.values[tuple(idx)]
        return out.reshape(shape)


def trilinear(gx, gy, gz, V, x, y, z):
    """Convenience trilinear interpolation (flat extrapolation)."""
    return LinearInterpND([gx, gy, gz], V)(x, y, z)


def stack_interp_l3_lx(gx, gy, gz, g_l3, g_lx, V):
    """The designs' full stack: trilinear in (x,y,z) + linear in lambda3 +
    linear in logxi.  V has shape (nx, ny, nz, n_l3, n_lx); size-1 lambda3 /
    logxi axes mean 'constant in that pseudo-state'.  Returns a callable
    f(x, y, z, l3, lx)."""
    return LinearInterpND([gx, gy, gz, g_l3, g_lx], V)


# ---------------------------------------------------------------- FOC helper
def foc_invest_rate(mu, c, Gamma, theta, floor=1e-8, i_max=None):
    """Single-sector production FOC inverted for the rate:
        delta/c = Gamma*theta/(1+theta*i) * mu  =>
        i = (Gamma*theta*mu*c/delta - 1)/theta,
    then feasibility-guarded (1 + theta*i >= floor)."""
    mu = np.asarray(mu, dtype=float)
    c = np.asarray(c, dtype=float)
    i = (Gamma * theta * mu * c / DELTA - 1.0) / theta
    return feasible_invest(i, theta, floor=floor, i_max=i_max)


def solve_focs(v_logK, v_Z, Z, A_g, v_logR=None, lk=None, lr=None,
               i_r_exog=None, A_d=A_D,
               Gamma_d=GAMMA_D, theta_d=THETA_D,
               Gamma_g=GAMMA_G, theta_g=THETA_G,
               psi0=PSI_0, psi1=PSI_1, delta=DELTA,
               floor=1e-8, c_floor=1e-8):
    """Closed-form solve of the production FOC + budget system given lifted
    costates.  With log utility the budget collapses to a scalar equation in
    c = C/K:

        mu_d = v_logK - Z*v_Z,   mu_g = v_logK + (1-Z)*v_Z
        1 + theta_j*i_j = Gamma_j*theta_j*mu_j*c/delta
        i_r = (0.5*psi0*exp(0.5*(lk-lr))*v_logR*c/delta)^2   [psi1 = 1/2]

        =>  q^2*c^2 + (1+G)*c - B = 0,
            B = (A_d + 1/theta_d)(1-Z) + (A_g + 1/theta_g)Z  [- i_r_exog]
            G = (Gamma_d*mu_d*(1-Z) + Gamma_g*mu_g*Z)/delta
            q = psi1*psi0*exp(psi1*(lk-lr))*v_logR/delta     [0 if no R&D]

    solved with the cancellation-safe root c = 2B/((1+G)+sqrt((1+G)^2+4q^2B)).

    R&D modes: v_logR (+ lk, lr) given -> interior i_r FOC (psi1 must be 1/2
    for the closed form); i_r_exog given -> fixed R&D spending; neither ->
    no-R&D regime (i_r = None in the output).

    Returns dict(i_d, i_g, i_r, c, mu_d, mu_g, interior) where `interior`
    flags points at which no feasibility clamp fired and c stayed > c_floor.
    After clamping, c is recomputed from the budget so the outputs are
    budget-consistent (FOCs then hold only at interior points — recorded, not
    hidden)."""
    v_logK = np.asarray(v_logK, dtype=float)
    v_Z = np.asarray(v_Z, dtype=float)
    Z = np.asarray(Z, dtype=float)

    mu_d = v_logK - Z * v_Z
    mu_g = v_logK + (1.0 - Z) * v_Z

    B = (A_d + 1.0 / theta_d) * (1.0 - Z) + (A_g + 1.0 / theta_g) * Z
    if i_r_exog is not None:
        B = B - np.asarray(i_r_exog, dtype=float)
    G = (Gamma_d * mu_d * (1.0 - Z) + Gamma_g * mu_g * Z) / delta

    rd_active = v_logR is not None
    if rd_active:
        if psi1 != 0.5:
            raise ValueError("closed-form i_r branch requires psi1 = 1/2")
        if lk is None or lr is None:
            raise ValueError("v_logR mode needs lk and lr")
        q = psi1 * psi0 * clip_exp(psi1 * (np.asarray(lk, dtype=float)
                                           - np.asarray(lr, dtype=float))) \
            * np.asarray(v_logR, dtype=float) / delta
        q = np.maximum(q, 0.0)      # negative v_logR => corner i_r = 0
    else:
        q = np.zeros_like(B)

    one_G = 1.0 + G
    disc = one_G ** 2 + 4.0 * q ** 2 * B
    bad = (disc <= 0.0) | (one_G + np.sqrt(np.maximum(disc, 0.0)) <= 0.0)
    denom = one_G + np.sqrt(np.maximum(disc, 0.0))
    c = np.where(bad, c_floor, 2.0 * B / np.where(denom == 0.0, 1.0, denom))
    c = np.maximum(c, c_floor)

    i_d_raw = (Gamma_d * theta_d * mu_d * c / delta - 1.0) / theta_d
    i_g_raw = (Gamma_g * theta_g * mu_g * c / delta - 1.0) / theta_g
    i_d = feasible_invest(i_d_raw, theta_d, floor=floor)
    i_g = feasible_invest(i_g_raw, theta_g, floor=floor)
    if rd_active:
        i_r_raw = (q * c) ** 2
        i_r = np.maximum(i_r_raw, 0.0)
    elif i_r_exog is not None:
        i_r = np.asarray(i_r_exog, dtype=float) * np.ones_like(B)
    else:
        i_r = None

    # budget-consistent c after any clamps
    c_out = (A_d - i_d) * (1.0 - Z) + (A_g - i_g) * Z
    if i_r is not None:
        c_out = c_out - i_r
    interior = (~bad) & (i_d == i_d_raw) & (i_g == i_g_raw) \
        & (c_out > c_floor)
    c_out = np.maximum(c_out, c_floor)
    return dict(i_d=i_d, i_g=i_g, i_r=i_r, c=c_out,
                mu_d=mu_d, mu_g=mu_g, interior=interior)


# ================================================================= self-test
def _selftest():
    rng = np.random.RandomState(0)
    report = []

    def check(name, err, tol):
        ok = err < tol
        report.append((name, err, tol, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: err={err:.3e} "
              f"(tol {tol:.0e})")
        return ok

    print("== zoo_lift_common self-test ==")

    # 1. transforms round-trip
    lk = rng.uniform(4, 7, 4096)
    Z = rng.uniform(0.01, 0.99, 4096)
    lr = rng.uniform(1, 6, 4096)
    lkd, lkg = logKd_of(lk, Z), logKg_of(lk, Z)
    lk2, Z2 = lk_Z_of_sectors(lkd, lkg)
    check("sector-split round-trip lk", np.max(np.abs(lk2 - lk)), 1e-12)
    check("sector-split round-trip Z", np.max(np.abs(Z2 - Z)), 1e-12)
    check("s_of definition", np.max(np.abs(s_of(lk, lr) - (lr - lk))), 0.0
          + 1e-15)
    check("s_g identity s_g = lr - lkg",
          np.max(np.abs(s_g_of(lk, Z, lr) - (lr - lkg))), 1e-12)
    assert np.isfinite(glog(0.0)) and np.isfinite(glog(-3.0))
    check("glog floor at 1e-12", abs(glog(0.0) - np.log(1e-12)), 1e-12)

    # 2. interpolation
    gx = np.array([0.0, 0.3, 1.0, 2.0])
    gy = np.array([-1.0, 0.5, 2.0])
    gz = np.array([0.0, 1.0])
    a, bx, by, bz = 0.7, -1.3, 2.1, 0.4
    f = lambda x, y, z: a + bx * x + by * y + bz * z
    V = f(gx[:, None, None], gy[None, :, None], gz[None, None, :])
    itp = LinearInterpND([gx, gy, gz], V)
    # exact at nodes
    Xn, Yn, Zn = np.meshgrid(gx, gy, gz, indexing="ij")
    check("interp exact at nodes",
          np.max(np.abs(itp(Xn, Yn, Zn) - V)), 1e-13)
    # exact for a (multi)linear function anywhere inside
    xs = rng.uniform(0, 2, 2000); ys = rng.uniform(-1, 2, 2000)
    zs = rng.uniform(0, 1, 2000)
    check("interp exact on linear fn (interior)",
          np.max(np.abs(itp(xs, ys, zs) - f(xs, ys, zs))), 1e-12)
    # flat clamp outside
    check("interp flat clamp outside",
          abs(itp(5.0, 10.0, -3.0) - f(2.0, 2.0, 0.0)), 1e-12)
    # trilinear wrapper agrees
    check("trilinear wrapper",
          np.max(np.abs(trilinear(gx, gy, gz, V, xs, ys, zs)
                        - itp(xs, ys, zs))), 1e-14)
    # 5-D stack with size-1 logxi axis (constant in that pseudo-state)
    g_l3 = np.array([0.0, 1.0 / 6, 1.0 / 3])
    g_lx = np.array([0.0])
    V5 = V[:, :, :, None, None] + 3.0 * g_l3[None, None, None, :, None] \
        + 0.0 * g_lx[None, None, None, None, :]
    st = stack_interp_l3_lx(gx, gy, gz, g_l3, g_lx, V5)
    l3s = rng.uniform(0, 1 / 3, 2000)
    check("5-D stack (l3 linear, size-1 lx axis)",
          np.max(np.abs(st(xs, ys, zs, l3s, np.full_like(xs, 9.9))
                        - (f(xs, ys, zs) + 3.0 * l3s))), 1e-12)

    # 3. logN transform
    Yv = np.linspace(0, 4, 4001)
    eps = 1e-6
    l3 = 1.0 / 3
    check("logN value continuity at yhat",
          abs(logN(Y_HAT, l3, True) - logN(Y_HAT, 0.0, False)), 1e-15)
    slope_num = (logN(Y_HAT + eps, l3, True)
                 - logN(Y_HAT - eps, l3, True)) / (2 * eps)
    check("logN slope continuity at yhat",
          abs(slope_num - dlogN_dY(Y_HAT, l3, True)), 1e-6)
    check("pre-damage slope == lambda1+lambda2*Y",
          np.max(np.abs(dlogN_dY(Yv, 0.0, False)
                        - (LAMBDA_1 + LAMBDA_2 * Yv))), 1e-15)
    Yp = np.linspace(2.5, 4, 2001)
    check("post-damage slope == lambda1+lambda2*Y+lambda3*(Y-yhat)",
          np.max(np.abs(dlogN_dY(Yp, l3, True)
                        - (LAMBDA_1 + LAMBDA_2 * Yp + l3 * (Yp - Y_HAT)))),
          1e-15)
    # numeric derivative of the level matches the slope everywhere (post)
    num = (logN(Yp[1:], l3, True) - logN(Yp[:-1], l3, True)) \
        / (Yp[1] - Yp[0])
    mid = 0.5 * (Yp[1:] + Yp[:-1])
    check("d(logN)/dY == slope (post, midpoint FD)",
          np.max(np.abs(num - dlogN_dY(mid, l3, True))), 1e-9)
    Vtrue = rng.randn(100)
    Yr = rng.uniform(2.5, 4, 100)
    check("v/V transform round-trip",
          np.max(np.abs(V_from_v(v_from_V(Vtrue, Yr, l3, True), Yr, l3, True)
                        - Vtrue)), 1e-15)

    # 4. guards
    assert np.isfinite(clip_exp(1e6)) and clip_exp(1e6) == np.exp(35.0)
    check("clip_exp == exp inside range",
          abs(clip_exp(3.2) - np.exp(3.2)), 1e-15)
    ig = feasible_invest(np.array([-1.0, 0.05]), THETA_D)
    check("feasibility floor 1+theta*i >= 1e-8",
          abs((1 + THETA_D * ig[0]) - 1e-8), 1e-12)
    assert ig[1] == 0.05

    # 5. FOC closed form: forward-construct an interior optimum, recover it
    n = 2048
    Zt = rng.uniform(0.05, 0.95, n)
    lkt = rng.uniform(4, 7, n)
    lrt = rng.uniform(1, 6, n)
    i_d0 = rng.uniform(0.02, 0.11, n)
    i_g0 = rng.uniform(0.02, 0.10, n)
    i_r0 = rng.uniform(0.001, 0.008, n)
    c0 = (A_D - i_d0) * (1 - Zt) + (A_G_PRE - i_g0) * Zt - i_r0
    assert np.all(c0 > 0)
    mu_d0 = DELTA * (1 + THETA_D * i_d0) / (GAMMA_D * THETA_D * c0)
    mu_g0 = DELTA * (1 + THETA_G * i_g0) / (GAMMA_G * THETA_G * c0)
    vZ0 = mu_g0 - mu_d0                       # mu_g - mu_d = v_Z
    vlK0 = mu_d0 + Zt * vZ0
    vlR0 = DELTA * np.sqrt(i_r0) \
        / (0.5 * PSI_0 * np.exp(0.5 * (lkt - lrt)) * c0)
    sol = solve_focs(vlK0, vZ0, Zt, A_G_PRE, v_logR=vlR0, lk=lkt, lr=lrt)
    check("FOC solve recovers i_d", np.max(np.abs(sol["i_d"] - i_d0)), 1e-10)
    check("FOC solve recovers i_g", np.max(np.abs(sol["i_g"] - i_g0)), 1e-10)
    check("FOC solve recovers i_r", np.max(np.abs(sol["i_r"] - i_r0)), 1e-10)
    check("FOC solve recovers c", np.max(np.abs(sol["c"] - c0)), 1e-10)
    assert bool(np.all(sol["interior"]))
    # no-R&D branch
    c1 = (A_D - i_d0) * (1 - Zt) + (A_G_POST - i_g0) * Zt
    mu_d1 = DELTA * (1 + THETA_D * i_d0) / (GAMMA_D * THETA_D * c1)
    mu_g1 = DELTA * (1 + THETA_G * i_g0) / (GAMMA_G * THETA_G * c1)
    vZ1 = mu_g1 - mu_d1
    vlK1 = mu_d1 + Zt * vZ1
    sol1 = solve_focs(vlK1, vZ1, Zt, A_G_POST)
    check("FOC (no R&D) recovers i_d",
          np.max(np.abs(sol1["i_d"] - i_d0)), 1e-10)
    check("FOC (no R&D) recovers i_g",
          np.max(np.abs(sol1["i_g"] - i_g0)), 1e-10)
    assert sol1["i_r"] is None
    # single-sector helper consistency
    check("foc_invest_rate matches joint solve",
          np.max(np.abs(foc_invest_rate(mu_d0, c0, GAMMA_D, THETA_D)
                        - i_d0)), 1e-10)

    n_fail = sum(1 for *_x, ok in report if not ok)
    print(f"== {len(report) - n_fail}/{len(report)} checks passed ==")
    if n_fail:
        raise SystemExit(f"SELF-TEST FAILED: {n_fail} checks")


if __name__ == "__main__":
    _selftest()
