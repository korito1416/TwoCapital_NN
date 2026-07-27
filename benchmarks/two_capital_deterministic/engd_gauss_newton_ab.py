"""
ENGD / Gauss-Newton natural-gradient on the value-net (TREATED) vs Adam+L-BFGS (CONTROL).

Method (from MEMORY: engd-gauss-newton-value-block):
  Replace the optimizer on the small value net by the damped Gauss-Newton /
  natural-gradient step
        delta_theta = (J^T J + lambda I)^{-1} J^T R
  with J = dR/dtheta the per-collocation residual Jacobian (N x P), R the strong
  HJB residual at the N=256 Z collocation points. Net is ~2k params so the dense
  path is trivial; we use the residual-space Woodbury form
        delta = J^T (J J^T + lambda I)^{-1} R       (N x N solve)
  and Marquardt-adapt lambda via the gain ratio.

TRUE-ERROR validation vs FD (never the loss number, since the GN metric differs):
  max|i_d - i_d_FD|, ||v_nn - v_fd||_inf, max|v'_nn - slope_FD|.

Both arms share seed/init/Adam-warmup budget. CONTROL then runs L-BFGS;
TREATED then runs the SAME number of "outer" steps as ENGD GN updates.
Tested on A_d=0.13 (well-conditioned) and A_d=0.05 (de-invest, near-degenerate).
"""
import os, sys, time
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

NSEED = int(os.environ.get("NSEED", "5"))
ADAM_STEPS = int(os.environ.get("ADAM_STEPS", "3000"))
LBFGS_MAXITER = int(os.environ.get("LBFGS_MAXITER", "4000"))
GN_STEPS = int(os.environ.get("GN_STEPS", "300"))


def build_problem(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zg = np.linspace(0.02, 0.98, 256).reshape(-1, 1).astype(np.float32)
    Zt = tf.constant(Zg)
    v_fd = np.interp(Zg.ravel(), fd["Z"], fd["v"]).reshape(-1, 1).astype(np.float32)
    id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])
    slope_fd = np.interp(Zg.ravel(), fd["Z"], fd["slope"])
    mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
    v0, vN = M.boundary_values(P)
    return dict(P=P, fd=fd, Zg=Zg, Zt=Zt, v_fd=v_fd, id_fd=id_fd, slope_fd=slope_fd,
                mI=mI, v0=v0, vN=vN)


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def value_fn(net, Z, v0, vN):
    return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)


def residual_vec(net, prob):
    """Per-collocation strong HJB residual R (N,1) -- matches make_lg in the diag file."""
    Zt, P, v0, vN = prob["Zt"], prob["P"], prob["v0"], prob["vN"]
    dl = P["delta"]; A_d = P["A_d"]; A_g = P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value_fn(net, Zt, v0, vN)
    vp = clamp(inner.gradient(v, Zt), Zt)
    q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
    Abar = (1 - Zt) * A_d + Zt * A_g
    c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
    i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
    phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
    phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
    mu = Zt * (1 - Zt) * (phi_g - phi_d)
    R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
    return R  # (N,1)


def make_lg(net, prob, precond=False):
    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            R = residual_vec(net, prob)
            mu_dummy = tf.ones_like(R)
            loss = tf.reduce_mean(tf.square(R / mu_dummy))
        return loss, outer.gradient(loss, net.trainable_variables)
    return lg


def adam(net, lg, steps, lr=2e-3):
    opt = tf.keras.optimizers.Adam(lr)
    for _ in range(steps):
        L, g = lg(); opt.apply_gradients(zip(g, net.trainable_variables))
    return float(L)


def lbfgs(net, lg, maxiter):
    vars_ = net.trainable_variables
    shapes = [v.shape for v in vars_]; sizes = [int(tf.size(v)) for v in vars_]
    def setf(x):
        i = 0
        for v, s, n in zip(vars_, shapes, sizes):
            v.assign(x[i:i + n].reshape(s).astype(np.float32)); i += n
    def fg(x):
        setf(x); L, g = lg()
        return float(L), np.concatenate([gi.numpy().ravel() for gi in g]).astype(np.float64)
    x0 = np.concatenate([v.numpy().ravel() for v in vars_]).astype(np.float64)
    r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxfun": 2 * maxiter})
    setf(r.x); return float(r.fun)


def supervised(net, prob, steps=4000):
    Zt, v_fd = prob["Zt"], prob["v_fd"]; v0, vN = prob["v0"], prob["vN"]
    opt = tf.keras.optimizers.Adam(2e-3)
    @tf.function
    def step():
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value_fn(net, Zt, v0, vN) - v_fd))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L
    for _ in range(steps):
        L = step()
    return float(L)


# ---------------- ENGD / Gauss-Newton natural-gradient ------------------------

def _flat_vars(net):
    return [v for v in net.trainable_variables]


def _jacobian_and_R(net, prob):
    """Return J (N x P) and R (N,) and the flat shapes for reassembly."""
    vars_ = _flat_vars(net)
    with tf.GradientTape() as tape:
        R = residual_vec(net, prob)          # (N,1)
        Rf = tf.reshape(R, [-1])             # (N,)
    jac = tape.jacobian(Rf, vars_, experimental_use_pfor=True)  # list of (N, *var.shape)
    N = int(Rf.shape[0])
    cols = [tf.reshape(j, [N, -1]) for j in jac]
    J = tf.concat(cols, axis=1)              # (N, P)
    return J.numpy().astype(np.float64), Rf.numpy().astype(np.float64), vars_


def _apply_delta(vars_, delta):
    i = 0
    for v in vars_:
        n = int(tf.size(v))
        v.assign_add(tf.reshape(tf.constant(delta[i:i + n].astype(np.float32)), v.shape))
        i += n


def _set_from_flat(vars_, x):
    i = 0
    for v in vars_:
        n = int(tf.size(v))
        v.assign(tf.reshape(tf.constant(x[i:i + n].astype(np.float32)), v.shape))
        i += n


def _flat(vars_):
    return np.concatenate([v.numpy().ravel() for v in vars_]).astype(np.float64)


def engd_gn(net, prob, lg, steps, lam0=1e-3):
    """Damped Gauss-Newton natural gradient via residual-space Woodbury.
    delta = J^T (J J^T + lam I)^{-1} R ; minimize 0.5||R||^2 ; Marquardt lambda."""
    lam = lam0
    L0, _ = lg(); loss = float(L0)
    for it in range(steps):
        J, R, vars_ = _jacobian_and_R(net, prob)         # J:(N,P) R:(N,)
        N = J.shape[0]
        # current SSE objective f = 0.5||R||^2  (loss = mean square = (1/N)*||R||^2)
        f0 = 0.5 * float(R @ R)
        x0 = _flat(vars_)
        # residual-space solve: (J J^T + lam I) y = R ; delta = -J^T y
        G = J @ J.T                                       # (N,N)
        accepted = False
        for _try in range(8):
            A = G + lam * np.eye(N)
            try:
                y = np.linalg.solve(A, R)
            except np.linalg.LinAlgError:
                lam *= 10.0; continue
            delta = -(J.T @ y)                            # (P,)
            _set_from_flat(vars_, x0 + delta)
            R_new = residual_vec(net, prob).numpy().ravel().astype(np.float64)
            f_new = 0.5 * float(R_new @ R_new)
            # predicted decrease (GN model): f0 - 0.5||R + J delta||^2
            pred = f0 - 0.5 * float((R + J @ delta) @ (R + J @ delta))
            rho = (f0 - f_new) / pred if pred > 1e-300 else -1.0
            if f_new < f0 and rho > 0:
                accepted = True
                if rho > 0.75:
                    lam = max(lam / 3.0, 1e-12)
                elif rho < 0.25:
                    lam = min(lam * 2.0, 1e8)
                break
            else:
                _set_from_flat(vars_, x0)                 # revert
                lam = min(lam * 10.0, 1e10)
        if not accepted:
            break
        loss = 2.0 * f_new / N
    return loss


# ---------------- reporting ---------------------------------------------------

def measure(net, prob):
    Zt, P = prob["Zt"], prob["P"]; v0, vN = prob["v0"], prob["vN"]
    Zg = prob["Zg"]; mI = prob["mI"]
    id_fd = prob["id_fd"]; slope_fd = prob["slope_fd"]; v_fd = prob["v_fd"]
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value_fn(net, Zt, v0, vN)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    v_np = v.numpy().ravel()
    i_d, i_g, c = M.controls(Zg.ravel(), vp, P)
    err_id = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    err_v = float(np.max(np.abs(v_np[mI] - v_fd.ravel()[mI])))
    err_slope = float(np.max(np.abs(vp[mI] - slope_fd[mI])))
    deinvest = bool((i_d[mI] < 0).any())
    return dict(err_id=err_id, err_v=err_v, err_slope=err_slope, deinvest=deinvest,
                id_lo=float(i_d[mI].min()), id_hi=float(i_d[mI].max()))


def spectral_vprime_mode(net, prob):
    """Smallest nonzero eigenvalue of J^T J (the mu_Z-suppressed v' direction proxy)."""
    J, R, _ = _jacobian_and_R(net, prob)
    s = np.linalg.svd(J, compute_uv=False)
    s2 = s ** 2
    s2 = s2[s2 > 1e-30]
    if len(s2) == 0:
        return np.nan, np.nan
    return float(s2.max()), float(s2.min())


def run_case(A_d, seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    prob = build_problem(A_d)
    id_fd, mI = prob["id_fd"], prob["mI"]

    # ---- CONTROL: shared Adam warmup + L-BFGS ----
    tf.random.set_seed(seed); np.random.seed(seed)
    netC = make_net()
    lgC = make_lg(netC, prob)
    adam(netC, lgC, ADAM_STEPS)
    lbfgs(netC, lgC, LBFGS_MAXITER)
    mC = measure(netC, prob)

    # ---- TREATED: SAME seed/init/Adam warmup + ENGD Gauss-Newton ----
    tf.random.set_seed(seed); np.random.seed(seed)
    netT = make_net()
    lgT = make_lg(netT, prob)
    adam(netT, lgT, ADAM_STEPS)
    smax0, smin0 = spectral_vprime_mode(netT, prob)
    engd_gn(netT, prob, lgT, GN_STEPS, lam0=1e-3)
    smax1, smin1 = spectral_vprime_mode(netT, prob)
    mT = measure(netT, prob)

    print(f"--- A_d={A_d} seed={seed} ---", flush=True)
    print(f"  FD i_d interior [{id_fd[mI].min():+.4f},{id_fd[mI].max():+.4f}] deinvest={bool((id_fd[mI]<0).any())}", flush=True)
    print(f"  CONTROL (Adam+LBFGS): max|i_d-FD|={mC['err_id']:.3e}  |v-vfd|inf={mC['err_v']:.3e}  max|v'-FD|={mC['err_slope']:.3e}  deinv={mC['deinvest']}", flush=True)
    print(f"  TREATED (Adam+ENGD ): max|i_d-FD|={mT['err_id']:.3e}  |v-vfd|inf={mT['err_v']:.3e}  max|v'-FD|={mT['err_slope']:.3e}  deinv={mT['deinvest']}", flush=True)
    print(f"  ratio TREATED/CONTROL: err_id={mT['err_id']/max(mC['err_id'],1e-30):.3f}  err_slope={mT['err_slope']/max(mC['err_slope'],1e-30):.3f}", flush=True)
    print(f"  J^TJ eig (post-Adam) max/min={smax0:.2e}/{smin0:.2e} kappa={smax0/max(smin0,1e-30):.2e}  (post-ENGD) max/min={smax1:.2e}/{smin1:.2e} kappa={smax1/max(smin1,1e-30):.2e}", flush=True)
    return mC, mT


def main():
    t0 = time.time()
    for A_d in [0.13, 0.05]:
        print(f"\n========== A_d={A_d} ==========", flush=True)
        rows = []
        for s in range(1, NSEED + 1):
            rows.append(run_case(A_d, s))
        errC = np.array([r[0]["err_id"] for r in rows])
        errT = np.array([r[1]["err_id"] for r in rows])
        slC = np.array([r[0]["err_slope"] for r in rows])
        slT = np.array([r[1]["err_slope"] for r in rows])
        print(f"\n  SUMMARY A_d={A_d} over {NSEED} seeds:", flush=True)
        print(f"    CONTROL max|i_d-FD|: median={np.median(errC):.3e} min={errC.min():.3e} max={errC.max():.3e}", flush=True)
        print(f"    TREATED max|i_d-FD|: median={np.median(errT):.3e} min={errT.min():.3e} max={errT.max():.3e}", flush=True)
        print(f"    CONTROL max|v'-FD| : median={np.median(slC):.3e}", flush=True)
        print(f"    TREATED max|v'-FD| : median={np.median(slT):.3e}", flush=True)
        wins = int((errT < errC).sum())
        print(f"    TREATED beats CONTROL on err_id in {wins}/{NSEED} seeds; median ratio={np.median(errT/np.maximum(errC,1e-30)):.3f}", flush=True)
    print(f"\nelapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
