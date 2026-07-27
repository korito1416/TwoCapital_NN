"""
A/B test: residual-based adaptive collocation (RAR/RAD) vs uniform-Z collocation,
on the FAILING de-invest case (A_d=0.05) small value-NN benchmark.

Both arms: SAME scratch init/seed/budget = Adam(5000) + L-BFGS(4000), precond=False
(raw residual loss, so we isolate the SAMPLING-side intervention, not in-loss weighting).

CONTROL : fixed Zt = linspace(0.02, 0.98, 256).
TREATED : each resample period, draw a 4096-pt uniform pool, compute pointwise R via
          the model's own closed-form residual, and resample 256 collocation points
          proportional to R^2 (np.random.choice with p=R^2/sum). Refreshed every
          REFRESH Adam steps and once per L-BFGS restart segment.

MEASURE for both: final raw RMS HJB residual on a fixed dense eval grid, and TRUE
accuracy max|i_d - FD| over the masked interior and over the de-invest band, vs FD.

Run on multiple seeds (resampling is stochastic).
"""
import os, sys
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

A_d = 0.05
P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
dl, A_g = P["delta"], P["A_g"]
Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
ad, ag = P["alpha_d"], P["alpha_g"]
v0, vN = M.boundary_values(P)
fd = solve_fd(P, n=4000)

# Fixed DENSE evaluation grid (identical for both arms; never used for training in TREATED)
Zg_eval = np.linspace(0.02, 0.98, 512).reshape(-1, 1).astype(np.float32)
id_fd_eval = np.interp(Zg_eval.ravel(), fd["Z"], fd["i_d"])
mI = (Zg_eval.ravel() >= 0.1) & (Zg_eval.ravel() <= 0.9)
# de-invest band per FD truth
deinvest_mask = mI & (id_fd_eval < 0.0)

POOL_N = 4096
NCOLL = 256
REFRESH = 500   # Adam steps between resamples


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def value(net, Z):
    return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)


def residual_tf(net, Zt):
    """Pointwise raw HJB residual R(Z) as a tf tensor (column)."""
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt)
    q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
    Abar = (1 - Zt) * A_d + Zt * A_g
    c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
    i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
    phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
    phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
    mu = Zt * (1 - Zt) * (phi_g - phi_d)
    R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
    return R


def make_lg(net, Z_holder):
    """Loss-and-grad closure that reads the CURRENT collocation tensor from Z_holder[0]."""
    @tf.function
    def lg():
        Zt = Z_holder[0]
        with tf.GradientTape() as outer:
            R = residual_tf(net, Zt)
            loss = tf.reduce_mean(tf.square(R))   # raw (precond=False)
        return loss, outer.gradient(loss, net.trainable_variables)
    return lg


def resample(net, rng, pool):
    """Draw NCOLL Z proportional to R^2 from a uniform pool; return float32 column."""
    R = residual_tf(net, tf.constant(pool)).numpy().ravel()
    w = R ** 2
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        sel = rng.choice(len(pool), size=NCOLL, replace=True)
    else:
        sel = rng.choice(len(pool), size=NCOLL, replace=True, p=w / s)
    return pool[sel].reshape(-1, 1).astype(np.float32)


def adam(net, lg, Z_holder, steps, treated, rng, lr=2e-3):
    opt = tf.keras.optimizers.Adam(lr)
    for k in range(steps):
        if treated and (k % REFRESH == 0):
            pool = rng.uniform(0.02, 0.98, POOL_N).reshape(-1, 1).astype(np.float32)
            Z_holder[0] = tf.constant(resample(net, rng, pool))
        L, g = lg(); opt.apply_gradients(zip(g, net.trainable_variables))
    return float(L)


def lbfgs(net, lg, Z_holder, treated, rng, maxiter=4000, refresh_segments=8):
    """L-BFGS in segments; resample the collocation set between segments (TREATED)."""
    vars_ = net.trainable_variables
    shapes = [v.shape for v in vars_]; sizes = [int(tf.size(v)) for v in vars_]
    def setf(x):
        i = 0
        for v, s, n in zip(vars_, shapes, sizes):
            v.assign(x[i:i + n].reshape(s).astype(np.float32)); i += n
    def fg(x):
        setf(x); L, g = lg()
        return float(L), np.concatenate([gi.numpy().ravel() for gi in g]).astype(np.float64)
    if not treated:
        x0 = np.concatenate([v.numpy().ravel() for v in vars_]).astype(np.float64)
        r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                     options={"maxiter": maxiter, "maxfun": 2 * maxiter})
        setf(r.x); return float(r.fun)
    seg = max(1, maxiter // refresh_segments)
    fun = np.nan
    for _ in range(refresh_segments):
        pool = rng.uniform(0.02, 0.98, POOL_N).reshape(-1, 1).astype(np.float32)
        Z_holder[0] = tf.constant(resample(net, rng, pool))
        x0 = np.concatenate([v.numpy().ravel() for v in vars_]).astype(np.float64)
        r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                     options={"maxiter": seg, "maxfun": 2 * seg})
        setf(r.x); fun = float(r.fun)
    return fun


def eval_metrics(net):
    Zt = tf.constant(Zg_eval)
    R = residual_tf(net, Zt).numpy().ravel()
    rms = float(np.sqrt(np.mean(R ** 2)))
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    i_d, i_g, c = M.controls(Zg_eval.ravel(), vp, P)
    err_int = float(np.max(np.abs(i_d[mI] - id_fd_eval[mI])))
    err_band = (float(np.max(np.abs(i_d[deinvest_mask] - id_fd_eval[deinvest_mask])))
                if deinvest_mask.any() else float("nan"))
    return rms, err_int, err_band, bool((i_d[mI] < 0).any())


def run_arm(seed, treated):
    tf.random.set_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net()
    Z_holder = [tf.constant(np.linspace(0.02, 0.98, NCOLL).reshape(-1, 1).astype(np.float32))]
    lg = make_lg(net, Z_holder)
    adam(net, lg, Z_holder, 5000, treated, rng)
    lbfgs(net, lg, Z_holder, treated, rng)
    return eval_metrics(net)


def main():
    seeds = [0, 1, 2, 3, 4]
    print(f"FD truth: i_d[interior] in [{id_fd_eval[mI].min():+.4f}, {id_fd_eval[mI].max():+.4f}], "
          f"de-invest band pts={int(deinvest_mask.sum())} of {int(mI.sum())} interior", flush=True)
    print(f"FD max|residual|={fd['max_abs_residual']:.2e}\n", flush=True)
    res = {"control": [], "treated": []}
    for s in seeds:
        for arm, treated in [("control", False), ("treated", True)]:
            rms, ei, eb, dein = run_arm(s, treated)
            res[arm].append((rms, ei, eb, dein))
            print(f"seed={s} {arm:8s} RMS_resid={rms:.3e}  max|i_d-FD|_int={ei:.4f}  "
                  f"band={eb:.4f}  deinvest={dein}", flush=True)
    print("\n=== SUMMARY (median over seeds) ===", flush=True)
    for arm in ("control", "treated"):
        a = np.array([(r[0], r[1], r[2]) for r in res[arm]])
        print(f"{arm:8s} RMS_resid med={np.median(a[:,0]):.3e} "
              f"[{a[:,0].min():.3e},{a[:,0].max():.3e}]  "
              f"max|i_d-FD|_int med={np.median(a[:,1]):.4f} "
              f"[{a[:,1].min():.4f},{a[:,1].max():.4f}]  "
              f"band med={np.median(a[:,2]):.4f}", flush=True)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "rar_resampling_ab.npz"),
             control=np.array([(r[0], r[1], r[2]) for r in res["control"]]),
             treated=np.array([(r[0], r[1], r[2]) for r in res["treated"]]))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
