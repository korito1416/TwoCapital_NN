"""
Independent verification of precond_w_pointwise_vs_raw.
Re-runs the SAME A/B (identical init per seed per arm) on DIFFERENT seeds (5,6,7)
to check whether the pointwise 1/(|mu_Z|+eps) weight gives a reproducible,
meaningful improvement in BOTH residual and FD accuracy, or is seed noise.

Reuses logic from precond_ab_multiseed.py verbatim (cloned init, unweighted
residual reported for both arms, FD error vs ground truth).
"""
import os, sys, time
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
Zg = np.linspace(0.02, 0.98, 256).reshape(-1, 1).astype(np.float32)
Zt = tf.constant(Zg)
v_fd = np.interp(Zg.ravel(), fd["Z"], fd["v"]).reshape(-1, 1).astype(np.float32)
id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])
mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def value(net, Z):
    return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)


def make_lg(net, precond):
    @tf.function
    def lg():
        with tf.GradientTape() as outer:
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
            w = (tf.abs(mu) + 5e-3) if precond else tf.ones_like(mu)
            loss = tf.reduce_mean(tf.square(R / w))
        return loss, outer.gradient(loss, net.trainable_variables)
    return lg


def raw_residual_rms(net):
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
    return float(tf.sqrt(tf.reduce_mean(tf.square(R))))


def adam(net, lg, steps, lr=2e-3):
    opt = tf.keras.optimizers.Adam(lr)
    for _ in range(steps):
        L, g = lg(); opt.apply_gradients(zip(g, net.trainable_variables))
    return float(L)


def lbfgs(net, lg, maxiter=4000):
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
    r = minimize(fg, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": 2 * maxiter})
    setf(r.x); return float(r.fun)


def supervised(net, steps=4000):
    opt = tf.keras.optimizers.Adam(2e-3)
    @tf.function
    def step():
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value(net, Zt) - v_fd))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L
    for _ in range(steps):
        L = step()
    return float(L)


def fd_err(net):
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    i_d, i_g, c = M.controls(Zg.ravel(), vp, P)
    return float(np.max(np.abs(i_d[mI] - id_fd[mI])))


def get_weights(net):
    return [w.numpy().copy() for w in net.weights]


def set_weights(net, ws):
    for w, v in zip(net.weights, ws):
        w.assign(v)


print(f"FD truth: i_d in [{id_fd[mI].min():+.4f}, {id_fd[mI].max():+.4f}] (de-invests)\n", flush=True)

SEEDS = [int(s) for s in sys.argv[1:]] or [5, 6, 7]
print(f"VERIFY seeds: {SEEDS}\n", flush=True)
resA = {True: {"res": [], "fd": []}, False: {"res": [], "fd": []}}
resB = {True: {"res": [], "fd": []}, False: {"res": [], "fd": []}}

for seed in SEEDS:
    t0 = time.time()
    print(f"==== SEED {seed} ====", flush=True)
    tf.random.set_seed(seed); np.random.seed(seed)
    net0 = make_net()
    supervised(net0)
    init_w = get_weights(net0)
    for precond in (False, True):
        net = make_net(); set_weights(net, init_w)
        lbfgs(net, make_lg(net, precond))
        res = raw_residual_rms(net); err = fd_err(net)
        resA[precond]["res"].append(res); resA[precond]["fd"].append(err)
        tag = "precond" if precond else "raw    "
        print(f"  (A) {tag}  rawRMSres={res:.3e}  max|i_d-FD|={err:.4f}", flush=True)

    tf.random.set_seed(seed + 100); np.random.seed(seed + 100)
    netb0 = make_net()
    init_wb = get_weights(netb0)
    for precond in (False, True):
        net = make_net(); set_weights(net, init_wb)
        adam(net, make_lg(net, precond), 5000)
        lbfgs(net, make_lg(net, precond))
        res = raw_residual_rms(net); err = fd_err(net)
        resB[precond]["res"].append(res); resB[precond]["fd"].append(err)
        tag = "precond" if precond else "raw    "
        print(f"  (B) {tag}  rawRMSres={res:.3e}  max|i_d-FD|={err:.4f}", flush=True)
    print(f"  [seed {seed} took {time.time()-t0:.0f}s]", flush=True)


def summ(d, label):
    print(f"\n=== SUMMARY {label} (mean +/- std over {len(SEEDS)} seeds) ===", flush=True)
    for precond in (False, True):
        tag = "precond" if precond else "raw    "
        r = np.array(d[precond]["res"]); f = np.array(d[precond]["fd"])
        print(f"  {tag}  rawRMSres={r.mean():.3e}+/-{r.std():.1e}   "
              f"max|i_d-FD|={f.mean():.4f}+/-{f.std():.4f}   (fd per-seed={np.round(f,4)})", flush=True)


summ(resA, "PATH A (FD-seed+LBFGS)")
summ(resB, "PATH B (scratch+Adam+LBFGS)")
print("\nDONE", flush=True)
