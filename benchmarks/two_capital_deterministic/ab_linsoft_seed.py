"""
Seed-parameterized re-run of the linear-vs-softplus VALUE HEAD A/B (DIRECT embedding,
the clean production-faithful comparison). Identical seed/init/budget in both arms.

Usage:  python ab_linsoft_seed.py <SEED1> [<SEED2> ...]
For each seed, runs:
  CONTROL  : value = head(body)  with final_activation=None      (linear head)
  TREATED  : value = head(body)  with final_activation=softplus  (production-faithful)
Reports, for each arm: supervised mse, pre-HJB resid/err, FINAL resid_rms,
max|i_d-FD| over Z in [0.1,0.9], de-invest recovery, softplus-floor fraction.
"""
import os
import sys
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


def make_body(head_activation, seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    out = tf.keras.layers.Dense(1, activation=head_activation)(h)
    return tf.keras.Model(inp, out)


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def value_direct(net, Z):
    return net(2.0 * Z - 1.0)


def make_lg(net, precond=True):
    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                inner.watch(Zt); v = value_direct(net, Zt)
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


def adam(net, lg, steps, lr=2e-3):
    opt = tf.keras.optimizers.Adam(lr); L = None
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
    r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxfun": 2 * maxiter})
    setf(r.x); return float(r.fun)


def supervised(net, steps=4000):
    opt = tf.keras.optimizers.Adam(2e-3)
    @tf.function
    def step():
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value_direct(net, Zt) - v_fd))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L
    L = None
    for _ in range(steps):
        L = step()
    return float(L)


def true_residual_rms(net):
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value_direct(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt)
    R = M.hjb_residual(Zg.ravel(), v.numpy().ravel(), vp.numpy().ravel(), P)
    return float(np.sqrt(np.mean(R[mI] ** 2)))


def report(net, name, softplus=False):
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value_direct(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    i_d, i_g, c = M.controls(Zg.ravel(), vp, P)
    err = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    res = true_residual_rms(net)
    raw = net(2.0 * Zt - 1.0).numpy().ravel()
    floor = float(np.mean(raw[mI] < 1e-4)) if softplus else float("nan")
    deinv = bool((i_d[mI] < 0).any())
    print(f"  {name:30s} resid_rms={res:.3e}  max|i_d-FD|={err:.4f}  "
          f"de-invest={deinv}  floorfrac={floor:.3f}", flush=True)
    return dict(resid=res, err=err, deinvest=deinv, floor=floor)


def run_arm(head_act, label, softplus, seed):
    print(f"--- {label} ---", flush=True)
    net = make_body(head_act, seed)
    smse = supervised(net)
    print(f"  supervised-fit mse={smse:.2e}", flush=True)
    report(net, "pre-HJB", softplus)
    adam(net, make_lg(net, True), 3000)
    lbfgs(net, make_lg(net, True))
    return report(net, "FINAL (FD-seed+Adam+LBFGS)", softplus)


seeds = [int(s) for s in sys.argv[1:]] or [int(os.environ.get("AB_SEED", "1"))]
print(f"FD truth: v in [{v_fd.min():+.3f},{v_fd.max():+.3f}]  "
      f"i_d in [{id_fd[mI].min():+.4f}, {id_fd[mI].max():+.4f}] (de-invests)\n", flush=True)
results = {}
for sd in seeds:
    print(f"\n################ SEED {sd} ################", flush=True)
    cf = run_arm(None, f"CONTROL linear  seed={sd}", False, sd)
    tf_ = run_arm("softplus", f"TREATED softplus seed={sd}", True, sd)
    results[sd] = (cf, tf_)

print("\n================ SUMMARY (DIRECT head) ================", flush=True)
print(f"{'seed':>5} {'arm':>9}  {'resid_rms':>11}  {'max|i_d-FD|':>11}  {'deinvest':>8}  {'floor':>6}", flush=True)
for sd, (cf, tfa) in results.items():
    print(f"{sd:>5} {'linear':>9}  {cf['resid']:.3e}  {cf['err']:11.4f}  {str(cf['deinvest']):>8}  {cf['floor']:6.3f}", flush=True)
    print(f"{sd:>5} {'softplus':>9}  {tfa['resid']:.3e}  {tfa['err']:11.4f}  {str(tfa['deinvest']):>8}  {tfa['floor']:6.3f}", flush=True)
print("\nDONE", flush=True)
