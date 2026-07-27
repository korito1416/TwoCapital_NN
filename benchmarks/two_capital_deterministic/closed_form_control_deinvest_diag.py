"""
Diagnostic on the FAILING de-invest case (A_d=0.05): OPTIMIZATION problem or LOSS problem?
Graph-mode (@tf.function) so it is fast.

  (A) FD-SEED STABILITY: fit v_nn to the FD value, then HJB residual + L-BFGS.
      STAYS at FD de-invest -> loss is fine, earlier failure was optimization.
      DRIFTS away          -> HJB residual does not pin the de-invest solution (structural).
  (B) FROM-SCRATCH + L-BFGS: can a strong optimizer find it without the seed?
"""
import os
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

tf.random.set_seed(0); np.random.seed(0)
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


def report(net, name):
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    i_d, i_g, c = M.controls(Zg.ravel(), vp, P)
    err = np.max(np.abs(i_d[mI] - id_fd[mI]))
    print(f"  {name:36s} max|i_d-FD|={err:.4f}  i_d[{i_d[mI].min():+.4f},{i_d[mI].max():+.4f}]"
          f"  de-invest={bool((i_d[mI]<0).any())}  ->{'STAYS/RECOVERS' if err<0.01 else 'WRONG'}", flush=True)
    return err


print(f"FD truth: i_d in [{id_fd[mI].min():+.4f}, {id_fd[mI].max():+.4f}] (de-invests)\n", flush=True)
print("(A) FD-SEED STABILITY:", flush=True)
net = make_net(); print(f"  supervised-fit mse={supervised(net):.2e}", flush=True); report(net, "after FD-seed (pre-HJB)")
lbfgs(net, make_lg(net, True)); report(net, "FD-seed + HJB(precond)+LBFGS")
net2 = make_net(); supervised(net2); lbfgs(net2, make_lg(net2, False)); report(net2, "FD-seed + HJB(raw)+LBFGS")
print("\n(B) FROM-SCRATCH + L-BFGS:", flush=True)
net3 = make_net(); adam(net3, make_lg(net3, True), 5000); lbfgs(net3, make_lg(net3, True)); report(net3, "scratch + Adam + LBFGS(precond)")
print("\nDONE", flush=True)
