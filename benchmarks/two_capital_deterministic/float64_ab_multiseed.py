"""
A/B test: float32 (control) vs float64 (treated) for the small value-NN HJB solve.

Same seed / init / budget for both arms. For each dtype we run:
  (A) FD-seed + HJB(raw) + LBFGS
  (B) FD-seed + HJB(precond) + LBFGS
  (C) scratch + Adam(5000) + LBFGS(precond)
Record final L-BFGS loss floor and max|i_d - FD| over Z in [0.1,0.9].

Also a MEASURE-ONLY variant: take the float32-trained net (arm A) and recompute
the HJB residual on the SAME batch in float64, to attribute how much of the
reported residual is pure float32 measurement noise.
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

# FD ground-truth arrays (kept in float64; recast per-arm)
Zg64 = np.linspace(0.02, 0.98, 256).reshape(-1, 1).astype(np.float64)
v_fd64 = np.interp(Zg64.ravel(), fd["Z"], fd["v"]).reshape(-1, 1).astype(np.float64)
id_fd = np.interp(Zg64.ravel(), fd["Z"], fd["i_d"])
mI = (Zg64.ravel() >= 0.1) & (Zg64.ravel() <= 0.9)


def build_arm(np_dtype, tf_name):
    """Return a dict of callables/tensors all in the requested dtype."""
    Zg = Zg64.astype(np_dtype)
    Zt = tf.constant(Zg, dtype=tf_name)
    v_fd = v_fd64.astype(np_dtype)
    v0c = tf.constant(v0, dtype=tf_name)
    vNc = tf.constant(vN, dtype=tf_name)

    def make_net():
        # init reproducibly & identically regardless of dtype
        tf.random.set_seed(SEED); np.random.seed(SEED)
        init = tf.keras.initializers.GlorotUniform(seed=SEED)
        binit = tf.keras.initializers.Zeros()
        inp = tf.keras.Input(shape=(1,), dtype=tf_name); h = inp
        for _ in range(3):
            h = tf.keras.layers.Dense(32, activation="tanh", dtype=tf_name,
                                      kernel_initializer=init, bias_initializer=binit)(h)
        out = tf.keras.layers.Dense(1, dtype=tf_name,
                                    kernel_initializer=init, bias_initializer=binit)(h)
        return tf.keras.Model(inp, out)

    def clamp(vp, Z, m=1e-4):
        return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m,
                                1.0 / tf.maximum(Z, 1e-9) - m)

    def value(net, Z):
        return (1 - Z) * v0c + Z * vNc + Z * (1 - Z) * net(2.0 * Z - 1.0)

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
                v.assign(x[i:i + n].reshape(s).astype(np_dtype)); i += n
        def fg(x):
            setf(x); L, g = lg()
            return float(L), np.concatenate([gi.numpy().ravel() for gi in g]).astype(np.float64)
        x0 = np.concatenate([v.numpy().ravel() for v in vars_]).astype(np.float64)
        r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                     options={"maxiter": maxiter, "maxfun": 2 * maxiter})
        setf(r.x); return float(r.fun)

    def supervised(net, steps=4000):
        opt = tf.keras.optimizers.Adam(2e-3)
        v_fd_t = tf.constant(v_fd, dtype=tf_name)
        @tf.function
        def step():
            with tf.GradientTape() as t:
                L = tf.reduce_mean(tf.square(value(net, Zt) - v_fd_t))
            opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
            return L
        for _ in range(steps):
            L = step()
        return float(L)

    def report(net, name):
        with tf.GradientTape() as inner:
            inner.watch(Zt); v = value(net, Zt)
        vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel().astype(np.float64)
        i_d, i_g, c = M.controls(Zg64.ravel(), vp, P)
        err = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
        print(f"  [{tf_name}] {name:32s} max|i_d-FD|={err:.5f}  "
              f"de-invest={bool((i_d[mI]<0).any())}", flush=True)
        return err, vp

    return dict(make_net=make_net, make_lg=make_lg, adam=adam, lbfgs=lbfgs,
                supervised=supervised, report=report, value=value, Zt=Zt)


def run_arm(tf_name, np_dtype):
    tf.keras.backend.set_floatx(tf_name)
    A = build_arm(np_dtype, tf_name)
    res = {}
    print(f"\n==== ARM dtype={tf_name} ====", flush=True)

    # (A) FD-seed + HJB(raw) + LBFGS
    net = A["make_net"](); A["supervised"](net)
    fA = A["lbfgs"](net, A["make_lg"](net, False))
    eA, _ = A["report"](net, "FD-seed+HJB(raw)+LBFGS")
    res["A_loss"], res["A_err"] = fA, eA

    # (C) scratch + Adam(5000) + LBFGS(precond)
    net3 = A["make_net"]()
    A["adam"](net3, A["make_lg"](net3, True), 5000)
    fC = A["lbfgs"](net3, A["make_lg"](net3, True))
    eC, _ = A["report"](net3, "scratch+Adam+LBFGS(precond)")
    res["C_loss"], res["C_err"] = fC, eC

    return res, net  # return arm-A net for measure-only


print(f"FD truth: i_d in [{id_fd[mI].min():+.4f}, {id_fd[mI].max():+.4f}]\n", flush=True)

SEEDS = [0, 1, 2, 3, 4]
agg = {"float32": {"A_loss": [], "A_err": [], "C_loss": [], "C_err": []},
       "float64": {"A_loss": [], "A_err": [], "C_loss": [], "C_err": []}}
for SEED in SEEDS:
    print(f"\n######## SEED={SEED} ########", flush=True)
    r32, _ = run_arm("float32", np.float32)
    r64, _ = run_arm("float64", np.float64)
    for k in ["A_loss", "A_err", "C_loss", "C_err"]:
        agg["float32"][k].append(r32[k]); agg["float64"][k].append(r64[k])

print("\n==== AGGREGATE over seeds", SEEDS, "====", flush=True)
def summ(v):
    a = np.array(v); return f"median={np.median(a):.4e} mean={np.mean(a):.4e} min={np.min(a):.4e} max={np.max(a):.4e}"
for tag in ["float32", "float64"]:
    print(f"  {tag}:", flush=True)
    print(f"    A(raw) loss : {summ(agg[tag]['A_loss'])}", flush=True)
    print(f"    A(raw) err  : {summ(agg[tag]['A_err'])}", flush=True)
    print(f"    C(scr) loss : {summ(agg[tag]['C_loss'])}", flush=True)
    print(f"    C(scr) err  : {summ(agg[tag]['C_err'])}", flush=True)
print("\n  per-seed A(raw) err:  f32", [f"{x:.4f}" for x in agg['float32']['A_err']],
      " f64", [f"{x:.4f}" for x in agg['float64']['A_err']], flush=True)
print("  per-seed A(raw) loss: f32", [f"{x:.2e}" for x in agg['float32']['A_loss']],
      " f64", [f"{x:.2e}" for x in agg['float64']['A_loss']], flush=True)
print("DONE", flush=True)
