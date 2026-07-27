"""
ADVERSARIAL VERIFY of the float64-vs-float32 A/B claim ("no-effect").
Re-runs the SAME arms (A=FD-seed+HJB(raw)+LBFGS, C=scratch+Adam+LBFGS-precond)
with NEW seeds (passed on argv) to check the prior single-seed result was noise.

Records, per dtype: L-BFGS loss floor AND max|i_d-FD| (a lower loss with worse
FD-accuracy is a FALSE win). Also the seed-independent MEASURE-ONLY check.
"""
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

Zg64 = np.linspace(0.02, 0.98, 256).reshape(-1, 1).astype(np.float64)
v_fd64 = np.interp(Zg64.ravel(), fd["Z"], fd["v"]).reshape(-1, 1).astype(np.float64)
id_fd = np.interp(Zg64.ravel(), fd["Z"], fd["i_d"])
mI = (Zg64.ravel() >= 0.1) & (Zg64.ravel() <= 0.9)


def build_arm(np_dtype, tf_name, SEED):
    Zg = Zg64.astype(np_dtype)
    Zt = tf.constant(Zg, dtype=tf_name)
    v_fd = v_fd64.astype(np_dtype)
    v0c = tf.constant(v0, dtype=tf_name)
    vNc = tf.constant(vN, dtype=tf_name)

    def make_net():
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

    def rawres(net):
        # raw (unweighted) RMS HJB residual, in float64, for an honest cross-dtype scale
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
        return float(np.sqrt(np.mean(R.numpy().astype(np.float64) ** 2)))

    def report(net):
        with tf.GradientTape() as inner:
            inner.watch(Zt); v = value(net, Zt)
        vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel().astype(np.float64)
        i_d, i_g, c = M.controls(Zg64.ravel(), vp, P)
        return float(np.max(np.abs(i_d[mI] - id_fd[mI])))

    return dict(make_net=make_net, make_lg=make_lg, adam=adam, lbfgs=lbfgs,
                supervised=supervised, report=report, rawres=rawres)


def run_arm(tf_name, np_dtype, SEED):
    tf.keras.backend.set_floatx(tf_name)
    A = build_arm(np_dtype, tf_name, SEED)
    res = {}
    # (A) FD-seed + HJB(raw) + LBFGS  -- the clean loss-floor A/B
    net = A["make_net"](); A["supervised"](net)
    res["A_loss"] = A["lbfgs"](net, A["make_lg"](net, False))
    res["A_rawres"] = A["rawres"](net)
    res["A_err"] = A["report"](net)
    # (C) scratch + Adam(5000) + LBFGS(precond)
    net3 = A["make_net"]()
    A["adam"](net3, A["make_lg"](net3, True), 5000)
    res["C_loss"] = A["lbfgs"](net3, A["make_lg"](net3, True))
    res["C_rawres"] = A["rawres"](net3)
    res["C_err"] = A["report"](net3)
    return res, net


SEEDS = [int(s) for s in sys.argv[1:]] or [11, 12]
print(f"FD truth: i_d in [{id_fd[mI].min():+.4f}, {id_fd[mI].max():+.4f}]", flush=True)
print(f"VERIFY SEEDS = {SEEDS}\n", flush=True)

agg = {"float32": {k: [] for k in ["A_loss","A_rawres","A_err","C_loss","C_rawres","C_err"]},
       "float64": {k: [] for k in ["A_loss","A_rawres","A_err","C_loss","C_rawres","C_err"]}}

net32_last = None
for SEED in SEEDS:
    print(f"######## SEED={SEED} ########", flush=True)
    r32, net32 = run_arm("float32", np.float32, SEED)
    r64, _ = run_arm("float64", np.float64, SEED)
    net32_last = (net32, SEED)
    print(f"  f32: A loss={r32['A_loss']:.4e} rawres={r32['A_rawres']:.3e} err={r32['A_err']:.5f} | "
          f"C loss={r32['C_loss']:.4e} rawres={r32['C_rawres']:.3e} err={r32['C_err']:.5f}", flush=True)
    print(f"  f64: A loss={r64['A_loss']:.4e} rawres={r64['A_rawres']:.3e} err={r64['A_err']:.5f} | "
          f"C loss={r64['C_loss']:.4e} rawres={r64['C_rawres']:.3e} err={r64['C_err']:.5f}", flush=True)
    for k in agg["float32"]:
        agg["float32"][k].append(r32[k]); agg["float64"][k].append(r64[k])

# MEASURE-ONLY (seed-independent): same weights, residual in each dtype
print("\n==== MEASURE-ONLY (last f32 net, residual recomputed in f64) ====", flush=True)
net32, SEED = net32_last
tf.keras.backend.set_floatx("float32")
A32 = build_arm(np.float32, "float32", SEED)
L32 = float(A32["make_lg"](net32, False)()[0])
tf.keras.backend.set_floatx("float64")
A64 = build_arm(np.float64, "float64", SEED)
net64c = A64["make_net"]()
for w32, w64 in zip(net32.weights, net64c.weights):
    w64.assign(w32.numpy().astype(np.float64))
L64 = float(A64["make_lg"](net64c, False)()[0])
print(f"  same weights: f32-loss={L32:.6e}  f64-loss={L64:.6e}  abs gap={abs(L32-L64):.3e}", flush=True)

print("\n==== AGGREGATE over", SEEDS, "====", flush=True)
def med(v): return np.median(np.array(v))
for tag in ["float32", "float64"]:
    a = agg[tag]
    print(f"  {tag}: A loss med={med(a['A_loss']):.4e} rawres med={med(a['A_rawres']):.3e} err med={med(a['A_err']):.5f}", flush=True)
    print(f"           C loss med={med(a['C_loss']):.4e} rawres med={med(a['C_rawres']):.3e} err med={med(a['C_err']):.5f}", flush=True)
print("\n  per-seed A rawres: f32", [f"{x:.3e}" for x in agg['float32']['A_rawres']],
      "f64", [f"{x:.3e}" for x in agg['float64']['A_rawres']], flush=True)
print("  per-seed A err   : f32", [f"{x:.4f}" for x in agg['float32']['A_err']],
      "f64", [f"{x:.4f}" for x in agg['float64']['A_err']], flush=True)
print("  per-seed C err   : f32", [f"{x:.4f}" for x in agg['float32']['C_err']],
      "f64", [f"{x:.4f}" for x in agg['float64']['C_err']], flush=True)
print("DONE", flush=True)
