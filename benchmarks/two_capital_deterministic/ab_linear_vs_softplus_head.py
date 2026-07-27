"""
A/B TEST: linear value-output head (control) vs softplus value-output head (treated).

Production SimulationStochasticJumps.py builds the value net with
final_activation='softplus' (V = logK + v, v forced >= 0 & gradient-saturating).
Here the TRUE v(Z) = V - logK is strongly NEGATIVE (FD: -4.94 .. -0.19), so a
softplus head is a structural mismatch. This script quantifies the penalty by
training the SAME small value-NN twice with IDENTICAL seed / init / budget:

  CONTROL  : final_activation = None     (linear, current benchmark)
  TREATED  : final_activation = softplus (production-faithful)

We test TWO faithful embeddings of the head:
  (1) DIRECT value head:   value(Z) = head(net_body(Z))         <- closest to production
                           (softplus forces value >= 0 everywhere)
  (2) ANCHORED inner net:  value(Z) = (1-Z)v0 + Z vN + Z(1-Z) head(net_body(Z))
                           (boundary-anchored ansatz used by the benchmark; the
                            softplus only constrains the interior correction's sign)

For BOTH arms we report: final HJB residual (the loss), TRUE accuracy
max|i_d - FD| over Z in [0.1,0.9], de-invest recovery, and softplus-floor fraction.
"""
import os
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

SEED = 0
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


def make_body(head_activation):
    """3x32 tanh body + final Dense with given activation on the OUTPUT."""
    tf.random.set_seed(SEED)
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    out = tf.keras.layers.Dense(1, activation=head_activation)(h)
    return tf.keras.Model(inp, out)


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def value_direct(net, Z):
    # value(Z) = head(body(2Z-1))  -- production-faithful (softplus -> value>=0)
    return net(2.0 * Z - 1.0)


def value_anchored(net, Z):
    # boundary-anchored: softplus only constrains the interior correction
    return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)


def make_lg(net, value_fn, precond=True):
    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                inner.watch(Zt); v = value_fn(net, Zt)
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
    L = None
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


def supervised(net, value_fn, steps=4000):
    opt = tf.keras.optimizers.Adam(2e-3)
    @tf.function
    def step():
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value_fn(net, Zt) - v_fd))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L
    L = None
    for _ in range(steps):
        L = step()
    return float(L)


def true_residual_rms(net, value_fn):
    """Raw (un-preconditioned) HJB residual RMS = the honest 'loss' to compare."""
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value_fn(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt)
    v_np = v.numpy().ravel(); vp_np = vp.numpy().ravel()
    R = M.hjb_residual(Zg.ravel(), v_np, vp_np, P)
    return float(np.sqrt(np.mean(R[mI] ** 2)))


def report(net, value_fn, name, softplus=False):
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value_fn(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    i_d, i_g, c = M.controls(Zg.ravel(), vp, P)
    err = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    res = true_residual_rms(net, value_fn)
    raw = net(2.0 * Zt - 1.0).numpy().ravel()
    floor = float(np.mean(raw[mI] < 1e-4)) if softplus else float("nan")
    deinv = bool((i_d[mI] < 0).any())
    print(f"  {name:34s} resid_rms={res:.3e}  max|i_d-FD|={err:.4f}  "
          f"i_d[{i_d[mI].min():+.4f},{i_d[mI].max():+.4f}]  de-invest={deinv}  "
          f"floorfrac={floor:.3f}", flush=True)
    return dict(name=name, resid=res, err=err, deinvest=deinv, floor=floor,
                idmin=float(i_d[mI].min()), idmax=float(i_d[mI].max()))


def run_arm(head_act, value_fn, label, softplus):
    print(f"\n=== {label} ===", flush=True)
    net = make_body(head_act)
    smse = supervised(net, value_fn)
    print(f"  supervised-fit mse={smse:.2e}", flush=True)
    r_seed = report(net, value_fn, "after FD-seed (pre-HJB)", softplus)
    adam(net, make_lg(net, value_fn, True), 3000)
    lbfgs(net, make_lg(net, value_fn, True))
    r_final = report(net, value_fn, "FD-seed + Adam + LBFGS", softplus)
    return r_seed, r_final


print(f"FD truth: v in [{v_fd.min():+.3f},{v_fd.max():+.3f}]  "
      f"i_d in [{id_fd[mI].min():+.4f}, {id_fd[mI].max():+.4f}] (de-invests)", flush=True)

print("\n################ EMBEDDING (1): DIRECT value head ################")
c1s, c1f = run_arm(None, value_direct, "CONTROL  (linear head, direct)", False)
t1s, t1f = run_arm("softplus", value_direct, "TREATED  (softplus head, direct)", True)

print("\n################ EMBEDDING (2): ANCHORED inner net ################")
c2s, c2f = run_arm(None, value_anchored, "CONTROL  (linear head, anchored)", False)
t2s, t2f = run_arm("softplus", value_anchored, "TREATED  (softplus head, anchored)", True)

print("\n================ SUMMARY ================", flush=True)
def line(tag, r):
    print(f"{tag:42s} resid={r['resid']:.3e}  max|i_d-FD|={r['err']:.4f}  "
          f"deinvest={r['deinvest']}  floor={r['floor']:.3f}", flush=True)
line("(1) DIRECT  CONTROL  linear  final", c1f)
line("(1) DIRECT  TREATED  softplus final", t1f)
line("(2) ANCHOR  CONTROL  linear  final", c2f)
line("(2) ANCHOR  TREATED  softplus final", t2f)
print("\nDONE", flush=True)
