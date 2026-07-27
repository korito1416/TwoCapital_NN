"""
Multi-seed adversarial verification of value_only_lbfgs_polish.

FAITHFUL to lbfgs_polish_ab.py except:
  - SEED is taken from argv (so we can re-run with DIFFERENT random seeds).
  - We assert the comparison is FAIR: CONTROL and TREATED start from the
    IDENTICAL Adam-trained weights (same budget, same init), and the only
    difference is the L-BFGS phase.
  - We explicitly flag a "FALSE WIN": residual drops but FD-accuracy worsens.

Run:  python lbfgs_polish_ab_seeds.py 0 1 2
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

ADAM_STEPS = int(os.environ.get("ADAM_STEPS", "5000"))
ADAM_LR = 2e-3
LBFGS_MAXITER = 4000

Zg = np.linspace(0.02, 0.98, 256).reshape(-1, 1).astype(np.float32)
Zt = tf.constant(Zg)
mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def value(net, Z, v0, vN):
    return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)


def make_lg(net, P, v0, vN, precond):
    dl, A_g, A_d = P["delta"], P["A_g"], P["A_d"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                inner.watch(Zt); v = value(net, Zt, v0, vN)
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


def rms_residual(net, P, v0, vN):
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt, v0, vN)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    Zr = Zg.ravel()
    R = M.hjb_residual(Zr, v.numpy().ravel(), vp, P)
    return float(np.sqrt(np.mean(R ** 2)))


def adam(net, lg, steps, lr):
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
                 options={"maxiter": maxiter, "maxfun": 2 * maxiter,
                          "ftol": 1e-15, "gtol": 1e-12, "maxcor": 50})
    setf(r.x)
    print(f"    [lbfgs] nit={r.nit} nfev={r.nfev} status={r.status} msg={str(r.message)[:60]}", flush=True)
    return float(r.fun), int(r.nit)


def control_err(net, P, v0, vN):
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt, v0, vN)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    i_d, i_g, c = M.controls(Zg.ravel(), vp, P)
    return i_d


def clone_weights(src):
    dst = make_net()
    for a, b in zip(dst.variables, src.variables):
        a.assign(b)
    return dst


def run_case(A_d, precond, seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    v0, vN = M.boundary_values(P)
    fd = solve_fd(P, n=4000)
    id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])

    net = make_net()
    adam(net, make_lg(net, P, v0, vN, precond), ADAM_STEPS, ADAM_LR)

    res_ctrl = rms_residual(net, P, v0, vN)
    id_ctrl = control_err(net, P, v0, vN)
    err_ctrl = float(np.max(np.abs(id_ctrl[mI] - id_fd[mI])))

    net_t = clone_weights(net)
    # FAIRNESS assert: identical start
    assert abs(rms_residual(net_t, P, v0, vN) - res_ctrl) < 1e-9
    _, nit = lbfgs(net_t, make_lg(net_t, P, v0, vN, precond), LBFGS_MAXITER)
    res_treat = rms_residual(net_t, P, v0, vN)
    id_treat = control_err(net_t, P, v0, vN)
    err_treat = float(np.max(np.abs(id_treat[mI] - id_fd[mI])))

    res_ratio = res_treat / res_ctrl
    err_ratio = err_treat / max(err_ctrl, 1e-12)
    false_win = (res_ratio < 0.9) and (err_ratio > 1.05)
    tag = f"seed={seed} A_d={A_d} precond={precond}"
    print(f"\n=== {tag} ===", flush=True)
    print(f"  FD i_d interior [{id_fd[mI].min():+.4f},{id_fd[mI].max():+.4f}] deinvest={bool((id_fd[mI]<0).any())}", flush=True)
    print(f"  CONTROL : RMS_resid={res_ctrl:.4e}  max|i_d-FD|={err_ctrl:.4e}", flush=True)
    print(f"  TREATED : RMS_resid={res_treat:.4e}  max|i_d-FD|={err_treat:.4e}  nit={nit}", flush=True)
    print(f"  resid T/C={res_ratio:.3e}  err T/C={err_ratio:.3e}  FALSE_WIN={false_win}", flush=True)
    return {"seed": seed, "A_d": A_d, "precond": precond,
            "res_ctrl": res_ctrl, "err_ctrl": err_ctrl,
            "res_treat": res_treat, "err_treat": err_treat,
            "res_ratio": res_ratio, "err_ratio": err_ratio,
            "nit": nit, "false_win": false_win}


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0]
    results = []
    for seed in seeds:
        for A_d in [0.05, 0.13]:
            for precond in [False, True]:
                results.append(run_case(A_d, precond, seed))
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)
