"""
A/B test: smooth (softplus) floor vs hard tf.maximum clamp inside log(1+theta*i)
at the de-invest corner (A_d=0.05).

CONTROL  : phi uses tf.math.log(tf.maximum(1+theta*i, 1e-8))   [hard kink]
TREATED  : phi uses tf.math.log(1e-8 + softplus(1+theta*i))    [C2 smooth floor]

Everything else identical: same seed/init/budget. Pipeline =
FD-seed (supervised) -> HJB(precond)+L-BFGS, the exact pipeline the de-invest
case already needs. We measure final HJB loss AND true accuracy max|i_d - FD|
in the interior band (Z in [0.1,0.9]) AND specifically in the de-invest band
(where FD i_d < 0).
"""
import os
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

SEED = int(os.environ.get("AB_SEED", "13"))
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
mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)        # interior band
mD = mI & (id_fd < 0.0)                                 # de-invest band (FD i_d<0)


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def value(net, Z):
    return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)


def make_lg(net, precond, smooth):
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
            arg_d = 1 + td * i_d; arg_g = 1 + tg * i_g
            if smooth:
                # C2 smooth lower bound: log(1e-8 + softplus(arg)). softplus(arg)>0
                # everywhere and ~= max(arg,0) away from 0, so no kink in the band.
                phi_d = ad + Gd * tf.math.log(1e-8 + tf.math.softplus(arg_d))
                phi_g = ag + Gg * tf.math.log(1e-8 + tf.math.softplus(arg_g))
            else:
                phi_d = ad + Gd * tf.math.log(tf.maximum(arg_d, 1e-8))
                phi_g = ag + Gg * tf.math.log(tf.maximum(arg_g, 1e-8))
            mu = Zt * (1 - Zt) * (phi_g - phi_d)
            R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
            w = (tf.abs(mu) + 5e-3) if precond else tf.ones_like(mu)
            loss = tf.reduce_mean(tf.square(R / w))
        return loss, outer.gradient(loss, net.trainable_variables)
    return lg


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


def measure(net):
    """Return (err_interior, err_deinvest, true HJB-loss-raw under hard-clamp eval)."""
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt)
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    i_d, i_g, c = M.controls(Zg.ravel(), vp, P)
    err_I = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    err_D = float(np.max(np.abs(i_d[mD] - id_fd[mD]))) if mD.any() else float("nan")
    # true raw residual (NumPy, consistent metric independent of which loss was trained)
    res = M.hjb_residual(Zg.ravel(), value(net, Zt).numpy().ravel(), vp, P)
    res_I = float(np.sqrt(np.mean(res[mI] ** 2)))
    return err_I, err_D, res_I, i_d


def run_arm(smooth, tag):
    # IDENTICAL init via identical seed
    tf.random.set_seed(SEED); np.random.seed(SEED)
    net = make_net()
    mse = supervised(net)
    e0I, e0D, r0, _ = measure(net)
    print(f"[{tag}] supervised mse={mse:.2e}  pre-HJB: err_I={e0I:.4f} err_D={e0D:.4f} resRMS={r0:.3e}", flush=True)
    Lf = lbfgs(net, make_lg(net, precond=True, smooth=smooth))
    eI, eD, rI, i_d = measure(net)
    print(f"[{tag}] post LBFGS: trained-loss={Lf:.3e}  err_I={eI:.4f} err_D={eD:.4f} "
          f"true-resRMS_I={rI:.3e}  deinvest_present={bool((i_d[mI]<0).any())}", flush=True)
    return dict(tag=tag, trained_loss=Lf, err_I=eI, err_D=eD, res_I=rI)


print(f"FD truth: i_d interior in [{id_fd[mI].min():+.4f}, {id_fd[mI].max():+.4f}]; "
      f"de-invest points in band = {int(mD.sum())} of {int(mI.sum())}\n", flush=True)

control = run_arm(smooth=False, tag="CONTROL hard-clamp ")
print(flush=True)
treated = run_arm(smooth=True,  tag="TREATED softplus    ")

print("\n=== SUMMARY ===", flush=True)
print(f"               trained_loss   err_I      err_D      true_resRMS_I", flush=True)
for d in (control, treated):
    print(f"{d['tag']:20s} {d['trained_loss']:.3e}   {d['err_I']:.4f}   {d['err_D']:.4f}   {d['res_I']:.3e}", flush=True)
print("\nDONE", flush=True)
