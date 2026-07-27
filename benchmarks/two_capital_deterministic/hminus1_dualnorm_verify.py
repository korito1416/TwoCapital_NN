"""
VERIFICATION re-run of hminus1_dualnorm_ab.py with DIFFERENT seeds {3,4,5}.
Same net / init / optimizer budget (Adam 5000 + L-BFGS 4000), same true-error-vs-FD
metrics.  Goal: confirm (or refute) the reported 'backfires' verdict reproduces.
"""
import os, json
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

tf.keras.backend.set_floatx("float32")

Zg = np.linspace(0.02, 0.98, 256).reshape(-1, 1).astype(np.float32)
Zt = tf.constant(Zg)
dZ = float(Zg[1, 0] - Zg[0, 0])
mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m,
                            1.0 / tf.maximum(Z, 1e-9) - m)


def make_ctx(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    v0, vN = M.boundary_values(P)
    fd = solve_fd(P, n=4000)
    v_fd = np.interp(Zg.ravel(), fd["Z"], fd["v"]).reshape(-1, 1).astype(np.float32)
    slope_fd = np.interp(Zg.ravel(), fd["Z"], fd["slope"])
    id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])
    return dict(P=P, v0=float(v0), vN=float(vN), v_fd=v_fd, slope_fd=slope_fd, id_fd=id_fd)


def value(net, Z, v0, vN):
    return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)


def residual_pieces(net, ctx):
    P = ctx["P"]; dl = P["delta"]; A_d = P["A_d"]; A_g = P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt, ctx["v0"], ctx["vN"])
    vp = clamp(inner.gradient(v, Zt), Zt)
    q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
    Abar = (1 - Zt) * A_d + Zt * A_g
    c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
    i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
    phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
    phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
    mu = Zt * (1 - Zt) * (phi_g - phi_d)
    R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
    return R, mu, v


def make_lg(net, ctx, mode):
    dl = ctx["P"]["delta"]

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            R, mu, _ = residual_pieces(net, ctx)
            if mode == "l2":
                loss = tf.reduce_mean(tf.square(R))
            elif mode == "hm1":
                muf = tf.stop_gradient(mu[:, 0])
                N = tf.shape(muf)[0]
                coef = muf / dZ
                fwd = muf > 0.0
                diag = -dl + tf.where(fwd, coef, -coef)
                sup = tf.where(fwd, -coef, tf.zeros_like(coef))
                sub = tf.where(fwd, tf.zeros_like(coef), coef)
                sup_sh = tf.concat([sup[:-1], [0.0]], axis=0)
                sub_sh = tf.concat([[0.0], sub[1:]], axis=0)
                diags = tf.stack([sup_sh, diag, sub_sh], axis=0)
                rhs = tf.reshape(R, [1, N, 1])
                z = tf.linalg.tridiagonal_solve(
                    tf.expand_dims(diags, 0), rhs, diagonals_format="compact")
                loss = tf.reduce_mean(tf.square(z))
            else:
                raise ValueError(mode)
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


def strong_l2_value(net, ctx):
    """Always report the COMMON, comparable metric: RMS of strong residual."""
    R, _, _ = residual_pieces(net, ctx)
    return float(np.sqrt(np.mean(R.numpy()**2)))


def true_error(net, ctx):
    P = ctx["P"]
    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(net, Zt, ctx["v0"], ctx["vN"])
    vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
    v_np = v.numpy().ravel()
    i_d, i_g, c = M.controls(Zg.ravel(), vp, P)
    e_id = float(np.max(np.abs(i_d[mI] - ctx["id_fd"][mI])))
    e_v = float(np.max(np.abs(v_np[mI] - ctx["v_fd"].ravel()[mI])))
    e_vp = float(np.max(np.abs(vp[mI] - ctx["slope_fd"][mI])))
    de = bool((i_d[mI] < 0).any())
    return dict(max_id_err=e_id, vinf_err=e_v, max_vp_err=e_vp, deinvest=de)


def run_arm(A_d, mode, seed, adam_steps=5000):
    tf.random.set_seed(seed); np.random.seed(seed)
    ctx = make_ctx(A_d)
    net = make_net()
    lg = make_lg(net, ctx, mode)
    adam(net, lg, adam_steps)
    lbfgs(net, lg)
    err = true_error(net, ctx)
    # common comparable strong-L2 RMS residual on the SAME grid
    err["strong_rms"] = strong_l2_value(net, ctx)
    err.update(A_d=A_d, mode=mode, seed=seed)
    return err, ctx


def main():
    seeds = [3, 4, 5]
    results = []
    for A_d in [0.13, 0.05]:
        ctx0 = make_ctx(A_d)
        print(f"\n=== A_d={A_d}  FD truth: i_d in "
              f"[{ctx0['id_fd'][mI].min():+.4f},{ctx0['id_fd'][mI].max():+.4f}] "
              f"deinvest={bool((ctx0['id_fd'][mI]<0).any())} ===", flush=True)
        for mode in ["l2", "hm1"]:
            errs = []
            for s in seeds:
                e, _ = run_arm(A_d, mode, s)
                errs.append(e); results.append(e)
                print(f"  {mode:4s} seed={s}  max|i_d-FD|={e['max_id_err']:.4f}  "
                      f"||v-v_fd||inf={e['vinf_err']:.4f}  max|v'-FD|={e['max_vp_err']:.4f}  "
                      f"strongRMS={e['strong_rms']:.3e}  deinvest={e['deinvest']}", flush=True)
            ei = np.array([e['max_id_err'] for e in errs])
            ev = np.array([e['vinf_err'] for e in errs])
            evp = np.array([e['max_vp_err'] for e in errs])
            print(f"  >> {mode:4s} MEDIAN max|i_d-FD|={np.median(ei):.4f} "
                  f"||v||inf={np.median(ev):.4f} max|v'|={np.median(evp):.4f} "
                  f"(min_id={ei.min():.4f})", flush=True)
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "hminus1_dualnorm_verify.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSAVED outputs/hminus1_dualnorm_verify.json", flush=True)


if __name__ == "__main__":
    main()
