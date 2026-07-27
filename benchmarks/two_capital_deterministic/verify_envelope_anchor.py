"""
INDEPENDENT VERIFICATION of envelope-bounds-and-certificate A/B.

Re-runs CONTROL (pure L2 strong-residual value-NN, Adam+L-BFGS) vs
TREATED (same seed/init/budget + analytic anchors a1 above-chord, a2 concavity,
a3 corner-slope) on the HARD de-invest pocket A_d=0.05, with FRESH seeds
NOT used in the original report (original used seeds 1,2,3).

Fairness checks enforced:
  - identical net init per seed (same make_net + seed reset before BOTH arms)
  - identical Adam steps / LR / L-BFGS budget for both arms
  - TRUE error vs FD ground truth only (max|i_d-FD| interior, max|v-FD|)
  - we ALSO report a W_SLOPE=0 (anchors a1+a2 only) variant to isolate the
    exact anchors from the partially-wrong corner-slope anchor.
"""
import json
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

tf.keras.backend.set_floatx("float32")

SEEDS = [7, 11, 13]          # fresh seeds, NOT in original {1,2,3}
A_DS = [0.05]                # the hard de-invest pocket (weak-id case)
N_ADAM = 6000
ADAM_LR = 2e-3
N_COLLO = 256

W_CHORD = 1.0e-1
W_CONCAVE = 1.0e-2
W_SLOPE = 5.0e-2


def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zg = np.linspace(0.02, 0.98, N_COLLO).reshape(-1, 1).astype(np.float32)
    v_fd = np.interp(Zg.ravel(), fd["Z"], fd["v"]).astype(np.float32)
    id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])
    v0, vN = M.boundary_values(P)
    sl0 = float(M.perturbation_slope(np.array([0.02]), P)[0])
    sl1 = float(M.perturbation_slope(np.array([0.98]), P)[0])
    return P, fd, Zg, v_fd, id_fd, v0, vN, sl0, sl1


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m,
                            1.0 / tf.maximum(Z, 1e-9) - m)


def make_lossfns(net, P, Zt, v0, vN, sl0, sl1, treated, w_slope):
    dl = P["delta"]; A_d = P["A_d"]; A_g = P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]

    def value(Z):
        return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)

    def residual_and_anchor():
        with tf.GradientTape() as t2:
            t2.watch(Zt)
            with tf.GradientTape() as t1:
                t1.watch(Zt); v = value(Zt)
            vp_raw = t1.gradient(v, Zt)
        vpp = t2.gradient(vp_raw, Zt)
        vp = clamp(vp_raw, Zt)
        q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
        Abar = (1 - Zt) * A_d + Zt * A_g
        c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
        i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
        phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
        phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
        mu = Zt * (1 - Zt) * (phi_g - phi_d)
        R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
        chord = (1 - Zt) * v0 + Zt * vN
        above = v - chord
        return R, vp, vpp, above, i_d

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            R, vp, vpp, above, _ = residual_and_anchor()
            loss = tf.reduce_mean(tf.square(R))
            if treated:
                loss += W_CHORD * tf.reduce_mean(tf.square(tf.nn.relu(-above)))
                loss += W_CONCAVE * tf.reduce_mean(tf.square(tf.nn.relu(vpp)))
                wlo = tf.cast(Zt < 0.1, tf.float32)
                whi = tf.cast(Zt > 0.9, tf.float32)
                denom = tf.maximum(tf.reduce_sum(wlo) + tf.reduce_sum(whi), 1.0)
                anch = (tf.reduce_sum(wlo * tf.square(vp - sl0))
                        + tf.reduce_sum(whi * tf.square(vp - sl1))) / denom
                loss += w_slope * anch
        return loss, outer.gradient(loss, net.trainable_variables)
    return lg, value, residual_and_anchor


def adam(net, lg, steps, lr):
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
                 options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-15, "gtol": 1e-12})
    setf(r.x); return r.nit


def evaluate(value, ranch, Zg, v_fd, id_fd, delta):
    mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
    R, vp, vpp, above, i_d = ranch()
    v_nn = value(tf.constant(Zg)).numpy().ravel()
    i_d = i_d.numpy().ravel()
    err_id = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    err_v = float(np.max(np.abs(v_nn - v_fd)))
    Rinf = float(np.max(np.abs(R.numpy())))
    return err_id, err_v, Rinf, Rinf / delta


def run_arm(seed, treated, w_slope, P, Zt, v0, vN, sl0, sl1, Zg, v_fd, id_fd):
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_net()
    lg, value, ranch = make_lossfns(net, P, Zt, v0, vN, sl0, sl1, treated, w_slope)
    adam(net, lg, N_ADAM, ADAM_LR)
    nit = lbfgs(net, lg)
    return evaluate(value, ranch, Zg, v_fd, id_fd, P["delta"]), nit


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zg, v_fd, id_fd, v0, vN, sl0, sl1 = build_case(A_d)
        Zt = tf.constant(Zg)
        delta = P["delta"]
        mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
        print(f"\n##### A_d={A_d} FD i_d interior [{id_fd[mI].min():+.4f},{id_fd[mI].max():+.4f}] "
              f"deinvest={bool((id_fd[mI]<0).any())} sl0={sl0:.3f} sl1={sl1:.3f}", flush=True)
        for seed in SEEDS:
            (eC, nC) = run_arm(seed, False, W_SLOPE, P, Zt, v0, vN, sl0, sl1, Zg, v_fd, id_fd)
            (eT, nT) = run_arm(seed, True, W_SLOPE, P, Zt, v0, vN, sl0, sl1, Zg, v_fd, id_fd)
            (eN, nN) = run_arm(seed, True, 0.0, P, Zt, v0, vN, sl0, sl1, Zg, v_fd, id_fd)
            row = {"A_d": A_d, "seed": seed,
                   "err_id_ctrl": eC[0], "err_v_ctrl": eC[1], "Rinf_ctrl": eC[2], "cert_ctrl": eC[3],
                   "err_id_treat": eT[0], "err_v_treat": eT[1], "Rinf_treat": eT[2], "cert_treat": eT[3],
                   "err_id_noslope": eN[0], "err_v_noslope": eN[1], "Rinf_noslope": eN[2], "cert_noslope": eN[3],
                   "id_ratio": eT[0] / max(eC[0], 1e-12),
                   "id_ratio_noslope": eN[0] / max(eC[0], 1e-12),
                   "v_ratio": eT[1] / max(eC[1], 1e-12),
                   "cert_valid_ctrl": bool(eC[3] >= eC[1]),
                   "cert_valid_treat": bool(eT[3] >= eT[1])}
            print(f" seed={seed} CONTROL: id={eC[0]:.4e} v={eC[1]:.4e} R={eC[2]:.3e} cert_valid={row['cert_valid_ctrl']} nit={nC}", flush=True)
            print(f" seed={seed} TREATED: id={eT[0]:.4e} v={eT[1]:.4e} R={eT[2]:.3e} cert_valid={row['cert_valid_treat']} nit={nT}", flush=True)
            print(f" seed={seed} NOSLOPE: id={eN[0]:.4e} v={eN[1]:.4e} R={eN[2]:.3e} nit={nN}", flush=True)
            print(f"   -> id_ratio(T/C)={row['id_ratio']:.3f} id_ratio_noslope(N/C)={row['id_ratio_noslope']:.3f} v_ratio={row['v_ratio']:.3f}", flush=True)
            results.append(row)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
