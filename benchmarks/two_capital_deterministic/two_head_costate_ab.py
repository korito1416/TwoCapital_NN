"""
A/B test of the two-head FOC-driven costate method (key: two-head-foc-driven-costate).

HYPOTHESIS: the v'-information exists in the FOC / costate structure, NOT in the
flat strong-residual R. So parameterize the marginal value p=v' with a DEDICATED
head and train it predominantly through the WELL-CONDITIONED channel (the costate /
adjoint equation R'(Z) = dR/dZ, which carries the LARGE dFOC/dv' gradient), routing
around the flat R. A soft derivative tie keeps p_head consistent with autodiff(value).

CONTROL  = single-head value-NN, p via autodiff, pure L2(strong-residual R).
           (Adam + tight L-BFGS) -- identical to the existing benchmark scripts.
TREATED  = TWO heads: value(Z) and p_head(Z). Controls computed from p_head via the
           closed-form FOC inversion M.controls. Loss:
              w_R   * mean(R^2)                      (weak global anchor, SMALL)
            + w_cs  * mean(Rprime^2)                 (costate/adjoint -- the strong channel)
            + w_tie * mean((autodiff(value) - p_head)^2)
            + w_anc * corner anchor of p_head to perturbation_slope (very weak)
           Same seed / init budget as control.

R (HJB strong residual, the SAME object both arms; used for true-error-free
   reporting too):
   R = delta*log c - delta*v + (1-Z)phi_d + Z phi_g + mu * p
Rprime = d/dZ R  computed by autodiff of the residual expression w.r.t. Z, with the
   value's v supplied by the value-head and p (=v') supplied by p_head, p'(=v'')
   by autodiff of p_head. (Envelope: controls are already FOC-optimal in p, so the
   only Z-dependence that matters flows through p_head -> the dFOC/dp gradient.)

TRUE ERROR (the ONLY metric): max|i_d - i_d_FD| over Z in [0.1,0.9], max|p - p_fd|,
max|v - v_fd|. Measured in BOTH A_d=0.13 (well-cond) and A_d=0.05 (weak-id pocket).
"""
import json
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

tf.keras.backend.set_floatx("float32")

SEEDS = [1, 2, 3]
A_DS = [0.05, 0.13]
N_ADAM = 6000
ADAM_LR = 2e-3
N_COLLO = 256

# treated loss weights
W_R = 1.0e-2     # strong residual demoted to a weak global anchor
W_CS = 1.0       # costate/adjoint residual -- the WELL-CONDITIONED channel
W_TIE = 1.0      # soft derivative tie autodiff(value) ~ p_head
W_ANC = 1.0e-3   # very weak corner anchor (perturbation slope is only approximate)


def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zg = np.linspace(0.02, 0.98, N_COLLO).reshape(-1, 1).astype(np.float32)
    v_fd = np.interp(Zg.ravel(), fd["Z"], fd["v"]).astype(np.float32)
    id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])
    p_fd = np.interp(Zg.ravel(), fd["Z"], fd["slope"]).astype(np.float32)
    v0, vN = M.boundary_values(P)
    sl0 = float(M.perturbation_slope(np.array([0.02]), P)[0])
    sl1 = float(M.perturbation_slope(np.array([0.98]), P)[0])
    return P, fd, Zg, v_fd, id_fd, p_fd, v0, vN, sl0, sl1


def make_net(out_init_bias=0.0):
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    out = tf.keras.layers.Dense(1)(h)
    return tf.keras.Model(inp, out)


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m,
                            1.0 / tf.maximum(Z, 1e-9) - m)


def residual_expr(Z, v, p, P):
    """Strong HJB residual R(Z) given v and slope p (both tf tensors). Controls
    are the closed-form FOC inversion in p (M.controls, in tf)."""
    dl, A_d, A_g = P["delta"], P["A_d"], P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]
    q_d = 1 - Z * p; q_g = 1 + (1 - Z) * p
    Abar = (1 - Z) * A_d + Z * A_g
    c = dl * (Abar + (1 - Z) / td + Z / tg) / (dl + (1 - Z) * Gd * q_d + Z * Gg * q_g)
    i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
    phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
    phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
    mu = Z * (1 - Z) * (phi_g - phi_d)
    R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Z) * phi_d + Z * phi_g + mu * p
    return R, i_d, c


# ---------------- CONTROL: single value-head, autodiff p, pure L2(R) ----------
def make_control_lossfn(net, P, Zt, v0, vN):
    def value(Z):
        return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            with tf.GradientTape() as t1:
                t1.watch(Zt); v = value(Zt)
            vp = clamp(t1.gradient(v, Zt), Zt)
            R, _, _ = residual_expr(Zt, v, vp, P)
            loss = tf.reduce_mean(tf.square(R))
        return loss, outer.gradient(loss, net.trainable_variables)

    def eval_fn(Zg):
        Zc = tf.constant(Zg)
        with tf.GradientTape() as t1:
            t1.watch(Zc); v = value(Zc)
        vp = clamp(t1.gradient(v, Zc), Zc).numpy().ravel()
        return value(Zc).numpy().ravel(), vp
    return lg, eval_fn


# ---------------- TREATED: two heads (value, p_head); FOC-driven costate -------
def make_treated_lossfn(vnet, pnet, P, Zt, v0, vN, sl0, sl1):
    def value(Z):
        return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * vnet(2.0 * Z - 1.0)

    def p_head(Z):
        # raw marginal-value head; clamp to the admissible q>0 band
        return clamp(pnet(2.0 * Z - 1.0), Z)

    allvars = vnet.trainable_variables + pnet.trainable_variables
    wlo = tf.cast(Zt < 0.1, tf.float32)
    whi = tf.cast(Zt > 0.9, tf.float32)
    denom = tf.maximum(tf.reduce_sum(wlo) + tf.reduce_sum(whi), 1.0)

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            # autodiff slope of the VALUE head (for the tie)
            with tf.GradientTape() as t1:
                t1.watch(Zt); v = value(Zt)
            v_slope = clamp(t1.gradient(v, Zt), Zt)
            # costate channel: Rprime = d/dZ R, with p from p_head and v from value-head.
            with tf.GradientTape() as tcs:
                tcs.watch(Zt)
                p = p_head(Zt)
                R_cs, _, _ = residual_expr(Zt, value(Zt), p, P)
            Rprime = tcs.gradient(R_cs, Zt)
            # weak global strong-residual anchor (uses p_head as the slope)
            R, _, _ = residual_expr(Zt, value(Zt), p_head(Zt), P)
            p_now = p_head(Zt)

            loss = W_R * tf.reduce_mean(tf.square(R))
            loss += W_CS * tf.reduce_mean(tf.square(Rprime))
            loss += W_TIE * tf.reduce_mean(tf.square(v_slope - p_now))
            anch = (tf.reduce_sum(wlo * tf.square(p_now - sl0))
                    + tf.reduce_sum(whi * tf.square(p_now - sl1))) / denom
            loss += W_ANC * anch
        return loss, outer.gradient(loss, allvars)

    def eval_fn(Zg):
        Zc = tf.constant(Zg)
        p = p_head(Zc).numpy().ravel()
        return value(Zc).numpy().ravel(), p
    return lg, eval_fn, allvars


def adam(lg, vars_, steps, lr):
    opt = tf.keras.optimizers.Adam(lr)
    L = None
    for _ in range(steps):
        L, g = lg(); opt.apply_gradients(zip(g, vars_))
    return float(L)


def lbfgs(lg, vars_, maxiter=4000):
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


def true_error(eval_fn, Zg, P, v_fd, id_fd, p_fd):
    mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
    v_nn, p_nn = eval_fn(Zg)
    i_d, _, _ = M.controls(Zg.ravel(), p_nn, P)
    err_id = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    err_p = float(np.max(np.abs(p_nn[mI] - p_fd[mI])))
    err_v = float(np.max(np.abs(v_nn - v_fd)))
    deinvest = bool((i_d[mI] < 0).any())
    return err_id, err_p, err_v, deinvest


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zg, v_fd, id_fd, p_fd, v0, vN, sl0, sl1 = build_case(A_d)
        Zt = tf.constant(Zg)
        mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd[mI].min():+.4f},{id_fd[mI].max():+.4f}] "
              f"deinvest={bool((id_fd[mI]<0).any())}  pert sl0={sl0:.3f} sl1={sl1:.3f}", flush=True)
        for seed in SEEDS:
            row = {"A_d": A_d, "seed": seed}

            # ---- CONTROL ----
            tf.random.set_seed(seed); np.random.seed(seed)
            net = make_net()
            lg_c, eval_c = make_control_lossfn(net, P, Zt, v0, vN)
            adam(lg_c, net.trainable_variables, N_ADAM, ADAM_LR)
            nit_c = lbfgs(lg_c, net.trainable_variables)
            eid_c, ep_c, ev_c, di_c = true_error(eval_c, Zg, P, v_fd, id_fd, p_fd)
            print(f"  seed={seed} CONTROL: max|i_d-FD|={eid_c:.4e} max|p-FD|={ep_c:.4e} "
                  f"max|v-FD|={ev_c:.4e} deinv={di_c} nit={nit_c}", flush=True)

            # ---- TREATED ----  same seed/init
            tf.random.set_seed(seed); np.random.seed(seed)
            vnet = make_net(); pnet = make_net()
            lg_t, eval_t, allv = make_treated_lossfn(vnet, pnet, P, Zt, v0, vN, sl0, sl1)
            adam(lg_t, allv, N_ADAM, ADAM_LR)
            nit_t = lbfgs(lg_t, allv)
            eid_t, ep_t, ev_t, di_t = true_error(eval_t, Zg, P, v_fd, id_fd, p_fd)
            print(f"  seed={seed} TREATED: max|i_d-FD|={eid_t:.4e} max|p-FD|={ep_t:.4e} "
                  f"max|v-FD|={ev_t:.4e} deinv={di_t} nit={nit_t}", flush=True)

            row.update(dict(eid_c=eid_c, ep_c=ep_c, ev_c=ev_c,
                            eid_t=eid_t, ep_t=ep_t, ev_t=ev_t,
                            id_ratio=eid_t / max(eid_c, 1e-12),
                            p_ratio=ep_t / max(ep_c, 1e-12),
                            v_ratio=ev_t / max(ev_c, 1e-12)))
            print(f"    -> id_ratio(T/C)={row['id_ratio']:.3f} p_ratio={row['p_ratio']:.3f} "
                  f"v_ratio={row['v_ratio']:.3f}", flush=True)
            results.append(row)

    # aggregate per A_d
    print("\n===== SUMMARY (median over seeds) =====", flush=True)
    for A_d in A_DS:
        rs = [r for r in results if r["A_d"] == A_d]
        med = lambda k: float(np.median([r[k] for r in rs]))
        print(f"A_d={A_d}: eid_c={med('eid_c'):.4e} eid_t={med('eid_t'):.4e} "
              f"id_ratio={med('id_ratio'):.3f} | p_ratio={med('p_ratio'):.3f} v_ratio={med('v_ratio'):.3f}",
              flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
