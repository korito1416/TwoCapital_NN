"""
A/B test of the HJI envelope/anchor method (method key: envelope-bounds-and-certificate).

CONTROL  = pure L2(strong-residual) value-NN (Adam + tight L-BFGS), as in the
           existing benchmark scripts.
TREATED  = same seed / init / budget, PLUS analytic-anchor penalties that inject
           EXTERNAL information the flat-in-v' residual lacks:
             (a1) above-chord hinge:   v(Z) >= (1-Z) v0 + Z vN   <=>  net(.)>=0
                  (verified true on the FD truth for both A_d cases).
             (a2) concavity hinge:     v'' <= 0  (penalize v''>0).  FD v is concave.
             (a3) corner-slope anchors: v'(Z) -> closed-form one-capital FOC slope
                  as Z->0,1, matched on a thin boundary collocation band.
           These are HARD analytic facts (corners are closed form; the sign
           structure is exact), so they pin v (hence v') beyond the flat residual.

CERTIFICATE (b): after training, form the discounted L-infinity contraction bound
           ||v_nn - v_true||_inf <= ||R||_inf / delta  and check it BRACKETS the
           actual max|v_nn - v_fd| (validity) and report the slack ratio. delta is
           small (~0.01) so the bound is loose (~100x) but must be valid.

TRUE ERROR (the ONLY metric, never the loss number, since the norm differs):
   max|i_d - i_d_FD| over the interior Z in [0.1,0.9], and max|v_nn - v_fd|.

Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python envelope_anchor_ab.py
(login node OK; it is small.)
"""
import json
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd

tf.keras.backend.set_floatx("float32")

DELTA_CERT = None  # filled per-P (the model delta)
SEEDS = [1, 2, 3]
A_DS = [0.05, 0.13]   # hard de-invest pocket + easy case
N_ADAM = 6000
ADAM_LR = 2e-3
N_COLLO = 256

# anchor weights (treated only)
W_CHORD = 1.0e-1     # above-chord hinge  net>=0
W_CONCAVE = 1.0e-2   # concavity hinge   v''<=0
W_SLOPE = 5.0e-2     # corner-slope anchor on boundary band


def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zg = np.linspace(0.02, 0.98, N_COLLO).reshape(-1, 1).astype(np.float32)
    v_fd = np.interp(Zg.ravel(), fd["Z"], fd["v"]).astype(np.float32)
    id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])
    v0, vN = M.boundary_values(P)
    # closed-form corner slopes from the one-capital FOC (perturbation_slope is the
    # leading-order v'(Z); exact at the corners up to the heterogeneity expansion).
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


def make_lossfns(net, P, Zt, v0, vN, sl0, sl1, treated):
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
        vpp = t2.gradient(vp_raw, Zt)          # second derivative (for concavity)
        vp = clamp(vp_raw, Zt)
        q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
        Abar = (1 - Zt) * A_d + Zt * A_g
        c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
        i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
        phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
        phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
        mu = Zt * (1 - Zt) * (phi_g - phi_d)
        R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
        # net(.) value = (v - chord)/(Z(1-Z)) ; above-chord <=> v>=chord
        chord = (1 - Zt) * v0 + Zt * vN
        above = v - chord                      # >=0 required (FD-verified)
        return R, vp, vpp, above, i_d

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            R, vp, vpp, above, _ = residual_and_anchor()
            loss = tf.reduce_mean(tf.square(R))
            if treated:
                # (a1) above-chord hinge: penalize v < chord  (net<0)
                loss += W_CHORD * tf.reduce_mean(tf.square(tf.nn.relu(-above)))
                # (a2) concavity hinge: penalize v'' > 0
                loss += W_CONCAVE * tf.reduce_mean(tf.square(tf.nn.relu(vpp)))
                # (a3) corner-slope anchor on the boundary band (Z<0.1 -> sl0, Z>0.9 -> sl1)
                wlo = tf.cast(Zt < 0.1, tf.float32)
                whi = tf.cast(Zt > 0.9, tf.float32)
                denom = tf.maximum(tf.reduce_sum(wlo) + tf.reduce_sum(whi), 1.0)
                anch = (tf.reduce_sum(wlo * tf.square(vp - sl0))
                        + tf.reduce_sum(whi * tf.square(vp - sl1))) / denom
                loss += W_SLOPE * anch
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


def evaluate(value, residual_and_anchor, Zg, P, v_fd, id_fd, delta):
    mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
    R, vp, vpp, above, i_d = residual_and_anchor()
    v_nn = value(tf.constant(Zg)).numpy().ravel()
    i_d = i_d.numpy().ravel()
    err_id = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    err_v = float(np.max(np.abs(v_nn - v_fd)))
    Rinf = float(np.max(np.abs(R.numpy())))
    cert = Rinf / delta
    return err_id, err_v, Rinf, cert


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zg, v_fd, id_fd, v0, vN, sl0, sl1 = build_case(A_d)
        Zt = tf.constant(Zg)
        delta = P["delta"]
        mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd[mI].min():+.4f},{id_fd[mI].max():+.4f}] "
              f"deinvest={bool((id_fd[mI]<0).any())}  corner-slopes sl0={sl0:.3f} sl1={sl1:.3f}", flush=True)
        for seed in SEEDS:
            row = {"A_d": A_d, "seed": seed}
            for treated in (False, True):
                tf.random.set_seed(seed); np.random.seed(seed)
                net = make_net()
                lg, value, ranch = make_lossfns(net, P, Zt, v0, vN, sl0, sl1, treated)
                adam(net, lg, N_ADAM, ADAM_LR)
                nit = lbfgs(net, lg)
                err_id, err_v, Rinf, cert = evaluate(value, ranch, Zg, P, v_fd, id_fd, delta)
                tag = "TREATED" if treated else "CONTROL"
                cert_valid = cert >= err_v
                slack = cert / max(err_v, 1e-12)
                print(f"  seed={seed} {tag}: max|i_d-FD|={err_id:.4e}  max|v-FD|={err_v:.4e}  "
                      f"||R||inf={Rinf:.3e}  cert={cert:.3e} valid={cert_valid} slack={slack:.1f}x nit={nit}",
                      flush=True)
                key = "treat" if treated else "ctrl"
                row[f"err_id_{key}"] = err_id
                row[f"err_v_{key}"] = err_v
                row[f"Rinf_{key}"] = Rinf
                row[f"cert_{key}"] = cert
                row[f"cert_valid_{key}"] = bool(cert_valid)
                row[f"slack_{key}"] = slack
            row["id_ratio"] = row["err_id_treat"] / max(row["err_id_ctrl"], 1e-12)
            row["v_ratio"] = row["err_v_treat"] / max(row["err_v_ctrl"], 1e-12)
            print(f"    -> id_ratio(T/C)={row['id_ratio']:.3f}  v_ratio(T/C)={row['v_ratio']:.3f}", flush=True)
            results.append(row)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
