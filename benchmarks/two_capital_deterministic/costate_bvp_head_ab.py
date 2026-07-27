"""
A/B test of the BOUNDARY-ANCHORED COSTATE BVP HEAD (method key:
boundary-anchored-costate-bvp-head).

IDEA (route around the flat-in-v' residual):
  The pointwise HJB residual is nearly flat in v' in the de-invest pocket (weak
  identification). But the FOC / envelope structure pins v' through the feasible
  multiplier q_d (controls are well identified). So we PARAMETERIZE q_d(Z)>0 with
  a softplus head, BACK OUT the slope exactly from the envelope inverse
      v'(Z) = (1 - q_d(Z)) / Z ,   q_g = 1 + (1-Z) v' ,
  and DEFINE the value by integrating that slope on a fixed sorted collocation
  grid (cumulative trapezoid), anchored at the LEFT closed-form corner v(0)=v0.
  The RIGHT corner v(1)=vN is enforced as a HARD penalty.  This turns the problem
  into a two-corner-anchored BVP -- exactly the structure solve_fd exploits --
  rather than pointwise least squares over an under-determined v'.

  Corner-blend on q_d controls the 1/Z amplification near Z->0:
      q_d(Z) = (1-Z)*qd0 + Z*qd1 + Z(1-Z)*softplus(net(2Z-1))
  with qd0,qd1 chosen so v'(Z) stays finite at the corners (qd0->1 as Z->0).

CONTROL  = single value-NN, pure L2(strong residual R), same seed/init/budget
           (mirrors closed_form_control_deinvest_diag / envelope_anchor CONTROL).
TREATED  = the q_d back-out costate head + cumulative-integral value + hard
           right-endpoint anchor.  Loss = mean(R^2) + w_end*(v(1)-vN)^2.

TRUE ERROR (the ONLY metric): max|i_d - i_d_FD| over interior Z in [0.1,0.9],
max|v'-fd slope|, max|v-v_fd|.  Both regimes: A_d=0.05 (weak-id de-invest pocket,
the decisive test) and A_d=0.13 (well conditioned -- guard against regression).

Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python costate_bvp_head_ab.py   (login node OK)
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
W_ENDS = [1.0, 10.0, 100.0]   # sweep the hard right-endpoint anchor weight

# Fixed SORTED collocation grid (interior, avoids the Z=0,1 singular corners).
Znp = np.linspace(0.02, 0.98, N_COLLO).astype(np.float32)
Zt = tf.constant(Znp.reshape(-1, 1))
dZ = np.diff(Znp).astype(np.float32)           # spacing for trapezoid
dZt = tf.constant(dZ.reshape(-1, 1))


def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    v_fd = np.interp(Znp, fd["Z"], fd["v"]).astype(np.float32)
    sl_fd = np.interp(Znp, fd["Z"], fd["slope"]).astype(np.float32)
    id_fd = np.interp(Znp, fd["Z"], fd["i_d"]).astype(np.float32)
    v0, vN = M.boundary_values(P)
    # corner q_d so the backed-out v'=(1-q_d)/Z stays finite at the corners.
    # qd0 chosen so v'(Z->0) -> sl0 (closed-form one-capital FOC slope):
    #   q_d = 1 - Z v'  => qd0 = 1 (then v'(0) finite, anchored by interior net).
    # qd1 chosen so v'(Z->1) -> sl1: q_d(1) = 1 - 1*sl1 = 1 - sl1.
    sl0 = float(M.perturbation_slope(np.array([0.0]), P)[0])
    sl1 = float(M.perturbation_slope(np.array([1.0]), P)[0])
    qd0 = 1.0                       # keeps v'(0)=(1-qd0)/0 -> 0/0 finite
    qd1 = 1.0 - sl1                 # v'(1) = (1-qd1)/1 = sl1
    return P, fd, v_fd, sl_fd, id_fd, v0, vN, qd0, qd1


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m,
                            1.0 / tf.maximum(Z, 1e-9) - m)


# ----------------------- CONTROL: plain value-NN, L2(R) ----------------------
def make_control(net, P, v0, vN):
    dl = P["delta"]; A_d = P["A_d"]; A_g = P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]

    def value(Z):
        return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)

    def pieces():
        with tf.GradientTape() as t1:
            t1.watch(Zt); v = value(Zt)
        vp = clamp(t1.gradient(v, Zt), Zt)
        q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
        Abar = (1 - Zt) * A_d + Zt * A_g
        c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
        i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
        phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
        phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
        mu = Zt * (1 - Zt) * (phi_g - phi_d)
        R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
        return R, vp, v, i_d

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            R, vp, v, _ = pieces()
            loss = tf.reduce_mean(tf.square(R))
        return loss, outer.gradient(loss, net.trainable_variables)

    def eval_fields():
        R, vp, v, i_d = pieces()
        return (R.numpy().ravel(), vp.numpy().ravel(),
                v.numpy().ravel(), i_d.numpy().ravel())
    return lg, eval_fields


# ------------- TREATED: q_d back-out costate head + BVP integration ----------
def make_treated(net, P, v0, vN, qd0, qd1, w_end):
    dl = P["delta"]; A_d = P["A_d"]; A_g = P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]

    def fields():
        raw = net(2.0 * Zt - 1.0)                          # head output
        # corner-blended feasible multiplier q_d(Z) > 0
        q_d = (1 - Zt) * qd0 + Zt * qd1 + Zt * (1 - Zt) * tf.nn.softplus(raw)
        q_d = tf.maximum(q_d, 1e-4)                        # strictly feasible
        vp = (1.0 - q_d) / tf.maximum(Zt, 1e-6)            # exact envelope inverse
        vp = clamp(vp, Zt)
        q_g = 1 + (1 - Zt) * vp
        Abar = (1 - Zt) * A_d + Zt * A_g
        c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
        i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
        phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
        phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
        mu = Zt * (1 - Zt) * (phi_g - phi_d)
        # value by cumulative trapezoid of the backed-out slope, anchored v(z0)~v0.
        vpf = tf.reshape(vp, [-1])
        seg = 0.5 * (vpf[1:] + vpf[:-1]) * tf.reshape(dZt, [-1])      # trapezoid segs
        v_int = v0 + tf.concat([[0.0], tf.cumsum(seg)], axis=0)       # v(0.02)=v0
        v_int = tf.reshape(v_int, [-1, 1])
        R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v_int + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
        v_right = v_int[-1, 0]                                        # v(0.98)
        return R, vp, v_int, i_d, v_right

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            R, vp, v_int, _, v_right = fields()
            # right-corner target: vN is v(1); add the analytic last sliver
            # v(1) ~= v(0.98) + 0.02*v'(~1) ; use vp at last node for the sliver.
            v1 = v_right + 0.02 * vp[-1, 0]
            loss = tf.reduce_mean(tf.square(R)) + w_end * tf.square(v1 - vN)
        return loss, outer.gradient(loss, net.trainable_variables)

    def eval_fields():
        R, vp, v_int, i_d, _ = fields()
        return (R.numpy().ravel(), vp.numpy().ravel(),
                v_int.numpy().ravel(), i_d.numpy().ravel())
    return lg, eval_fields


# ------------------------------- optimizers ----------------------------------
def adam(net, lg, steps, lr):
    opt = tf.keras.optimizers.Adam(lr); L = None
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


def metrics(eval_fields, v_fd, sl_fd, id_fd):
    mI = (Znp >= 0.1) & (Znp <= 0.9)
    R, vp, v, i_d = eval_fields()
    err_id = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    err_sl = float(np.max(np.abs(vp[mI] - sl_fd[mI])))
    err_v = float(np.max(np.abs(v - v_fd)))
    vp_half = float(np.interp(0.5, Znp, vp))
    return err_id, err_sl, err_v, vp_half, float(np.max(np.abs(R)))


def main():
    results = []
    for A_d in A_DS:
        P, fd, v_fd, sl_fd, id_fd, v0, vN, qd0, qd1 = build_case(A_d)
        mI = (Znp >= 0.1) & (Znp <= 0.9)
        sl_half_fd = float(np.interp(0.5, Znp, sl_fd))
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd[mI].min():+.4f},{id_fd[mI].max():+.4f}] "
              f"deinvest={bool((id_fd[mI]<0).any())}  v'(0.5)_FD={sl_half_fd:.4f}  "
              f"qd0={qd0:.3f} qd1={qd1:.3f}", flush=True)

        # ---- CONTROL (seed-averaged), independent of w_end ----
        ctrl_rows = []
        for seed in SEEDS:
            tf.random.set_seed(seed); np.random.seed(seed)
            net = make_net()
            lg, ev = make_control(net, P, v0, vN)
            adam(net, lg, N_ADAM, ADAM_LR); nit = lbfgs(net, lg)
            eid, esl, ev_, vph, Rinf = metrics(ev, v_fd, sl_fd, id_fd)
            ctrl_rows.append((eid, esl, ev_, vph, Rinf))
            print(f"  [CTRL] seed={seed}: max|i_d-FD|={eid:.4e} max|v'-FD|={esl:.4e} "
                  f"max|v-FD|={ev_:.4e} v'(0.5)={vph:.4f} ||R||inf={Rinf:.2e}", flush=True)
        cm = np.mean(np.array(ctrl_rows), axis=0)
        print(f"  [CTRL mean] max|i_d-FD|={cm[0]:.4e} max|v'-FD|={cm[1]:.4e} "
              f"max|v-FD|={cm[2]:.4e} v'(0.5)={cm[3]:.4f}", flush=True)

        # ---- TREATED, sweeping w_end ----
        for w_end in W_ENDS:
            tr_rows = []
            for seed in SEEDS:
                tf.random.set_seed(seed); np.random.seed(seed)
                net = make_net()
                lg, ev = make_treated(net, P, v0, vN, qd0, qd1, w_end)
                adam(net, lg, N_ADAM, ADAM_LR); nit = lbfgs(net, lg)
                eid, esl, ev_, vph, Rinf = metrics(ev, v_fd, sl_fd, id_fd)
                tr_rows.append((eid, esl, ev_, vph, Rinf))
                print(f"  [TREAT w={w_end:g}] seed={seed}: max|i_d-FD|={eid:.4e} "
                      f"max|v'-FD|={esl:.4e} max|v-FD|={ev_:.4e} v'(0.5)={vph:.4f} "
                      f"||R||inf={Rinf:.2e}", flush=True)
            tm = np.mean(np.array(tr_rows), axis=0)
            id_ratio = tm[0] / max(cm[0], 1e-12)
            sl_ratio = tm[1] / max(cm[1], 1e-12)
            print(f"  [TREAT mean w={w_end:g}] max|i_d-FD|={tm[0]:.4e} max|v'-FD|={tm[1]:.4e} "
                  f"max|v-FD|={tm[2]:.4e} v'(0.5)={tm[3]:.4f}  "
                  f"=> id_ratio(T/C)={id_ratio:.3f} sl_ratio={sl_ratio:.3f}", flush=True)
            results.append({"A_d": A_d, "w_end": w_end,
                            "ctrl_id": float(cm[0]), "ctrl_sl": float(cm[1]),
                            "ctrl_v": float(cm[2]), "ctrl_vph": float(cm[3]),
                            "treat_id": float(tm[0]), "treat_sl": float(tm[1]),
                            "treat_v": float(tm[2]), "treat_vph": float(tm[3]),
                            "id_ratio": float(id_ratio), "sl_ratio": float(sl_ratio),
                            "fd_vph": sl_half_fd})
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
