"""
A/B test of HOWARD policy-time-iteration NN (continuous-time EGM / PINN policy iteration).
Method key: howard-policy-time-iteration-nn.

HYPOTHESIS: the v'-INFORMATION lives in the FOC (dFOC/dv' is LARGE -> controls
well-identified), NOT in the strong residual (mu_Z = dR/dv' is 20-220x SMALLER ->
residual flat in v' in the de-invest pocket). So instead of fitting v' from the flat
strong residual, READ v' through the well-conditioned FOC inversion and RE-DERIVE v'
as the gradient of a well-posed LINEAR policy-evaluation solve -- exactly the machinery
solve_fd uses.

CONTROL  = pure L2(strong nonlinear residual) single value-NN, Adam + tight L-BFGS
           (identical to the existing benchmark baseline).
TREATED  = Howard outer loop, K sweeps, warm-started:
   (1) POLICY step: vp_k = clamp(autodiff(value)); i_d,i_g,c = M.controls(Z,vp_k,P);
       FREEZE these (tf.stop_gradient) -- they parametrise the operator this sweep.
   (2) POLICY-EVALUATION step: with the frozen policy the HJB is LINEAR in v:
         R_lin = delta*log(c_froz) - delta*v + (1-Z)phi_d_froz + Z*phi_g_froz
                 + mu_froz*autodiff(value)
       Train net_v to min mean(R_lin^2) by short Adam + tight L-BFGS (ftol=1e-15).
       v IS being re-fit, but only against a LINEAR, well-posed residual whose
       v'-coupling (mu_froz*v') is the SAME machinery as the FD upwind transport.
   (3) re-read vp_{k+1} = clamp(autodiff(value)); repeat.
   Controls reported each sweep from M.controls(Z, vp). v' is NEVER fit from the
   strong nonlinear residual.

Matched budget: CONTROL gets N_ADAM Adam + one L-BFGS(maxiter). TREATED splits the
SAME total Adam budget across K sweeps and the SAME total L-BFGS budget across sweeps.

TRUE ERROR (only metric): max|i_d - i_d_FD| over Z in [0.1,0.9], and max|v'-slope_FD|,
max|v-v_FD|. Reported per regime. PASS = id_ratio(T/C)<1 in the A_d=0.05 pocket AND
id_ratio(T/C)<=1 (no harm) at A_d=0.13.

Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python howard_policy_iter_ab.py   (login node OK)
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
N_COLLO = 256

# Total matched budgets (shared by CONTROL and TREATED):
TOTAL_ADAM = 6000
TOTAL_LBFGS = 4000
ADAM_LR = 2e-3
K_SWEEPS = 12           # Howard outer sweeps


def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zg = np.linspace(0.02, 0.98, N_COLLO).reshape(-1, 1).astype(np.float32)
    v_fd = np.interp(Zg.ravel(), fd["Z"], fd["v"]).astype(np.float32)
    id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])
    slope_fd = np.interp(Zg.ravel(), fd["Z"], fd["slope"]).astype(np.float32)
    v0, vN = M.boundary_values(P)
    return P, fd, Zg, v_fd, id_fd, slope_fd, v0, vN


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m,
                            1.0 / tf.maximum(Z, 1e-9) - m)


def value_fn(net, Z, v0, vN):
    return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)


# ------------------------------------------------------------------ controls
def vp_controls(net, Zt, P, v0, vN):
    """Return (v, vp_clamped, i_d, i_g, c, phi_d, phi_g, mu) all as tf tensors."""
    dl, A_d, A_g = P["delta"], P["A_d"], P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]
    with tf.GradientTape() as t1:
        t1.watch(Zt); v = value_fn(net, Zt, v0, vN)
    vp = clamp(t1.gradient(v, Zt), Zt)
    q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
    Abar = (1 - Zt) * A_d + Zt * A_g
    c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
    i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
    phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
    phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
    mu = Zt * (1 - Zt) * (phi_g - phi_d)
    return v, vp, i_d, i_g, c, phi_d, phi_g, mu


# ------------------------------------------------------------------ CONTROL loss
def make_strong_lg(net, P, Zt, v0, vN):
    dl = P["delta"]
    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            v, vp, i_d, i_g, c, phi_d, phi_g, mu = vp_controls(net, Zt, P, v0, vN)
            R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
            loss = tf.reduce_mean(tf.square(R))
        return loss, outer.gradient(loss, net.trainable_variables)
    return lg


# ------------------------------------------------------------------ TREATED policy-eval loss
def make_polyeval_lg(net, P, Zt, v0, vN, c_froz, phi_d_froz, phi_g_froz, mu_froz):
    """LINEAR-in-v policy-evaluation residual with FROZEN policy.
    flow_froz = delta*log(c_froz) + (1-Z)phi_d_froz + Z phi_g_froz are constants;
    only v (= value(net)) and vp (= autodiff value) vary; coupling mu_froz*vp is
    linear in the network output's gradient (well-posed transport)."""
    dl = P["delta"]
    flow_froz = dl * tf.math.log(tf.maximum(c_froz, 1e-8)) + (1 - Zt) * phi_d_froz + Zt * phi_g_froz
    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            with tf.GradientTape() as t1:
                t1.watch(Zt); v = value_fn(net, Zt, v0, vN)
            vp = t1.gradient(v, Zt)           # raw gradient (linear in params via net)
            R_lin = flow_froz - dl * v + mu_froz * vp
            loss = tf.reduce_mean(tf.square(R_lin))
        return loss, outer.gradient(loss, net.trainable_variables)
    return lg


# ------------------------------------------------------------------ optimizers
def adam(net, lg, steps, lr=ADAM_LR):
    opt = tf.keras.optimizers.Adam(lr)
    L = None
    for _ in range(steps):
        L, g = lg(); opt.apply_gradients(zip(g, net.trainable_variables))
    return float(L) if L is not None else 0.0


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
                 options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-15, "gtol": 1e-12})
    setf(r.x); return r.nit


# ------------------------------------------------------------------ evaluate
def evaluate(net, Zt, Zg, P, v0, vN, v_fd, id_fd, slope_fd):
    mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
    v, vp, i_d, i_g, c, phi_d, phi_g, mu = vp_controls(net, Zt, P, v0, vN)
    v = v.numpy().ravel(); vp = vp.numpy().ravel(); i_d = i_d.numpy().ravel()
    err_id = float(np.max(np.abs(i_d[mI] - id_fd[mI])))
    err_v = float(np.max(np.abs(v - v_fd)))
    err_slope = float(np.max(np.abs(vp[mI] - slope_fd[mI])))
    return err_id, err_v, err_slope


def run_control(net, P, Zt, v0, vN, Zg, v_fd, id_fd, slope_fd):
    lg = make_strong_lg(net, P, Zt, v0, vN)
    adam(net, lg, TOTAL_ADAM)
    nit = lbfgs(net, lg, TOTAL_LBFGS)
    return evaluate(net, Zt, Zg, P, v0, vN, v_fd, id_fd, slope_fd) + (nit,)


def run_howard(net, P, Zt, v0, vN, Zg, v_fd, id_fd, slope_fd):
    adam_per = max(TOTAL_ADAM // K_SWEEPS, 1)
    lbfgs_per = max(TOTAL_LBFGS // K_SWEEPS, 1)
    total_nit = 0
    sweep_errs = []
    for k in range(K_SWEEPS):
        # (1) POLICY step: read v', compute & FREEZE the policy
        _, vp, i_d, i_g, c, phi_d, phi_g, mu = vp_controls(net, Zt, P, v0, vN)
        c_f = tf.stop_gradient(c); pd_f = tf.stop_gradient(phi_d)
        pg_f = tf.stop_gradient(phi_g); mu_f = tf.stop_gradient(mu)
        # (2) POLICY-EVALUATION: train net on the LINEAR residual with frozen policy
        lg = make_polyeval_lg(net, P, Zt, v0, vN, c_f, pd_f, pg_f, mu_f)
        adam(net, lg, adam_per)
        total_nit += lbfgs(net, lg, lbfgs_per)
        # (3) re-read controls for logging
        e = evaluate(net, Zt, Zg, P, v0, vN, v_fd, id_fd, slope_fd)
        sweep_errs.append(e[0])
    return evaluate(net, Zt, Zg, P, v0, vN, v_fd, id_fd, slope_fd) + (total_nit, sweep_errs)


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zg, v_fd, id_fd, slope_fd, v0, vN = build_case(A_d)
        Zt = tf.constant(Zg)
        mI = (Zg.ravel() >= 0.1) & (Zg.ravel() <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd[mI].min():+.4f},{id_fd[mI].max():+.4f}] "
              f"deinvest={bool((id_fd[mI]<0).any())}  FD resid={fd['max_abs_residual']:.1e}", flush=True)
        for seed in SEEDS:
            row = {"A_d": A_d, "seed": seed}
            # CONTROL
            tf.random.set_seed(seed); np.random.seed(seed)
            net_c = make_net()
            ec_id, ec_v, ec_sl, ec_nit = run_control(net_c, P, Zt, v0, vN, Zg, v_fd, id_fd, slope_fd)
            print(f"  seed={seed} CONTROL: max|i_d-FD|={ec_id:.4e}  max|v'-FD|={ec_sl:.4e}  "
                  f"max|v-FD|={ec_v:.4e}  nit={ec_nit}", flush=True)
            # TREATED (same seed/init)
            tf.random.set_seed(seed); np.random.seed(seed)
            net_t = make_net()
            et_id, et_v, et_sl, et_nit, sweep_errs = run_howard(net_t, P, Zt, v0, vN, Zg, v_fd, id_fd, slope_fd)
            print(f"  seed={seed} TREATED: max|i_d-FD|={et_id:.4e}  max|v'-FD|={et_sl:.4e}  "
                  f"max|v-FD|={et_v:.4e}  nit={et_nit}", flush=True)
            print(f"           sweep err_id trace: " +
                  " ".join(f"{x:.3e}" for x in sweep_errs), flush=True)
            row.update(err_id_ctrl=ec_id, err_v_ctrl=ec_v, err_sl_ctrl=ec_sl,
                       err_id_treat=et_id, err_v_treat=et_v, err_sl_treat=et_sl,
                       id_ratio=et_id / max(ec_id, 1e-12), v_ratio=et_v / max(ec_v, 1e-12),
                       sl_ratio=et_sl / max(ec_sl, 1e-12))
            print(f"    -> id_ratio(T/C)={row['id_ratio']:.3f}  v'_ratio(T/C)={row['sl_ratio']:.3f}  "
                  f"v_ratio(T/C)={row['v_ratio']:.3f}", flush=True)
            results.append(row)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
