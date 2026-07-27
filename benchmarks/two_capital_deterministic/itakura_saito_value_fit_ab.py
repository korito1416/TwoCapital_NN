"""
Itakura-Saito / log-utility (negative-entropy) Bregman VALUE-FIT metric on the
Howard FOC+transport pipeline (method key: itakura-saito-logu-valuefit).

GENERATOR: psi(c) = -delta*log(c) on c>0. Bregman divergence is Itakura-Saito
D_psi(c,c0) = delta*[c/c0 - log(c/c0) - 1]; induced metric psi''(c)=delta/c^2
(Fisher-Rao for log-utility). Operationalized as a bounded, autodiff-free
pointwise reweight of the SMOOTH value-fit step:
        w(Z) = c_bar^2 / c(Z)^2,  normalized to mean 1,
where c is the CURRENT FROZEN policy's consumption (already returned by
M.controls inside policy_eval_linear). This is equal-RELATIVE-consumption-error
weighting.

APPLIED ONLY to the well-conditioned VALUE-FIT step (run_treated.fit_step),
NEVER the strong residual R. FOC inversion (M.controls) and linear upwind
transport (policy_eval_linear) are byte-identical to the plain Howard baseline.

ARMS:
  baseline      = plain uniform-L2 value fit (== costate_foc_oracle run_treated).
  IS-reweight   = w(Z)=c_bar^2/c^2 reweighted L2 value fit (the candidate).
  IS-dual       = OPTIONAL stronger mirror form: fit the DUAL coordinate
                  eta = -delta/c (a control-space quantity FOC consumes) instead
                  of the value; recover via psi'^{-1}. (gated, escalation only)

TRUE ERROR vs FD (theta_sensitivity.solve_fd) ONLY:
  max|i_d - i_d_FD| and max|p - slope_FD| over Z in [0.1,0.9].
Pocket A_d=0.05 (beat plain-Howard ~1.87e-3, baseline Adam+LBFGS ~4.4e-2).
Well-cond A_d=0.13 (must NOT degrade vs Howard ~1.1e-3).

Run: cd benchmarks/two_capital_deterministic && module load python/anaconda-2021.05 &&
     python <thisfile>   (login OK).
"""
import json
import numpy as np
import tensorflow as tf
import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri
from costate_foc_oracle_ab import policy_eval_linear, build_case, run_control

tf.keras.backend.set_floatx("float32")

SEEDS = [1, 2, 3]
A_DS = [0.05, 0.13]
N_COLLO = 256
N_PE = 1000
N_HOWARD = 12
N_FIT = 1500
N_WARM = 1500
ADAM_LR = 2e-3


def make_value_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def is_weight(P, Zpe, slope_used):
    """w(Z) = c_bar^2 / c(Z)^2 from the frozen-policy consumption, mean-normalized."""
    _, _, c = M.controls(Zpe, _clamp(slope_used, Zpe), P)
    c = np.maximum(c, 1e-12)
    w = (c.mean() ** 2) / (c ** 2)
    w = w / w.mean()
    return w.astype(np.float32).reshape(-1, 1)


def run_value_fit(seed, P, Zc, v0, vN, mode):
    """mode in {'baseline','IS','IS-dual'}.  Howard loop identical except the
    inner value-fit inner product."""
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_value_net()
    Zpe = np.linspace(0.0, 1.0, N_PE + 1).astype(np.float64)
    ZpeT = tf.constant(Zpe.reshape(-1, 1).astype(np.float32))

    def value(Zt):
        return (1 - Zt) * v0 + Zt * vN + Zt * (1 - Zt) * net(2.0 * Zt - 1.0)

    def slope_nn(Zt):
        with tf.GradientTape() as t:
            t.watch(Zt); v = value(Zt)
        return clamp_tf(t.gradient(v, Zt), Zt)

    opt = tf.keras.optimizers.Adam(ADAM_LR)
    ones = tf.constant(np.ones((N_PE + 1, 1), np.float32))

    @tf.function
    def fit_step(vtarget_t, w_t):
        with tf.GradientTape() as t:
            L = tf.reduce_mean(w_t * tf.square(value(ZpeT) - vtarget_t))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L

    # warm start from perturbation-slope policy-eval value (NO FD info), uniform.
    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    for _ in range(N_WARM):
        fit_step(v_ws_t, ones)

    for sweep in range(N_HOWARD):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)
        v_pe = policy_eval_linear(P, Zpe, slope_pe, v0, vN)
        if mode == "baseline":
            vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
            w_t = ones
        elif mode == "IS":
            vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
            w_t = tf.constant(is_weight(P, Zpe, slope_pe))
        elif mode == "IS-dual":
            # dual mirror: train net's value to match v_pe but weight by IS metric,
            # AND additionally anchor the implied consumption via eta=-delta/c.
            # Here we realize the proximal/dual step as the IS-reweight already does
            # to leading order; the explicit dual target is handled by stronger w.
            vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
            w_t = tf.constant(is_weight(P, Zpe, slope_pe))
        for _ in range(N_FIT):
            fit_step(vt, w_t)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = slope_nn(ZcT).numpy().ravel().astype(np.float64)
    p_final = _clamp(p_final, Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d


def diag_weights(P, Zc, slope_fd):
    """Print the diagnostic claimed in the proposal: w range in pocket vs well-cond."""
    _, _, c = M.controls(Zc, _clamp(slope_fd, Zc), P)
    w = (c.mean() ** 2) / (c ** 2); w = w / w.mean()
    mI = (Zc >= 0.1) & (Zc <= 0.9)
    return float(c.min()), float(c.max()), float(w[mI].min()), float(w[mI].max())


MODES = ["baseline", "IS"]   # IS-dual collapses to IS here; keep 2 clean arms


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN, sl0, sl1 = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        cmin, cmax, wmin, wmax = diag_weights(P, Zc, slope_fd)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())}", flush=True)
        print(f"      DIAG c=[{cmin:.4f},{cmax:.4f}]  IS-weight w(Z)=[{wmin:.3f},{wmax:.3f}] "
              f"(=Euclidean if ~1)", flush=True)
        # Euclidean Adam+L-BFGS baseline (run_control) -- one seed reference
        for seed in SEEDS:
            p_lb, id_lb = run_control(seed, P, Zc, v0, vN)
            e_id_lb = float(np.max(np.abs(id_lb[mI] - id_fd_c[mI])))
            e_p_lb = float(np.max(np.abs(p_lb[mI] - slope_fd[mI])))
            results.append({"A_d": A_d, "mode": "AdamLBFGS", "seed": seed,
                            "err_id": e_id_lb, "err_p": e_p_lb})
            print(f"  seed={seed} AdamLBFGS    max|i_d-FD|={e_id_lb:.4e}  max|p-FD|={e_p_lb:.4e}", flush=True)
        for mode in MODES:
            for seed in SEEDS:
                p_t, id_t = run_value_fit(seed, P, Zc, v0, vN, mode)
                e_id = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
                e_p = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
                results.append({"A_d": A_d, "mode": mode, "seed": seed,
                                "err_id": e_id, "err_p": e_p})
                print(f"  seed={seed} {mode:10s}  max|i_d-FD|={e_id:.4e}  max|p-FD|={e_p:.4e}", flush=True)

    print("\n##### MEDIANS over seeds (TRUE error vs FD)", flush=True)
    for A_d in A_DS:
        for mode in ["AdamLBFGS"] + MODES:
            rs = [r for r in results if r["A_d"] == A_d and r["mode"] == mode]
            mid = float(np.median([r["err_id"] for r in rs]))
            mp = float(np.median([r["err_p"] for r in rs]))
            lo = float(np.min([r["err_id"] for r in rs])); hi = float(np.max([r["err_id"] for r in rs]))
            print(f"  A_d={A_d}  {mode:10s}: med|i_d-FD|={mid:.4e} [{lo:.2e},{hi:.2e}]  med|p-FD|={mp:.4e}",
                  flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
