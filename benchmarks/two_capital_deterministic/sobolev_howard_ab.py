"""
Sobolev-anchored Howard fit (method key: sobolev-anchored-howard-fit).

COMBINE: keep the Howard FOC+transport routing of costate_foc_oracle_ab.py
(boundary-ansatz value net; slope via autodiff; M.controls FOC inversion;
LINEAR upwind policy-evaluation solve; 12 Howard sweeps; perturbation warm start),
and CHANGE ONLY the supervised fit_step.

DIAGNOSED FLOOR: i_d is read by AUTODIFF of a value net supervised ONLY on v.
The value error is ~1e-3 in the pocket but the AUTODIFF-SLOPE error is ~1e-1.
Add an H1/Sobolev derivative-matching term so the net fits the quantity the FOC
actually consumes (v'):

    L = mean((value - v_pe)^2) + lambda * mean((autodiff_v'(value) - p_pe)^2)

where p_pe = slope_from_value(Zpe, v_pe) is the well-conditioned oracle slope
(neighbor FD of the transport-solved value). autodiff_v'(value) is computed with
a nested GradientTape inside fit_step.

ARMS in this harness:
  BASELINE  = plain-Howard (value-only fit)  == run_treated of the original script.
  TREATED   = Sobolev fit, lambda in {0.1, 1, 10}, plus an inverse-Dirichlet
              auto-weight variant (balance value vs derivative gradient norms).
  Corners anchored to perturbation_slope where autodiff v' is noisiest (mu->0).

TRUE ERROR vs FD (the ONLY metric): max|i_d-i_d_FD| and max|p-slope_FD| over
Z in [0.1,0.9], per regime. Pocket = A_d=0.05 (beat plain-Howard ~1.5e-3 toward
1e-4). Well-cond = A_d=0.13 (must NOT degrade; id_ratio(T/plain) <= 1.2).

Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python sobolev_howard_ab.py   (login OK).
"""
import json
import numpy as np
import tensorflow as tf
import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri

tf.keras.backend.set_floatx("float32")

SEEDS = [1, 2, 3, 4, 5, 6]
A_DS = [0.05, 0.13]
N_COLLO = 256
N_PE = 1000
N_HOWARD = 12
N_FIT = 1500
ADAM_LR = 2e-3
# lambda sweep + an inverse-Dirichlet auto-weight arm (lambda="auto")
LAMBDAS = [0.0, 0.1, 1.0, 10.0, "auto"]   # 0.0 == plain-Howard baseline


def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zc = np.linspace(0.02, 0.98, N_COLLO).astype(np.float64)
    slope_fd = np.interp(Zc, fd["Z"], fd["slope"])
    id_fd_c = np.interp(Zc, fd["Z"], fd["i_d"])
    v0, vN = M.boundary_values(P)
    sl0 = float(M.perturbation_slope(np.array([0.02]), P)[0])
    sl1 = float(M.perturbation_slope(np.array([0.98]), P)[0])
    return P, fd, Zc, slope_fd, id_fd_c, v0, vN, sl0, sl1


def policy_eval_linear(P, Zpe, slope_pe, v0, vN, dtau=2.0, max_iter=200000, tol=1e-12):
    n = len(Zpe) - 1; dZ = Zpe[1] - Zpe[0]
    p = _clamp(slope_pe.copy(), Zpe)
    i_d, i_g, c = M.controls(Zpe, p, P)
    phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    mu = Zpe * (1.0 - Zpe) * (phi_g - phi_d)
    flow = P["delta"] * np.log(np.maximum(c, 1e-300)) + (1.0 - Zpe) * phi_d + Zpe * phi_g
    idx = np.arange(1, n)
    Abar = M.A_bar(Zpe, P)
    c_sym = P["delta"] * (1.0 + P["theta_d"] * Abar) / (P["theta_d"] * (P["delta"] + P["Gamma_d"]))
    v = (np.log(c_sym) + ((1 - Zpe) * P["alpha_d"] + Zpe * P["alpha_g"]) / P["delta"]
         + (P["Gamma_d"] / P["delta"]) * np.log(P["Gamma_d"] * P["theta_d"] * c_sym / P["delta"]))
    v[0], v[-1] = v0, vN
    mi = mu[idx]; fwd = mi > 0.0; coef = mi / dZ
    for _ in range(max_iter):
        diag = np.full(n - 1, 1.0 / dtau + P["delta"]); sub = np.zeros(n - 1); sup = np.zeros(n - 1)
        diag[fwd] += coef[fwd]; sup[fwd] -= coef[fwd]
        diag[~fwd] -= coef[~fwd]; sub[~fwd] += coef[~fwd]
        rhs = v[idx] / dtau + flow[idx]; rhs[0] -= sub[0] * v0; rhs[-1] -= sup[-1] * vN
        v_new = _tri(sub, diag, sup, rhs)
        step = np.max(np.abs(v_new - v[idx]))
        v[idx] = v_new
        if step < tol:
            break
    return v


def slope_from_value(Zpe, v):
    p = np.empty_like(v)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * (Zpe[1] - Zpe[0]))
    p[0] = (v[1] - v[0]) / (Zpe[1] - Zpe[0]); p[-1] = (v[-1] - v[-2]) / (Zpe[1] - Zpe[0])
    return _clamp(p, Zpe)


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def run_sobolev(seed, P, Zc, v0, vN, sl0, sl1, lam):
    """Howard loop identical to costate_foc_oracle_ab.run_treated, but fit_step
    adds a Sobolev derivative-matching term with weight `lam`.
    lam==0.0 reproduces the plain-Howard baseline (value-only fit).
    lam=='auto' uses inverse-Dirichlet auto-weighting (balance grad norms)."""
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_net()
    Zpe = np.linspace(0.0, 1.0, N_PE + 1).astype(np.float64)
    ZpeT = tf.constant(Zpe.reshape(-1, 1).astype(np.float32))
    # corner-anchor weight mask: emphasize derivative match where autodiff v' is
    # noisiest (the mu->0 corners). Build a smooth weight peaked at the corners.
    corner_w = np.exp(-((Zpe - 0.0) / 0.06) ** 2) + np.exp(-((Zpe - 1.0) / 0.06) ** 2)
    corner_wT = tf.constant((1.0 + 2.0 * corner_w).astype(np.float32).reshape(-1, 1))

    def value(Zt):
        return (1 - Zt) * v0 + Zt * vN + Zt * (1 - Zt) * net(2.0 * Zt - 1.0)

    def slope_nn(Zt):
        with tf.GradientTape() as t:
            t.watch(Zt); v = value(Zt)
        return clamp_tf(t.gradient(v, Zt), Zt)

    opt = tf.keras.optimizers.Adam(ADAM_LR)
    auto = isinstance(lam, str) and lam == "auto"
    lam_const = tf.constant(0.0 if auto else float(lam), tf.float32)

    @tf.function
    def fit_step_value(vtarget_t):
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value(ZpeT) - vtarget_t))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L

    @tf.function
    def fit_step_sobolev(vtarget_t, ptarget_t):
        with tf.GradientTape() as t:
            with tf.GradientTape() as inner:
                inner.watch(ZpeT); v = value(ZpeT)
            vp = inner.gradient(v, ZpeT)
            Lv = tf.reduce_mean(tf.square(v - vtarget_t))
            Lp = tf.reduce_mean(corner_wT * tf.square(vp - ptarget_t))
            L = Lv + lam_const * Lp
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L

    @tf.function
    def fit_step_auto(vtarget_t, ptarget_t):
        # inverse-Dirichlet auto-weight: scale Lp so its grad-norm matches Lv's
        vars_ = net.trainable_variables
        with tf.GradientTape(persistent=True) as t:
            with tf.GradientTape() as inner:
                inner.watch(ZpeT); v = value(ZpeT)
            vp = inner.gradient(v, ZpeT)
            Lv = tf.reduce_mean(tf.square(v - vtarget_t))
            Lp = tf.reduce_mean(corner_wT * tf.square(vp - ptarget_t))
        gv = t.gradient(Lv, vars_); gp = t.gradient(Lp, vars_)
        nv = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in gv]) + 1e-30)
        npp = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in gp]) + 1e-30)
        w = tf.stop_gradient(nv / npp)
        grads = [a + w * b for a, b in zip(gv, gp)]
        del t
        opt.apply_gradients(zip(grads, vars_))
        return Lv + w * Lp

    # warm start from perturbation-slope policy-eval value (no FD info)
    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    for _ in range(1500):
        fit_step_value(v_ws_t)

    for sweep in range(N_HOWARD):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)
        v_pe = policy_eval_linear(P, Zpe, slope_pe, v0, vN)
        p_pe = slope_from_value(Zpe, v_pe)
        # anchor corners to perturbation slope (noisiest autodiff region)
        p_pe[0] = sl0; p_pe[-1] = sl1
        vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
        pt = tf.constant(p_pe.astype(np.float32).reshape(-1, 1))
        for _ in range(N_FIT):
            if lam_const == 0.0 and not auto:
                fit_step_value(vt)
            elif auto:
                fit_step_auto(vt, pt)
            else:
                fit_step_sobolev(vt, pt)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = slope_nn(ZcT).numpy().ravel().astype(np.float64)
    p_final = _clamp(p_final, Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN, sl0, sl1 = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())} | slope [{slope_fd[mI].min():.3f},{slope_fd[mI].max():.3f}]",
              flush=True)
        for seed in SEEDS:
            for lam in LAMBDAS:
                p_t, id_t = run_sobolev(seed, P, Zc, v0, vN, sl0, sl1, lam)
                e_id = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
                e_p = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
                results.append({"A_d": A_d, "seed": seed, "lam": str(lam),
                                "err_id": e_id, "err_p": e_p})
                print(f"  A_d={A_d} seed={seed} lam={str(lam):>5} "
                      f"max|i_d-FD|={e_id:.4e}  max|p-FD|={e_p:.4e}", flush=True)
    print("\n##### MEDIANS over seeds (per A_d, per lambda)", flush=True)
    for A_d in A_DS:
        base = float(np.median([r["err_id"] for r in results if r["A_d"] == A_d and r["lam"] == "0.0"]))
        for lam in LAMBDAS:
            rs = [r for r in results if r["A_d"] == A_d and r["lam"] == str(lam)]
            mid = float(np.median([r["err_id"] for r in rs]))
            mp = float(np.median([r["err_p"] for r in rs]))
            print(f"  A_d={A_d} lam={str(lam):>5}: med id={mid:.4e}  med p={mp:.4e}"
                  f"  ratio(vs plain-Howard)={mid/max(base,1e-12):.3f}", flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
