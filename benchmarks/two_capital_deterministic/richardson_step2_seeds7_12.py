"""
STEP 2: A/B test of RICHARDSON-DE-BIASED policy-evaluation target inside the
Howard value-net loop, vs the plain-Howard (first-order upwind target) baseline,
in the SAME harness / seeds / budget as costate_foc_oracle_ab.py.

Method key: richardson-highorder-pe-transport-target.

BASELINE (plain Howard)  = the costate_foc_oracle_ab.py TREATED method exactly:
  Howard loop, each sweep: slope=autodiff(value net) -> controls (FOC) -> FREEZE
  policy -> FIRST-ORDER upwind linear policy-evaluation solve -> oracle value v_pe
  -> supervise the value net toward v_pe.  (~1.5e-3 floor in the A_d=0.05 pocket.)

TREATED (Richardson)     = identical, EXCEPT the oracle VALUE TARGET is de-biased:
  on the SAME frozen policy, solve PE at N_PE and 2*N_PE, extrapolate
  v_ext = 2*v(2N) - v(N) (interpolated onto the N_PE grid), and supervise toward
  v_ext.  This removes the O(dZ) upwind diffusion from the target the NN chases
  (STEP 1: target v-error 1.4e-3 -> 9.6e-5 in the pocket).

TRUE error vs FD: max|i_d - i_d_FD| over Z in [0.1,0.9], per regime, per seed.
PASS = TREATED beats BASELINE in the A_d=0.05 pocket (toward 1e-4) AND does NOT
degrade at A_d=0.13.

Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python richardson_pe_step2_ab.py  (login OK).
"""
import json
import numpy as np
import tensorflow as tf
import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri

tf.keras.backend.set_floatx("float32")

SEEDS = [7, 8, 9, 10, 11, 12]
A_DS = [0.05, 0.13]
N_COLLO = 256
N_PE = 1000
ADAM_LR = 2e-3
N_HOWARD = 12
N_FIT = 1500


def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zc = np.linspace(0.02, 0.98, N_COLLO).astype(np.float64)
    slope_fd = np.interp(Zc, fd["Z"], fd["slope"])
    id_fd_c = np.interp(Zc, fd["Z"], fd["i_d"])
    v0, vN = M.boundary_values(P)
    return P, fd, Zc, slope_fd, id_fd_c, v0, vN


def policy_eval_linear(P, Zpe, slope_pe, v0, vN, dtau=2.0, max_iter=200000, tol=1e-12):
    """First-order upwind linear policy-evaluation on a FROZEN policy."""
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


def policy_eval_richardson(P, Zpe_lo, slope_pe_lo, v0, vN):
    """Richardson-de-biased PE value target on the N_PE grid Zpe_lo, using the SAME
    frozen policy (slope read off the value net) evaluated on the lo and 2x grids.
    The frozen policy on the fine grid is the lo-grid slope interpolated up, so both
    solves freeze the IDENTICAL policy -> the only difference is operator dZ."""
    N_lo = len(Zpe_lo) - 1
    Zpe_hi = np.linspace(0.0, 1.0, 2 * N_lo + 1)
    slope_pe_hi = np.interp(Zpe_hi, Zpe_lo, slope_pe_lo)
    v_lo = policy_eval_linear(P, Zpe_lo, slope_pe_lo, v0, vN)
    v_hi = policy_eval_linear(P, Zpe_hi, slope_pe_hi, v0, vN)
    v_hi_on_lo = np.interp(Zpe_lo, Zpe_hi, v_hi)
    return 2.0 * v_hi_on_lo - v_lo


def make_value_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def run_howard(seed, P, Zc, v0, vN, richardson):
    """Shared Howard value-net loop. richardson=False -> plain upwind target
    (the costate_foc_oracle baseline). richardson=True -> de-biased target."""
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

    @tf.function
    def fit_step(vtarget_t):
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value(ZpeT) - vtarget_t))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L

    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    for _ in range(1500):
        fit_step(v_ws_t)

    for sweep in range(N_HOWARD):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)
        if richardson:
            v_pe = policy_eval_richardson(P, Zpe, slope_pe, v0, vN)
        else:
            v_pe = policy_eval_linear(P, Zpe, slope_pe, v0, vN)
        vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
        for _ in range(N_FIT):
            fit_step(vt)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = _clamp(slope_nn(ZcT).numpy().ravel().astype(np.float64), Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())}", flush=True)
        for seed in SEEDS:
            p_b, id_b = run_howard(seed, P, Zc, v0, vN, richardson=False)
            p_t, id_t = run_howard(seed, P, Zc, v0, vN, richardson=True)
            e_id_b = float(np.max(np.abs(id_b[mI] - id_fd_c[mI])))
            e_id_t = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
            e_p_b = float(np.max(np.abs(p_b[mI] - slope_fd[mI])))
            e_p_t = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
            row = {"A_d": A_d, "seed": seed,
                   "err_id_base": e_id_b, "err_id_rich": e_id_t,
                   "err_p_base": e_p_b, "err_p_rich": e_p_t,
                   "id_ratio": e_id_t / max(e_id_b, 1e-12)}
            results.append(row)
            print(f"  seed={seed} HOWARD-upwind  max|i_d-FD|={e_id_b:.4e}  max|p-FD|={e_p_b:.4e}", flush=True)
            print(f"  seed={seed} HOWARD-RICHRD  max|i_d-FD|={e_id_t:.4e}  max|p-FD|={e_p_t:.4e}"
                  f"  -> id_ratio(R/U)={row['id_ratio']:.3f}", flush=True)
    print("\n##### MEDIANS over seeds", flush=True)
    for A_d in A_DS:
        rs = [r for r in results if r["A_d"] == A_d]
        med = lambda k: float(np.median([r[k] for r in rs]))
        print(f"  A_d={A_d}: UPWIND id={med('err_id_base'):.4e}  RICHARDSON id={med('err_id_rich'):.4e}"
              f"  | UPWIND p={med('err_p_base'):.4e}  RICHARDSON p={med('err_p_rich'):.4e}"
              f"  id_ratio={med('id_ratio'):.3f}", flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
