"""
A/B test of the rho-WEIGHTED TRANSPORT-ENERGY INNER LOSS combined with the
Howard FOC routing (method key: rho-weighted-transport-energy-inner-loss).

CONTROL  = plain-Howard L2-to-oracle (the established workflow-#3 baseline):
   each sweep, FREEZE the autodiff-slope policy, do the LINEAR upwind tridiagonal
   policy-evaluation numpy solve -> oracle value v_pe, supervise the value net
   toward v_pe by L2.  (This is the ~1.5e-3-pocket Howard method, verbatim.)

TREATED  = SAME Howard routing, but the inner supervision target is REPLACED by an
   invariant-measure (rho)-weighted CONTINUOUS variational energy:
       J_rho = mean( rho(Z) * ( delta*v(Z) - mu_froz*v'(Z) - flow_froz )^2 )
   where
     * mu_froz, flow_froz = stop_gradient frozen FOC coefficients from the
       autodiff slope (M.controls), recomputed each sweep -- the SAME frozen
       policy the linear PE solve uses, but used as CONTINUOUS coefficients;
     * v'(Z) is autodiff of the boundary-ansatz value net (NOT a numpy slope),
       so the minimizer is the CONTINUOUS energy solution -- no O(dZ) upwind
       diffusion bias from the numpy target;
     * rho(Z) = exp(-cumtrapz(delta/clip(|mu_froz|,eps))) is the integrating
       factor that SYMMETRIZES the advection operator (GN Hessian ~ kappa(L),
       not kappa(L)^2).  rho is warm-started from the perturbation-slope policy
       and FROZEN for the first RHO_FREEZE sweeps (avoid mis-weighting from
       early-noisy autodiff mu), clipped to [RMIN,RMAX], normalised to mean 1.

Both arms share: boundary-ansatz value net v(Z)=(1-Z)v0+Z*vN+Z(1-Z)*net(2Z-1),
the autodiff slope clamp, the same warm start, the same #sweeps and #fit-steps,
the same seed/optimizer/budget.  Only the inner loss differs.

DIAGNOSTIC arm (treated_nooracle == TREATED itself): the J_rho minimizer uses NO
numpy oracle value target at all, so the gap between CONTROL and TREATED isolates
how much of the floor was numpy-target upwind diffusion vs net capacity.

TRUE ERROR (the ONLY metric): max|i_d - i_d_FD| over Z in [0.1,0.9], per regime.
Decisive test = A_d=0.05 de-invest pocket (beat plain-Howard ~1.5e-3 toward 1e-4?)
Guardrail    = A_d=0.13 well-conditioned (must NOT degrade vs plain-Howard).

Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python rho_weighted_energy_ab.py   (login OK).
"""
import json
import numpy as np
import tensorflow as tf
import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri

tf.keras.backend.set_floatx("float32")

SEEDS = [7, 8, 9, 11, 13]   # SEED-B verification: all DIFFERENT from original [1..6]
A_DS = [0.05, 0.13]
N_COLLO = 256
N_PE = 1000          # grid for the linear policy-evaluation solve / energy collocation
N_HOWARD = 12        # Howard policy-improvement sweeps
N_FIT = 1500         # supervised fit steps per sweep (same in both arms)
ADAM_LR = 2e-3
N_WARM = 1500        # warm-start steps toward perturbation-slope PE value
RHO_FREEZE = 3       # # sweeps to keep rho frozen at the warm-start weighting
RMIN, RMAX = 0.2, 50.0  # rho clip (after mean-1 normalisation reference)
MU_EPS = 1e-4        # floor on |mu| inside delta/mu integrand


# ---------------------------------------------------------------------------
#  Shared FD pieces
# ---------------------------------------------------------------------------

def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zc = np.linspace(0.02, 0.98, N_COLLO).astype(np.float64)
    slope_fd = np.interp(Zc, fd["Z"], fd["slope"])
    id_fd_c = np.interp(Zc, fd["Z"], fd["i_d"])
    v0, vN = M.boundary_values(P)
    return P, fd, Zc, slope_fd, id_fd_c, v0, vN


# ---------------------------------------------------------------------------
#  Linear policy-evaluation numpy oracle (CONTROL target). Returns v AND the
#  frozen mu/flow coefficients (reused to build rho/energy in TREATED).
# ---------------------------------------------------------------------------

def frozen_coeffs(P, Zpe, slope_pe):
    p = _clamp(slope_pe.copy(), Zpe)
    i_d, i_g, c = M.controls(Zpe, p, P)
    phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    mu = Zpe * (1.0 - Zpe) * (phi_g - phi_d)
    flow = P["delta"] * np.log(np.maximum(c, 1e-300)) + (1.0 - Zpe) * phi_d + Zpe * phi_g
    return mu, flow


def policy_eval_linear(P, Zpe, slope_pe, v0, vN, dtau=2.0, max_iter=200000, tol=1e-12):
    n = len(Zpe) - 1; dZ = Zpe[1] - Zpe[0]
    mu, flow = frozen_coeffs(P, Zpe, slope_pe)
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
        step = np.max(np.abs(v_new - v[idx])); v[idx] = v_new
        if step < tol:
            break
    return v


# ---------------------------------------------------------------------------
#  rho integrating-factor weight
# ---------------------------------------------------------------------------

def rho_weight(P, Zpe, mu):
    """rho(Z) = exp(-cumtrapz(delta/mu)) integrating factor that symmetrizes the
    advection operator d/dZ via  d/dZ(rho*mu*v) form. We use |mu| floored so the
    weight is finite at the (interior) stagnation point where mu->0; the sign of
    the integrand is taken from delta/mu so the exponent grows where transport is
    slow (the pocket), concentrating the energy there. Normalised to mean 1, then
    clipped to [RMIN,RMAX]."""
    dZ = Zpe[1] - Zpe[0]
    mu_s = np.sign(mu)
    mu_s[mu_s == 0] = 1.0
    integrand = P["delta"] / (mu_s * np.maximum(np.abs(mu), MU_EPS))
    # cumulative trapezoid from the LEFT boundary
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dZ)])
    cum = cum - cum.mean()                         # center the exponent for stability
    cum = np.clip(cum, -20.0, 20.0)
    rho = np.exp(-cum)
    rho = rho / rho.mean()                         # mean-1
    rho = np.clip(rho, RMIN, RMAX)
    rho = rho / rho.mean()
    return rho


# ---------------------------------------------------------------------------
#  Net + slope
# ---------------------------------------------------------------------------

def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


# ---------------------------------------------------------------------------
#  CONTROL: plain-Howard L2-to-oracle
# ---------------------------------------------------------------------------

def run_control(seed, P, Zc, v0, vN):
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_net()
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
    for _ in range(N_WARM):
        fit_step(v_ws_t)

    for sweep in range(N_HOWARD):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)
        v_pe = policy_eval_linear(P, Zpe, slope_pe, v0, vN)
        vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
        for _ in range(N_FIT):
            fit_step(vt)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = _clamp(slope_nn(ZcT).numpy().ravel().astype(np.float64), Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d


# ---------------------------------------------------------------------------
#  TREATED: rho-weighted transport-energy inner loss (same Howard routing)
# ---------------------------------------------------------------------------

def run_treated(seed, P, Zc, v0, vN):
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_net()
    delta = float(P["delta"])
    Zpe = np.linspace(0.0, 1.0, N_PE + 1).astype(np.float64)
    ZpeT = tf.constant(Zpe.reshape(-1, 1).astype(np.float32))

    def value(Zt):
        return (1 - Zt) * v0 + Zt * vN + Zt * (1 - Zt) * net(2.0 * Zt - 1.0)

    def slope_nn(Zt):
        with tf.GradientTape() as t:
            t.watch(Zt); v = value(Zt)
        return clamp_tf(t.gradient(v, Zt), Zt)

    opt = tf.keras.optimizers.Adam(ADAM_LR)

    # plain L2 warm-start fit_step (shared with control's warm start)
    @tf.function
    def l2_fit_step(vtarget_t):
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value(ZpeT) - vtarget_t))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L

    # rho-weighted continuous-energy inner step. mu_froz, flow_froz, rho are
    # numpy constants (stop-gradient by construction) recomputed each sweep.
    @tf.function
    def energy_fit_step(mu_t, flow_t, rho_t):
        with tf.GradientTape() as gt:
            with tf.GradientTape() as it:
                it.watch(ZpeT); v = value(ZpeT)
            vp = clamp_tf(it.gradient(v, ZpeT), ZpeT)
            resid = delta * v - mu_t * vp - flow_t
            L = tf.reduce_mean(rho_t * tf.square(resid))
        opt.apply_gradients(zip(gt.gradient(L, net.trainable_variables), net.trainable_variables))
        return L

    # warm start (same as control)
    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    for _ in range(N_WARM):
        l2_fit_step(v_ws_t)

    # frozen rho from the perturbation-slope policy (no FD info)
    mu_ws, _ = frozen_coeffs(P, Zpe, pert)
    rho_frozen = rho_weight(P, Zpe, mu_ws)
    rho_frozen_t = tf.constant(rho_frozen.astype(np.float32).reshape(-1, 1))

    rho_stats = []
    for sweep in range(N_HOWARD):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)  # autodiff slope
        mu_froz, flow_froz = frozen_coeffs(P, Zpe, slope_pe)          # frozen FOC coeffs
        if sweep < RHO_FREEZE:
            rho_t = rho_frozen_t
            rho_cur = rho_frozen
        else:
            rho_cur = rho_weight(P, Zpe, mu_froz)
            rho_t = tf.constant(rho_cur.astype(np.float32).reshape(-1, 1))
        rho_stats.append(float(rho_cur.max() / max(rho_cur.min(), 1e-12)))
        mu_t = tf.constant(mu_froz.astype(np.float32).reshape(-1, 1))
        flow_t = tf.constant(flow_froz.astype(np.float32).reshape(-1, 1))
        for _ in range(N_FIT):
            energy_fit_step(mu_t, flow_t, rho_t)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = _clamp(slope_nn(ZcT).numpy().ravel().astype(np.float64), Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d, float(np.median(rho_stats))


# ---------------------------------------------------------------------------

def main():
    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())} | slope [{slope_fd[mI].min():.3f},{slope_fd[mI].max():.3f}]",
              flush=True)
        for seed in SEEDS:
            p_c, id_c = run_control(seed, P, Zc, v0, vN)
            p_t, id_t, rho_ratio = run_treated(seed, P, Zc, v0, vN)
            e_id_c = float(np.max(np.abs(id_c[mI] - id_fd_c[mI])))
            e_id_t = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
            e_p_c = float(np.max(np.abs(p_c[mI] - slope_fd[mI])))
            e_p_t = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
            row = {"A_d": A_d, "seed": seed,
                   "err_id_ctrl": e_id_c, "err_id_treat": e_id_t,
                   "err_p_ctrl": e_p_c, "err_p_treat": e_p_t,
                   "id_ratio": e_id_t / max(e_id_c, 1e-12),
                   "rho_ratio": rho_ratio}
            results.append(row)
            print(f"  seed={seed} CONTROL(Howard)  max|i_d-FD|={e_id_c:.4e}  max|p-FD|={e_p_c:.4e}", flush=True)
            print(f"  seed={seed} TREATED(rho-E)   max|i_d-FD|={e_id_t:.4e}  max|p-FD|={e_p_t:.4e}"
                  f"  -> id_ratio(T/C)={row['id_ratio']:.3f}  max/min(rho)={rho_ratio:.2f}", flush=True)
    print("\n##### MEDIANS over seeds", flush=True)
    for A_d in A_DS:
        rs = [r for r in results if r["A_d"] == A_d]
        med = lambda k: float(np.median([r[k] for r in rs]))
        print(f"  A_d={A_d}: CONTROL id={med('err_id_ctrl'):.4e}  TREATED id={med('err_id_treat'):.4e}"
              f"  | CONTROL p={med('err_p_ctrl'):.4e}  TREATED p={med('err_p_treat'):.4e}"
              f"  id_ratio={med('id_ratio'):.3f}  rho={med('rho_ratio'):.2f}", flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
