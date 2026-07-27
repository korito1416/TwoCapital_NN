"""
A/B test of the SEPARATE-COSTATE-NET / FOC-ORACLE-TARGET method
(method key: separate-costate-net-foc-oracle-target).

HYPOTHESIS: the v' INFORMATION exists in the FOC + transport structure (the FD
success mechanism), NOT in the flat-in-v' strong residual. So learn the slope
p(Z)=v'(Z) DIRECTLY with a separate "costate" network, supervised on FOC-consistent
policy-evaluation targets, ROUTING AROUND the flat residual.

CONTROL  = the established single value-NN trained on L2(strong residual)
           (Adam + tight L-BFGS) -- the same construction as the other AB scripts.
           v' is read off the value net; controls via M.controls.

TREATED  = a separate costate net p_phi(Z) (deep-BSDE / PG-DPO style). Each sweep:
   (1) controls = M.controls(Z, p_phi, P)              [FOC inversion; well-cond]
   (2) FREEZE that policy and do the VERIFIED LINEAR upwind tridiagonal
       policy-evaluation solve (Thomas/solve_banded, sign-definite-mu interior
       transport -- exactly the FD success path) -> value v_pe(Z)
   (3) read back p_target = neighbor finite-difference slope of v_pe
   (4) supervise p_phi -> p_target by L2.  Anchor corners to M.perturbation_slope.
   Iterate the Howard sweep ~12 times (policy improvement). Controls read from the
   converged p_phi.

This isolates whether the NN can REPRESENT the pocket slope when the TARGET comes
from the well-conditioned FOC+transport oracle rather than the flat residual.

TRUE ERROR (the ONLY metric; never compare loss numbers across formulations):
   max|i_d - i_d_FD| over Z in [0.1,0.9], and max|p - slope_FD|, per regime.
   Decisive test = the A_d=0.05 de-invest pocket where L-BFGS STALLED.

Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python costate_foc_oracle_ab.py   (login OK).
"""
import json
import numpy as np
import tensorflow as tf
from scipy.linalg import solve_banded
from scipy.optimize import minimize
import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri

tf.keras.backend.set_floatx("float32")

SEEDS = [1, 2, 3]
A_DS = [0.05, 0.13]
N_COLLO = 256
N_PE = 1000          # grid for the linear policy-evaluation solve
N_ADAM = 6000
ADAM_LR = 2e-3
N_HOWARD = 12        # Howard policy-improvement sweeps (treated)
N_FIT = 1500         # supervised fit steps of p_phi -> p_target per sweep


# ---------------------------------------------------------------------------
#  Shared finite-difference pieces
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  TREATED: separate costate net + FOC policy-evaluation oracle (Howard)
# ---------------------------------------------------------------------------

def policy_eval_linear(P, Zpe, slope_pe, v0, vN, dtau=2.0, max_iter=200000, tol=1e-12):
    """LINEAR upwind tridiagonal policy-evaluation: freeze controls implied by
    slope_pe (FOC), solve the resulting LINEAR transport ODE for v. This is the
    SAME M-matrix contraction the FD solver uses, but with a FROZEN policy
    (no v' re-derivation inside) -> reads v' from the well-conditioned transport.
    Returns the solved value v on Zpe (the policy is held fixed throughout)."""
    n = len(Zpe) - 1; dZ = Zpe[1] - Zpe[0]
    p = _clamp(slope_pe.copy(), Zpe)
    i_d, i_g, c = M.controls(Zpe, p, P)
    phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    mu = Zpe * (1.0 - Zpe) * (phi_g - phi_d)              # frozen drift
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


def make_costate_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def run_treated(seed, P, Zc, v0, vN, sl0, sl1):
    """Costate route via the FOC policy-evaluation ORACLE. KEY representational
    choice: the network represents the (smooth) VALUE with the boundary ansatz,
    and the slope p=v' is read by AUTODIFF -- NOT a direct (spiky, 2-decade-range)
    slope net, which is the representation that fails. Howard loop:
      (1) slope = autodiff(value net)  ->  controls via M.controls
      (2) FREEZE policy, LINEAR upwind policy-eval solve -> oracle value v_pe
      (3) supervise the value net toward v_pe (well-conditioned, smooth target)
    The well-conditioned FOC+transport oracle (numpy-Howard) reaches i_d err ~1e-4
    in the de-invest pocket; this measures what the NN representation retains."""
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_costate_net()
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

    # warm start the value net from the perturbation-slope policy-eval value (no FD info)
    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    for _ in range(1500):
        fit_step(v_ws_t)

    for sweep in range(N_HOWARD):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)  # (1) controls implied
        v_pe = policy_eval_linear(P, Zpe, slope_pe, v0, vN)           # (2) linear PE solve
        vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))      # (3) oracle value target
        for _ in range(N_FIT):                                        #     supervise the VALUE
            fit_step(vt)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = slope_nn(ZcT).numpy().ravel().astype(np.float64)
    p_final = _clamp(p_final, Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d


# ---------------------------------------------------------------------------
#  CONTROL: single value-NN on L2(strong residual) (Adam + tight L-BFGS)
# ---------------------------------------------------------------------------

def make_value_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def run_control(seed, P, Zc, v0, vN):
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_value_net()
    dl = P["delta"]; A_d = P["A_d"]; A_g = P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]
    Zt = tf.constant(Zc.reshape(-1, 1).astype(np.float32))

    def value(Z):
        return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)

    @tf.function
    def lg():
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                inner.watch(Zt); v = value(Zt)
            vp = clamp_tf(inner.gradient(v, Zt), Zt)
            q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
            Abar = (1 - Zt) * A_d + Zt * A_g
            c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
            i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
            phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
            phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
            mu = Zt * (1 - Zt) * (phi_g - phi_d)
            R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + (1 - Zt) * phi_d + Zt * phi_g + mu * vp
            loss = tf.reduce_mean(tf.square(R))
        return loss, outer.gradient(loss, net.trainable_variables)

    opt = tf.keras.optimizers.Adam(ADAM_LR)
    for _ in range(N_ADAM):
        L, g = lg(); opt.apply_gradients(zip(g, net.trainable_variables))

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
    minimize(fg, x0, jac=True, method="L-BFGS-B",
             options={"maxiter": 4000, "maxfun": 8000, "ftol": 1e-15, "gtol": 1e-12})

    with tf.GradientTape() as inner:
        inner.watch(Zt); v = value(Zt)
    vp = clamp_tf(inner.gradient(v, Zt), Zt).numpy().ravel().astype(np.float64)
    i_d, i_g, c = M.controls(Zc, vp, P)
    return vp, i_d


# ---------------------------------------------------------------------------

def main():
    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN, sl0, sl1 = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())} | slope [{slope_fd[mI].min():.3f},{slope_fd[mI].max():.3f}] "
              f"sl0={sl0:.3f} sl1={sl1:.3f}", flush=True)
        for seed in SEEDS:
            p_c, id_c = run_control(seed, P, Zc, v0, vN)
            p_t, id_t = run_treated(seed, P, Zc, v0, vN, sl0, sl1)
            e_id_c = float(np.max(np.abs(id_c[mI] - id_fd_c[mI])))
            e_id_t = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
            e_p_c = float(np.max(np.abs(p_c[mI] - slope_fd[mI])))
            e_p_t = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
            row = {"A_d": A_d, "seed": seed,
                   "err_id_ctrl": e_id_c, "err_id_treat": e_id_t,
                   "err_p_ctrl": e_p_c, "err_p_treat": e_p_t,
                   "id_ratio": e_id_t / max(e_id_c, 1e-12),
                   "p_ratio": e_p_t / max(e_p_c, 1e-12)}
            results.append(row)
            print(f"  seed={seed} CONTROL  max|i_d-FD|={e_id_c:.4e}  max|p-FD|={e_p_c:.4e}", flush=True)
            print(f"  seed={seed} TREATED  max|i_d-FD|={e_id_t:.4e}  max|p-FD|={e_p_t:.4e}"
                  f"  -> id_ratio(T/C)={row['id_ratio']:.3f}  p_ratio={row['p_ratio']:.3f}", flush=True)
    # aggregate medians per regime
    print("\n##### MEDIANS over seeds", flush=True)
    for A_d in A_DS:
        rs = [r for r in results if r["A_d"] == A_d]
        med = lambda k: float(np.median([r[k] for r in rs]))
        print(f"  A_d={A_d}: CONTROL id={med('err_id_ctrl'):.4e}  TREATED id={med('err_id_treat'):.4e}"
              f"  | CONTROL p={med('err_p_ctrl'):.4e}  TREATED p={med('err_p_treat'):.4e}"
              f"  id_ratio={med('id_ratio'):.3f}", flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
