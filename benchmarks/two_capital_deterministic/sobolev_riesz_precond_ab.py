"""
Sobolev-RIESZ-preconditioned Howard value fit (method key: sobolev-riesz-howard).

DISTINCT from the REFUTED naive H^1 (sobolev_howard_ab.py), which added an
explicit slope-target term  L = ||v-v_pe||^2 + lam*||autodiff_v' - p_pe||^2 .
That moved the fixed point onto a NOISY/biased slope target p_pe (neighbor-FD of
the transport value + perturbation-corner anchors) and was 25-135x WORSE; even
the auto-weight arm was 2.3-3.4x worse. Lesson: the slope target is the wrong
DATA term; you must NOT change what the fit converges to.

THIS arm keeps the data-fidelity term as the well-conditioned L2 VALUE fit ONLY
(same fixed point as plain-Howard), and applies the H^1 geometry as a RIESZ-MAP
PRECONDITIONER on the value-FIT update -- a mirror/natural-gradient step whose
distance generator is the H^1 inner product. It SMOOTHS the descent direction
(removes high-frequency components that pollute the autodiff slope) without
re-targeting the fit.

DISTANCE GENERATOR
------------------
psi(u) = 1/2 <u, G u>_grid,  G = I + beta * L,  L = -d^2/dZ^2 (Dirichlet, on the
PE grid). G is SPD; D_psi(a,b)=1/2 <a-b, G (a-b)> is the H^1(beta)-seminorm
squared distance. Mirror/NG descent in this geometry preconditions the L2-fit
gradient by G^{-1} (the Riesz map / discrete Green's function), i.e. it is a
LOW-PASS smoothing of the update. High Z-frequencies (the autodiff-slope noise)
are damped by 1/(1+beta*k^2); the smooth value content passes unchanged.

WHERE IT APPLIES (NOT the strong residual): the supervised value-FIT step inside
the Howard loop -- the L2 regression value(Z) -> v_pe(Z) where v_pe is the
output of the well-conditioned LINEAR upwind policy-evaluation solve. The strong
HJB residual is never touched.

ARMS:
  lam==0.0  baseline plain-Howard (value-only L2 fit) == run_treated original.
  beta>0    Riesz-preconditioned value fit (THIS proposal), beta in a small sweep.
Same warm start, same 12 Howard sweeps, same N_FIT, same Adam.

TRUE ERROR vs FD (ONLY metric): max|i_d-FD|, max|p-FD| over Z in [0.1,0.9].
Pocket A_d=0.05 (beat ~1.5e-3); well-cond A_d=0.13 (must not degrade).
"""
import json
import numpy as np
import tensorflow as tf
import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri

tf.keras.backend.set_floatx("float32")

SEEDS = [1, 2, 3]
A_DS = [0.05, 0.13]
N_COLLO = 256
N_PE = 1000
N_HOWARD = 12
N_FIT = 1500
ADAM_LR = 2e-3
# beta = H^1 preconditioner strength in the Riesz map (I + beta*L)^{-1}. 0 == plain.
BETAS = [0.0, 1e-3, 1e-2]


def build_case(A_d):
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    fd = solve_fd(P, n=4000)
    Zc = np.linspace(0.02, 0.98, N_COLLO).astype(np.float64)
    slope_fd = np.interp(Zc, fd["Z"], fd["slope"])
    id_fd_c = np.interp(Zc, fd["Z"], fd["i_d"])
    v0, vN = M.boundary_values(P)
    return P, fd, Zc, slope_fd, id_fd_c, v0, vN


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


def make_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def build_riesz_factor(npe, dZ, beta):
    """Cholesky/banded factor of G = I + beta*L on the PE grid (Dirichlet L).
    Returns a callable that applies G^{-1} (Riesz smoothing) to a (npe+1,) array,
    holding the two endpoints fixed (the boundary ansatz pins them anyway)."""
    if beta <= 0.0:
        return lambda r: r
    n = npe + 1
    main = np.full(n, 1.0 + 2.0 * beta / dZ**2)
    off = np.full(n - 1, -beta / dZ**2)
    # Dirichlet at endpoints: keep rows 0 and n-1 as identity
    main[0] = main[-1] = 1.0
    off[0] = 0.0; off[-1] = 0.0
    ab = np.zeros((3, n))
    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off
    from scipy.linalg import solve_banded
    def apply_Ginv(r):
        return solve_banded((1, 1), ab, r)
    return apply_Ginv


def run_riesz(seed, P, Zc, v0, vN, beta):
    """Howard loop identical to run_treated, but the VALUE-fit gradient is
    Riesz-preconditioned by G^{-1}=(I+beta L)^{-1} in the SAMPLE (grid) space
    BEFORE backprop to params. We realize the Riesz map by smoothing the residual
    (value - v_pe) on the grid: the L2 fit gradient wrt the net is
        d/dtheta mean( (value - v_pe)^2 ) = (2/N) J^T (value - v_pe),
    J = d value / d theta on the grid. Replacing the residual r=(value-v_pe) by
    G^{-1} r gives the H^1-natural-gradient step  (2/N) J^T G^{-1} r, i.e. mirror
    descent with potential psi=1/2<u,G u>. Implemented by a custom loss whose
    gradient wrt 'value' samples is the smoothed residual (stop-grad trick)."""
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_net()
    Zpe = np.linspace(0.0, 1.0, N_PE + 1).astype(np.float64)
    dZ = Zpe[1] - Zpe[0]
    ZpeT = tf.constant(Zpe.reshape(-1, 1).astype(np.float32))
    apply_Ginv = build_riesz_factor(N_PE, dZ, beta)

    def value(Zt):
        return (1 - Zt) * v0 + Zt * vN + Zt * (1 - Zt) * net(2.0 * Zt - 1.0)

    def slope_nn(Zt):
        with tf.GradientTape() as t:
            t.watch(Zt); v = value(Zt)
        return clamp_tf(t.gradient(v, Zt), Zt)

    opt = tf.keras.optimizers.Adam(ADAM_LR)

    @tf.function
    def fit_step_plain(vtarget_t):
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value(ZpeT) - vtarget_t))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L

    @tf.function
    def _value_grid():
        return value(ZpeT)

    @tf.function
    def _apply_vjp(coef):
        with tf.GradientTape() as t:
            surrogate = tf.reduce_sum(value(ZpeT) * coef)
        g = t.gradient(surrogate, net.trainable_variables)
        opt.apply_gradients(zip(g, net.trainable_variables))

    def fit_step_riesz(vtarget_np):
        # 1st pass: read value on grid to form the smoothed cotangent G^{-1} r.
        v_np = _value_grid().numpy().ravel()
        r = (v_np - vtarget_np).astype(np.float64)            # raw residual
        r_smooth = apply_Ginv(r)                              # G^{-1} r  (Riesz map)
        coef = tf.constant((2.0 / len(r)) * r_smooth.astype(np.float32).reshape(-1, 1))
        # 2nd pass: VJP J^T coef = grad of <value, stopgrad(coef)> -> the H^1-NG step.
        _apply_vjp(coef)
        return float(np.mean(r * r))

    # warm start (value-only L2, no preconditioner -- same start as baseline)
    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    for _ in range(1500):
        fit_step_plain(v_ws_t)

    for sweep in range(N_HOWARD):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)
        v_pe = policy_eval_linear(P, Zpe, slope_pe, v0, vN)
        if beta <= 0.0:
            vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
            for _ in range(N_FIT):
                fit_step_plain(vt)
        else:
            vt_np = v_pe.astype(np.float64)
            for _ in range(N_FIT):
                fit_step_riesz(vt_np)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = slope_nn(ZcT).numpy().ravel().astype(np.float64)
    p_final = _clamp(p_final, Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())} | slope [{slope_fd[mI].min():.3f},{slope_fd[mI].max():.3f}]",
              flush=True)
        for seed in SEEDS:
            for beta in BETAS:
                p_t, id_t = run_riesz(seed, P, Zc, v0, vN, beta)
                e_id = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
                e_p = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
                results.append({"A_d": A_d, "seed": seed, "beta": beta,
                                "err_id": e_id, "err_p": e_p})
                print(f"  A_d={A_d} seed={seed} beta={beta:>7.0e} "
                      f"max|i_d-FD|={e_id:.4e}  max|p-FD|={e_p:.4e}", flush=True)
    print("\n##### MEDIANS over seeds (per A_d, per beta)", flush=True)
    for A_d in A_DS:
        base = float(np.median([r["err_id"] for r in results if r["A_d"] == A_d and r["beta"] == 0.0]))
        for beta in BETAS:
            rs = [r for r in results if r["A_d"] == A_d and r["beta"] == beta]
            mid = float(np.median([r["err_id"] for r in rs]))
            mp = float(np.median([r["err_p"] for r in rs]))
            print(f"  A_d={A_d} beta={beta:>7.0e}: med id={mid:.4e}  med p={mp:.4e}"
                  f"  ratio(vs plain)={mid/max(base,1e-12):.3f}", flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
