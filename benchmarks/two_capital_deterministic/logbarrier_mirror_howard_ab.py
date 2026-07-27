"""
Log-barrier / negative-entropy MIRROR map on the Howard FOC+transport pipeline
(method key: logbarrier-mirror-admissibility).

DESIGN
------
The admissible set for the costate slope p=v' is the OPEN INTERVAL
    p in ( -1/(1-Z) , 1/Z )    <=>    q_d = 1 - Z p > 0  AND  q_g = 1 + (1-Z) p > 0
which is EXACTLY the FOC growth-factor constraint 1+theta_d i_d > 0, 1+theta_g i_g > 0.
In (q_d, q_g) coordinates this is the positive ORTHANT. The native mirror map for a
positive orthant / probability-like constraint is the NEGATIVE ENTROPY potential

    psi(q) = q_d (log q_d - 1) + q_g (log q_g - 1)         (separable Burg/neg-entropy)

with Bregman divergence  D_psi(q, q~) = sum_j [ q_j log(q_j/q~_j) - q_j + q~_j ]
(a generalized KL / I-divergence), and Hessian (induced metric)
    grad^2 psi = diag( 1/q_d , 1/q_g ).
Pulled back to the slope p (q_d,q_g affine in p, fixed Z), the induced 1-D metric is
    g(p) = (Z^2 / q_d) + ((1-Z)^2 / q_g)   = barrier curvature of the admissible interval.
g(p) -> +inf as p approaches EITHER endpoint -> mirror steps never leave the set and
SLOW DOWN near saturation, instead of clipping (which is what clamp_tf / _clamp do now).

WHERE IT APPLIES (NOT the strong residual)
------------------------------------------
On the Howard inner VALUE-FIT / FOC-inversion step of costate_foc_oracle_ab.py::
run_treated (== torch_egm/howard.py). Two coupled uses:
  (A) MIRROR-PRECONDITIONED slope read for the controls. The autodiff slope p_nn is
      mapped to the dual coordinate (eta = grad psi(q) = log q), the well-conditioned
      transport-oracle slope p_pe defines the dual target eta_pe, and we take a mirror
      (Bregman) average toward the oracle in DUAL space, then map back. Equivalent to a
      mirror-descent step on the control with the neg-entropy potential: it lives on the
      FOC inversion + transport oracle, never on R.
  (B) BARRIER-METRIC value fit. The supervised value-fit loss is reweighted pointwise by
      the barrier curvature g(p) so the value net spends accuracy where the admissible
      interval is TIGHT (corners, where q_d or q_g -> 0). This is a diagonal Riemannian
      metric on the VALUE FIT (the smooth, well-conditioned target), NOT on R.

This is structurally DIFFERENT from the refuted residual re-norming: dR/dp = mu_Z, and
NONE of these terms multiply R or invert mu_Z. The metric is the CONSTRAINT barrier of
the FOC, whose Jacobian di/dp is 3-10x larger than mu_Z (the well-conditioned map).

ARMS
----
  BASELINE  = plain Howard value-only fit (== run_treated).
  MIRROR-A  = dual-space (neg-entropy) Bregman pull of the slope toward the oracle.
  MIRROR-B  = barrier-curvature-weighted value fit.
  MIRROR-AB = both.
beta = mirror step / dual mixing weight in {0.0(=baseline),0.25,0.5}.

TRUE ERROR vs FD (ONLY metric): max|i_d-i_d_FD|, max|p-slope_FD| over Z in [0.1,0.9].
Pocket A_d=0.05 (beat plain-Howard ~1.5e-3 toward 1e-4). Well-cond A_d=0.13 (must NOT
degrade vs Howard ~1.1e-3).

Run: cd benchmarks/two_capital_deterministic && module load python/anaconda-2021.05 &&
     python logbarrier_mirror_howard_ab.py    (login OK).
"""
import json
import numpy as np
import tensorflow as tf
import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri
from costate_foc_oracle_ab import policy_eval_linear, build_case

tf.keras.backend.set_floatx("float32")

SEEDS = [1, 2, 3, 4, 5]
A_DS = [0.05, 0.13]
N_COLLO = 256
N_PE = 1000
N_HOWARD = 12
N_FIT = 1500
N_WARM = 1500
ADAM_LR = 2e-3
BETAS = [0.0, 0.25, 0.5]          # 0.0 == plain-Howard baseline
MARGIN = 1e-4                      # admissibility margin (matches clamp_tf m)


def q_of_p(p, Z):
    """Growth factors q_d=1-Zp, q_g=1+(1-Z)p (the positive-orthant coordinates)."""
    return 1.0 - Z * p, 1.0 + (1.0 - Z) * p


def barrier_metric(p, Z, floorq=1e-3):
    """Induced 1-D metric g(p)=Z^2/q_d + (1-Z)^2/q_g (neg-entropy Hessian pullback)."""
    qd, qg = q_of_p(p, Z)
    qd = np.maximum(qd, floorq); qg = np.maximum(qg, floorq)
    return Z * Z / qd + (1.0 - Z) * (1.0 - Z) / qg


def mirror_pull_slope(p_nn, p_pe, Z, beta, floorq=1e-6):
    """Neg-entropy (Burg) Bregman pull of p_nn toward the oracle p_pe in DUAL space.

    Mirror map: eta_j = grad psi(q_j) = log q_j. Take the geodesic (dual-linear) step
        eta_new = (1-beta) eta_nn + beta eta_pe   for each of q_d, q_g,
    i.e. a GEOMETRIC mean of the growth factors -> q_new = q_nn^(1-beta) q_pe^beta.
    Map back: q_d_new = 1-Z p  and  q_g_new = 1+(1-Z) p are two affine images of the
    SAME p, so invert each and average the two implied p's by their barrier curvature
    (the metric-consistent reconciliation). This is one mirror-descent step toward the
    well-conditioned transport oracle, living entirely on the FOC inversion.
    """
    qd_n, qg_n = q_of_p(p_nn, Z); qd_p, qg_p = q_of_p(p_pe, Z)
    qd_n = np.maximum(qd_n, floorq); qg_n = np.maximum(qg_n, floorq)
    qd_p = np.maximum(qd_p, floorq); qg_p = np.maximum(qg_p, floorq)
    qd_new = qd_n ** (1.0 - beta) * qd_p ** beta      # geometric (dual-linear) mean
    qg_new = qg_n ** (1.0 - beta) * qg_p ** beta
    # invert each affine image to p; reconcile by barrier curvature weights
    p_from_d = (1.0 - qd_new) / np.maximum(Z, 1e-9)
    p_from_g = (qg_new - 1.0) / np.maximum(1.0 - Z, 1e-9)
    wd = Z * Z / qd_new                                # 1/var in dual coordinate d
    wg = (1.0 - Z) * (1.0 - Z) / qg_new
    p_new = (wd * p_from_d + wg * p_from_g) / (wd + wg)
    return _clamp(p_new, Z)


def make_value_net():
    inp = tf.keras.Input(shape=(1,)); h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=MARGIN):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


def run_mirror(seed, P, Zc, v0, vN, beta, use_A, use_B):
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
    def fit_step(vtarget_t, w_t):
        with tf.GradientTape() as t:
            L = tf.reduce_mean(w_t * tf.square(value(ZpeT) - vtarget_t))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L

    ones = tf.constant(np.ones((N_PE + 1, 1), np.float32))

    # warm start from the perturbation-slope policy-eval value (NO FD info)
    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    for _ in range(N_WARM):
        fit_step(v_ws_t, ones)

    for sweep in range(N_HOWARD):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)
        # (A) mirror pull of the slope used for controls toward the transport oracle.
        #     We first get the oracle slope by a transport solve at the CURRENT slope,
        #     reading back the well-conditioned slope, then mirror-mix, then RE-solve.
        v_pe0 = policy_eval_linear(P, Zpe, slope_pe, v0, vN)
        p_oracle = np.empty_like(v_pe0)
        dZ = Zpe[1] - Zpe[0]
        p_oracle[1:-1] = (v_pe0[2:] - v_pe0[:-2]) / (2.0 * dZ)
        p_oracle[0] = (v_pe0[1] - v_pe0[0]) / dZ; p_oracle[-1] = (v_pe0[-1] - v_pe0[-2]) / dZ
        p_oracle = _clamp(p_oracle, Zpe)
        if use_A and beta > 0.0:
            slope_used = mirror_pull_slope(slope_pe, p_oracle, Zpe, beta)
        else:
            slope_used = slope_pe
        v_pe = policy_eval_linear(P, Zpe, slope_used, v0, vN)
        vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
        # (B) barrier-curvature-weighted value fit (normalize to mean 1 so LR is comparable)
        if use_B:
            g = barrier_metric(slope_used, Zpe)
            w = (g / np.mean(g)).astype(np.float32).reshape(-1, 1)
            w_t = tf.constant(w)
        else:
            w_t = ones
        for _ in range(N_FIT):
            fit_step(vt, w_t)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = slope_nn(ZcT).numpy().ravel().astype(np.float64)
    p_final = _clamp(p_final, Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d


ARMS = [("baseline", 0.0, False, False),
        ("mirrorA", 0.25, True, False),
        ("mirrorA", 0.5, True, False),
        ("mirrorB", 0.0, False, True),
        ("mirrorAB", 0.25, True, True),
        ("mirrorAB", 0.5, True, True)]


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN, sl0, sl1 = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())}", flush=True)
        for arm, beta, uA, uB in ARMS:
            tag = f"{arm}(b={beta})"
            eids, eps = [], []
            for seed in SEEDS:
                p_t, id_t = run_mirror(seed, P, Zc, v0, vN, beta, uA, uB)
                e_id = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
                e_p = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
                eids.append(e_id); eps.append(e_p)
                results.append({"A_d": A_d, "arm": arm, "beta": beta, "seed": seed,
                                "err_id": e_id, "err_p": e_p})
            print(f"  {tag:16s} med|i_d-FD|={np.median(eids):.4e}  "
                  f"med|p-FD|={np.median(eps):.4e}  (seeds {[f'{x:.2e}' for x in eids]})", flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
