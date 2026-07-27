"""
COMBINED method: Howard routing (around the flat weak-id residual) + ENGD/Gauss-Newton
on the WELL-CONDITIONED inner LINEAR policy-evaluation solve.
(method key: howard-engd-inner-gn-solve)

BASELINE  (PLAIN-HOWARD): costate_foc_oracle_ab.run_treated. Each Howard sweep:
   (1) slope = autodiff(value net) -> controls via M.controls (FOC inversion).
   (2) FREEZE policy, numpy LINEAR upwind transport solve -> oracle value v_pe.
   (3) supervise value net toward v_pe by ADAM (N_FIT steps)  <-- the inner floor.

TREATED   (HOWARD-ENGD): IDENTICAL outer Howard loop, but the inner step (3) is
replaced by a damped Gauss-Newton / energy-NG solve of the FROZEN-POLICY LINEAR
residual, evaluated DIRECTLY on the net (no numpy oracle target -- oracle-free):

   R_lin(Z) = delta*log(c_froz) + (1-Z)*phi_d_froz + Z*phi_g_froz
              - delta*value(Z) + mu_froz(Z) * d/dZ value(Z)

with c_froz, phi_d_froz, phi_g_froz, mu_froz FROZEN (stop_gradient) from the
controls implied by the CURRENT net slope at the start of the sweep. R_lin is
LINEAR in the net output + its Z-gradient -> the GN model is EXACT (Muller-
Zeinhofer ICML23) and a few damped GN steps drive it to near single precision.

CRITICAL impl note: R_lin contains autodiff v' = d/dZ value, so the residual
Jacobian J = dR_lin/dtheta carries the MIXED second derivative
d/dtheta (d value / dZ). We use NESTED GradientTapes: an inner tape (watch Z) for
v', then tape.jacobian(R_lin, theta) over the whole graph so the mixed term is
captured. We VERIFY this with a finite-difference Jacobian check at startup.

We reuse engd_gauss_newton_ab.engd_gn's residual-space Woodbury Marquardt logic,
re-implemented locally for the LINEAR R_lin (vector-valued, frozen coefficients).

DIAGNOSTICS (validate the COMBINE thesis): per sweep we log the inner GN linear-
residual decay (expect fast / ~exact for the linear op) and kappa(J^T J) (expect
BOUNDED/modest here -- the well-conditioning, in contrast to the strong-residual
ENGD run where kappa blows up).

TRUE error vs FD (the ONLY metric): max|i_d - i_d_FD| over Z in [0.1,0.9], per regime.
Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python howard_engd_inner_gn_ab.py   (login OK).
"""
import json
import numpy as np
import tensorflow as tf

import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri
from costate_foc_oracle_ab import (
    build_case, policy_eval_linear, make_costate_net, run_treated as run_howard_plain,
)

tf.keras.backend.set_floatx("float32")

SEEDS = [4, 5, 6]
A_DS = [0.05, 0.13]
N_COLLO = 256
N_PE = 1000          # numpy warm-start policy-eval grid
N_GN = 256           # grid for the GN linear residual collocation (NxN dense solve)
N_HOWARD = 12
# GN inner-solve budget per sweep
GN_INNER_STEPS = 12
WARM_FIT = 1500
WARM_LR = 2e-3


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m, 1.0 / tf.maximum(Z, 1e-9) - m)


# ---------------------------------------------------------------------------
#  TREATED: Howard outer loop + Gauss-Newton inner LINEAR policy-eval solve
# ---------------------------------------------------------------------------

def run_treated_engd(seed, P, Zc, v0, vN, sl0, sl1, verbose=False):
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_costate_net()
    Zpe = np.linspace(0.0, 1.0, N_PE + 1).astype(np.float64)        # numpy warm-start grid
    Zgn = np.linspace(0.0, 1.0, N_GN + 1).astype(np.float64)        # GN residual collocation
    ZpeT = tf.constant(Zpe.reshape(-1, 1).astype(np.float32))
    ZgnT = tf.constant(Zgn.reshape(-1, 1).astype(np.float32))
    dl = P["delta"]

    def value(Zt):
        return (1 - Zt) * v0 + Zt * vN + Zt * (1 - Zt) * net(2.0 * Zt - 1.0)

    def value_and_slope(Zt):
        with tf.GradientTape() as t:
            t.watch(Zt); v = value(Zt)
        vp = clamp_tf(t.gradient(v, Zt), Zt)
        return v, vp

    def slope_nn(Zt):
        _, vp = value_and_slope(Zt)
        return vp

    # ---- inner LINEAR frozen-policy residual (R_lin), linear in net o/p + slope ----
    def lin_residual(flow_t, mu_t):
        # flow_t = delta*log c_froz + (1-Z)phi_d_froz + Z phi_g_froz  (frozen const)
        # mu_t   = frozen drift                                        (frozen const)
        # R_lin  = flow - delta*value + mu * v'    (v' = autodiff slope of the LIVE net)
        v, vp = value_and_slope(ZgnT)
        return flow_t - dl * v + mu_t * vp        # (N,1)

    vars_ = net.trainable_variables

    def jacobian_and_R(flow_t, mu_t):
        with tf.GradientTape() as tape:
            R = lin_residual(flow_t, mu_t)        # (N,1) -- nested tape inside for v'
            Rf = tf.reshape(R, [-1])              # (N,)
        jac = tape.jacobian(Rf, vars_, experimental_use_pfor=True)
        N = int(Rf.shape[0])
        cols = [tf.reshape(j, [N, -1]) for j in jac]
        J = tf.concat(cols, axis=1)               # (N,P)
        return J.numpy().astype(np.float64), Rf.numpy().astype(np.float64)

    def flat():
        return np.concatenate([v.numpy().ravel() for v in vars_]).astype(np.float64)

    def set_from_flat(x):
        i = 0
        for v in vars_:
            n = int(tf.size(v))
            v.assign(tf.reshape(tf.constant(x[i:i + n].astype(np.float32)), v.shape)); i += n

    # warm start (numpy oracle value from perturbation policy -- no FD info), Adam
    opt = tf.keras.optimizers.Adam(WARM_LR)
    @tf.function
    def warm_step(vt):
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value(ZpeT) - vt))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables), net.trainable_variables))
        return L
    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    for _ in range(WARM_FIT):
        warm_step(v_ws_t)

    kappa_log = []; res_log = []
    for sweep in range(N_HOWARD):
        # (1) FOC inversion from current net slope -> freeze the policy coefficients (on Zgn)
        slope_pe = slope_nn(ZgnT).numpy().ravel().astype(np.float64)
        slope_pe = _clamp(slope_pe, Zgn)
        i_d, i_g, c = M.controls(Zgn, slope_pe, P)
        phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        mu_froz = Zgn * (1.0 - Zgn) * (phi_g - phi_d)
        flow = dl * np.log(np.maximum(c, 1e-300)) + (1.0 - Zgn) * phi_d + Zgn * phi_g
        flow_t = tf.constant(flow.astype(np.float32).reshape(-1, 1))
        mu_t = tf.constant(mu_froz.astype(np.float32).reshape(-1, 1))

        # (2)+(3) GN solve of the FROZEN LINEAR residual (replaces inner Adam supervision)
        lam = 1e-3
        for it in range(GN_INNER_STEPS):
            J, R = jacobian_and_R(flow_t, mu_t)
            N = J.shape[0]
            f0 = 0.5 * float(R @ R)
            x0 = flat()
            G = J @ J.T
            accepted = False
            for _try in range(8):
                A = G + lam * np.eye(N)
                try:
                    y = np.linalg.solve(A, R)
                except np.linalg.LinAlgError:
                    lam *= 10.0; continue
                delta = -(J.T @ y)
                set_from_flat(x0 + delta)
                R_new = lin_residual(flow_t, mu_t).numpy().ravel().astype(np.float64)
                f_new = 0.5 * float(R_new @ R_new)
                pred = f0 - 0.5 * float((R + J @ delta) @ (R + J @ delta))
                rho = (f0 - f_new) / pred if pred > 1e-300 else -1.0
                if f_new < f0 and rho > 0:
                    accepted = True
                    lam = max(lam / 3.0, 1e-12) if rho > 0.75 else (min(lam * 2.0, 1e8) if rho < 0.25 else lam)
                    break
                else:
                    set_from_flat(x0); lam = min(lam * 10.0, 1e10)
            if not accepted:
                break
        # diagnostics for the LAST GN evaluation of this sweep
        J, R = jacobian_and_R(flow_t, mu_t)
        s = np.linalg.svd(J, compute_uv=False); s2 = s ** 2; s2 = s2[s2 > 1e-30]
        kappa = float(s2.max() / s2.min()) if len(s2) else np.nan
        rms = float(np.sqrt(np.mean(R ** 2)))
        kappa_log.append(kappa); res_log.append(rms)
        if verbose:
            print(f"      sweep={sweep:2d} R_lin rms={rms:.3e}  kappa(J^TJ)={kappa:.3e}", flush=True)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = slope_nn(ZcT).numpy().ravel().astype(np.float64)
    p_final = _clamp(p_final, Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d, dict(kappa=kappa_log, res=res_log)


# ---------------------------------------------------------------------------
#  Jacobian verification (mixed second derivative via nested tapes)
# ---------------------------------------------------------------------------

def verify_jacobian(P, v0, vN, seed=1):
    """FD check: confirm the analytic R_lin Jacobian (which carries the mixed
    d/dtheta d value/dZ term) matches a finite-difference Jacobian."""
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_costate_net()
    Zsmall = np.linspace(0.1, 0.9, 5).astype(np.float64)
    ZT = tf.constant(Zsmall.reshape(-1, 1).astype(np.float32))
    dl = P["delta"]

    def value(Zt):
        return (1 - Zt) * v0 + Zt * vN + Zt * (1 - Zt) * net(2.0 * Zt - 1.0)

    def lin_residual():
        with tf.GradientTape() as t:
            t.watch(ZT); v = value(ZT)
        vp = clamp_tf(t.gradient(v, ZT), ZT)
        # arbitrary frozen coeffs
        flow = tf.constant(np.linspace(0.5, 0.7, 5).astype(np.float32).reshape(-1, 1))
        mu = tf.constant(np.linspace(-0.02, 0.03, 5).astype(np.float32).reshape(-1, 1))
        return flow - dl * v + mu * vp

    vars_ = net.trainable_variables
    with tf.GradientTape() as tape:
        R = tf.reshape(lin_residual(), [-1])
    jac = tape.jacobian(R, vars_, experimental_use_pfor=True)
    cols = [tf.reshape(j, [int(R.shape[0]), -1]) for j in jac]
    J = tf.concat(cols, axis=1).numpy().astype(np.float64)

    x0 = np.concatenate([v.numpy().ravel() for v in vars_]).astype(np.float64)
    def set_from_flat(x):
        i = 0
        for v in vars_:
            n = int(tf.size(v))
            v.assign(tf.reshape(tf.constant(x[i:i + n].astype(np.float32)), v.shape)); i += n
    def Rof():
        return tf.reshape(lin_residual(), [-1]).numpy().astype(np.float64)
    # FD on a handful of params
    eps = 1e-3
    cols_fd = []
    idxs = [0, 5, 50, 100, len(x0) - 1]
    for j in idxs:
        xp = x0.copy(); xp[j] += eps; set_from_flat(xp); Rp = Rof()
        xm = x0.copy(); xm[j] -= eps; set_from_flat(xm); Rm = Rof()
        cols_fd.append((Rp - Rm) / (2 * eps))
    set_from_flat(x0)
    Jfd = np.stack(cols_fd, axis=1)
    Jan = J[:, idxs]
    rel = np.max(np.abs(Jan - Jfd)) / max(np.max(np.abs(Jfd)), 1e-12)
    return rel


# ---------------------------------------------------------------------------

def main():
    relerr = verify_jacobian(M.load_calibration("A_g_prime_prime"),
                             *M.boundary_values(M.load_calibration("A_g_prime_prime")))
    # NOTE: float32 FD noise floor ~1.5e-3 here; the SAME check in float64 gives ~1e-10
    # (mixed d/dtheta dv/dZ term confirmed captured via nested tapes), so threshold is loose.
    print(f"##### Jacobian FD-check (mixed 2nd-deriv, float32 FD) max rel err = {relerr:.3e} "
          f"({'PASS' if relerr < 5e-3 else 'FAIL -- biased GN!'})", flush=True)

    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN, sl0, sl1 = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior [{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())}", flush=True)
        for seed in SEEDS:
            # BASELINE = plain Howard (Adam inner supervision), SAME harness
            p_h, id_h = run_howard_plain(seed, P, Zc, v0, vN, sl0, sl1)
            # TREATED = Howard + GN inner linear solve
            p_t, id_t, diag = run_treated_engd(seed, P, Zc, v0, vN, sl0, sl1, verbose=(seed == SEEDS[0]))
            e_id_h = float(np.max(np.abs(id_h[mI] - id_fd_c[mI])))
            e_id_t = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
            e_p_h = float(np.max(np.abs(p_h[mI] - slope_fd[mI])))
            e_p_t = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
            kmax = float(np.nanmax(diag["kappa"])); rmin = float(np.nanmin(diag["res"]))
            row = {"A_d": A_d, "seed": seed,
                   "err_id_howard": e_id_h, "err_id_engd": e_id_t,
                   "err_p_howard": e_p_h, "err_p_engd": e_p_t,
                   "id_ratio": e_id_t / max(e_id_h, 1e-12),
                   "kappa_max": kmax, "res_min": rmin}
            results.append(row)
            print(f"  seed={seed} HOWARD(Adam) max|i_d-FD|={e_id_h:.4e}  max|p-FD|={e_p_h:.4e}", flush=True)
            print(f"  seed={seed} HOWARD-ENGD  max|i_d-FD|={e_id_t:.4e}  max|p-FD|={e_p_t:.4e}"
                  f"  -> id_ratio(ENGD/Howard)={row['id_ratio']:.3f}  kappa_max={kmax:.2e}  R_lin_min={rmin:.2e}", flush=True)

    print("\n##### MEDIANS over seeds", flush=True)
    for A_d in A_DS:
        rs = [r for r in results if r["A_d"] == A_d]
        med = lambda k: float(np.median([r[k] for r in rs]))
        print(f"  A_d={A_d}: HOWARD id={med('err_id_howard'):.4e}  ENGD id={med('err_id_engd'):.4e}"
              f"  id_ratio={med('id_ratio'):.3f}  kappa_max(med)={med('kappa_max'):.2e}", flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
