"""
mu_Z-GATED HOWARD / STRONG-RESIDUAL BLEND
(method key: mu-gated-howard-strongresidual-blend)

GOAL: fix the A_d=0.13 well-conditioned REGRESSION of plain Howard
(id_ratio median 3.45x vs the strong-residual control) WITHOUT losing the
A_d=0.05 de-invest-pocket win (~1.5e-3 median where the strong residual fails
at ~1e-1 because the residual is FLAT in v').

MECHANISM (combine two CONSISTENT operators that share the SAME fixed point):
  - Howard / FOC-transport oracle  : WELL-conditioned where mu_Z -> 0 (the pocket),
        because it routes v' through the FOC inversion + sign-definite LINEAR
        transport (the FD success path). But it OVER-SMOOTHS where mu_Z is large.
  - Strong-residual L2(R)          : already ~3e-4 where mu_Z is large (A_d=0.13),
        but FLAT (uninformative) where mu_Z -> 0.

  Define a gate from the FROZEN drift  w(Z) = eps / (mu(Z)^2 + eps)
      w -> 1  where mu -> 0   (pocket)     -> use HOWARD value target
      w -> 0  where mu large  (well-cond)  -> use STRONG RESIDUAL
  ONE value net (same boundary ansatz, slope by autodiff) trained on
      L = mean[ w * (value - v_pe_oracle)^2 / s_v^2 ]  +  lam_R * mean[ (1-w) * R^2 ]
  Both terms are minimized by the TRUE solution, so the blend is unbiased; the
  gate only decides WHICH well-conditioned operator supervises each Z.

ARMS (same harness/seed/budget as costate_foc_oracle_ab.py):
  (i)   plain-Howard      = run_treated  (value net + autodiff slope + PE oracle)
  (ii)  strong-residual   = run_control  (single value net on L2(R), Adam+L-BFGS)
  (iii) mu-gated blend     = run_gated   (this method)

TRUE ERROR (ONLY metric): max|i_d - i_d_FD| over Z in [0.1,0.9], per regime.
PASS = gated matches strong-residual at A_d=0.13 (id err ~ control, ratio<=~1.2)
       AND retains the pocket win at A_d=0.05 (median < 1.5e-3).

Run: srun --account=pi-lhansen --partition=caslake --time=0:25:00 \
        --cpus-per-task=4 --mem=8G python mu_gated_howard_ab.py   (login OK).
"""
import json
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

import two_capital_model as M
from theta_sensitivity import solve_fd, _clamp, _tri
from costate_foc_oracle_ab import (
    build_case, policy_eval_linear, run_treated, run_control,
    N_PE, ADAM_LR,
)

tf.keras.backend.set_floatx("float32")

SEEDS = [1, 2, 3]
A_DS = [0.05, 0.13]
N_COLLO = 256

# gated-method hyperparameters
N_HOWARD_G = 12       # Howard sweeps (same as plain Howard)
N_FIT_G = 1500        # supervised steps per sweep
LAM_R = 1.0           # weight on the strong-residual term
EPS_SWEEP = [None]    # None => auto-calibrate eps from the median mu^2 per regime


def make_value_net():
    inp = tf.keras.Input(shape=(1,))
    h = inp
    for _ in range(3):
        h = tf.keras.layers.Dense(32, activation="tanh")(h)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))


def clamp_tf(vp, Z, m=1e-4):
    return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m,
                            1.0 / tf.maximum(Z, 1e-9) - m)


def run_gated(seed, P, Zc, v0, vN, eps=None, verbose=False):
    """mu_Z-gated blend of the Howard value-target loss and the strong-residual
    loss, on a SINGLE value net (boundary ansatz, slope by autodiff).

    Each Howard sweep:
      (1) slope = autodiff(value net)  ->  controls via M.controls (FOC inversion)
      (2) FREEZE policy, LINEAR upwind PE solve -> oracle value target v_pe(Z)
      (3) compute the FROZEN drift mu(Z) on the PE grid -> gate w(Z)=eps/(mu^2+eps)
      (4) train the value net on  w*(value - v_pe)^2/s_v^2 + lam_R*(1-w)*R^2
    The R term is the SAME strong residual as run_control, evaluated through the
    SAME value net (autodiff slope), so both operators are consistent.
    """
    tf.random.set_seed(seed); np.random.seed(seed)
    net = make_value_net()
    Zpe = np.linspace(0.0, 1.0, N_PE + 1).astype(np.float64)
    ZpeT = tf.constant(Zpe.reshape(-1, 1).astype(np.float32))

    dl = P["delta"]; A_d = P["A_d"]; A_g = P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]

    def value(Zt):
        return (1 - Zt) * v0 + Zt * vN + Zt * (1 - Zt) * net(2.0 * Zt - 1.0)

    def slope_nn(Zt):
        with tf.GradientTape() as t:
            t.watch(Zt); v = value(Zt)
        return clamp_tf(t.gradient(v, Zt), Zt)

    opt = tf.keras.optimizers.Adam(ADAM_LR)

    # ----- warm start from the perturbation-slope PE value (no FD info) -----
    pert = M.perturbation_slope(Zpe, P)
    v_ws = policy_eval_linear(P, Zpe, pert, v0, vN)
    v_ws_t = tf.constant(v_ws.astype(np.float32).reshape(-1, 1))
    s_v = float(np.std(v_ws)) + 1e-6   # value scale for normalizing the howard term

    @tf.function
    def ws_step(vt):
        with tf.GradientTape() as t:
            L = tf.reduce_mean(tf.square(value(ZpeT) - vt))
        opt.apply_gradients(zip(t.gradient(L, net.trainable_variables),
                                net.trainable_variables))
        return L
    for _ in range(1500):
        ws_step(v_ws_t)

    @tf.function
    def gated_step(vt, wt):
        """Blend on the PE grid: howard value-fit (weight w) + strong residual (1-w)."""
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                inner.watch(ZpeT); v = value(ZpeT)
            vp = clamp_tf(inner.gradient(v, ZpeT), ZpeT)
            # strong residual R through the SAME net
            q_d = 1 - ZpeT * vp; q_g = 1 + (1 - ZpeT) * vp
            Abar = (1 - ZpeT) * A_d + ZpeT * A_g
            c = dl * (Abar + (1 - ZpeT) / td + ZpeT / tg) / \
                (dl + (1 - ZpeT) * Gd * q_d + ZpeT * Gg * q_g)
            i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
            phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
            phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
            mu = ZpeT * (1 - ZpeT) * (phi_g - phi_d)
            R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v + \
                (1 - ZpeT) * phi_d + ZpeT * phi_g + mu * vp
            L_how = tf.reduce_mean(wt * tf.square((v - vt) / s_v))
            L_res = LAM_R * tf.reduce_mean((1.0 - wt) * tf.square(R))
            L = L_how + L_res
        opt.apply_gradients(zip(outer.gradient(L, net.trainable_variables),
                                net.trainable_variables))
        return L_how, L_res

    gate_stat = {}
    for sweep in range(N_HOWARD_G):
        slope_pe = slope_nn(ZpeT).numpy().ravel().astype(np.float64)  # (1)
        v_pe = policy_eval_linear(P, Zpe, slope_pe, v0, vN)           # (2)
        # (3) frozen drift mu on the PE grid -> gate
        i_d_f, i_g_f, _ = M.controls(Zpe, _clamp(slope_pe, Zpe), P)
        phi_d_f = M.phi(i_d_f, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g_f = M.phi(i_g_f, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        mu_f = Zpe * (1.0 - Zpe) * (phi_g_f - phi_d_f)
        if eps is None:
            # auto-calibrate: eps = median(mu^2) over the interior so the gate
            # separates "small mu" (pocket) from "large mu" (well-cond)
            mInt = (Zpe >= 0.05) & (Zpe <= 0.95)
            eps_use = float(np.median(mu_f[mInt] ** 2)) + 1e-30
        else:
            eps_use = eps
        w = eps_use / (mu_f ** 2 + eps_use)
        gate_stat = {"min_abs_mu": float(np.min(np.abs(mu_f[(Zpe>=0.1)&(Zpe<=0.9)]))),
                     "max_abs_mu": float(np.max(np.abs(mu_f[(Zpe>=0.1)&(Zpe<=0.9)]))),
                     "eps": eps_use,
                     "w_mean": float(np.mean(w)),
                     "w_min": float(np.min(w)), "w_max": float(np.max(w))}
        vt = tf.constant(v_pe.astype(np.float32).reshape(-1, 1))
        wt = tf.constant(w.astype(np.float32).reshape(-1, 1))
        for _ in range(N_FIT_G):                                     # (4)
            gated_step(vt, wt)

    ZcT = tf.constant(Zc.reshape(-1, 1).astype(np.float32))
    p_final = slope_nn(ZcT).numpy().ravel().astype(np.float64)
    p_final = _clamp(p_final, Zc)
    i_d, i_g, c = M.controls(Zc, p_final, P)
    return p_final, i_d, gate_stat


def main():
    results = []
    for A_d in A_DS:
        P, fd, Zc, slope_fd, id_fd_c, v0, vN, sl0, sl1 = build_case(A_d)
        mI = (Zc >= 0.1) & (Zc <= 0.9)
        print(f"\n##### A_d={A_d}  FD i_d interior "
              f"[{id_fd_c[mI].min():+.5f},{id_fd_c[mI].max():+.5f}] "
              f"deinvest={bool((id_fd_c[mI]<0).any())} | "
              f"slope [{slope_fd[mI].min():.3f},{slope_fd[mI].max():.3f}]", flush=True)
        for seed in SEEDS:
            p_c, id_c = run_control(seed, P, Zc, v0, vN)
            p_t, id_t = run_treated(seed, P, Zc, v0, vN, sl0, sl1)
            p_g, id_g, gs = run_gated(seed, P, Zc, v0, vN)
            e_id_c = float(np.max(np.abs(id_c[mI] - id_fd_c[mI])))
            e_id_t = float(np.max(np.abs(id_t[mI] - id_fd_c[mI])))
            e_id_g = float(np.max(np.abs(id_g[mI] - id_fd_c[mI])))
            e_p_c = float(np.max(np.abs(p_c[mI] - slope_fd[mI])))
            e_p_t = float(np.max(np.abs(p_t[mI] - slope_fd[mI])))
            e_p_g = float(np.max(np.abs(p_g[mI] - slope_fd[mI])))
            row = {"A_d": A_d, "seed": seed,
                   "err_id_strongres": e_id_c, "err_id_howard": e_id_t,
                   "err_id_gated": e_id_g,
                   "err_p_strongres": e_p_c, "err_p_howard": e_p_t,
                   "err_p_gated": e_p_g,
                   "gated_vs_howard": e_id_g / max(e_id_t, 1e-12),
                   "gated_vs_strongres": e_id_g / max(e_id_c, 1e-12),
                   "gate": gs}
            results.append(row)
            print(f"  seed={seed} STRONGRES max|i_d-FD|={e_id_c:.4e}", flush=True)
            print(f"  seed={seed} HOWARD    max|i_d-FD|={e_id_t:.4e}", flush=True)
            print(f"  seed={seed} GATED     max|i_d-FD|={e_id_g:.4e}"
                  f"  (g/howard={row['gated_vs_howard']:.3f} "
                  f"g/strongres={row['gated_vs_strongres']:.3f})", flush=True)
            print(f"           gate: min|mu|={gs['min_abs_mu']:.3e} "
                  f"max|mu|={gs['max_abs_mu']:.3e} eps={gs['eps']:.3e} "
                  f"w[min,mean,max]=[{gs['w_min']:.3f},{gs['w_mean']:.3f},"
                  f"{gs['w_max']:.3f}]", flush=True)

    print("\n##### MEDIANS over seeds", flush=True)
    for A_d in A_DS:
        rs = [r for r in results if r["A_d"] == A_d]
        med = lambda k: float(np.median([r[k] for r in rs]))
        print(f"  A_d={A_d}: STRONGRES id={med('err_id_strongres'):.4e}  "
              f"HOWARD id={med('err_id_howard'):.4e}  "
              f"GATED id={med('err_id_gated'):.4e}", flush=True)
    print("\nJSON " + json.dumps(results), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
