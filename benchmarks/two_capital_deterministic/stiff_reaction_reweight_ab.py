"""
Synthetic-stiff-reaction reweighting A/B test (method: synthetic-stiff-reaction-reweighting).

The deterministic two-capital benchmark has NO jump term. To unit-test the
"reaction-residual reweighting" idea (which targets the multiplicative jump-stiffness
axis present in the full model), we INJECT a synthetic stiff reaction term

        -c_reac(Z) * v

into BOTH the NN residual AND the FD operator, so FD ground truth is preserved.
c_reac(Z) is exp-ramped across Z to span ~2-3 decades (e.g. 0.01 .. 10), so the
effective discount delta + c_reac(Z) spans the operator spectrum across decades.

CONTROL arm : plain L2 of the strong residual.
TREATED arm : same residual but the (reaction-bearing) residual reweighted by
              1/(delta + c_reac(Z)) so stiff-reaction regions do not dominate
              the gradient.

Both arms: SAME seed / init / optimizer budget (Adam + L-BFGS).
TRUE ERROR is measured vs the reaction-augmented FD solve:
    max|i_d - i_d_FD|   and   ||v - v_fd||_inf
reported globally AND split into a LOW-stiffness and HIGH-stiffness subregion.
We NEVER compare the loss numbers (reweighting changes the loss scale).
"""
import os
import sys
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from scipy.linalg import solve_banded
import two_capital_model as M

# ----------------------------------------------------------------------------
# Synthetic stiff reaction c_reac(Z): exp-ramped, spans [CLO, CHI] across Z in (0,1)
# ----------------------------------------------------------------------------
DECADE_SPAN = float(os.environ.get("DECADE_SPAN", "3.0"))  # decades of stiffness
CLO = float(os.environ.get("CLO", "0.01"))                  # low end of c_reac
CHI = CLO * (10.0 ** DECADE_SPAN)                           # high end of c_reac


def c_reac_np(Z):
    Z = np.asarray(Z, dtype=np.float64)
    return CLO * (10.0 ** (DECADE_SPAN * Z))


def c_reac_tf(Z):
    return CLO * tf.pow(tf.constant(10.0, Z.dtype), DECADE_SPAN * Z)


# ----------------------------------------------------------------------------
# FD solve WITH the injected reaction -c_reac(Z)*v  (ground truth)
# ----------------------------------------------------------------------------
def _clamp_np(p, Z, margin=1e-7):
    lo = -1.0 / np.maximum(1.0 - Z, 1e-9) + margin
    hi = 1.0 / np.maximum(Z, 1e-9) - margin
    return np.clip(p, lo, hi)


def _tri(sub, diag, sup, rhs):
    ab = np.zeros((3, len(diag)))
    ab[0, 1:] = sup[:-1]; ab[1, :] = diag; ab[2, :-1] = sub[1:]
    return solve_banded((1, 1), ab, rhs)


def solve_fd_reac(P, n=4000, dtau=2.0, max_iter=200000, tol=1e-12):
    """Semi-implicit upwind false transient for v(Z) WITH -c_reac(Z)*v reaction.

    Boundary values are the one-capital values but with the discount delta
    replaced by delta + c_reac at the endpoints, so the BCs are consistent with
    the reaction-augmented operator (single-capital steady states).
    """
    Z = np.linspace(0.0, 1.0, n + 1); dZ = 1.0 / n
    dl = P["delta"]
    cr = c_reac_np(Z)
    # reaction-consistent boundary values: at Z=0 (dirty only) and Z=1 (green only)
    # the operator is  0 = delta*log c + phi - (delta + c_reac)*v  with mu=0.
    def one_cap_reac(A, alpha, Gamma, theta, dl_eff):
        c = dl * (1.0 + theta * A) / (theta * (dl + Gamma))  # control unaffected by reaction
        flow = dl * np.log(c) + alpha + Gamma * np.log(Gamma * theta * c / dl)
        return flow / dl_eff
    v0 = one_cap_reac(P["A_d"], P["alpha_d"], P["Gamma_d"], P["theta_d"], dl + cr[0])
    vN = one_cap_reac(P["A_g"], P["alpha_g"], P["Gamma_g"], P["theta_g"], dl + cr[-1])

    Abar = M.A_bar(Z, P)
    c_sym = dl * (1.0 + P["theta_d"] * Abar) / (P["theta_d"] * (dl + P["Gamma_d"]))
    v = (np.log(c_sym) + ((1 - Z) * P["alpha_d"] + Z * P["alpha_g"]) / dl
         + (P["Gamma_d"] / dl) * np.log(P["Gamma_d"] * P["theta_d"] * c_sym / dl))
    v[0], v[-1] = v0, vN
    idx = np.arange(1, n)
    cr_i = cr[idx]
    it = 0
    for it in range(max_iter):
        p = np.empty_like(v)
        p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
        p[0] = (v[1] - v[0]) / dZ; p[-1] = (v[-1] - v[-2]) / dZ
        p = _clamp_np(p, Z)
        i_d, i_g, c = M.controls(Z, p, P)
        phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        mu = Z * (1.0 - Z) * (phi_g - phi_d)
        flow = dl * np.log(np.maximum(c, 1e-300)) + (1.0 - Z) * phi_d + Z * phi_g
        mi = mu[idx]; fwd = mi > 0.0; coef = mi / dZ
        # discount is now (delta + c_reac) on the diagonal
        diag = (1.0 / dtau) + (dl + cr_i); sub = np.zeros(n - 1); sup = np.zeros(n - 1)
        diag[fwd] += coef[fwd]; sup[fwd] -= coef[fwd]
        diag[~fwd] -= coef[~fwd]; sub[~fwd] += coef[~fwd]
        rhs = v[idx] / dtau + flow[idx]; rhs[0] -= sub[0] * v0; rhs[-1] -= sup[-1] * vN
        v_new = _tri(sub, diag, sup, rhs)
        step = np.max(np.abs(v_new - v[idx]))
        v[idx] = v_new
        if step < tol:
            break
    p = np.empty_like(v)
    p[1:-1] = (v[2:] - v[:-2]) / (2.0 * dZ)
    p[0] = (v[1] - v[0]) / dZ; p[-1] = (v[-1] - v[-2]) / dZ
    slope = _clamp_np(p, Z)
    i_d, i_g, c = M.controls(Z, slope, P)
    return {"Z": Z, "v": v, "slope": slope, "i_d": i_d, "i_g": i_g, "c": c,
            "v0": v0, "vN": vN, "iters": it + 1}


# ----------------------------------------------------------------------------
# NN setup (matches closed_form_control_deinvest_diag.py)
# ----------------------------------------------------------------------------
def main():
    SEED = int(os.environ.get("SEED", "0"))
    tf.random.set_seed(SEED); np.random.seed(SEED)

    A_d = 0.05
    P = M.load_calibration("A_g_prime_prime"); P["A_d"] = A_d
    dl, A_g = P["delta"], P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]

    fd = solve_fd_reac(P, n=4000)
    v0, vN = fd["v0"], fd["vN"]

    Zg = np.linspace(0.02, 0.98, 256).reshape(-1, 1).astype(np.float32)
    Zt = tf.constant(Zg)
    cr_t = c_reac_tf(Zt)  # (256,1) float32
    v_fd = np.interp(Zg.ravel(), fd["Z"], fd["v"]).reshape(-1, 1).astype(np.float32)
    id_fd = np.interp(Zg.ravel(), fd["Z"], fd["i_d"])
    Zr = Zg.ravel()
    mI = (Zr >= 0.1) & (Zr <= 0.9)
    # stiffness subregions (by c_reac value): LOW = bottom third of Z, HIGH = top third
    mLOW = mI & (Zr <= 0.4)
    mHIGH = mI & (Zr >= 0.6)

    def make_net():
        inp = tf.keras.Input(shape=(1,)); h = inp
        for _ in range(3):
            h = tf.keras.layers.Dense(32, activation="tanh")(h)
        return tf.keras.Model(inp, tf.keras.layers.Dense(1)(h))

    def clamp(vp, Z, m=1e-4):
        return tf.clip_by_value(vp, -1.0 / tf.maximum(1 - Z, 1e-9) + m,
                                1.0 / tf.maximum(Z, 1e-9) - m)

    def value(net, Z):
        return (1 - Z) * v0 + Z * vN + Z * (1 - Z) * net(2.0 * Z - 1.0)

    def make_lg(net, reweight):
        @tf.function
        def lg():
            with tf.GradientTape() as outer:
                with tf.GradientTape() as inner:
                    inner.watch(Zt); v = value(net, Zt)
                vp = clamp(inner.gradient(v, Zt), Zt)
                q_d = 1 - Zt * vp; q_g = 1 + (1 - Zt) * vp
                Abar = (1 - Zt) * A_d + Zt * A_g
                c = dl * (Abar + (1 - Zt) / td + Zt / tg) / (dl + (1 - Zt) * Gd * q_d + Zt * Gg * q_g)
                i_d = Gd * c * q_d / dl - 1 / td; i_g = Gg * c * q_g / dl - 1 / tg
                phi_d = ad + Gd * tf.math.log(tf.maximum(1 + td * i_d, 1e-8))
                phi_g = ag + Gg * tf.math.log(tf.maximum(1 + tg * i_g, 1e-8))
                mu = Zt * (1 - Zt) * (phi_g - phi_d)
                # HJB residual WITH injected reaction -c_reac*v
                R = (dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v
                     + (1 - Zt) * phi_d + Zt * phi_g + mu * vp - cr_t * v)
                # reweight by 1/(delta + c_reac) so stiff regions don't dominate
                w = (dl + cr_t) if reweight else tf.ones_like(R)
                loss = tf.reduce_mean(tf.square(R / w))
            return loss, outer.gradient(loss, net.trainable_variables)
        return lg

    def adam(net, lg, steps, lr=2e-3):
        opt = tf.keras.optimizers.Adam(lr)
        L = None
        for _ in range(steps):
            L, g = lg(); opt.apply_gradients(zip(g, net.trainable_variables))
        return float(L)

    def lbfgs(net, lg, maxiter=4000):
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
                     options={"maxiter": maxiter, "maxfun": 2 * maxiter})
        setf(r.x); return float(r.fun)

    def errors(net):
        with tf.GradientTape() as inner:
            inner.watch(Zt); v = value(net, Zt)
        vp = clamp(inner.gradient(v, Zt), Zt).numpy().ravel()
        i_d, i_g, c = M.controls(Zr, vp, P)
        vnn = value(net, Zt).numpy().ravel()
        def idmax(m): return float(np.max(np.abs(i_d[m] - id_fd[m])))
        def vmax(m): return float(np.max(np.abs(vnn[m] - v_fd.ravel()[m])))
        return {
            "id_all": idmax(mI), "id_low": idmax(mLOW), "id_high": idmax(mHIGH),
            "v_all": vmax(mI), "v_low": vmax(mLOW), "v_high": vmax(mHIGH),
        }

    print(f"=== SEED={SEED}  DECADE_SPAN={DECADE_SPAN}  c_reac in [{CLO:.3g}, {CHI:.3g}] ===", flush=True)
    print(f"FD truth: i_d in [{id_fd[mI].min():+.4f}, {id_fd[mI].max():+.4f}]  "
          f"v in [{v_fd.ravel()[mI].min():.3f},{v_fd.ravel()[mI].max():.3f}]  iters={fd['iters']}\n", flush=True)

    ADAM_STEPS = int(os.environ.get("ADAM_STEPS", "5000"))
    LBFGS_ITERS = int(os.environ.get("LBFGS_ITERS", "4000"))

    results = {}
    for arm, reweight in [("CONTROL (plain L2)", False), ("TREATED (reweighted)", True)]:
        tf.random.set_seed(SEED); np.random.seed(SEED)
        net = make_net()
        lg = make_lg(net, reweight)
        adam(net, lg, ADAM_STEPS)
        lbfgs(net, lg, LBFGS_ITERS)
        e = errors(net)
        results[arm] = e
        print(f"{arm}", flush=True)
        print(f"   max|i_d-FD|  all={e['id_all']:.4e}  LOW-stiff={e['id_low']:.4e}  HIGH-stiff={e['id_high']:.4e}", flush=True)
        print(f"   ||v-v_fd||inf all={e['v_all']:.4e}  LOW-stiff={e['v_low']:.4e}  HIGH-stiff={e['v_high']:.4e}\n", flush=True)

    c = results["CONTROL (plain L2)"]; t = results["TREATED (reweighted)"]
    print("=== SUMMARY (true error vs FD; lower is better) ===", flush=True)
    for k, lbl in [("id_all", "max|i_d-FD| all"), ("id_high", "max|i_d-FD| HIGH-stiff"),
                   ("id_low", "max|i_d-FD| LOW-stiff"), ("v_high", "||v-vfd|| HIGH-stiff"),
                   ("v_low", "||v-vfd|| LOW-stiff")]:
        cc, tt = c[k], t[k]
        chg = (tt - cc) / cc * 100 if cc != 0 else float('nan')
        verdict = "BETTER" if tt < cc * 0.97 else ("WORSE" if tt > cc * 1.03 else "~same")
        print(f"  {lbl:24s} control={cc:.4e}  treated={tt:.4e}  ({chg:+6.1f}%)  {verdict}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
