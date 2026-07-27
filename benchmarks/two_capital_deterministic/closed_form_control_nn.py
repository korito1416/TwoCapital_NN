"""
Step 0 test of the CONTROL-ELIMINATION (semi-analytic controls) method, on the
deterministic two-capital benchmark where the FD solver is ground truth.

Idea under test: do NOT train control networks. Train ONLY a value net v(Z); obtain
the controls i_d, i_g in CLOSED FORM from v'(Z) via the FOC map (two_capital_model
.controls/.consumption), with the admissibility slope clamp (q_d,q_g>0). The only
loss is the HJB residual (preconditioned by 1/(|mu|+eps)); BCs are hard-constrained.

The decisive question (the user's doubt): the HJB has no analytic solution for v, but
the CONTROLS given v' are analytic. Can a smooth NN value function, trained on the
HJB residual with substituted closed-form controls, RECOVER the FD solution -- INCLUDING
the sign of i_d where FD de-invests? We test a de-invest calibration (A_d=0.05 -> FD
i_d<0) and the base (A_d=0.1303 -> FD i_d>0), and compare i_d(Z), i_g(Z), v'(Z) to FD.

Small 1-D problem; runs quickly on CPU.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_capital_model as M
from theta_sensitivity import solve_fd

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)
tf.random.set_seed(0); np.random.seed(0)


def make_value_net(n_units=32, n_layers=3):
    inp = tf.keras.Input(shape=(1,))
    h = inp
    for _ in range(n_layers):
        h = tf.keras.layers.Dense(n_units, activation="tanh")(h)
    out = tf.keras.layers.Dense(1)(h)
    return tf.keras.Model(inp, out)


def slope_clamp(vp, Z, margin=1e-4):
    lo = -1.0 / tf.maximum(1.0 - Z, 1e-9) + margin
    hi = 1.0 / tf.maximum(Z, 1e-9) - margin
    return tf.clip_by_value(vp, lo, hi)


def run_case(A_d, tag, n_iter=30000, lr=1e-3):
    P = M.load_calibration("A_g_prime_prime")
    P["A_d"] = float(A_d)
    dl, A_g = P["delta"], P["A_g"]
    Gd, Gg, td, tg = P["Gamma_d"], P["Gamma_g"], P["theta_d"], P["theta_g"]
    ad, ag = P["alpha_d"], P["alpha_g"]

    # FD ground truth
    fd = solve_fd(P, n=4000)
    v0, vN = M.boundary_values(P)

    net = make_value_net()
    # collocation grid (fixed, 1-D)
    Zg = np.linspace(0.02, 0.98, 256).reshape(-1, 1).astype(np.float32)
    Zt = tf.constant(Zg)

    def value_and_slope(Z):
        with tf.GradientTape() as tp:
            tp.watch(Z)
            raw = net(2.0 * Z - 1.0)
            # hard BC: v(0)=v0, v(1)=vN exactly
            v = (1.0 - Z) * v0 + Z * vN + Z * (1.0 - Z) * raw
        vp = tp.gradient(v, Z)
        return v, vp

    eps = 5e-3

    @tf.function
    def train_step():
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                inner.watch(Zt)
                raw = net(2.0 * Zt - 1.0)
                v = (1.0 - Zt) * v0 + Zt * vN + Zt * (1.0 - Zt) * raw
            vp = inner.gradient(v, Zt)
            vp = slope_clamp(vp, Zt)
            q_d = 1.0 - Zt * vp
            q_g = 1.0 + (1.0 - Zt) * vp
            Abar = (1.0 - Zt) * P["A_d"] + Zt * A_g
            c = dl * (Abar + (1.0 - Zt) / td + Zt / tg) / (
                dl + (1.0 - Zt) * Gd * q_d + Zt * Gg * q_g)
            i_d = Gd * c * q_d / dl - 1.0 / td
            i_g = Gg * c * q_g / dl - 1.0 / tg
            phi_d = ad + Gd * tf.math.log(tf.maximum(1.0 + td * i_d, 1e-8))
            phi_g = ag + Gg * tf.math.log(tf.maximum(1.0 + tg * i_g, 1e-8))
            mu = Zt * (1.0 - Zt) * (phi_g - phi_d)
            R = dl * tf.math.log(tf.maximum(c, 1e-8)) - dl * v \
                + (1.0 - Zt) * phi_d + Zt * phi_g + mu * vp
            loss = tf.reduce_mean(tf.square(R / (tf.abs(mu) + eps)))
        g = outer.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(g, net.trainable_variables))
        return loss

    opt = tf.keras.optimizers.Adam(lr)
    for it in range(n_iter):
        L = train_step()
        if it % 5000 == 0:
            print(f"[{tag}] iter {it:6d}  HJB-loss={float(L):.3e}", flush=True)

    # evaluate NN controls vs FD
    v, vp = value_and_slope(Zt)
    vp = slope_clamp(vp, Zt).numpy().ravel()
    Zf = Zg.ravel()
    i_d_nn, i_g_nn, c_nn = M.controls(Zf, vp, P)
    # FD reference interpolated to grid
    i_d_fd = np.interp(Zf, fd["Z"], fd["i_d"])
    i_g_fd = np.interp(Zf, fd["Z"], fd["i_g"])
    vp_fd = np.interp(Zf, fd["Z"], fd["slope"])

    # metrics on interior
    m = (Zf >= 0.1) & (Zf <= 0.9)
    err_id = np.max(np.abs(i_d_nn[m] - i_d_fd[m]))
    err_vp = np.max(np.abs(vp[m] - vp_fd[m]))
    fd_deinv = (i_d_fd[m] < 0).any()
    nn_deinv = (i_d_nn[m] < 0).any()
    sign_ok = np.mean(np.sign(i_d_nn[m]) == np.sign(i_d_fd[m]))
    print(f"[{tag}] A_d={A_d}: max|i_d_nn-i_d_fd|={err_id:.4f}  max|v'-v'_fd|={err_vp:.4f}")
    print(f"[{tag}]   i_d range: NN [{i_d_nn[m].min():+.4f},{i_d_nn[m].max():+.4f}]  FD [{i_d_fd[m].min():+.4f},{i_d_fd[m].max():+.4f}]")
    print(f"[{tag}]   FD de-invests: {fd_deinv}   NN de-invests: {nn_deinv}   sign-match: {sign_ok*100:.0f}%")
    return dict(tag=tag, A_d=A_d, Z=Zf, i_d_nn=i_d_nn, i_g_nn=i_g_nn, vp_nn=vp,
                i_d_fd=i_d_fd, i_g_fd=i_g_fd, vp_fd=vp_fd,
                err_id=err_id, err_vp=err_vp, fd_deinv=fd_deinv, nn_deinv=nn_deinv)


def main():
    cases = [run_case(0.05, "deinvest"), run_case(0.1303, "base")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for r, row in zip(cases, ax):
        m = (r["Z"] >= 0.1) & (r["Z"] <= 0.9)
        for a, key, fdkey, lab in zip(row, ["i_d_nn", "i_g_nn", "vp_nn"],
                                      ["i_d_fd", "i_g_fd", "vp_fd"],
                                      [r"$i^d$", r"$i^g$", r"$v'$"]):
            a.plot(r["Z"][m], r[fdkey][m], "k-", lw=2.4, label="FD (truth)")
            a.plot(r["Z"][m], r[key][m], "r--", lw=1.8, label="closed-form-control NN")
            a.axhline(0, color="grey", lw=0.7, ls=":")
            a.set_xlabel("Z"); a.set_ylabel(lab)
            a.set_title(f"{r['tag']} (A_d={r['A_d']}): {lab}")
            a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("Closed-form-control NN vs FD ground truth (does it recover de-invest?)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OD, "closed_form_control_nn_vs_fd.png")
    fig.savefig(p, dpi=150); print("saved", p)

    print("\n==== VERDICT ====")
    for r in cases:
        ok = (r["err_id"] < 0.01) and (r["nn_deinv"] == r["fd_deinv"])
        print(f"  {r['tag']:9s} A_d={r['A_d']}: i_d err={r['err_id']:.4f}  "
              f"de-invest FD={r['fd_deinv']} NN={r['nn_deinv']}  -> {'PASS' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
