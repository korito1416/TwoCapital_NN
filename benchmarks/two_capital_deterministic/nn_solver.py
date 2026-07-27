"""
Deep-Galerkin (DGM-PIA style) neural-network solver for the deterministic
two-capital HJB, mirroring the parent project's approach but for the reduced
1-D problem in Z. A single network approximates v(Z); the slope v'(Z) is taken
by automatic differentiation, and the controls/consumption are the closed-form
FOC expressions (so no separate control network is needed here). The loss is the
mean-squared HJB residual plus soft anchoring to the one-capital boundary values.
"""

import numpy as np
import tensorflow as tf

import two_capital_model as M


def _build_net(width=64, depth=4):
    inp = tf.keras.Input(shape=(1,))
    h = inp
    for _ in range(depth):
        h = tf.keras.layers.Dense(width, activation="tanh")(h)
    out = tf.keras.layers.Dense(1)(h)
    return tf.keras.Model(inp, out)


class _WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then cosine decay to final_frac*base (mirrors the project)."""

    def __init__(self, base, total, warmup_frac=0.05, final_frac=0.005):
        self.base = float(base)
        self.total = int(total)
        self.warmup = max(int(warmup_frac * total), 1)
        self.final = self.base * float(final_frac)

    def __call__(self, step):
        s = tf.cast(step, tf.float32)
        warm = self.base * s / float(self.warmup)
        prog = tf.clip_by_value((s - self.warmup) / float(max(self.total - self.warmup, 1)), 0.0, 1.0)
        cos = self.final + 0.5 * (self.base - self.final) * (1.0 + tf.cos(np.pi * prog))
        return tf.where(s < self.warmup, warm, cos)

    def get_config(self):
        return {"base": self.base, "total": self.total, "warmup": self.warmup, "final": self.final}


def _tf_pieces(Z, slope, p):
    """TF version of consumption/controls/phi/drift (vectorized, column tensors)."""
    q_d = 1.0 - Z * slope
    q_g = 1.0 + (1.0 - Z) * slope
    Abar = (1.0 - Z) * p["A_d"] + Z * p["A_g"]
    num = p["delta"] * (Abar + (1.0 - Z) / p["theta_d"] + Z / p["theta_g"])
    den = p["delta"] + (1.0 - Z) * p["Gamma_d"] * q_d + Z * p["Gamma_g"] * q_g
    c = num / den
    i_d = p["Gamma_d"] * c * q_d / p["delta"] - 1.0 / p["theta_d"]
    i_g = p["Gamma_g"] * c * q_g / p["delta"] - 1.0 / p["theta_g"]
    phi_d = p["alpha_d"] + p["Gamma_d"] * tf.math.log(tf.maximum(1.0 + p["theta_d"] * i_d, 1e-12))
    phi_g = p["alpha_g"] + p["Gamma_g"] * tf.math.log(tf.maximum(1.0 + p["theta_g"] * i_g, 1e-12))
    drift = Z * (1.0 - Z) * (phi_g - phi_d)
    return c, i_d, i_g, phi_d, phi_g, drift


def solve_nn(p, iters=80000, batch=1024, lr=2e-3, width=64, depth=4,
             bc_weight=50.0, seed=0, verbose=False):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    net = _build_net(width, depth)
    sched = _WarmupCosine(lr, iters)
    opt = tf.keras.optimizers.Adam(sched, global_clipnorm=1.0)
    v0, vN = M.boundary_values(p)
    pf = {k: (tf.constant(v, tf.float32) if isinstance(v, float) else v) for k, v in p.items()
          if k != "a_g_choice"}
    z_bc = tf.constant([[0.0], [1.0]], dtype=tf.float32)
    v_bc = tf.constant([[v0], [vN]], dtype=tf.float32)

    @tf.function
    def train_step():
        z = tf.random.uniform((batch, 1), 0.0, 1.0, dtype=tf.float32)
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape_z:
                tape_z.watch(z)
                v = net(z, training=True)
            slope = tape_z.gradient(v, z)
            c, i_d, i_g, phi_d, phi_g, drift = _tf_pieces(z, slope, pf)
            resid = (pf["delta"] * tf.math.log(tf.maximum(c, 1e-12)) - pf["delta"] * v
                     + (1.0 - z) * phi_d + z * phi_g + drift * slope)
            loss_pde = tf.reduce_mean(resid ** 2)
            v_bc_pred = net(z_bc, training=True)
            loss_bc = tf.reduce_mean((v_bc_pred - v_bc) ** 2)
            loss = loss_pde + bc_weight * loss_bc
        grads = tape.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(grads, net.trainable_variables))
        return loss_pde, loss_bc

    for it in range(iters):
        lp, lb = train_step()
        if verbose and (it % 2000 == 0 or it == iters - 1):
            print(f"  [NN] iter {it:6d}  pde_rmse={float(lp)**0.5:.3e}  bc={float(lb):.3e}")

    # Evaluate on a dense grid with autodiff slope.
    Zg = np.linspace(0.0, 1.0, 4001, dtype=np.float32).reshape(-1, 1)
    zt = tf.constant(Zg)
    with tf.GradientTape() as tape_z:
        tape_z.watch(zt)
        vt = net(zt, training=False)
    slope_t = tape_z.gradient(vt, zt)
    Z = Zg.ravel().astype(np.float64)
    v = vt.numpy().ravel().astype(np.float64)
    slope = slope_t.numpy().ravel().astype(np.float64)
    i_d, i_g, c = M.controls(Z, slope, p)
    resid = M.hjb_residual(Z, v, slope, p)
    return {
        "method": "NN", "Z": Z, "v": v, "slope": slope,
        "i_d": i_d, "i_g": i_g, "c": c, "ratio": i_g / i_d,
        "residual": resid, "max_abs_residual": float(np.max(np.abs(resid[1:-1]))),
    }


if __name__ == "__main__":
    P = M.load_calibration("A_g_prime_prime")
    out = solve_nn(P, verbose=True)
    print(f"NN done, max|residual|={out['max_abs_residual']:.3e}")
    for z in (0.1, 0.3, 0.5, 0.7, 0.9):
        k = int(z * (len(out["Z"]) - 1))
        print(f"  Z={z:.1f}  i_d={out['i_d'][k]:.5f}  i_g={out['i_g'][k]:.5f}  i_g/i_d={out['ratio'][k]:.4f}")
