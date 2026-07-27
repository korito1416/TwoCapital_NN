"""
DGM-PIA neural solver for the two-capital-with-shocks HJB (second-order). Mirrors
nn_dgm_solver.py (deterministic) but the value-network residual now carries the
v'' diffusion term and the sigma corrections; the FOCs/controls are unchanged.
Reuses the project's FeedForwardSubNet / DGM gated net, the 3-loss two-step DGM-PIA,
optional FD-supervised warm start, optional preconditioning, and L-BFGS polish.
"""
import os
import sys

import numpy as np
import tensorflow as tf

import two_capital_shock_model as M
# shared NN utilities from the deterministic benchmark (on path via M -> DET)
_DET = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "two_capital_deterministic"))
if _DET not in sys.path:
    sys.path.insert(0, _DET)
from nn_dgm_solver import subnet_class           # noqa: E402
from feedforward_subnet import setup_optimizers, stratified_uniform  # noqa: E402
from params import investment_rate_activation     # noqa: E402


class DGMShock:
    def __init__(self, P, num_iterations=200000, batch_size=256, lr_v=1e-4, lr_c=4e-3,
                 num_neurons=32, num_layers=4, z_min=0.01, z_max=0.99, bc_weight=1.0,
                 arch="forwardnet", precond=False, precond_eps=5e-3):
        self.P = P; self.bs = batch_size; self.z_min, self.z_max = z_min, z_max
        self.bc_weight = bc_weight; self.arch = arch
        self.precond = precond; self.precond_eps = precond_eps
        self.v0, self.vN = M.boundary_values(P)
        FFN = subnet_class(arch)
        common = dict(num_hiddens=[num_neurons] * num_layers, use_bias=True, dim=1)
        self.v_nn = FFN({**common, "activation": "swish", "final_activation": None, "nn_name": "v_nn"})
        self.i_d_nn = FFN({**common, "activation": "tanh",
                           "final_activation": investment_rate_activation(P["theta_d"]), "nn_name": "i_d_nn"})
        self.i_g_nn = FFN({**common, "activation": "tanh",
                           "final_activation": investment_rate_activation(P["theta_g"]), "nn_name": "i_g_nn"})
        for net in (self.v_nn, self.i_d_nn, self.i_g_nn):
            net.build((None, 1))
        opt_params = {"learning_rates": [lr_v, lr_c], "num_iterations": num_iterations,
                      "learning_rate_schedule_type": "warmup_cosine",
                      "extra": {"warmup_steps": max(1, num_iterations // 100), "min_lr": 1e-6},
                      "gradient_clip_norm": 1.0}
        setup_optimizers(opt_params)
        self.opt = opt_params["optimizers"]

    def sample(self):
        return stratified_uniform(self.z_min, self.z_max, self.bs)

    def pde_rhs(self, Z):
        P = self.P
        with tf.GradientTape() as t2:
            t2.watch(Z)
            with tf.GradientTape() as t1:
                t1.watch(Z)
                v = self.v_nn(Z)
            vp = t1.gradient(v, Z)
        vpp = t2.gradient(vp, Z)
        i_d = self.i_d_nn(Z); i_g = self.i_g_nn(Z)
        c = (1.0 - Z) * (P["A_d"] - i_d) + Z * (P["A_g"] - i_g)
        inside_d = 1.0 + P["theta_d"] * i_d
        inside_g = 1.0 + P["theta_g"] * i_g
        phi_d = P["alpha_d"] + P["Gamma_d"] * tf.math.log(tf.maximum(inside_d, 1e-8))
        phi_g = P["alpha_g"] + P["Gamma_g"] * tf.math.log(tf.maximum(inside_g, 1e-8))
        sd2, sg2 = P["sigma_d"] ** 2, P["sigma_g"] ** 2
        logK_drift = (1.0 - Z) * phi_d + Z * phi_g - 0.5 * (sd2 * (1.0 - Z) ** 2 + sg2 * Z ** 2)
        v1 = (phi_g - phi_d + (1.0 - Z) * sd2 - Z * sg2) * Z * (1.0 - Z)
        v2 = 0.5 * Z ** 2 * (1.0 - Z) ** 2 * (sd2 + sg2)
        rhs = P["delta"] * tf.math.log(tf.maximum(c, 1e-8)) + logK_drift + v1 * vp + v2 * vpp
        pv = P["delta"] * v
        marg = P["delta"] / tf.maximum(c, 1e-8)
        q_d = 1.0 - Z * vp
        q_g = 1.0 + (1.0 - Z) * vp
        FOC_d = -marg + P["Gamma_d"] * P["theta_d"] / tf.maximum(inside_d, 1e-8) * q_d
        FOC_g = -marg + P["Gamma_g"] * P["theta_g"] / tf.maximum(inside_g, 1e-8) * q_g
        # weight for optional preconditioning: sensitivity of R to v' ~ v1 + diffusion coupling
        w = tf.abs(v1) + v2 / tf.maximum(Z * (1.0 - Z), 1e-3) + self.precond_eps
        return rhs, pv, c, inside_g, inside_d, FOC_d, FOC_g, w

    def _hjb(self, rhs, pv, w):
        r = rhs - pv
        if self.precond:
            r = r / w
        return tf.sqrt(tf.reduce_mean(tf.square(r)))

    def _feas(self, c, ig, idd):
        eps = 1e-7
        nc = tf.cast(c < 1e-8, tf.float32); ng = tf.cast(ig < 1e-8, tf.float32); nd = tf.cast(idd < 1e-8, tf.float32)
        n_bad = tf.reduce_sum(nc) + tf.reduce_sum(ng) + tf.reduce_sum(nd)
        loss = (tf.sqrt(tf.reduce_mean(tf.square(-c * nc + eps)))
                + tf.sqrt(tf.reduce_mean(tf.square(-ig * ng + eps)))
                + tf.sqrt(tf.reduce_mean(tf.square(-idd * nd + eps))))
        return n_bad, loss

    def loss_value(self, Z):
        rhs, pv, c, ig, idd, FOC_d, FOC_g, w = self.pde_rhs(Z)
        nb, lf = self._feas(c, ig, idd)
        if nb > 0:
            return lf
        bc = tf.constant([[self.z_min], [self.z_max]], tf.float32)
        bcl = tf.reduce_mean((self.v_nn(bc) - tf.constant([[self.v0], [self.vN]], tf.float32)) ** 2)
        return (self._hjb(rhs, pv, w) + tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))
                + tf.sqrt(tf.reduce_mean(tf.square(FOC_d))) + self.bc_weight * bcl)

    def loss_control(self, Z):
        rhs, pv, c, ig, idd, FOC_d, FOC_g, w = self.pde_rhs(Z)
        nb, lf = self._feas(c, ig, idd)
        if nb > 0:
            return lf
        return (-tf.reduce_mean(rhs - pv) + tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))
                + tf.sqrt(tf.reduce_mean(tf.square(FOC_d))))

    @tf.function
    def train_step(self):
        Z = self.sample()
        with tf.GradientTape() as t:
            lv = self.loss_value(Z)
        self.opt[0].apply_gradients(zip(t.gradient(lv, self.v_nn.trainable_variables),
                                        self.v_nn.trainable_variables))
        Z = self.sample()
        cv = self.i_d_nn.trainable_variables + self.i_g_nn.trainable_variables
        with tf.GradientTape() as t:
            lc = self.loss_control(Z)
        self.opt[1].apply_gradients(zip(t.gradient(lc, cv), cv))
        return lv, lc

    def supervised_fit(self, Zt, vt, idt, igt, iters=20000, lr=1e-3, verbose=False):
        Z = tf.constant(np.asarray(Zt, np.float32).reshape(-1, 1))
        V = tf.constant(np.asarray(vt, np.float32).reshape(-1, 1))
        ID = tf.constant(np.asarray(idt, np.float32).reshape(-1, 1))
        IG = tf.constant(np.asarray(igt, np.float32).reshape(-1, 1))
        opt = tf.keras.optimizers.Adam(lr)
        vl = self.v_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_g_nn.trainable_variables
        n = int(Z.shape[0]); bs = min(1024, n)

        @tf.function
        def step():
            idx = tf.random.uniform([bs], 0, n, dtype=tf.int32)
            z = tf.gather(Z, idx)
            with tf.GradientTape() as t:
                loss = (tf.reduce_mean((self.v_nn(z) - tf.gather(V, idx)) ** 2)
                        + tf.reduce_mean((self.i_d_nn(z) - tf.gather(ID, idx)) ** 2)
                        + tf.reduce_mean((self.i_g_nn(z) - tf.gather(IG, idx)) ** 2))
            opt.apply_gradients(zip(t.gradient(loss, vl), vl))
            return loss
        for it in range(iters):
            l = step()
            if verbose and it % 4000 == 0:
                print(f"  [sup] it {it} mse={float(l):.3e}", flush=True)
        return float(l)

    def lbfgs_polish(self, n_points=4000, maxiter=5000):
        import scipy.optimize as so
        vl = self.v_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_g_nn.trainable_variables
        shapes = [v.shape for v in vl]; sizes = [int(np.prod(s)) for s in shapes]

        def setf(x):
            x = tf.constant(x, tf.float32); i = 0
            for v, s, sz in zip(vl, shapes, sizes):
                v.assign(tf.reshape(x[i:i + sz], s)); i += sz
        Z = stratified_uniform(self.z_min, self.z_max, n_points)
        bc = tf.constant([[self.z_min], [self.z_max]], tf.float32)
        vbc = tf.constant([[self.v0], [self.vN]], tf.float32)

        @tf.function
        def lg():
            with tf.GradientTape() as t:
                rhs, pv, c, ig, idd, FOC_d, FOC_g, w = self.pde_rhs(Z)
                r = (rhs - pv) / w if self.precond else (rhs - pv)
                loss = (tf.reduce_mean(tf.square(r)) + tf.reduce_mean(tf.square(FOC_d))
                        + tf.reduce_mean(tf.square(FOC_g)) + self.bc_weight * tf.reduce_mean(tf.square(self.v_nn(bc) - vbc)))
            g = t.gradient(loss, vl)
            return loss, tf.concat([tf.reshape(gi, [-1]) for gi in g], 0)

        def func(x):
            setf(x); loss, g = lg(); return float(loss.numpy()), g.numpy().astype(np.float64)
        x0 = tf.concat([tf.reshape(v, [-1]) for v in vl], 0).numpy().astype(np.float64)
        res = so.minimize(func, x0, jac=True, method="L-BFGS-B",
                          options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-16, "gtol": 1e-14})
        setf(res.x); return res

    def eval_losses(self, n=4096):
        Z = stratified_uniform(self.z_min, self.z_max, n)
        rhs, pv, c, ig, idd, FOC_d, FOC_g, w = self.pde_rhs(Z)
        return (float(tf.sqrt(tf.reduce_mean(tf.square(rhs - pv)))),
                float(tf.sqrt(tf.reduce_mean(tf.square(FOC_d)))),
                float(tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))))

    def evaluate_grid(self, n=2000):
        Z = np.linspace(0.0, 1.0, n + 1, dtype=np.float32).reshape(-1, 1)
        zt = tf.constant(Z)
        with tf.GradientTape() as t1:
            t1.watch(zt); v = self.v_nn(zt)
        vp = t1.gradient(v, zt).numpy().ravel().astype(np.float64)
        i_d = self.i_d_nn(zt).numpy().ravel().astype(np.float64)
        i_g = self.i_g_nn(zt).numpy().ravel().astype(np.float64)
        Zf = Z.ravel().astype(np.float64)
        c = (1 - Zf) * (self.P["A_d"] - i_d) + Zf * (self.P["A_g"] - i_g)
        return {"arch": self.arch, "Z": Zf, "v": v.numpy().ravel().astype(np.float64),
                "slope": vp, "i_d": i_d, "i_g": i_g, "c": c, "ratio": i_g / i_d,
                "final_losses": self.eval_losses()}


def solve_nn_shock(P, iters=120000, seed=0, verbose=False, log_every=5000, lbfgs_iters=5000,
                   arch="forwardnet", fd_ref=None, sup_iters=20000, **kw):
    tf.random.set_seed(seed); np.random.seed(seed)
    m = DGMShock(P, num_iterations=iters, arch=arch, **kw)
    if fd_ref is not None and sup_iters > 0:
        msk = (np.asarray(fd_ref["Z"]) >= 0.05) & (np.asarray(fd_ref["Z"]) <= 0.95)
        m.supervised_fit(np.asarray(fd_ref["Z"])[msk], np.asarray(fd_ref["v"])[msk],
                         np.asarray(fd_ref["i_d"])[msk], np.asarray(fd_ref["i_g"])[msk],
                         iters=int(sup_iters), verbose=verbose)
        if verbose:
            print(f"  [shock/{arch}] after warm start: {m.eval_losses()}", flush=True)
    for it in range(iters):
        lv, lc = m.train_step()
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"  [shock] it {it:6d} loss_v={float(lv):.3e} | HJB,FOCd,FOCg={m.eval_losses()}", flush=True)
    if lbfgs_iters and lbfgs_iters > 0:
        res = m.lbfgs_polish(maxiter=int(lbfgs_iters))
        if verbose:
            print(f"  [shock] L-BFGS: success={res.success} nit={res.nit}; {m.eval_losses()}", flush=True)
    out = m.evaluate_grid()
    return out


if __name__ == "__main__":
    P = M.load_calibration("A_g_prime_prime")
    out = solve_nn_shock(P, iters=int(8e4), verbose=True, lbfgs_iters=3000)
    print("final (HJB,FOC_d,FOC_g)=", out["final_losses"])
