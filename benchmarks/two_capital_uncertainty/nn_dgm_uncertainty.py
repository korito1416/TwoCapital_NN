"""
DGM-PIA neural solver for the ROBUST two-capital HJB with log(xi) as a pseudo-state.

The network learns v(Z, logxi) on the slab Z in [z_min,z_max] x logxi in [-3,5]
(exactly as models/PostDamagePostTech.py carries logxi as an input). Derivatives are
taken in Z only; xi = exp(logxi) is a parameter dimension entering the robustness drag
  -1/(2 xi)[(1-Z)^2 sd^2 q_d^2 + Z^2 sg^2 q_g^2]
added to the shock HJB residual. Controls i_d(Z,logxi), i_g(Z,logxi) share the input.
3-loss DGM-PIA (HJB residual + FOC_d + FOC_g), warmup-cosine LR, optional FD-supervised
warm start and preconditioning, mirroring nn_dgm_shock.py. The xi-dependent boundary
v_j(xi) = const_j - sigma_j^2/(2 xi delta) is imposed as Dirichlet data at Z in {z_min,z_max}.
"""
import os
import sys

import numpy as np
import tensorflow as tf

# shock model (carries xi robustness) + shared NN utilities
_SHOCK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "two_capital_shock"))
_DET = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "two_capital_deterministic"))
for p in (_SHOCK, _DET):
    if p not in sys.path:
        sys.path.insert(0, p)
import two_capital_shock_model as M                      # noqa: E402
from nn_dgm_solver import subnet_class                   # noqa: E402
from feedforward_subnet import setup_optimizers, stratified_uniform  # noqa: E402
from params import investment_rate_activation            # noqa: E402


def _const_boundary(P):
    """xi-independent part of the one-capital boundary values (the bracket)."""
    def c_of(A, Gamma, theta):
        return P["delta"] * (1.0 + theta * A) / (theta * (P["delta"] + Gamma))
    cd = c_of(P["A_d"], P["Gamma_d"], P["theta_d"])
    cg = c_of(P["A_g"], P["Gamma_g"], P["theta_g"])
    bd = (np.log(cd) + P["alpha_d"] / P["delta"]
          + (P["Gamma_d"] / P["delta"]) * np.log(P["Gamma_d"] * P["theta_d"] * cd / P["delta"])
          - P["sigma_d"] ** 2 / (2.0 * P["delta"]))
    bg = (np.log(cg) + P["alpha_g"] / P["delta"]
          + (P["Gamma_g"] / P["delta"]) * np.log(P["Gamma_g"] * P["theta_g"] * cg / P["delta"])
          - P["sigma_g"] ** 2 / (2.0 * P["delta"]))
    return float(bd), float(bg)


class DGMUncertainty:
    def __init__(self, P, num_iterations=200000, batch_size=512, lr_v=1e-4, lr_c=4e-3,
                 num_neurons=32, num_layers=4, z_min=0.01, z_max=0.99,
                 logxi_min=-3.0, logxi_max=5.0, bc_weight=1.0,
                 arch="forwardnet", precond=False, precond_eps=5e-3):
        self.P = P; self.bs = batch_size; self.z_min, self.z_max = z_min, z_max
        self.lxmin, self.lxmax = logxi_min, logxi_max
        self.bc_weight = bc_weight; self.arch = arch
        self.precond = precond; self.precond_eps = precond_eps
        self.bd_const, self.bg_const = _const_boundary(P)
        self.sd2 = P["sigma_d"] ** 2; self.sg2 = P["sigma_g"] ** 2
        FFN = subnet_class(arch)
        # dim = OUTPUT width (scalar) -- the 2-D INPUT (Z, logxi) is set by build((None, 2)).
        common = dict(num_hiddens=[num_neurons] * num_layers, use_bias=True, dim=1)
        self.v_nn = FFN({**common, "activation": "swish", "final_activation": None, "nn_name": "v_nn"})
        self.i_d_nn = FFN({**common, "activation": "tanh",
                           "final_activation": investment_rate_activation(P["theta_d"]), "nn_name": "i_d_nn"})
        self.i_g_nn = FFN({**common, "activation": "tanh",
                           "final_activation": investment_rate_activation(P["theta_g"]), "nn_name": "i_g_nn"})
        for net in (self.v_nn, self.i_d_nn, self.i_g_nn):
            net.build((None, 2))
        opt_params = {"learning_rates": [lr_v, lr_c], "num_iterations": num_iterations,
                      "learning_rate_schedule_type": "warmup_cosine",
                      "extra": {"warmup_steps": max(1, num_iterations // 100), "min_lr": 1e-6},
                      "gradient_clip_norm": 1.0}
        setup_optimizers(opt_params)
        self.opt = opt_params["optimizers"]

    def sample(self):
        Z = stratified_uniform(self.z_min, self.z_max, self.bs)
        lx = stratified_uniform(self.lxmin, self.lxmax, self.bs)
        return Z, lx

    def pde_rhs(self, Z, lx):
        P = self.P
        xi = tf.exp(lx)
        with tf.GradientTape() as t2:
            t2.watch(Z)
            with tf.GradientTape() as t1:
                t1.watch(Z)
                v = self.v_nn(tf.concat([Z, lx], 1))
            vp = t1.gradient(v, Z)
        vpp = t2.gradient(vp, Z)
        X = tf.concat([Z, lx], 1)
        i_d = self.i_d_nn(X); i_g = self.i_g_nn(X)
        c = (1.0 - Z) * (P["A_d"] - i_d) + Z * (P["A_g"] - i_g)
        inside_d = 1.0 + P["theta_d"] * i_d
        inside_g = 1.0 + P["theta_g"] * i_g
        phi_d = P["alpha_d"] + P["Gamma_d"] * tf.math.log(tf.maximum(inside_d, 1e-8))
        phi_g = P["alpha_g"] + P["Gamma_g"] * tf.math.log(tf.maximum(inside_g, 1e-8))
        sd2, sg2 = self.sd2, self.sg2
        logK_drift = (1.0 - Z) * phi_d + Z * phi_g - 0.5 * (sd2 * (1.0 - Z) ** 2 + sg2 * Z ** 2)
        v1 = (phi_g - phi_d + (1.0 - Z) * sd2 - Z * sg2) * Z * (1.0 - Z)
        v2 = 0.5 * Z ** 2 * (1.0 - Z) ** 2 * (sd2 + sg2)
        q_d = 1.0 - Z * vp
        q_g = 1.0 + (1.0 - Z) * vp
        drag = -(1.0 / (2.0 * xi)) * ((1.0 - Z) ** 2 * sd2 * q_d ** 2 + Z ** 2 * sg2 * q_g ** 2)
        rhs = P["delta"] * tf.math.log(tf.maximum(c, 1e-8)) + logK_drift + v1 * vp + v2 * vpp + drag
        pv = P["delta"] * v
        marg = P["delta"] / tf.maximum(c, 1e-8)
        FOC_d = -marg + P["Gamma_d"] * P["theta_d"] / tf.maximum(inside_d, 1e-8) * q_d
        FOC_g = -marg + P["Gamma_g"] * P["theta_g"] / tf.maximum(inside_g, 1e-8) * q_g
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

    def _bc_loss(self):
        # xi-dependent Dirichlet data at Z in {z_min, z_max} for a batch of logxi
        lx = stratified_uniform(self.lxmin, self.lxmax, self.bs)
        xi = tf.exp(lx)
        z0 = tf.fill(tf.shape(lx), tf.constant(self.z_min, tf.float32))
        zN = tf.fill(tf.shape(lx), tf.constant(self.z_max, tf.float32))
        v0_t = self.bd_const - self.sd2 / (2.0 * xi * self.P["delta"])
        vN_t = self.bg_const - self.sg2 / (2.0 * xi * self.P["delta"])
        v0_p = self.v_nn(tf.concat([z0, lx], 1))
        vN_p = self.v_nn(tf.concat([zN, lx], 1))
        return tf.reduce_mean((v0_p - v0_t) ** 2) + tf.reduce_mean((vN_p - vN_t) ** 2)

    def loss_value(self, Z, lx):
        rhs, pv, c, ig, idd, FOC_d, FOC_g, w = self.pde_rhs(Z, lx)
        nb, lf = self._feas(c, ig, idd)
        if nb > 0:
            return lf
        return (self._hjb(rhs, pv, w) + tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))
                + tf.sqrt(tf.reduce_mean(tf.square(FOC_d))) + self.bc_weight * self._bc_loss())

    def loss_control(self, Z, lx):
        rhs, pv, c, ig, idd, FOC_d, FOC_g, w = self.pde_rhs(Z, lx)
        nb, lf = self._feas(c, ig, idd)
        if nb > 0:
            return lf
        return (-tf.reduce_mean(rhs - pv) + tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))
                + tf.sqrt(tf.reduce_mean(tf.square(FOC_d))))

    @tf.function
    def train_step(self):
        Z, lx = self.sample()
        with tf.GradientTape() as t:
            lv = self.loss_value(Z, lx)
        self.opt[0].apply_gradients(zip(t.gradient(lv, self.v_nn.trainable_variables),
                                        self.v_nn.trainable_variables))
        Z, lx = self.sample()
        cv = self.i_d_nn.trainable_variables + self.i_g_nn.trainable_variables
        with tf.GradientTape() as t:
            lc = self.loss_control(Z, lx)
        self.opt[1].apply_gradients(zip(t.gradient(lc, cv), cv))
        return lv, lc

    def supervised_fit(self, Zt, LXt, vt, idt, igt, iters=20000, lr=1e-3, verbose=False):
        Z = tf.constant(np.asarray(Zt, np.float32).reshape(-1, 1))
        LX = tf.constant(np.asarray(LXt, np.float32).reshape(-1, 1))
        V = tf.constant(np.asarray(vt, np.float32).reshape(-1, 1))
        ID = tf.constant(np.asarray(idt, np.float32).reshape(-1, 1))
        IG = tf.constant(np.asarray(igt, np.float32).reshape(-1, 1))
        opt = tf.keras.optimizers.Adam(lr)
        vl = self.v_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_g_nn.trainable_variables
        n = int(Z.shape[0]); bs = min(2048, n)

        @tf.function
        def step():
            idx = tf.random.uniform([bs], 0, n, dtype=tf.int32)
            z = tf.gather(Z, idx); lx = tf.gather(LX, idx)
            X = tf.concat([z, lx], 1)
            with tf.GradientTape() as t:
                loss = (tf.reduce_mean((self.v_nn(X) - tf.gather(V, idx)) ** 2)
                        + tf.reduce_mean((self.i_d_nn(X) - tf.gather(ID, idx)) ** 2)
                        + tf.reduce_mean((self.i_g_nn(X) - tf.gather(IG, idx)) ** 2))
            opt.apply_gradients(zip(t.gradient(loss, vl), vl))
            return loss
        for it in range(iters):
            l = step()
            if verbose and it % 4000 == 0:
                print(f"  [sup] it {it} mse={float(l):.3e}", flush=True)
        return float(l)

    def eval_losses(self, n=8192):
        Z = stratified_uniform(self.z_min, self.z_max, n)
        lx = stratified_uniform(self.lxmin, self.lxmax, n)
        rhs, pv, c, ig, idd, FOC_d, FOC_g, w = self.pde_rhs(Z, lx)
        return (float(tf.sqrt(tf.reduce_mean(tf.square(rhs - pv)))),
                float(tf.sqrt(tf.reduce_mean(tf.square(FOC_d)))),
                float(tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))))

    def evaluate_grid(self, xis, n=2000):
        """Evaluate v', i_d, i_g on a Z-grid at each requested xi."""
        Zg = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
        out = {"Z": Zg.astype(np.float64), "xis": list(xis), "by_xi": {}}
        for xi in xis:
            lx = np.full_like(Zg, np.log(xi))
            zt = tf.constant(Zg.reshape(-1, 1)); lxt = tf.constant(lx.reshape(-1, 1))
            with tf.GradientTape() as t1:
                t1.watch(zt); v = self.v_nn(tf.concat([zt, lxt], 1))
            vp = t1.gradient(v, zt).numpy().ravel().astype(np.float64)
            X = tf.concat([zt, lxt], 1)
            i_d = self.i_d_nn(X).numpy().ravel().astype(np.float64)
            i_g = self.i_g_nn(X).numpy().ravel().astype(np.float64)
            out["by_xi"][float(xi)] = {"slope": vp, "i_d": i_d, "i_g": i_g,
                                       "v": v.numpy().ravel().astype(np.float64)}
        out["final_losses"] = self.eval_losses()
        return out


def solve_nn_uncertainty(P, iters=150000, seed=0, verbose=False, log_every=5000,
                         arch="forwardnet", fd_ref=None, sup_iters=20000,
                         eval_xis=(0.05, 0.1, 148.4), **kw):
    tf.random.set_seed(seed); np.random.seed(seed)
    m = DGMUncertainty(P, num_iterations=iters, arch=arch, **kw)
    if fd_ref is not None and sup_iters > 0:
        m.supervised_fit(fd_ref["Z"], fd_ref["logxi"], fd_ref["v"], fd_ref["i_d"], fd_ref["i_g"],
                         iters=int(sup_iters), verbose=verbose)
        if verbose:
            print(f"  [unc/{arch}] after warm start: {m.eval_losses()}", flush=True)
    for it in range(iters):
        lv, lc = m.train_step()
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"  [unc] it {it:6d} loss_v={float(lv):.3e} | HJB,FOCd,FOCg={m.eval_losses()}", flush=True)
    return m.evaluate_grid(eval_xis)


if __name__ == "__main__":
    P = M.load_calibration("A_g_prime_prime")
    P["sigma_d"] = P["sigma_g"] = 0.2  # large sigma so robustness is visible
    out = solve_nn_uncertainty(P, iters=int(4e4), verbose=True)
    print("final (HJB,FOC_d,FOC_g)=", out["final_losses"])
