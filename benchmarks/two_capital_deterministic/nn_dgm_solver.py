"""
Faithful DGM-PIA neural solver for the deterministic two-capital model, using the
PROJECT's actual networks/method (models/feedforward_subnet.py, the regime
objective_fn structure in models/PostDamagePostTech.py).

Mirrors the project exactly:
  * separate value net `v_nn` and control nets `i_d_nn`, `i_g_nn`
    (NOT closed-form controls);
  * the same summed-hidden-layer FeedForwardSubNet, swish value / tanh controls,
    the custom bounded investment activation 1 - (1+1/theta)/(exp(2x)+1)
    (range (-1/theta, 1)) from params.investment_rate_activation;
  * THREE losses (value: HJB residual + FOC_d + FOC_g; control: Hamiltonian
    + FOC_d + FOC_g), a feasibility guard, and the two-step train_step
    (value via optimizer 0, controls via optimizer 1) with warmup-cosine LR.

Only necessary deviation: the value output activation is LINEAR (not softplus),
because in this sub-model v(Z) = V - logK is negative (the project's softplus
enforces positivity for its v = V - logN transform, which does not apply here).

The reduced HJB (V(logK,Z)=logK+v(Z), V_logK=1, V_Z=v'):
  rhs = delta*log c + (1-Z)phi_d + Z phi_g + Z(1-Z)(phi_g-phi_d) v'
  pv  = delta*v ;   residual = rhs - pv
  FOC_d = -delta/c + Gamma_d theta_d/(1+theta_d i_d) (1 - Z v')
  FOC_g = -delta/c + Gamma_g theta_g/(1+theta_g i_g) (1 + (1-Z) v')
with c = (1-Z)(A_d - i_d) + Z(A_g - i_g) from the control networks.
"""

import importlib.util
import os

import numpy as np
import tensorflow as tf

import two_capital_model as M
# Project networks/method (models/ is on sys.path via two_capital_model):
from feedforward_subnet import setup_optimizers, stratified_uniform
from params import investment_rate_activation

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))


def subnet_class(arch="forwardnet"):
    """Return the FeedForwardSubNet class for the chosen architecture:
    'forwardnet' = models/ summed-hidden MLP; 'dgm' = models_dgm/ gated DGM net
    (Sirignano-Spiliopoulos). Both share the same config/call interface."""
    if arch == "dgm":
        path = os.path.join(_ROOT, "models_dgm", "feedforward_subnet.py")
        spec = importlib.util.spec_from_file_location("dgm_subnet", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.FeedForwardSubNet
    from feedforward_subnet import FeedForwardSubNet  # models/ (forwardnet)
    return FeedForwardSubNet


class DGMTwoCapital:
    def __init__(self, P, num_iterations=200000, batch_size=256,
                 lr_v=1e-4, lr_c=4e-3, num_neurons=32, num_layers=4,
                 z_min=0.01, z_max=0.99, bc_weight=1.0, arch="forwardnet",
                 precond=False, precond_eps=1e-3):
        self.P = P
        self.bs = batch_size
        self.z_min, self.z_max = z_min, z_max
        self.bc_weight = bc_weight
        self.arch = arch
        self.precond = precond
        self.precond_eps = precond_eps
        self.v0, self.vN = M.boundary_values(P)

        FFN = subnet_class(arch)
        common = dict(num_hiddens=[num_neurons] * num_layers, use_bias=True, dim=1)
        self.v_nn = FFN({**common, "activation": "swish",
                         "final_activation": None, "nn_name": "v_nn"})
        self.i_d_nn = FFN({**common, "activation": "tanh",
                           "final_activation": investment_rate_activation(P["theta_d"]),
                           "nn_name": "i_d_nn"})
        self.i_g_nn = FFN({**common, "activation": "tanh",
                           "final_activation": investment_rate_activation(P["theta_g"]),
                           "nn_name": "i_g_nn"})
        for net in (self.v_nn, self.i_d_nn, self.i_g_nn):
            net.build((None, 1))

        opt_params = {
            "learning_rates": [lr_v, lr_c],
            "num_iterations": num_iterations,
            "learning_rate_schedule_type": "warmup_cosine",
            "extra": {"warmup_steps": max(1, num_iterations // 100), "min_lr": 1e-6},
            "gradient_clip_norm": 1.0,
        }
        setup_optimizers(opt_params)
        self.opt = opt_params["optimizers"]  # [value_opt, control_opt]

    def sample(self):
        return stratified_uniform(self.z_min, self.z_max, self.bs)

    def pde_rhs(self, Z):
        P = self.P
        with tf.GradientTape() as tape_z:
            tape_z.watch(Z)
            v = self.v_nn(Z)
        v_prime = tape_z.gradient(v, Z)
        i_d = self.i_d_nn(Z)
        i_g = self.i_g_nn(Z)
        c = (1.0 - Z) * (P["A_d"] - i_d) + Z * (P["A_g"] - i_g)
        inside_d = 1.0 + P["theta_d"] * i_d
        inside_g = 1.0 + P["theta_g"] * i_g
        phi_d = P["alpha_d"] + P["Gamma_d"] * tf.math.log(tf.maximum(inside_d, 1e-8))
        phi_g = P["alpha_g"] + P["Gamma_g"] * tf.math.log(tf.maximum(inside_g, 1e-8))
        mu = Z * (1.0 - Z) * (phi_g - phi_d)
        rhs = P["delta"] * tf.math.log(tf.maximum(c, 1e-8)) + (1.0 - Z) * phi_d + Z * phi_g + mu * v_prime
        pv = P["delta"] * v
        marg_util_c = P["delta"] / tf.maximum(c, 1e-8)
        q_d = 1.0 - Z * v_prime
        q_g = 1.0 + (1.0 - Z) * v_prime
        FOC_d = -marg_util_c + P["Gamma_d"] * P["theta_d"] / tf.maximum(inside_d, 1e-8) * q_d
        FOC_g = -marg_util_c + P["Gamma_g"] * P["theta_g"] / tf.maximum(inside_g, 1e-8) * q_g
        return rhs, pv, c, inside_g, inside_d, FOC_d, FOC_g, mu

    def _hjb_term(self, rhs, pv, mu):
        """HJB residual, optionally preconditioned by 1/(|mu|+eps). Since the raw
        HJB residual ~ mu * (v' error), dividing by |mu| makes the loss directly
        sensitive to the v' error (well-conditioned), curing the flat valley."""
        r = rhs - pv
        if self.precond:
            r = r / (tf.abs(mu) + self.precond_eps)
        return tf.sqrt(tf.reduce_mean(tf.square(r)))

    def _feasibility(self, c, inside_g, inside_d):
        eps = 1e-7
        nc = tf.cast(c < 1e-8, tf.float32)
        ng = tf.cast(inside_g < 1e-8, tf.float32)
        nd = tf.cast(inside_d < 1e-8, tf.float32)
        n_bad = tf.reduce_sum(nc) + tf.reduce_sum(ng) + tf.reduce_sum(nd)
        loss = (tf.sqrt(tf.reduce_mean(tf.square(-c * nc + eps)))
                + tf.sqrt(tf.reduce_mean(tf.square(-inside_g * ng + eps)))
                + tf.sqrt(tf.reduce_mean(tf.square(-inside_d * nd + eps))))
        return n_bad, loss

    def loss_value(self, Z):
        rhs, pv, c, ig, idd, FOC_d, FOC_g, mu = self.pde_rhs(Z)
        n_bad, lf = self._feasibility(c, ig, idd)
        if n_bad > 0:
            return lf
        bc = tf.constant([[self.z_min], [self.z_max]], dtype=tf.float32)
        vb = self.v_nn(bc)
        bc_loss = tf.reduce_mean((vb - tf.constant([[self.v0], [self.vN]], tf.float32)) ** 2)
        return (self._hjb_term(rhs, pv, mu)
                + tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))
                + tf.sqrt(tf.reduce_mean(tf.square(FOC_d)))
                + self.bc_weight * bc_loss)

    def loss_control(self, Z):
        rhs, pv, c, ig, idd, FOC_d, FOC_g, mu = self.pde_rhs(Z)
        n_bad, lf = self._feasibility(c, ig, idd)
        if n_bad > 0:
            return lf
        return (-tf.reduce_mean(rhs - pv)
                + tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))
                + tf.sqrt(tf.reduce_mean(tf.square(FOC_d))))

    @tf.function
    def train_step(self):
        Z = self.sample()
        with tf.GradientTape() as tape:
            lv = self.loss_value(Z)
        g = tape.gradient(lv, self.v_nn.trainable_variables)
        self.opt[0].apply_gradients(zip(g, self.v_nn.trainable_variables))

        Z = self.sample()
        cvars = self.i_d_nn.trainable_variables + self.i_g_nn.trainable_variables
        with tf.GradientTape() as tape:
            lc = self.loss_control(Z)
        g = tape.gradient(lc, cvars)
        self.opt[1].apply_gradients(zip(g, cvars))
        return lv, lc

    def supervised_fit(self, Zt, vt, idt, igt, iters=20000, lr=1e-3, verbose=False):
        """Warm-start: fit v_nn, i_d_nn, i_g_nn to the FD reference (MSE). This
        seeds the networks at the correct solution so the subsequent HJB+FOC
        residual refinement can be tested for whether it STAYS at the FD answer
        (consistent) or drifts back to the weak-identification plateau."""
        Z = tf.constant(np.asarray(Zt, np.float32).reshape(-1, 1))
        V = tf.constant(np.asarray(vt, np.float32).reshape(-1, 1))
        ID = tf.constant(np.asarray(idt, np.float32).reshape(-1, 1))
        IG = tf.constant(np.asarray(igt, np.float32).reshape(-1, 1))
        opt = tf.keras.optimizers.Adam(lr)
        vlist = (self.v_nn.trainable_variables + self.i_d_nn.trainable_variables
                 + self.i_g_nn.trainable_variables)
        n = int(Z.shape[0]); bs = min(1024, n)

        @tf.function
        def step():
            idx = tf.random.uniform([bs], 0, n, dtype=tf.int32)
            z = tf.gather(Z, idx)
            with tf.GradientTape() as t:
                loss = (tf.reduce_mean((self.v_nn(z) - tf.gather(V, idx)) ** 2)
                        + tf.reduce_mean((self.i_d_nn(z) - tf.gather(ID, idx)) ** 2)
                        + tf.reduce_mean((self.i_g_nn(z) - tf.gather(IG, idx)) ** 2))
            g = t.gradient(loss, vlist)
            opt.apply_gradients(zip(g, vlist))
            return loss

        for it in range(iters):
            l = step()
            if verbose and (it % 4000 == 0 or it == iters - 1):
                print(f"  [sup] it {it:6d}  fit_mse={float(l):.3e}", flush=True)
        return float(l)

    def lbfgs_polish(self, n_points=4000, maxiter=5000, seed=0):
        """Second-order L-BFGS polish (scipy) over ALL params jointly, minimizing
        the combined least-squares objective HJB^2 + FOC_d^2 + FOC_g^2 (+ bc). This
        is the standard PINN remedy for the Adam plateau: quasi-Newton navigates the
        ill-conditioned (weakly v'-identified) landscape far better than first-order
        Adam, typically driving the residual orders of magnitude lower."""
        import scipy.optimize as so
        tf.random.set_seed(seed)
        vlist = (self.v_nn.trainable_variables + self.i_d_nn.trainable_variables
                 + self.i_g_nn.trainable_variables)
        shapes = [v.shape for v in vlist]
        sizes = [int(np.prod(s)) for s in shapes]

        def set_flat(x):
            x = tf.constant(x, tf.float32); i = 0
            for v, s, sz in zip(vlist, shapes, sizes):
                v.assign(tf.reshape(x[i:i + sz], s)); i += sz

        Z = stratified_uniform(self.z_min, self.z_max, n_points)
        bc = tf.constant([[self.z_min], [self.z_max]], dtype=tf.float32)
        vbc = tf.constant([[self.v0], [self.vN]], dtype=tf.float32)

        @tf.function
        def loss_and_grad():
            with tf.GradientTape() as tape:
                rhs, pv, c, ig, idd, FOC_d, FOC_g, mu = self.pde_rhs(Z)
                r = rhs - pv
                if self.precond:
                    r = r / (tf.abs(mu) + self.precond_eps)
                loss = (tf.reduce_mean(tf.square(r))
                        + tf.reduce_mean(tf.square(FOC_d))
                        + tf.reduce_mean(tf.square(FOC_g))
                        + self.bc_weight * tf.reduce_mean(tf.square(self.v_nn(bc) - vbc)))
            g = tape.gradient(loss, vlist)
            g = tf.concat([tf.reshape(gi, [-1]) for gi in g], axis=0)
            return loss, g

        def func(x):
            set_flat(x)
            loss, g = loss_and_grad()
            return float(loss.numpy()), g.numpy().astype(np.float64)

        x0 = tf.concat([tf.reshape(v, [-1]) for v in vlist], axis=0).numpy().astype(np.float64)
        res = so.minimize(func, x0, jac=True, method="L-BFGS-B",
                          options={"maxiter": maxiter, "maxfun": maxiter * 2,
                                   "ftol": 1e-16, "gtol": 1e-14})
        set_flat(res.x)
        return res

    def eval_losses(self, n=4096):
        Z = stratified_uniform(self.z_min, self.z_max, n)
        rhs, pv, c, ig, idd, FOC_d, FOC_g, mu = self.pde_rhs(Z)
        return (float(tf.sqrt(tf.reduce_mean(tf.square(rhs - pv)))),
                float(tf.sqrt(tf.reduce_mean(tf.square(FOC_d)))),
                float(tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))))

    def evaluate_grid(self, n=4000):
        Z = np.linspace(0.0, 1.0, n + 1, dtype=np.float32).reshape(-1, 1)
        zt = tf.constant(Z)
        with tf.GradientTape() as tape_z:
            tape_z.watch(zt)
            v = self.v_nn(zt)
        vp = tape_z.gradient(v, zt).numpy().ravel().astype(np.float64)
        i_d = self.i_d_nn(zt).numpy().ravel().astype(np.float64)
        i_g = self.i_g_nn(zt).numpy().ravel().astype(np.float64)
        Zf = Z.ravel().astype(np.float64)
        c = (1 - Zf) * (self.P["A_d"] - i_d) + Zf * (self.P["A_g"] - i_g)
        resid = M.hjb_residual(Zf, v.numpy().ravel().astype(np.float64), vp, self.P)
        return {"method": "NN-DGM", "arch": self.arch, "Z": Zf,
                "v": v.numpy().ravel().astype(np.float64),
                "slope": vp, "i_d": i_d, "i_g": i_g, "c": c, "ratio": i_g / i_d,
                "residual": resid, "max_abs_residual": float(np.max(np.abs(resid[1:-1])))}


def solve_nn_dgm(P, iters=200000, seed=0, verbose=False, log_every=5000,
                 lbfgs_iters=5000, arch="forwardnet", fd_ref=None, sup_iters=20000, **kw):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    m = DGMTwoCapital(P, num_iterations=iters, arch=arch, **kw)
    if fd_ref is not None and sup_iters > 0:
        if verbose:
            print(f"  [DGM/{arch}] FD-supervised warm start ({sup_iters} iters)...", flush=True)
        msk = (np.asarray(fd_ref["Z"]) >= 0.05) & (np.asarray(fd_ref["Z"]) <= 0.95)
        m.supervised_fit(np.asarray(fd_ref["Z"])[msk], np.asarray(fd_ref["v"])[msk],
                         np.asarray(fd_ref["i_d"])[msk], np.asarray(fd_ref["i_g"])[msk],
                         iters=int(sup_iters), verbose=verbose)
        if verbose:
            print(f"  [DGM/{arch}] after warm start: (HJB,FOC_d,FOC_g)={m.eval_losses()}", flush=True)
    for it in range(iters):
        lv, lc = m.train_step()
        if verbose and (it % log_every == 0 or it == iters - 1):
            pde, fd_, fg_ = m.eval_losses()
            print(f"  [DGM] it {it:6d}  loss_v={float(lv):.3e} loss_c={float(lc):.3e}  "
                  f"| HJB={pde:.3e} FOC_d={fd_:.3e} FOC_g={fg_:.3e}", flush=True)
    if verbose:
        print(f"  [DGM] Adam done; eval (HJB,FOC_d,FOC_g)={m.eval_losses()}", flush=True)
    if lbfgs_iters and lbfgs_iters > 0:
        res = m.lbfgs_polish(maxiter=int(lbfgs_iters), seed=seed)
        if verbose:
            print(f"  [DGM] L-BFGS done: success={res.success} nit={res.nit} "
                  f"fun={res.fun:.3e}; eval (HJB,FOC_d,FOC_g)={m.eval_losses()}", flush=True)
    out = m.evaluate_grid()
    out["final_losses"] = m.eval_losses()
    return out


if __name__ == "__main__":
    P = M.load_calibration("A_g_prime_prime")
    out = solve_nn_dgm(P, iters=int(2e5), verbose=True)
    print(f"DGM done. final (HJB,FOC_d,FOC_g)={out['final_losses']}")
    for z in (0.1, 0.3, 0.5, 0.7, 0.9):
        k = int(z * (len(out["Z"]) - 1))
        print(f"  Z={z:.1f}  i_d={out['i_d'][k]:.5f}  i_g={out['i_g'][k]:.5f}  v'={out['slope'][k]:.4f}")
