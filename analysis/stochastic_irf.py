#!/usr/bin/env python3
"""Stochastic impulse responses on the NN solution (Mike point 2), for ONE xi.

Two objects, from one Monte-Carlo pass (baseline path + 4 perturbed paths, common
random numbers), under the ROBUST h-distorted no-jump diffusion measure:

(A) STATE / economic stochastic IRF -- the variational (first-variation / "stochastic
    response") process Lambda_t = dX_t/dX_0 . e_shock (Hansen-Souganidis 2025 eq 4;
    Barnett-Brock-Hansen-Zhang mitigation 2025 sec 5), realised as the common-random
    finite-difference tangent (x_perturbed - x)/eps, path-averaged with 10/90 bands,
    for a marginal shock in each of the 4 initial coordinates (Capital=logK,
    GreenShare=Z, Temperature=Y, Technology=logR = Haoyang's --m0). Plus the responses
    of derived quantities (emissions E, consumption C, investments i_d/i_g/i_r).
    The deterministic IRF is the sigma=0 degenerate case of this.

(B) PRICED / marginal-value decomposition -- DV(X0).e_shock = E~[ int Dis_t (Lambda_t.Scf_t) dt ]
    split into flow i (delta grad-utility), flow ii (jump-intensity gradient) and flow iii
    (post-jump continuation-value gradient), for each shock. This captures the technology-jump
    channel that the raw state tangent understates (R&D's bite is via bringing the breakthrough
    forward). Reuses SVRDDecomposer.cashflow_terms and the robust jump discount.

Run one xi per job; overlay across xi with plot_stochastic_irf.py.
"""
from __future__ import annotations
import argparse, os, sys, time
from collections import defaultdict, OrderedDict
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "analysis"))
import tensorflow as tf  # noqa: E402
from compute_svrd_decomposition import SVRDDecomposer, TF_FLOAT, pvalue  # noqa: E402

SHOCKS = ["Capital", "GreenShare", "Temperature", "Technology"]
SHOCK_IDX = {"Capital": 0, "GreenShare": 1, "Temperature": 2, "Technology": 3}
DERIVED = ["E", "C", "i_d", "i_g", "i_r"]
STATEK = ["logK", "Z", "Y", "logR"]
FLOWKEYS = ["flow_i", "flow_ii_damage", "flow_ii_tech", "flow_iii_damage", "flow_iii_tech"]


class StochasticIRF(SVRDDecomposer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._c_nojump = tf.function(self.nojump_step, reduce_retracing=True)
        self._c_cashflow = tf.function(self.cashflow_terms, reduce_retracing=True)
        self._c_astep = tf.function(self._analytic_step, reduce_retracing=True)
        self._c_djvp = tf.function(self._derived_jvp, reduce_retracing=True)

    def derived(self, xt: np.ndarray) -> dict:
        logk, z = xt[:, 0], xt[:, 1]
        E = pvalue("η") * pvalue("A_d") * (1.0 - z) * np.exp(logk)
        cur = self.current_objects(tf.convert_to_tensor(xt.astype(np.float32), TF_FLOAT))
        cok = cur["c_over_k"].numpy().ravel()
        return dict(E=E, C=cok * np.exp(logk), i_d=cur["i_d"].numpy().ravel(),
                    i_g=cur["i_g"].numpy().ravel(), i_r=cur["i_r"].numpy().ravel())

    # ---- analytic (NN-derivative) tangent: JVP of the exact production maps, no eps ----
    def _derived_tf(self, x):
        """(n,5) = [E, C, i_d, i_g, i_r] as a differentiable function of x (order == DERIVED)."""
        cur = self.current_objects(x)
        logk, z = x[:, 0:1], x[:, 1:2]
        K = tf.exp(logk)
        E = pvalue("η") * pvalue("A_d") * (1.0 - z) * K
        C = cur["c_over_k"] * K
        return tf.concat([E, C, cur["i_d"], cur["i_g"], cur["i_r"]], axis=1)

    def _analytic_step(self, x, lams, shocks, dt):
        """Advance baseline x by the exact nojump_step, and each of the 4 state tangents
        lams[:,:,c] by the forward-mode JVP of that SAME map (forward-over-reverse: the
        inner value-gradient tape is differentiated by the outer ForwardAccumulator, so the
        tangent carries the exact Hessian-vector ∂²V·Λ and policy-Jacobian ∂i·Λ terms).
        lams: (n,4,4) [path, state-coord, shock];  returns x_next (n,4), lams_next (n,4,4)."""
        x_next = None
        cols = []
        for c in range(4):
            with tf.autodiff.ForwardAccumulator(x, lams[:, :, c]) as acc:
                xn = self.nojump_step(x, shocks, dt)
            if c == 0:
                x_next = xn
            cols.append(acc.jvp(xn))                      # (n,4) = advanced tangent for shock c
        return x_next, tf.stack(cols, axis=2)             # (n,4,4)

    def _derived_jvp(self, x, lams):
        """Derived-quantity response d(E,C,i_d,i_g,i_r)/d(shock) = grad(·)(x) . Lambda, per shock.
        Returns (n,5,4) [path, derived-quantity, shock]. Pure NN derivative, no finite difference."""
        cols = []
        for c in range(4):
            with tf.autodiff.ForwardAccumulator(x, lams[:, :, c]) as acc:
                der = self._derived_tf(x)
            cols.append(acc.jvp(der))                     # (n,5)
        return tf.stack(cols, axis=2)                     # (n,5,4)

    def simulate_all(self, eps, n_paths, years, dt, seed, y0, store_points=70, do_priced=True):
        rng = np.random.default_rng(seed)
        x = self.initial_x(n_paths, y0)
        xperts = []
        for c in SHOCKS:
            xb = x.numpy().copy(); xb[:, SHOCK_IDX[c]] += eps
            xperts.append(tf.convert_to_tensor(xb, TF_FLOAT))
        discount = tf.ones((n_paths, 1), TF_FLOAT)
        steps = int(round(years / dt)); sqrt_dt = np.sqrt(dt)
        every = max(1, steps // store_points)
        dt32 = np.float32(dt); eps32 = np.float32(eps)

        times = []
        resp = {c: defaultdict(list) for c in SHOCKS}   # state IRF: (mean,p10,p90)
        lev = defaultdict(list)
        priced = {c: {fk: tf.zeros((n_paths,), TF_FLOAT) for fk in FLOWKEYS} for c in SHOCKS}

        def stat(v):
            return (float(np.mean(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90)))

        def record_state(t):
            xn = x.numpy(); db = self.derived(xn)
            times.append(t)
            for c in SHOCKS:
                lam = (xperts[SHOCK_IDX[c]].numpy() - xn) / eps
                dp = self.derived(xperts[SHOCK_IDX[c]].numpy())
                R = {k: lam[:, i] for i, k in enumerate(STATEK)}
                for k in DERIVED:
                    R[k] = (dp[k] - db[k]) / eps
                for k, v in R.items():
                    resp[c][k].append(stat(v))
            for k, idx in [("Ylev", 2), ("logKlev", 0), ("logRlev", 3), ("Zlev", 1)]:
                lev[k].append(stat(xn[:, idx]))
            lev["Elev"].append(stat(db["E"]))

        for t in range(steps):
            if t % every == 0:
                record_state(t * dt)
            if do_priced:
                terms = self._c_cashflow(x)
                w = tf.squeeze(discount, axis=1) * dt32
                flows = {"flow_i": terms["flow_i"],
                         "flow_ii_damage": terms["flow_ii_damage"], "flow_ii_tech": terms["flow_ii_tech"],
                         "flow_iii_damage": terms["flow_iii_damage"], "flow_iii_tech": terms["flow_iii_tech"]}
                for c in SHOCKS:
                    lam = (xperts[SHOCK_IDX[c]] - x) / eps32
                    for fk in FLOWKEYS:
                        priced[c][fk] += tf.reduce_sum(lam * flows[fk], axis=1) * w
                discount = discount * tf.exp(-(pvalue("δ") + terms["robust_intensity_sum"]) * dt32)
            s = tf.convert_to_tensor(rng.normal(0.0, sqrt_dt, size=(n_paths, 4)), TF_FLOAT)
            x = self._c_nojump(x, s, dt)
            xperts = [self._c_nojump(xp, s, dt) for xp in xperts]
        record_state(steps * dt)

        meta = self.scaling_factor(y0)
        out = {"t": np.array(times), "xi": self.xi,
               "scale_logR": meta["scale_logR_to_level_consumption"],
               "MU0": meta["MU0"], "C0": meta["C0"]}
        for c in SHOCKS:
            for k, v in resp[c].items():
                out[f"resp_{c}_{k}"] = np.array(v)
            if do_priced:
                for fk in FLOWKEYS:
                    vals = priced[c][fk].numpy()
                    out[f"priced_{c}_{fk}_mean"] = float(np.mean(vals))
                    out[f"priced_{c}_{fk}_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        for k, v in lev.items():
            out[f"lev_{k}"] = np.array(v)
        return out

    def simulate_all_analytic(self, n_paths, years, dt, seed, y0, store_points=70):
        """Analytic variant of simulate_all: the state tangent Lambda_t and the derived-quantity
        responses are propagated by exact NN derivatives (ForwardAccumulator JVP of nojump_step /
        the control nets) rather than a common-random finite difference. No eps -> no FD noise floor.
        Baseline path is the same nojump_step trajectory (identical shocks), so this is a clean
        drop-in for the state / economic IRF (the priced FK decomposition is unaffected / omitted)."""
        rng = np.random.default_rng(seed)
        x = self.initial_x(n_paths, y0)
        lams = np.zeros((n_paths, 4, 4), np.float32)
        for c, name in enumerate(SHOCKS):
            lams[:, SHOCK_IDX[name], c] = 1.0          # Lambda_0 = e_(shock coordinate)
        lams = tf.convert_to_tensor(lams, TF_FLOAT)
        steps = int(round(years / dt)); every = max(1, steps // store_points)
        sqrt_dt = np.sqrt(dt)

        times = []
        resp = {c: defaultdict(list) for c in SHOCKS}
        lev = defaultdict(list)

        def stat(v):
            return (float(np.mean(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90)))

        def record_state(t):
            xn = x.numpy(); db = self.derived(xn)
            dr = self._c_djvp(x, lams).numpy()          # (n,5,4) derived responses (analytic)
            ln = lams.numpy()                           # (n,4,4) state tangents
            times.append(t)
            for ci, c in enumerate(SHOCKS):
                R = {STATEK[k]: ln[:, k, ci] for k in range(4)}
                for di, dk in enumerate(DERIVED):
                    R[dk] = dr[:, di, ci]
                for k, v in R.items():
                    resp[c][k].append(stat(v))
            for k, idx in [("Ylev", 2), ("logKlev", 0), ("logRlev", 3), ("Zlev", 1)]:
                lev[k].append(stat(xn[:, idx]))
            lev["Elev"].append(stat(db["E"]))

        for t in range(steps):
            if t % every == 0:
                record_state(t * dt)
            s = tf.convert_to_tensor(rng.normal(0.0, sqrt_dt, size=(n_paths, 4)), TF_FLOAT)
            x, lams = self._c_astep(x, lams, s, dt)
        record_state(steps * dt)

        meta = self.scaling_factor(y0)
        out = {"t": np.array(times), "xi": self.xi, "analytic": True,
               "scale_logR": meta["scale_logR_to_level_consumption"],
               "MU0": meta["MU0"], "C0": meta["C0"]}
        for c in SHOCKS:
            for k, v in resp[c].items():
                out[f"resp_{c}_{k}"] = np.array(v)
        for k, v in lev.items():
            out[f"lev_{k}"] = np.array(v)
        return out


def parse_args():
    p = argparse.ArgumentParser(description="Stochastic (variational-process) IRF on the NN solution, one xi.")
    p.add_argument("--export-folder", required=True)
    p.add_argument("--xi", type=float, required=True)
    p.add_argument("--n-paths", type=int, default=512)
    p.add_argument("--years", type=float, default=60.0)
    p.add_argument("--dt", type=float, default=1.0 / 12.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--y0", type=float, default=1.2)
    p.add_argument("--store-points", type=int, default=70)
    p.add_argument("--no-priced", action="store_true",
                   help="skip the Feynman-Kac priced decomposition (state IRF only; much faster -> big N).")
    p.add_argument("--analytic", action="store_true",
                   help="propagate the tangent + derived responses by exact NN derivatives "
                        "(ForwardAccumulator JVP), not a finite difference. No eps, no FD noise floor. "
                        "State/economic IRF only (implies --no-priced).")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    export = Path(args.export_folder).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    irf_obj = StochasticIRF(str(export), args.xi, batch_size=max(args.n_paths, 64))
    if args.analytic:
        irf = irf_obj.simulate_all_analytic(
            args.n_paths, args.years, args.dt, args.seed, args.y0, args.store_points)
    else:
        irf = irf_obj.simulate_all(
            args.eps, args.n_paths, args.years, args.dt, args.seed, args.y0, args.store_points,
            do_priced=not args.no_priced)
    np.savez_compressed(out_dir / f"irf_xi_{args.xi:g}.npz", **irf)
    mode = "analytic" if args.analytic else f"FD(eps={args.eps:g})"
    print(f"xi={args.xi:g} [{mode}] done in {time.time()-t0:.0f}s -> {out_dir}/irf_xi_{args.xi:g}.npz")
    if not args.analytic and not args.no_priced:
        for c in SHOCKS:
            tot = sum(irf[f"priced_{c}_{fk}_mean"] for fk in FLOWKEYS)
            print(f"  priced DV.e_{c}: total={tot:+.4g}  "
                  + " ".join(f"{fk.replace('flow_','f')}={irf[f'priced_{c}_{fk}_mean']:+.3g}" for fk in FLOWKEYS))


if __name__ == "__main__":
    main()
