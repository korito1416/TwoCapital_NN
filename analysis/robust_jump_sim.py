"""Worst-case (robust) connected four-regime jump simulator.

Takes the CANONICAL reference jump dynamics (models/SimulationStochasticJumps.py:step_state)
and adds the two robustness distortions, so the simulated economy is under the worst-case measure:

  1. drift distortion h = -(1/xi) sigma' dV   -> added to the (logK,Z,Y,logR) drifts
  2. jump-intensity distortion g^l = exp(-(V^l - V)/xi):
       damage jump fires at  sum_l (1/L) J_n(y) g^l   (worst case: sooner + tilt to worse lambda3)
       tech   jump fires at   J_g g^{tech}            (worst case: good breakthrough fired LESS)

With xi -> inf this reduces EXACTLY to the reference simulator (h->0, g->1): that is the
verification (`--verify` compares one path against the reference given the same seed).

Single-path (matches the reference structure for verifiability); a vectorised version for large
ensembles is a follow-up. Never touches models/ (this lives in analysis/).
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import tensorflow as tf  # noqa: E402
from SimulationStochasticJumps import (  # noqa: E402
    RegimeModels, PARAMS, initial_state, draw_damage_lambda3, xi_label, STAGE_SPECS,
)

TF32 = tf.float32


class RobustJumpSim:
    """Wraps RegimeModels; adds worst-case drift (h) and jump-intensity (g) distortions."""

    def __init__(self, export_folder, xi, batch_size=8):
        self.ev = RegimeModels(export_folder, xi, batch_size=batch_size, include_all_stages=False)
        self.xi = float(xi)

    # --- value and its state-gradient for the CURRENT regime (for h) ---
    def value_and_grad(self, state):
        stage = self.ev.stage_for_state(int(state["tech_state"]), int(state["damage_state"]))
        model = self.ev.models[stage]
        vec = self.ev.state_vector(stage, state)          # python list, length 6/7/8
        x = tf.constant([vec], dtype=TF32)
        # differentiate v_nn w.r.t. the FIRST FOUR inputs = (logK, Z, Y, logR-or-Ag) real states
        with tf.GradientTape() as tape:
            tape.watch(x)
            v = model.v_nn(x, training=False)
        g = tape.gradient(v, x)[0].numpy()                # (width,)
        return float(tf.squeeze(v).numpy()), g            # g[0:4] = d/d(logK,Z,Y, 4th slot)

    # --- value of a TARGET (post-jump) regime evaluated at the given post-jump state (for g) ---
    def value_at(self, tech_state, damage_state, state, lambda3=None, y=None):
        stage = self.ev.stage_for_state(int(tech_state), int(damage_state))
        s = dict(state); s["tech_state"] = tech_state; s["damage_state"] = damage_state
        if lambda3 is not None:
            s["lambda3"] = lambda3
        if y is not None:
            s["Y"] = y
        if int(tech_state) >= 2:                          # post-tech: green productivity = breakthrough
            s["A_g"] = float(PARAMS["A_g_prime_prime"])
        vec = self.ev.state_vector(stage, s)
        v = self.ev.models[stage].v_nn(tf.constant([vec], dtype=TF32), training=False)
        return float(tf.squeeze(v).numpy())

    def _g(self, cont, cur):
        return float(np.exp(np.clip(-(cont - cur) / self.xi, -50.0, 50.0)))

    def robust_step(self, state, policy, rng, dt, robust=True):
        p = PARAMS
        log_k, z, y, log_r = state["logK"], state["Z"], state["Y"], state["logR"]
        tech_state, damage_state = int(state["tech_state"]), int(state["damage_state"])
        k = float(np.exp(log_k))
        i_d, i_g = float(policy["i_d"]), float(policy["i_g"])
        i_r = float(policy["i_r"]) if not np.isnan(policy["i_r"]) else 0.0
        V_cur = float(policy["V"])

        sd, sg, sk = float(p["σ_d"]), float(p["σ_g"]), float(p["σ_κ"])
        ad, ag_, gd, gg = float(p["α_d"]), float(p["α_g"]), float(p["Γ_d"]), float(p["Γ_g"])
        td, tg = float(p["θ_d"]), float(p["θ_g"])
        inside_d = max(1.0 + td * i_d, 1e-8); inside_g = max(1.0 + tg * i_g, 1e-8)
        phid = ad + gd * np.log(inside_d); phig = ag_ + gg * np.log(inside_g)

        vkk = 0.5 * (sd**2 * (1 - z) ** 2 + sg**2 * z**2)
        drift_lk = phid * (1 - z) + phig * z - vkk
        drift_z = (phig - phid - z * sg**2 + (1 - z) * sd**2) * z * (1 - z)
        E = float(p["η"]) * float(p["A_d"]) * (1 - z) * k
        drift_y = float(p["θ_bar"]) * E
        diff_y = float(p["ϛ"]) * E

        # ---- robust DRIFT distortion h = -(1/xi) sigma' dV ----
        if robust:
            _, g4 = self.value_and_grad(state)
            v_lk, v_z, v_y = g4[0], g4[1], g4[2]
            v_lr = g4[3] if tech_state < 2 else 0.0        # post-tech: 4th slot is A_g, no logR channel
            xi = self.xi
            h_d = -((v_lk - z * v_z) * (1 - z) * sd) / xi
            h_g = -((v_lk + (1 - z) * v_z) * z * sg) / xi
            h_y = -(v_y * diff_y) / xi
            h_r = -(v_lr * sk) / xi
            drift_lk += sd * (1 - z) * h_d + sg * z * h_g
            drift_z += -sd * z * (1 - z) * h_d + sg * z * (1 - z) * h_g
            drift_y += diff_y * h_y
        else:
            h_r = 0.0

        dwg, dwd, dwy, dwr = rng.normal(0.0, np.sqrt(dt), size=4)
        ns = dict(state)
        ns["logK"] = log_k + drift_lk * dt + sd * (1 - z) * dwd + sg * z * dwg
        ns["Z"] = float(np.clip(z + drift_z * dt - sd * z * (1 - z) * dwd + sg * z * (1 - z) * dwg, 1e-4, 0.9999))
        ns["Y"] = max(0.0, y + drift_y * dt + diff_y * dwy)
        if tech_state < 2:
            drift_lr = (-float(p["ζ"]) + float(p["ψ0"]) * np.exp(float(p["ψ1"]) * (np.log(max(i_r, 1e-12)) + log_k - log_r))
                        - 0.5 * sk**2 + (sk * h_r if robust else 0.0))
            ns["logR"] = log_r + drift_lr * dt + sk * dwr
        else:
            ns["logR"] = 0.0

        tech_event, damage_event = "", False

        # ---- DAMAGE jump with worst-case intensity sum_l (1/L) J_n g^l ----
        if damage_state == 0:
            yq = ns["Y"]
            jn = float(p["r1"]) * (np.exp(float(p["r2"]) / 2.0 * (yq - float(p["y_lower"])) ** 2) - 1.0)
            if yq <= float(p["y_lower"]):
                jn = 0.0
            L = int(PARAMS["L"]); yhat = float(p["y_upper"])
            lam_vals = list(PARAMS["λ3_values"])
            if robust and jn > 0.0:
                gs = [self._g(self.value_at(tech_state, 1, ns, lambda3=lm, y=yhat), V_cur) for lm in lam_vals]
            else:
                gs = [1.0] * L
            j_each = [(1.0 / L) * jn * gs[i] for i in range(L)]
            j_tot = sum(j_each)
            if rng.random() < 1.0 - np.exp(-j_tot * dt):
                damage_event = True
                ns["damage_state"] = 1
                probs = np.array(j_each) / max(j_tot, 1e-30)
                ns["lambda3"] = float(lam_vals[rng.choice(L, p=probs)])   # worst-case tilt over lambda3
                ns["Y"] = yhat

        # ---- TECH jump with worst-case intensity J_g g^{tech} ----
        if tech_state == 0:
            jg = self.ev.tech_jump_intensity_scale * np.exp(ns["logR"]) / float(p["varrho"])
            if self.ev.one_tech_jump_mode or self.ev.pi >= 1.0 - 1e-12:
                if robust and jg > 0.0:
                    g_tech = self._g(self.value_at(2, damage_state, ns), V_cur)
                else:
                    g_tech = 1.0
                if rng.random() < 1.0 - np.exp(-jg * g_tech * dt):
                    ns["tech_state"] = 2; ns["A_g"] = float(p["A_g_prime_prime"]); ns["logR"] = 0.0
                    tech_event = "tech_0_to_2"
            else:                                          # two-stage (pi<1) not used at pi=1; kept for completeness
                g1 = self._g(self.value_at(1, damage_state, ns), V_cur) if robust else 1.0
                g2 = self._g(self.value_at(2, damage_state, ns), V_cur) if robust else 1.0
                ji = self.ev.tech_jump_intensity_scale * (1 - self.ev.pi) * np.exp(ns["logR"]) / float(p["varrho"]) * g1
                jf = self.ev.tech_jump_intensity_scale * self.ev.pi * np.exp(ns["logR"]) / float(p["varrho"]) * g2
                if rng.random() < 1.0 - np.exp(-(ji + jf) * dt):
                    to2 = rng.random() < jf / max(ji + jf, 1e-30)
                    ns["tech_state"] = 2 if to2 else 1
                    ns["A_g"] = float(p["A_g_prime_prime"] if to2 else p["A_g_prime"])
                    if to2:
                        ns["logR"] = 0.0
                    tech_event = "tech_0_to_2" if to2 else "tech_0_to_1"
        elif tech_state == 1:
            jg = self.ev.tech_jump_intensity_scale * self.ev.pi * np.exp(ns["logR"]) / float(p["varrho"])
            g2 = self._g(self.value_at(2, damage_state, ns), V_cur) if robust else 1.0
            if rng.random() < 1.0 - np.exp(-jg * g2 * dt):
                ns["tech_state"] = 2; ns["A_g"] = float(p["A_g_prime_prime"]); ns["logR"] = 0.0
                tech_event = "tech_1_to_2"

        return ns, tech_event, damage_event

    def simulate(self, n_paths, years, dt, seed, y0, robust=True, store_points=61):
        rng = np.random.default_rng(seed)
        steps = int(round(years / dt)); every = max(1, steps // store_points)
        keys_alloc = ["C/Y", "I_d/Y", "I_g/Y", "I_r/Y"]
        keys_mv = ["V_logK", "V_Z", "V_Y", "V_logR"]
        acc = {k: 0.0 for k in keys_alloc + keys_mv}; cnt = 0
        regime_time = np.zeros(4)                          # fraction of (path,time) in each regime code
        Ad = float(PARAMS["A_d"])
        for _ in range(n_paths):
            state = initial_state(y0); policy = self.ev.evaluate(state)
            for s in range(steps + 1):
                if s % every == 0:
                    z = state["Z"]; ag = self.ev.green_productivity(int(state["tech_state"]), state["A_g"])
                    yk = Ad * (1 - z) + ag * z
                    _, g4 = self.value_and_grad(state)
                    ir = 0.0 if np.isnan(policy["i_r"]) else policy["i_r"]
                    acc["C/Y"] += policy["C_over_K"] / yk
                    acc["I_d/Y"] += policy["i_d"] * (1 - z) / yk
                    acc["I_g/Y"] += policy["i_g"] * z / yk
                    acc["I_r/Y"] += ir / yk
                    acc["V_logK"] += g4[0]; acc["V_Z"] += g4[1]; acc["V_Y"] += g4[2]
                    acc["V_logR"] += g4[3] if int(state["tech_state"]) < 2 else 0.0
                    cnt += 1
                    code = 2 * (1 if int(state["tech_state"]) >= 2 else 0) + (1 if int(state["damage_state"]) >= 1 else 0)
                    regime_time[code] += 1
                if s < steps:
                    state, _, _ = self.robust_step(state, policy, rng, dt, robust=robust)
                    policy = self.ev.evaluate(state)
        return {k: acc[k] / cnt for k in acc}, regime_time / regime_time.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-folder", required=True)
    ap.add_argument("--xi", type=float, required=True)
    ap.add_argument("--n-paths", type=int, default=100)
    ap.add_argument("--years", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=1.0 / 12.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--y0", type=float, default=1.2)
    ap.add_argument("--reference", action="store_true", help="robustness OFF (h=0,g=1) -> should match reference sim")
    ap.add_argument("--verify", action="store_true", help="one path, reference mode, print state at a few steps")
    args = ap.parse_args()
    sim = RobustJumpSim(args.export_folder, args.xi)
    if args.verify:
        rng = np.random.default_rng(args.seed)
        st = initial_state(args.y0); pol = sim.ev.evaluate(st)
        for s in range(1, 25):
            st, te, de = sim.robust_step(st, pol, rng, args.dt, robust=False)
            pol = sim.ev.evaluate(st)
            if s in (1, 6, 12, 24):
                print(f"step {s:3d}: logK={st['logK']:.5f} Z={st['Z']:.5f} Y={st['Y']:.5f} "
                      f"logR={st['logR']:.5f} tech={st['tech_state']} dmg={st['damage_state']}")
        return
    tables, regime_frac = sim.simulate(args.n_paths, args.years, args.dt, args.seed, args.y0,
                                       robust=not args.reference)
    tag = "REFERENCE(robust off)" if args.reference else f"ROBUST xi={args.xi}"
    print(f"[{tag}] n_paths={args.n_paths}  pooled path-average:")
    for k in ["C/Y", "I_d/Y", "I_g/Y", "I_r/Y"]:
        print(f"  {k:8} {tables[k]:+.4f}")
    for k in ["V_logK", "V_Z", "V_Y", "V_logR"]:
        print(f"  {k:8} {tables[k]:+.5f}")
    print(f"  regime time-share [PreDPreT,PostDPreT,PreDPostT,PostDPostT] = "
          + " ".join(f"{x:.3f}" for x in regime_frac))


if __name__ == "__main__":
    main()
