"""Vectorised worst-case connected four-regime jump simulator (batches all paths).

Same dynamics + robustness distortions as analysis/robust_jump_sim.py (which is verified exactly
against the canonical single-path reference), but every NN call is done on the full (n_paths) batch,
so thousands of paths are feasible. Regimes are handled by per-regime evaluation + selection.

pi=1 (OneJump) regimes, coded 0..3:
    0 PreDamagePreTech (tech0,dmg0)   1 PostDamagePreTech (tech0,dmg1)
    2 PreDamagePostTech(tech2,dmg0)   3 PostDamagePostTech(tech2,dmg1)
Damage jump: 0->1, 2->3.   Tech jump (breakthrough): 0->2, 1->3.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import tensorflow as tf  # noqa: E402
from SimulationStochasticJumps import RegimeModels, PARAMS  # noqa: E402

TF32 = tf.float32
REGIME_NAME = {0: "PreDamagePreTech", 1: "PostDamagePreTech", 2: "PreDamagePostTech", 3: "PostDamagePostTech"}


class VecRobustJumpSim:
    def __init__(self, export_folder, xi, batch_size=4096):
        self.ev = RegimeModels(export_folder, xi, batch_size=batch_size, include_all_stages=False)
        self.xi = np.float32(xi)
        self.lx = np.float32(np.log(xi))
        self.m = self.ev.models

    def _inp(self, regime, x, lam3, y_override=None):
        """tensor stage input (n, width) from x=(n,4)[logK,Z,Y,logR], lam3=(n,1). Differentiable in x."""
        logk, z, y, logr = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        if y_override is not None:
            y = tf.fill(tf.shape(y), np.float32(y_override))
        lxc = tf.fill(tf.shape(logk), self.lx)
        agpp = tf.fill(tf.shape(logk), np.float32(PARAMS["A_g_prime_prime"]))
        if regime == 0:      # PreDamagePreTech (7)
            return tf.concat([logk, z, y, logr, lxc, lxc, lxc], 1)
        if regime == 1:      # PostDamagePreTech (8)
            return tf.concat([logk, z, y, logr, lam3, lxc, lxc, lxc], 1)
        if regime == 2:      # PreDamagePostTech (6)
            return tf.concat([logk, z, y, agpp, lxc, lxc], 1)
        if regime == 3:      # PostDamagePostTech (7)
            return tf.concat([logk, z, y, lam3, agpp, lxc, lxc], 1)
        raise ValueError(regime)

    def _v(self, regime, x, lam3, y_override=None):
        return self.m[REGIME_NAME[regime]].v_nn(self._inp(regime, x, lam3, y_override), training=False)

    def _v_grad_controls(self, regime, x, lam3):
        """value, dV/dx (n,4), and controls i_d,i_g,i_r for `regime` on the batch."""
        mdl = self.m[REGIME_NAME[regime]]
        with tf.GradientTape() as tp:
            tp.watch(x)
            inp = self._inp(regime, x, lam3)
            v = mdl.v_nn(inp, training=False)
        grad = tp.gradient(v, x)                                  # (n,4); post-tech -> col3 ~ 0 (logR absent)
        i_d = mdl.i_d_nn(inp, training=False)
        i_g = mdl.i_g_nn(inp, training=False)
        i_r = tf.exp(-mdl.i_r_nn(inp, training=False)) if REGIME_NAME[regime] in (
            "PreDamagePreTech", "PostDamagePreTech") else tf.zeros_like(i_d)
        return v, grad, i_d, i_g, i_r

    def simulate(self, n_paths, years, dt, seed, y0, robust=True, store_points=61):
        p = PARAMS
        Ad, Ag_pre = np.float32(p["A_d"]), np.float32(p["A_g"])
        Agpp = np.float32(p["A_g_prime_prime"])
        sd, sg, sk = np.float32(p["σ_d"]), np.float32(p["σ_g"]), np.float32(p["σ_κ"])
        ad, ag_, gd, gg = (np.float32(p[k]) for k in ("α_d", "α_g", "Γ_d", "Γ_g"))
        td, tg = np.float32(p["θ_d"]), np.float32(p["θ_g"])
        eta, thb, sy = np.float32(p["η"]), np.float32(p["θ_bar"]), np.float32(p["ϛ"])
        zeta, ps0, ps1 = np.float32(p["ζ"]), np.float32(p["ψ0"]), np.float32(p["ψ1"])
        ylo, yhat, r1, r2 = (np.float32(p[k]) for k in ("y_lower", "y_upper", "r1", "r2"))
        lam1, lam2 = np.float32(p["λ1"]), np.float32(p["λ2"])
        varrho = np.float32(p["varrho"]); scale = np.float32(self.ev.tech_jump_intensity_scale)
        L = int(p["L"]); lam_vals = np.asarray(p["λ3_values"], np.float32)
        xi = self.xi

        rng = np.random.default_rng(seed)
        n = n_paths
        x = np.zeros((n, 4), np.float32)
        x[:, 0] = np.log(p["K0"]); x[:, 1] = p["Z0"]; x[:, 2] = y0; x[:, 3] = np.log(p["R0"])
        tech = np.zeros(n, np.int8); dmg = np.zeros(n, np.int8); lam3 = np.zeros(n, np.float32)

        steps = int(round(years / dt)); every = max(1, steps // store_points); sq = np.sqrt(dt)
        AK = {k: 0.0 for k in ["C/Y", "I_d/Y", "I_g/Y", "I_r/Y", "V_logK", "V_Z", "V_Y", "V_logR"]}
        cnt = 0; regime_time = np.zeros(4)
        from collections import defaultdict
        SERIES = defaultdict(list); TIMES = []
        def st3(a): return (float(np.nanmean(a)), float(np.nanpercentile(a, 10)), float(np.nanpercentile(a, 90)))

        def code(): return (2 * (tech >= 2).astype(int) + (dmg >= 1).astype(int))

        for s in range(steps + 1):
            xt = tf.constant(x, TF32); l3 = tf.constant(lam3[:, None], TF32)
            cc = code()
            # current-regime objects, per regime, then select
            V = np.zeros(n, np.float32); G = np.zeros((n, 4), np.float32)
            iD = np.zeros(n, np.float32); iG = np.zeros(n, np.float32); iR = np.zeros(n, np.float32)
            vals_at_x = {}                                      # regime -> value(n,) at current x (for tech g)
            for R in range(4):
                mask = cc == R
                v, g, i_d, i_g, i_r = self._v_grad_controls(R, xt, l3)
                vals_at_x[R] = v.numpy()[:, 0]
                if mask.any():
                    V[mask] = v.numpy()[mask, 0]; G[mask] = g.numpy()[mask]
                    iD[mask] = i_d.numpy()[mask, 0]; iG[mask] = i_g.numpy()[mask, 0]; iR[mask] = i_r.numpy()[mask, 0]

            z = x[:, 1]; logk = x[:, 0]; logr = x[:, 3]; k = np.exp(logk)
            ag_cur = np.where(tech >= 2, Agpp, Ag_pre)
            inside_d = np.maximum(1 + td * iD, 1e-8); inside_g = np.maximum(1 + tg * iG, 1e-8)
            phid = ad + gd * np.log(inside_d); phig = ag_ + gg * np.log(inside_g)
            vkk = 0.5 * (sd**2 * (1 - z) ** 2 + sg**2 * z**2)
            drift_lk = phid * (1 - z) + phig * z - vkk
            drift_z = (phig - phid - z * sg**2 + (1 - z) * sd**2) * z * (1 - z)
            E = eta * Ad * (1 - z) * k
            drift_y = thb * E; diff_y = sy * E

            # network outputs the transformed value (v = V + logN); true V_Y = dv_dY - (logN)_Y.
            # (logN)_Y = λ1 + λ2 Y  (pre-damage) + λ3 (Y - ŷ)  (post-damage; λ3=0 pre-damage -> one formula).
            logN_Y = lam1 + lam2 * x[:, 2] + lam3 * (x[:, 2] - yhat)
            if robust:
                v_lk, v_z, v_lr = G[:, 0], G[:, 1], np.where(tech < 2, G[:, 3], 0.0)
                v_y = G[:, 2] - logN_Y                              # true V_Y (transform-corrected)
                h_d = -((v_lk - z * v_z) * (1 - z) * sd) / xi
                h_g = -((v_lk + (1 - z) * v_z) * z * sg) / xi
                h_y = -(v_y * diff_y) / xi
                h_r = -(v_lr * sk) / xi
                drift_lk += sd * (1 - z) * h_d + sg * z * h_g
                drift_z += -sd * z * (1 - z) * h_d + sg * z * (1 - z) * h_g
                drift_y += diff_y * h_y
            else:
                h_r = np.zeros(n, np.float32)

            if s % every == 0:
                yk = Ad * (1 - z) + ag_cur * z
                cok = (Ad - iD) * (1 - z) + (ag_cur - iG) * z - np.where(tech < 2, iR, 0.0)
                AK["C/Y"] += float(np.mean(cok / yk)); AK["I_d/Y"] += float(np.mean(iD * (1 - z) / yk))
                AK["I_g/Y"] += float(np.mean(iG * z / yk)); AK["I_r/Y"] += float(np.mean(np.where(tech < 2, iR, 0.0) / yk))
                AK["V_logK"] += float(np.mean(G[:, 0])); AK["V_Z"] += float(np.mean(G[:, 1]))
                AK["V_Y"] += float(np.mean(G[:, 2] - logN_Y))     # report the economic V_Y, not the transformed gradient
                AK["V_logR"] += float(np.mean(np.where(tech < 2, G[:, 3], 0.0)))
                cnt += 1
                for R in range(4):
                    regime_time[R] += int((cc == R).sum())
                # ---- time-series recording (Haoyang-style path quantities, mean + 10/90 bands) ----
                K = np.exp(x[:, 0]); Rk = np.where(tech < 2, np.exp(x[:, 3]), np.nan)
                rd = np.where(tech < 2, iR, 0.0) / yk
                TIMES.append(s * dt)
                for nm, arr in [("K", K), ("Z", z), ("Y", x[:, 2]), ("R", Rk), ("E", E), ("A_g", ag_cur),
                                ("I_g", iG * z * K), ("I_d", iD * (1 - z) * K), ("C", cok * K), ("RD", rd),
                                ("I_g_Y", iG * z / yk), ("I_d_Y", iD * (1 - z) / yk), ("C_Y", cok / yk)]:
                    SERIES[nm].append(st3(arr))
                for R in range(4):
                    SERIES[f"regime{R}"].append((float((cc == R).mean()), 0.0, 0.0))
            if s == steps:
                break

            # diffuse
            dW = rng.normal(0.0, sq, size=(n, 4)).astype(np.float32)     # [g,d,y,r]
            x_new = x.copy()
            x_new[:, 0] = logk + drift_lk * dt + sd * (1 - z) * dW[:, 1] + sg * z * dW[:, 0]
            x_new[:, 1] = np.clip(z + drift_z * dt - sd * z * (1 - z) * dW[:, 1] + sg * z * (1 - z) * dW[:, 0], 1e-4, 0.9999)
            x_new[:, 2] = np.maximum(0.0, x[:, 2] + drift_y * dt + diff_y * dW[:, 2])
            pre = tech < 2
            drift_lr = -zeta + ps0 * np.exp(ps1 * (np.log(np.maximum(iR, 1e-12)) + logk - logr)) - 0.5 * sk**2 + (sk * h_r)
            x_new[:, 3] = np.where(pre, logr + drift_lr * dt + sk * dW[:, 3], 0.0)
            x = x_new

            # current-regime value re-evaluated at the POST-diffusion state (jumps fire post-diffusion,
            # so g = exp(-(V^post - V)/xi) must evaluate BOTH sides at this state, per the reference convention)
            if robust:
                xtp = tf.constant(x, TF32); l3p = tf.constant(lam3[:, None], TF32)
                Vpd = V.copy()
                for R in range(4):
                    m = cc == R
                    if m.any():
                        Vpd[m] = self._v(R, xtp, l3p).numpy()[m, 0]
            else:
                Vpd = V

            # ---- DAMAGE jump (dmg==0) ----
            dmg0 = dmg == 0
            if dmg0.any():
                yq = x[:, 2]
                jn = np.where(yq > ylo, r1 * (np.exp(r2 / 2 * (yq - ylo) ** 2) - 1.0), 0.0).astype(np.float32)
                xt2 = tf.constant(x, TF32)
                # post-damage continuation values at Y=yhat for each lambda3, for BOTH post-damage regimes
                gmat = np.ones((n, L), np.float32)
                if robust:
                    for li, lv in enumerate(lam_vals):
                        l3c = tf.fill((n, 1), np.float32(lv))
                        v_pre = self._v(1, xt2, l3c, y_override=yhat).numpy()[:, 0]   # PostDamagePreTech
                        v_pst = self._v(3, xt2, l3c, y_override=yhat).numpy()[:, 0]   # PostDamagePostTech
                        v_post = np.where(tech < 2, v_pre, v_pst)
                        gmat[:, li] = np.exp(np.clip(-(v_post - Vpd) / xi, -50, 50))
                j_each = (1.0 / L) * jn[:, None] * gmat                                # (n,L)
                j_tot = j_each.sum(1)
                fire = (rng.random(n) < (1 - np.exp(-j_tot * dt))) & dmg0 & (jn > 0)
                if fire.any():
                    probs = j_each[fire] / np.maximum(j_tot[fire, None], 1e-30)
                    u = rng.random(fire.sum())
                    idx = (np.cumsum(probs, 1) > u[:, None]).argmax(1)
                    lam3[fire] = lam_vals[idx]
                    dmg[fire] = 1
                    x[fire, 2] = yhat

            # ---- TECH jump (tech==0) -> breakthrough ----
            tech0 = tech == 0
            if tech0.any():
                jg = scale * np.exp(x[:, 3]) / varrho
                if robust:
                    xt3 = tf.constant(x, TF32); l3c = tf.constant(lam3[:, None], TF32)
                    v_post_tech = np.where(dmg < 1,
                                           self._v(2, xt3, l3c).numpy()[:, 0],   # PreDamagePostTech
                                           self._v(3, xt3, l3c).numpy()[:, 0])   # PostDamagePostTech
                    g_tech = np.exp(np.clip(-(v_post_tech - Vpd) / xi, -50, 50))
                else:
                    g_tech = np.ones(n, np.float32)
                fire = (rng.random(n) < (1 - np.exp(-jg * g_tech * dt))) & tech0
                if fire.any():
                    tech[fire] = 2; x[fire, 3] = 0.0

        tables = {k: AK[k] / cnt for k in AK}
        series = {k: np.array(v) for k, v in SERIES.items()}
        series["t"] = np.array(TIMES)
        return tables, regime_time / regime_time.sum(), series

    def simulate_irf(self, n_paths, years, dt, seed, y0, robust=True, store_points=61):
        """Common-random-number stochastic IRF on the coupled (worst-case) dynamics.
        Baseline + one perturbed group per structural Brownian channel (a 1-s.d. sigma-column shock
        to dirty-K, green-K, temperature, knowledge); all groups share the SAME Brownian increments
        and jump uniforms, so IRF_q = mean_paths(perturbed_q - baseline_q) is variance-reduced."""
        p = PARAMS
        Ad, Ag_pre, Agpp = np.float32(p["A_d"]), np.float32(p["A_g"]), np.float32(p["A_g_prime_prime"])
        sd, sg, sk = np.float32(p["σ_d"]), np.float32(p["σ_g"]), np.float32(p["σ_κ"])
        ad, ag_, gd, gg = (np.float32(p[k]) for k in ("α_d", "α_g", "Γ_d", "Γ_g"))
        td, tg = np.float32(p["θ_d"]), np.float32(p["θ_g"])
        eta, thb, sy = np.float32(p["η"]), np.float32(p["θ_bar"]), np.float32(p["ϛ"])
        zeta, ps0, ps1 = np.float32(p["ζ"]), np.float32(p["ψ0"]), np.float32(p["ψ1"])
        ylo, yhat, r1, r2 = (np.float32(p[k]) for k in ("y_lower", "y_upper", "r1", "r2"))
        lam1, lam2 = np.float32(p["λ1"]), np.float32(p["λ2"])
        varrho = np.float32(p["varrho"]); scale = np.float32(self.ev.tech_jump_intensity_scale)
        L = int(p["L"]); lam_vals = np.asarray(p["λ3_values"], np.float32); xi = self.xi
        K0, Z0, R0 = np.float32(p["K0"]), np.float32(p["Z0"]), np.float32(p["R0"])
        E0 = eta * Ad * (1 - Z0) * K0
        SH = [("DirtyCapital", np.array([sd * (1 - Z0), -sd * Z0 * (1 - Z0), 0, 0], np.float32)),
              ("GreenCapital", np.array([sg * Z0, sg * Z0 * (1 - Z0), 0, 0], np.float32)),
              ("Temperature",  np.array([0, 0, sy * E0, 0], np.float32)),
              ("Knowledge",    np.array([0, 0, 0, sk], np.float32))]
        names = [nm for nm, _ in SH]; Gn = 1 + len(SH); n = n_paths; N = Gn * n

        rng = np.random.default_rng(seed)
        base = np.array([np.log(K0), Z0, y0, np.log(R0)], np.float32)
        x = np.tile(base, (N, 1))
        for gi, (_, sh) in enumerate(SH):
            x[(gi + 1) * n:(gi + 2) * n] += sh
        tech = np.zeros(N, np.int8); dmg = np.zeros(N, np.int8); lam3 = np.zeros(N, np.float32)
        steps = int(round(years / dt)); every = max(1, steps // store_points); sq = np.sqrt(dt)
        QK = ["K", "Y", "R", "E", "C", "I_g", "I_d", "RD", "A_g", "posttech"]
        REC = {q: [] for q in QK}; TIMES = []

        def gmean(a): return a.reshape(Gn, n).mean(1)

        for s in range(steps + 1):
            xt = tf.constant(x, TF32); l3 = tf.constant(lam3[:, None], TF32)
            cc = 2 * (tech >= 2).astype(int) + (dmg >= 1).astype(int)
            V = np.zeros(N, np.float32); Gd = np.zeros((N, 4), np.float32)
            iD = np.zeros(N, np.float32); iG = np.zeros(N, np.float32); iR = np.zeros(N, np.float32)
            for R in range(4):
                m = cc == R
                v, g, i_d, i_g, i_r = self._v_grad_controls(R, xt, l3)
                if m.any():
                    V[m] = v.numpy()[m, 0]; Gd[m] = g.numpy()[m]
                    iD[m] = i_d.numpy()[m, 0]; iG[m] = i_g.numpy()[m, 0]; iR[m] = i_r.numpy()[m, 0]
            z = x[:, 1]; logk = x[:, 0]; logr = x[:, 3]; k = np.exp(logk)
            ag_cur = np.where(tech >= 2, Agpp, Ag_pre); yk = Ad * (1 - z) + ag_cur * z; E = eta * Ad * (1 - z) * k
            if s % every == 0:
                cok = (Ad - iD) * (1 - z) + (ag_cur - iG) * z - np.where(tech < 2, iR, 0.0)
                rd = np.where(tech < 2, iR, 0.0) / yk
                vals = {"K": k, "Y": x[:, 2], "R": np.where(tech < 2, np.exp(logr), 0.0), "E": E, "C": cok * k,
                        "I_g": iG * z * k, "I_d": iD * (1 - z) * k, "RD": rd, "A_g": ag_cur,
                        "posttech": (tech >= 2).astype(np.float32)}
                TIMES.append(s * dt)
                for q in QK:
                    REC[q].append(gmean(vals[q]))
            if s == steps:
                break
            inside_d = np.maximum(1 + td * iD, 1e-8); inside_g = np.maximum(1 + tg * iG, 1e-8)
            phid = ad + gd * np.log(inside_d); phig = ag_ + gg * np.log(inside_g)
            vkk = 0.5 * (sd**2 * (1 - z)**2 + sg**2 * z**2)
            drift_lk = phid * (1 - z) + phig * z - vkk
            drift_z = (phig - phid - z * sg**2 + (1 - z) * sd**2) * z * (1 - z)
            drift_y = thb * E; diff_y = sy * E
            if robust:
                logN_Y = lam1 + lam2 * x[:, 2] + lam3 * (x[:, 2] - yhat)      # (logN)_Y, regime-correct
                v_lk, v_z, v_lr = Gd[:, 0], Gd[:, 1], np.where(tech < 2, Gd[:, 3], 0.0)
                v_y = Gd[:, 2] - logN_Y                                        # true V_Y (transform-corrected)
                h_d = -((v_lk - z * v_z) * (1 - z) * sd) / xi; h_g = -((v_lk + (1 - z) * v_z) * z * sg) / xi
                h_y = -(v_y * diff_y) / xi; h_r = -(v_lr * sk) / xi
                drift_lk += sd * (1 - z) * h_d + sg * z * h_g
                drift_z += -sd * z * (1 - z) * h_d + sg * z * (1 - z) * h_g; drift_y += diff_y * h_y
            else:
                h_r = np.zeros(N, np.float32)
            dWb = rng.normal(0, sq, (n, 4)).astype(np.float32); dW = np.tile(dWb, (Gn, 1))   # CRN: shared shocks
            ud = np.tile(rng.random(n), Gn); ul = np.tile(rng.random(n), Gn); ut = np.tile(rng.random(n), Gn)
            xN = x.copy()
            xN[:, 0] = logk + drift_lk * dt + sd * (1 - z) * dW[:, 1] + sg * z * dW[:, 0]
            xN[:, 1] = np.clip(z + drift_z * dt - sd * z * (1 - z) * dW[:, 1] + sg * z * (1 - z) * dW[:, 0], 1e-4, 0.9999)
            xN[:, 2] = np.maximum(0.0, x[:, 2] + drift_y * dt + diff_y * dW[:, 2])
            pre = tech < 2
            drift_lr = -zeta + ps0 * np.exp(ps1 * (np.log(np.maximum(iR, 1e-12)) + logk - logr)) - 0.5 * sk**2 + sk * h_r
            xN[:, 3] = np.where(pre, logr + drift_lr * dt + sk * dW[:, 3], 0.0)
            x = xN
            # current-regime value at POST-diffusion state (consistent jump-distortion timing)
            if robust:
                xtp = tf.constant(x, TF32); l3p = tf.constant(lam3[:, None], TF32)
                Vpd = V.copy()
                for R in range(4):
                    m = cc == R
                    if m.any():
                        Vpd[m] = self._v(R, xtp, l3p).numpy()[m, 0]
            else:
                Vpd = V
            dmg0 = dmg == 0
            if dmg0.any():
                yq = x[:, 2]; jn = np.where(yq > ylo, r1 * (np.exp(r2 / 2 * (yq - ylo)**2) - 1.0), 0.0).astype(np.float32)
                xt2 = tf.constant(x, TF32); gmat = np.ones((N, L), np.float32)
                if robust:
                    for li, lv in enumerate(lam_vals):
                        l3c = tf.fill((N, 1), np.float32(lv))
                        v_post = np.where(tech < 2, self._v(1, xt2, l3c, y_override=yhat).numpy()[:, 0],
                                          self._v(3, xt2, l3c, y_override=yhat).numpy()[:, 0])
                        gmat[:, li] = np.exp(np.clip(-(v_post - Vpd) / xi, -50, 50))
                j_each = (1.0 / L) * jn[:, None] * gmat; j_tot = j_each.sum(1)
                fire = (ud < (1 - np.exp(-j_tot * dt))) & dmg0 & (jn > 0)
                if fire.any():
                    probs = j_each[fire] / np.maximum(j_tot[fire, None], 1e-30)
                    idx = (np.cumsum(probs, 1) > ul[fire, None]).argmax(1)
                    lam3[fire] = lam_vals[idx]; dmg[fire] = 1; x[fire, 2] = yhat
            tech0 = tech == 0
            if tech0.any():
                jg = scale * np.exp(x[:, 3]) / varrho
                if robust:
                    xt3 = tf.constant(x, TF32); l3c = tf.constant(lam3[:, None], TF32)
                    v_pt = np.where(dmg < 1, self._v(2, xt3, l3c).numpy()[:, 0], self._v(3, xt3, l3c).numpy()[:, 0])
                    g_tech = np.exp(np.clip(-(v_pt - Vpd) / xi, -50, 50))
                else:
                    g_tech = np.ones(N, np.float32)
                fire = (ut < (1 - np.exp(-jg * g_tech * dt))) & tech0
                if fire.any():
                    tech[fire] = 2; x[fire, 3] = 0.0

        out = {"t": np.array(TIMES), "shocks": np.array(names)}
        for q in QK:
            arr = np.array(REC[q])                       # (T, Gn)
            out[f"base_{q}"] = arr[:, 0]
            for gi, nm in enumerate(names):
                out[f"irf_{nm}_{q}"] = arr[:, gi + 1] - arr[:, 0]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-folder", required=True)
    ap.add_argument("--xi", type=float, required=True)
    ap.add_argument("--n-paths", type=int, default=2048)
    ap.add_argument("--years", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=1.0 / 12.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--y0", type=float, default=1.2)
    ap.add_argument("--reference", action="store_true")
    args = ap.parse_args()
    import time
    sim = VecRobustJumpSim(args.export_folder, args.xi, batch_size=args.n_paths)
    t0 = time.time()
    tables, rf, _series = sim.simulate(args.n_paths, args.years, args.dt, args.seed, args.y0, robust=not args.reference)
    tag = "REFERENCE" if args.reference else f"ROBUST xi={args.xi}"
    print(f"[{tag}] n_paths={args.n_paths} in {time.time()-t0:.0f}s  pooled path-average:")
    for k in ["C/Y", "I_d/Y", "I_g/Y", "I_r/Y"]:
        print(f"  {k:8} {tables[k]:+.4f}")
    for k in ["V_logK", "V_Z", "V_Y", "V_logR"]:
        print(f"  {k:8} {tables[k]:+.5f}")
    print(f"  regime time-share [PreDPreT,PostDPreT,PreDPostT,PostDPostT] = " + " ".join(f"{v:.3f}" for v in rf))


if __name__ == "__main__":
    main()
