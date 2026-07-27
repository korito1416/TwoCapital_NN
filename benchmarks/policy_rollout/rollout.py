"""Monte Carlo policy-evaluation rollout — the referee panel's arbitration experiment.

For each delivered solution: freeze its policy networks, simulate N stochastic paths
(Brownian shocks + damage/tech jumps, baseline law), accumulate realized discounted
utility

    J = E[ sum_t e^{-delta t} * delta * ( log(C/K)_t + logK_t - logN(Y_t) ) * dt ]
        (+ e^{-delta T} * V_net(x_T) terminal bootstrap, reported with and without)

and compare (a) J(own policy) against the network's claimed V(x0) — internal
consistency; (b) J across policies under common random numbers — dominance (the
planner problem is a max: a solution whose claimed V is higher but whose policy
delivers lower realized welfare has an overstated value function).

Conventions replicated EXACTLY from models/SimulationStochasticJumps.py step_state
(Euler steps, Z clip, Y floor, damage jump pins Y := y_upper, lambda3 drawn uniform,
OneJump pi=1 goes straight to breakthrough, i_r = exp(-i_r_nn)) and from the regime
files' value transform: the nets output v = V + logN, with
  pre-damage : logN = l1*Y + l2/2*Y^2
  post-damage: logN = l1*Y + l2/2*Y^2 + l3/2*(Y - y_upper)^2
(integrals of the (logN)_y lines at PreDamagePreTech.py:216 / PostDamagePreTech.py:213).

At xi large (e.g. 148.6, effectively uncertainty-neutral) J is the plain expected
utility and the comparison is exact. At small xi the same rollout evaluates policies
under the BASELINE law (not the robust value); reported with that caveat.

Usage:
  python rollout.py --run "reference=/abs/run" --run "nber_s1=/abs/run" \
      --xi 148.6 --n-paths 2000 --years 600 --seed 0 --out rollout_xi148.npz
  python rollout.py --self-test --run "reference=/abs/run"
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models"))
import tensorflow as tf
import SimulationStochasticJumps as SSJ
from params import PARAMS

# All runs in this study are OneJump pi=1; arm directories lack the marker files the
# inference helpers read (RUNA has them), so pin the mode explicitly.
SSJ.infer_one_tech_jump_mode = lambda folder: True
SSJ.infer_tech_jump_probability = lambda folder: 1.0
SSJ.infer_tech_jump_intensity_scale = lambda folder: 1.0

P = {k: float(PARAMS[k]) for k in
     ["A_d", "σ_d", "σ_g", "σ_κ", "α_d", "α_g", "Γ_d", "Γ_g", "θ_d", "θ_g",
      "η", "θ_bar", "ϛ", "ζ", "ψ0", "ψ1", "r1", "r2", "y_lower", "y_upper",
      "λ1", "λ2", "varrho", "δ", "K0", "Z0", "R0"]}
L3_VALUES = np.asarray(PARAMS["λ3_values"], dtype=np.float64)
Y0 = 1.1


def log_n(Y, lam3, damage_state):
    base = P["λ1"] * Y + 0.5 * P["λ2"] * Y ** 2
    post = 0.5 * lam3 * (Y - P["y_upper"]) ** 2
    return base + np.where(damage_state > 0, post, 0.0)


class Batched:
    """Batched policy/value evaluation on RegimeModels' nets (float32 in, float64 out)."""

    def __init__(self, root, xi):
        self.ev = SSJ.RegimeModels(root, xi)
        self.lx = float(np.log(xi))

    def eval_group(self, stage, logK, Z, Y, logR, lam3, a_g):
        n = len(logK)
        lx = np.full(n, self.lx)
        cols = {
            "PreDamagePreTech":  [logK, Z, Y, logR, lx, lx, lx],
            "PostDamagePreTech": [logK, Z, Y, logR, lam3, lx, lx, lx],
            "PreDamagePostTech": [logK, Z, Y, a_g, lx, lx],
            "PostDamagePostTech": [logK, Z, Y, lam3, a_g, lx, lx],
        }[stage]
        x = tf.constant(np.stack(cols, axis=1), dtype=tf.float32)
        m = self.ev.models[stage]
        v = m.v_nn(x, training=False).numpy().ravel().astype(np.float64)
        i_g = m.i_g_nn(x, training=False).numpy().ravel().astype(np.float64)
        i_d = m.i_d_nn(x, training=False).numpy().ravel().astype(np.float64)
        if SSJ.STAGE_SPECS[stage][2]:
            i_r = np.exp(-m.i_r_nn(x, training=False).numpy().ravel().astype(np.float64))
        else:
            i_r = np.zeros(n)
        return v, i_g, i_d, i_r


STAGES = {(0, 0): "PreDamagePreTech", (0, 1): "PostDamagePreTech",
          (2, 0): "PreDamagePostTech", (2, 1): "PostDamagePostTech"}


def rollout(root, xi, n, years, dt, draws, terminal_bootstrap=True, keep_diag=True, trace=None,
            no_jumps=False, record_stride=0):
    be = Batched(root, xi)
    scale = be.ev.tech_jump_intensity_scale
    W, U_dmg, U_tech, L3_IDX = draws          # (T,n,4), (T,n), (T,n), (T,n)
    T = W.shape[0]

    logK = np.full(n, np.log(P["K0"]));  Z = np.full(n, P["Z0"])
    Y = np.full(n, Y0);                  logR = np.full(n, np.log(P["R0"]))
    lam3 = np.zeros(n); a_g = np.zeros(n)
    tech = np.zeros(n, dtype=int); dmg = np.zeros(n, dtype=int)

    J = np.zeros(n)
    rec = {"t": [], "Ig": [], "Id": [], "Ir": [], "lk": [], "Z": [], "Y": [], "lr": [],
           "tech": [], "dmg": []} if record_stride else None
    HORIZONS = [60, 100, 200, 400]
    snaps = {}
    diag = {"Ig60": None, "mean_C_over_K_min": np.full(n, np.inf),
            "oob_logK": 0.0, "oob_Y": 0.0, "oob_logR": 0.0, "steps": 0}
    logK_hist = {}
    v0 = None

    for t in range(T):
        i_g = np.empty(n); i_d = np.empty(n); i_r = np.empty(n); v = np.empty(n)
        for (ts, ds), stage in STAGES.items():
            idx = np.where((tech == ts) & (dmg == ds))[0]
            if len(idx) == 0:
                continue
            vv, gg, dd_, rr = be.eval_group(stage, logK[idx], Z[idx], Y[idx],
                                            logR[idx], lam3[idx], a_g[idx])
            v[idx] = vv; i_g[idx] = gg; i_d[idx] = dd_; i_r[idx] = rr
        if t == 0:
            v0 = v.copy()

        a_g_cur = np.where(tech == 2, a_g, float(PARAMS["A_g"]))
        c_over_k = (P["A_d"] - i_d) * (1.0 - Z) + (a_g_cur - i_g) * Z - i_r
        c_safe = np.maximum(c_over_k, 1e-8)
        diag["mean_C_over_K_min"] = np.minimum(diag["mean_C_over_K_min"], c_over_k)

        flow = P["δ"] * (np.log(c_safe) + logK - log_n(Y, lam3, dmg))
        J += np.exp(-P["δ"] * t * dt) * flow * dt
        if rec is not None and t % record_stride == 0:
            K_lvl = np.exp(logK)
            rec["t"].append(t * dt)
            rec["Ig"].append((i_g * K_lvl * Z).astype(np.float32))
            rec["Id"].append((i_d * K_lvl * (1 - Z)).astype(np.float32))
            rec["Ir"].append((i_r * K_lvl).astype(np.float32))
            rec["lk"].append(logK.astype(np.float32)); rec["Z"].append(Z.astype(np.float32))
            rec["Y"].append(Y.astype(np.float32));    rec["lr"].append(logR.astype(np.float32))
            rec["tech"].append(tech.astype(np.int8)); rec["dmg"].append(dmg.astype(np.int8))

        # ----- Euler step (mirrors step_state) -----
        ins_d = np.maximum(1.0 + P["θ_d"] * i_d, 1e-8)
        ins_g = np.maximum(1.0 + P["θ_g"] * i_g, 1e-8)
        vkk = 0.5 * (P["σ_d"] ** 2 * (1 - Z) ** 2 + P["σ_g"] ** 2 * Z ** 2)
        drift_k = (P["α_d"] + P["Γ_d"] * np.log(ins_d)) * (1 - Z) \
                + (P["α_g"] + P["Γ_g"] * np.log(ins_g)) * Z - vkk
        drift_z = (P["α_g"] + P["Γ_g"] * np.log(ins_g)
                   - (P["α_d"] + P["Γ_d"] * np.log(ins_d))
                   - Z * P["σ_g"] ** 2 + (1 - Z) * P["σ_d"] ** 2) * Z * (1 - Z)
        emissions = P["η"] * P["A_d"] * (1 - Z) * np.exp(logK)
        dWg, dWd, dWy, dWr = W[t, :, 0], W[t, :, 1], W[t, :, 2], W[t, :, 3]

        # all updates computed from the OLD state (mirrors step_state exactly)
        pre = tech < 2
        drift_r = (-P["ζ"] + P["ψ0"] * np.exp(P["ψ1"] * (np.log(np.maximum(i_r, 1e-12))
                   + logK - logR)) - 0.5 * P["σ_κ"] ** 2)
        logK_new = logK + drift_k * dt + P["σ_d"] * (1 - Z) * dWd + P["σ_g"] * Z * dWg
        Z_new = np.clip(Z + drift_z * dt - P["σ_d"] * Z * (1 - Z) * dWd
                        + P["σ_g"] * Z * (1 - Z) * dWg, 1e-4, 0.9999)
        Y_new = np.maximum(0.0, Y + P["θ_bar"] * emissions * dt + P["ϛ"] * emissions * dWy)
        logR_new = np.where(pre, logR + drift_r * dt + P["σ_κ"] * dWr, 0.0)
        logK, Z, Y, logR = logK_new, Z_new, Y_new, logR_new

        # ----- damage jump (intensity at the NEW Y, matching step_state) -----
        can_dmg = dmg == 0
        j_dmg = P["r1"] * (np.exp(0.5 * P["r2"] * (Y - P["y_lower"]) ** 2) - 1.0)
        j_dmg = np.where(Y <= P["y_lower"], 0.0, j_dmg)
        hit_d = can_dmg & (U_dmg[t] < 1.0 - np.exp(-j_dmg * dt))
        if no_jumps:
            hit_d = np.zeros(n, dtype=bool)
        dmg = np.where(hit_d, 1, dmg)
        lam3 = np.where(hit_d, L3_VALUES[L3_IDX[t]], lam3)
        Y = np.where(hit_d, P["y_upper"], Y)

        # ----- tech jump (OneJump pi=1: straight to breakthrough) -----
        can_tech = tech == 0
        j_tech = scale * np.exp(logR) / P["varrho"]
        hit_t = can_tech & (U_tech[t] < 1.0 - np.exp(-j_tech * dt))
        if no_jumps:
            hit_t = np.zeros(n, dtype=bool)
        tech = np.where(hit_t, 2, tech)
        a_g = np.where(hit_t, float(PARAMS["A_g_prime_prime"]), a_g)
        logR = np.where(tech == 2, 0.0, logR)

        if keep_diag:
            diag["oob_logK"] += float(np.mean(logK > 7.0)); diag["oob_Y"] += float(np.mean(Y > 4.0))
            diag["oob_logR"] += float(np.mean((logR < 1.0) | (logR > 6.0)) if np.any(tech < 2) else 0.0)
            diag["steps"] += 1
        if keep_diag and t == int(60 / dt):
            diag["Ig60"] = float(np.mean(i_g * np.exp(logK) * Z))
        for h in HORIZONS:
            if t == int(round(h / dt)) - 1:
                vh = np.empty(n)
                for (ts2, ds2), stage2 in STAGES.items():
                    idx2 = np.where((tech == ts2) & (dmg == ds2))[0]
                    if len(idx2):
                        vh[idx2] = be.eval_group(stage2, logK[idx2], Z[idx2], Y[idx2],
                                                 logR[idx2], lam3[idx2], a_g[idx2])[0]
                Vh = vh - log_n(Y, lam3, dmg)
                snaps[h] = {"J_trunc": J.copy(),
                            "J_boot": J + np.exp(-P["δ"] * h) * Vh,
                            "oob_logK_now": float(np.mean(logK > 7.0))}
        if t == T - int(10 / dt) - 1:
            logK_hist["T-10y"] = logK.copy()
        if t == T - 1:
            logK_hist["T"] = logK.copy()
            logK_hist["flow_T"] = flow.copy()
        if trace is not None:
            trace.append([logK[0], Z[0], Y[0], logR[0], int(tech[0]), int(dmg[0])])

    # analytic tail: integral_T^inf e^{-dt} flow ~ e^{-dT} (flow_T + d*g*(t-T)) with per-path g
    if "T" in logK_hist and "T-10y" in logK_hist:
        g_est = (logK_hist["T"] - logK_hist["T-10y"]) / 10.0
        J_tail = J + np.exp(-P["δ"] * T * dt) * (logK_hist["flow_T"] / P["δ"] + g_est / P["δ"])
    else:
        J_tail = J.copy()

    if terminal_bootstrap:
        vT = np.empty(n)
        for (ts, ds), stage in STAGES.items():
            idx = np.where((tech == ts) & (dmg == ds))[0]
            if len(idx):
                vT[idx] = be.eval_group(stage, logK[idx], Z[idx], Y[idx],
                                        logR[idx], lam3[idx], a_g[idx])[0]
        VT = vT - log_n(Y, lam3, dmg)
        J_boot = J + np.exp(-P["δ"] * T * dt) * VT
    else:
        J_boot = J

    V0 = float(v0[0]) - float(log_n(np.array([Y0]), np.array([0.0]), np.array([0]))[0])
    for k in ("oob_logK", "oob_Y", "oob_logR"):
        diag[k] = diag[k] / max(diag["steps"], 1)
    return {"J": J, "J_boot": J_boot, "J_tail": J_tail, "V0_claimed": V0, "diag": diag,
            "snaps": snaps, "rec": rec,
            "frac_damage": float(np.mean(dmg)), "frac_tech": float(np.mean(tech == 2))}


def self_test(root, xi, seed=0):
    """Bit-level agreement of one vectorized path with step_state over 3 years."""
    dt = 1.0 / 12.0; T = 36; n = 1
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, np.sqrt(dt), size=(T, n, 4))
    U_dmg = rng.random((T, n)); U_tech = rng.random((T, n))
    L3_IDX = rng.integers(0, len(L3_VALUES), size=(T, n))

    ev = SSJ.RegimeModels(root, xi)
    state = SSJ.initial_state(Y0)

    class FakeRng:
        def __init__(self): self.t = 0
        def normal(self, loc, scale, size=4):
            w = W[self.t, 0]
            return np.array([w[0], w[1], w[2], w[3]])  # (d_w_g, d_w_d, d_w_y, d_w_r)
        def random(self):
            u = self._slots.pop(0); return u
        def integers(self, lo, hi): return L3_IDX[self.t, 0]

    fk = FakeRng()
    traj = []
    for t in range(T):
        fk.t = t
        pol = ev.evaluate(state)
        # step_state consumes: normals; then damage uniform iff damage_state==0; then tech iff tech<2
        fk._slots = []
        if state["damage_state"] == 0: fk._slots.append(U_dmg[t, 0])
        if state["tech_state"] == 0 or state["tech_state"] == 1: fk._slots.append(U_tech[t, 0])
        state, _, _ = SSJ.step_state(state, pol, ev, fk, dt)
        traj.append([state["logK"], state["Z"], state["Y"], state["logR"],
                     state["tech_state"], state["damage_state"]])
    ref = np.array(traj)

    # vectorized replay with identical draws + trajectory trace
    tr = []
    rollout(root, xi, n, years=T * dt, dt=dt, draws=(W, U_dmg, U_tech, L3_IDX),
            terminal_bootstrap=False, keep_diag=False, trace=tr)
    vec = np.array(tr)
    diff = np.abs(ref[:, :4] - vec[:, :4])
    err = np.max(diff)
    states_ok = np.array_equal(ref[:, 4:].astype(int), vec[:, 4:].astype(int))
    print(f"self-test: max |state diff| over 36 steps = {err:.2e}; jump states identical = {states_ok}")
    if err >= 1e-9:
        bad = np.argwhere(diff > 1e-12)
        t0, c0 = bad[0]
        names = ["logK", "Z", "Y", "logR"]
        print(f"FIRST divergence at step {t0}, coord {names[c0]}:")
        print("  step_state row:", ref[t0])
        print("  vectorized row:", vec[t0])
        if t0 > 0:
            print("  prev (equal)  :", ref[t0 - 1])
        print("  per-coord diff at t0:", diff[t0])
    assert err < 1e-9 and states_ok, "vectorized stepper DIVERGES from step_state"
    print("SELF-TEST PASSED (vectorized stepper == step_state)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=/abs/run/root")
    ap.add_argument("--xi", type=float, default=148.6)
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--years", type=float, default=600.0)
    ap.add_argument("--dt", type=float, default=1.0 / 12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rollout.npz")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-jumps", action="store_true")
    ap.add_argument("--record-stride", type=int, default=0)
    a = ap.parse_args()
    runs = [r.split("=", 1) for r in a.run]

    if a.self_test:
        self_test(runs[0][1], a.xi, a.seed)
        return

    T = int(round(a.years / a.dt)); n = a.n_paths
    rng = np.random.default_rng(a.seed)
    draws = (rng.normal(0.0, np.sqrt(a.dt), size=(T, n, 4)),
             rng.random((T, n)), rng.random((T, n)),
             rng.integers(0, len(L3_VALUES), size=(T, n)))

    out = {}
    Js = {}
    for label, root in runs:
        r = rollout(root, a.xi, n, a.years, a.dt, draws,
                    no_jumps=a.no_jumps, record_stride=a.record_stride)
        Js[label] = r
        se = float(np.std(r["J_tail"]) / np.sqrt(n))
        print(f"{label:14s} J_trunc={np.mean(r['J']):.4f}  J_tail={np.mean(r['J_tail']):.4f} (se {se:.4f})  "
              f"J_boot={np.mean(r['J_boot']):.4f}  V0_claimed={r['V0_claimed']:.4f}  "
              f"gap_tail={np.mean(r['J_tail']) - r['V0_claimed']:+.4f}  "
              f"minC/K={np.min(r['diag']['mean_C_over_K_min']):.4f}  "
              f"oob(logK,Y,logR)=({r['diag']['oob_logK']:.2f},{r['diag']['oob_Y']:.2f},{r['diag']['oob_logR']:.2f})  "
              f"P(dmg)={r['frac_damage']:.2f} P(tech)={r['frac_tech']:.2f} Ig60={r['diag']['Ig60']}")
        for h, sn in r["snaps"].items():
            print(f"    h={h:3d}y  J_trunc={np.mean(sn['J_trunc']):9.4f}  J_boot={np.mean(sn['J_boot']):9.4f}  oob_logK_now={sn['oob_logK_now']:.2f}")
            out[f"{label}|J{h}"] = sn["J_trunc"]; out[f"{label}|J{h}boot"] = sn["J_boot"]
        out[f"{label}|J"] = r["J"]; out[f"{label}|J_boot"] = r["J_boot"]; out[f"{label}|J_tail"] = r["J_tail"]
        out[f"{label}|V0"] = r["V0_claimed"]
        if r["rec"] is not None:
            out[f"{label}|rec_t"] = np.array(r["rec"]["t"])
            for q in ("Ig", "Id", "Ir", "lk", "Z", "Y", "lr", "tech", "dmg"):
                out[f"{label}|rec_{q}"] = np.stack(r["rec"][q])

    labels = [l for l, _ in runs]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for h in sorted(Js[labels[i]]["snaps"]):
                a_ = Js[labels[i]]["snaps"][h]; b_ = Js[labels[j]]["snaps"][h]
                for kind in ("J_trunc", "J_boot"):
                    d = a_[kind] - b_[kind]
                    print(f"paired d{kind}@{h}y  {labels[i]} - {labels[j]} = {np.mean(d):+.4f} (se {np.std(d)/np.sqrt(n):.4f})")
    np.savez(os.path.join(HERE, a.out), **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
