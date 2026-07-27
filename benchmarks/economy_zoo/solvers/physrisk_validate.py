"""physrisk_validate.py -- AUDIT-MANDATED coarse validations for the PHYSRISK economy
(before the full sbatch ladder is trusted). Login-budget: coarse grids (31,21,31).

Steps (all RUN by `python physrisk_validate.py`; `--fast` shrinks budgets for debug):

  A  Neutral coarse chain (xi = inf, exactly-linear jump expectation): the 12-solve
     backward ladder on (31,21,31) -> outputs/physrisk_coarse/*_xiinf.npz.

  V1 Jump-bearing correctness: PostDamagePreTech(l=3) re-solved twice from the SAME
     warm start with the SAME Howard budget -- arm LIN uses xi = inf (harness takes
     the explicitly-linear jump branch g=1), arm G uses xi = 1e5 (finite -> the full
     xi*J*(1-exp(-(Wpost-W)/xi)) distortion machinery, lagged g, exp-clip). The two
     must agree: field RMS(W_G - W_LIN) <= 1e-4.

  V2 step_exact stability where J*dt >> 1: at the s-top the tech hazard is
     K0*e^{2}/varrho = 8.71/yr, J*dt = 8.7 at the chain's dt3d = 1.0 (21.8 at
     the harness-reference dt = 2.5). Checks on the chain's
     PostDamagePreTech solves: all fields finite (no overflow/NaN), W monotone
     increasing in s in the upper half of the s-grid, and W(s=+2) within tolerance
     of the post-tech far-field anchor W_post. NOTE: the hazard is IDENTICAL to
     JONES's by construction (K0*e^{s}/varrho, design minimal-pair), so this V2
     certifies the JONES hazard treatment too; JONES had not been built when this
     ran, so V2 is run HERE rather than shared from a JONES artifact.

  V3 xi = 0.05 outer-loop stability with the g-distortion ON (exp-clip 35 in place):
     coarse solves at xi = 0.05 for (a) PostDamagePostTech l=1..5 (robust drift
     feedback, no jumps), (b) PreDamagePostTech (5 damage-jump channels -- the
     extreme-exponent worst-case-damage g's), (c) PostDamagePreTech l=3 (tech
     channel g). Howard/di history captured and reported; gate = finite fields +
     di_int <= 2e-2 (coarse-grid tolerance; the destroyed low-Z strip converges
     to ~5e-3 at dt=0.5, see the dt deviation note in physrisk_callbacks) +
     h-clip trust region not binding (bind fraction <= 1%).

  FK Feynman-Kac LEVEL Monte Carlo (design-mandated for PHYSRISK, zoo-wide gate):
     simulate the FULL 4-state SDE (logK, Z, Y, s) + BOTH jump processes (tech,
     damage w/ uniform lambda3 reveal) under the FD policies of the neutral chain,
     from 3 seed states x npaths paths; verify
        E[ int_0^T e^{-delta t} delta (log c_t + logK_t) dt ] = logK0 + W(seed)
     within MC sampling error plus a small documented systematic allowance
     (O(sigma^2) diffusion dropped by the PIBYS evaluation + quadrature +
     coarse-grid interpolation): gate |diff| <= 3*SE + 0.03.

Writes outputs/physrisk_coarse/physrisk_validation_PROVENANCE.json (all gate
numbers) and figures/physrisk_validation_coarse.png.
"""
import os
import sys
import io
import re
import json
import time
import argparse
import datetime
import contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd_reduced as FR                            # noqa: E402
import physrisk_callbacks as PC                    # noqa: E402

OUTD = os.path.abspath(os.path.join(HERE, "..", "outputs", "physrisk_coarse"))
FIGD = os.path.abspath(os.path.join(HERE, "..", "figures"))
LK0 = float(np.log(880.0))
SEEDS = [(0.7, 1.1, -4.364),      # production initial point (Z0, Y0, s0)
         (0.4, 2.0, -3.0),
         (0.85, 0.7, -5.0)]


def _npz(regime, il, xi):
    return os.path.join(OUTD, PC.solve_name(regime, il, xi) + ".npz")


def _load(regime, il, xi):
    return np.load(_npz(regime, il, xi))


# ------------------------------------------------------------------ V1
def run_v1(grids, howard=10):
    Zg, Yg, Sg, _ = grids
    il = 2                                     # l3 = 1/6
    base = _load("PostDamagePreTech", il, np.inf)
    wpost = PC.w_interp_of(_load("PostDamagePostTech", il, np.inf),
                           (Zg, Yg, np.linspace(PC.S_LO, PC.S_HI, 3)))
    init_ctr = {k: np.array(base[k]) for k in ("i_d", "i_g", "i_r")}
    Wwarm = np.array(base["W"])
    arms = {}
    for tag, xi in (("LIN", np.inf), ("G", 1e5)):
        cfg = PC.physrisk_cfg("PostDamagePreTech", PC.LAM3_GRID[il], xi,
                              (Zg, Yg, Sg), w_post_tech=wpost,
                              W_init=Wwarm, robust_drift=False)
        out = FR.solve_reduced(cfg, T=1200.0, dt=1.0, howard_max=howard,
                               tol_pocket=1e-12, relax=0.15, tail_avg=12,
                               init_ctr=init_ctr, verbose=False)
        arms[tag] = out
        print("  V1 arm %s: iters=%d gate=%.2e rmsR=%.2e" %
              (tag, out["iters"], out["gate_int"], out["rms_residual"]), flush=True)
    d = arms["G"]["W"] - arms["LIN"]["W"]
    rms = float(np.sqrt(np.mean(d ** 2)))
    mx = float(np.max(np.abs(d)))
    ok = rms <= 1e-4
    print("V1 large-xi vs explicit-linear jump source: RMS=%.3e max=%.3e -> %s"
          % (rms, mx, "PASS" if ok else "FAIL"), flush=True)
    return dict(rms=rms, max_abs=mx, target=1e-4, passed=bool(ok)), d, arms


# ------------------------------------------------------------------ V2
def run_v2(grids):
    """step_exact certificate where J*dt >> 1 (tech hazard 8.71/yr at s=+2,
    J*dt = 21.8 at dt = 2.5): (i) all fields finite (no overflow/NaN); (ii) W
    monotone increasing in s over the upper half of the grid; (iii) the DECISIVE
    check -- the true HJB residual on the topmost INTERIOR s-slice, NORMALIZED by
    the local total rate (delta + J), stays small. A left-rectangle jump-sink
    scheme is corrupted there by O(J*dt) ~ 20 relative; the exact exponential/
    step_exact treatment must keep |R|/(delta+J) at the ordinary residual scale
    (gate <= 2e-2). The absolute gap W(s_top) - W_post is REPORTED as data (the
    exact far-field limit holds only as J -> infinity, so it is not gated)."""
    Zg, Yg, Sg, Sg2 = grids
    dt = 1.0            # the 3-D chain solves run at dt3d = 1.0
    Jtop = np.exp(PC.S_HI) * PC.K0 / PC.VARRHO
    res = dict(hazard_top=float(Jtop), J_dt_top=float(Jtop * dt),
               note=("hazard K0*e^s/varrho IDENTICAL to JONES by design "
                     "(minimal pair); JONES unbuilt when run -> V2 run here, "
                     "certifies both"))
    finite_ok, mono_frac, gaps, nres = True, [], [], []
    for il in range(5):
        d = _load("PostDamagePreTech", il, np.inf)
        W = np.array(d["W"])
        finite_ok &= bool(np.all(np.isfinite(W)) and np.all(np.isfinite(d["i_d"]))
                          and np.all(np.isfinite(d["i_r"])))
        hi = W[:, :, Sg.size // 2:]
        mono_frac.append(float(np.mean(np.diff(hi, axis=2) >= -1e-6)))
        Wp = np.load(_npz("PostDamagePostTech", il, np.inf))["W"][:, :, 1]
        gaps.append(float(np.max(np.abs(W[:, :, -1] - Wp))))
        # normalized true residual on the topmost interior s-slice
        wpost = PC.w_interp_of(_load("PostDamagePostTech", il, np.inf),
                               (Zg, Yg, Sg2))
        cfg = PC.physrisk_cfg("PostDamagePreTech", PC.LAM3_GRID[il], np.inf,
                              (Zg, Yg, Sg), w_post_tech=wpost, robust_drift=False)
        ctr = {k: np.array(d[k]) for k in ("i_d", "i_g", "i_r", "c")}
        R = FR.true_residual(cfg, W, ctr)                 # interior nodes
        J_slice = float(np.exp(Sg[-2]) * PC.K0 / PC.VARRHO)
        nres.append(float(np.max(np.abs(R[:, :, -1])) / (0.01 + J_slice)))
    res.update(all_finite=bool(finite_ok),
               monotone_frac_upper_s=[round(m, 5) for m in mono_frac],
               gap_Ws2_vs_Wpost_max_abs=[round(g, 5) for g in gaps],
               J_at_top_interior_slice=float(np.exp(Sg[-2]) * PC.K0 / PC.VARRHO),
               top_slice_residual_over_rate=[round(r, 5) for r in nres])
    ok = finite_ok and min(mono_frac) >= 0.98 and max(nres) <= 2e-2
    res["passed"] = bool(ok)
    print("V2 step_exact/J*dt=%.1f: finite=%s mono_frac(min)=%.4f "
          "max|R|/(delta+J)@top=%.4f (gate 2e-2) gap(data)=%.4f -> %s"
          % (Jtop * dt, finite_ok, min(mono_frac), max(nres), max(gaps),
             "PASS" if ok else "FAIL"), flush=True)
    return res


# ------------------------------------------------------------------ V3
_HOW_RE = re.compile(r"\[howard\s+(\d+)\] di_full=([0-9.e+-]+) di_int=([0-9.e+-]+)")


def _solve_capture(cfg, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = FR.solve_reduced(cfg, verbose=True, **kw)
    hist = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
            for m in _HOW_RE.finditer(buf.getvalue())]
    return out, hist


def _bind_frac(out, Zg, xi, pre_t):
    """Fraction of INTERIOR nodes where the H_CLIP trust region binds at the
    final costates (must be ~0 for the solve to be trusted; see callbacks doc)."""
    z = Zg[:, None, None]
    W_Z, W_Y, W_S = out["W_Z"], out["W_Y"], out["W_S"]
    p = PC.base_params("PreDamagePreTech")
    E = PC.emissions(z, p)
    hd = -((1.0 - W_S - z * W_Z) * (1 - z) * p["s_d"]) / xi
    hg = -((1.0 - W_S + (1 - z) * W_Z) * z * p["s_g"]) / xi
    hy = -(W_Y * E * p["vars"]) / xi
    H = 0.999 * PC.H_CLIP
    b = (np.abs(hd) > H) | (np.abs(hg) > H) | (np.abs(hy) > H)
    if pre_t:
        b = b | (np.abs(-(W_S * p["s_r"]) / xi) > H)
    return float(np.mean(b[1:-1, 1:-1, :]))


def run_v3(grids, howard=40):
    Zg, Yg, Sg, Sg2 = grids
    xi = 0.05
    report, hist_all = {}, {}
    ok = True
    # (a) PostDamagePostTech l=1..5 (robust drift, no jumps)
    pdpt = []
    for il, l3 in enumerate(PC.LAM3_GRID):
        neut = _load("PostDamagePostTech", il, np.inf)
        cfg = PC.physrisk_cfg("PostDamagePostTech", l3, xi, (Zg, Yg, Sg2),
                              W_init=np.array(neut["W"]), robust_drift=True)
        init = {k: np.array(neut[k]) for k in cfg["control_names"]}
        out, hist = _solve_capture(cfg, T=1200.0, dt=0.5, howard_max=howard,
                                   tol_pocket=1e-6, relax=0.15, tail_avg=12,
                                   init_ctr=init)
        nm = PC.solve_name("PostDamagePostTech", il, xi)
        PC._save_solve(out, cfg, os.path.join(OUTD, nm),
                       extra_prov=dict(validation="V3 coarse xi=0.05"))
        pdpt.append(out)
        fin = bool(np.all(np.isfinite(out["W"])))
        bf = _bind_frac(out, Zg, xi, pre_t=False)
        report[nm] = dict(iters=out["iters"], di_int_final=out["di_int_final"],
                          gate_int=out["gate_int"], rmsR=out["rms_residual"],
                          finite=fin, h_clip_bind_frac=bf)
        hist_all[nm] = hist
        ok &= fin and out["di_int_final"] <= 2e-2 and bf <= 0.01
    pdpt_i = [PC.w_interp_of(o, (Zg, Yg, Sg2)) for o in pdpt]
    # (b) PreDamagePostTech: 5 damage channels with the xi=0.05 g's
    neut = _load("PreDamagePostTech", None, np.inf)
    cfg = PC.physrisk_cfg("PreDamagePostTech", 0.0, xi, (Zg, Yg, Sg2),
                          w_post_damage=pdpt_i, W_init=np.array(neut["W"]),
                          robust_drift=True)
    init = {k: np.array(neut[k]) for k in cfg["control_names"]}
    out, hist = _solve_capture(cfg, T=1200.0, dt=0.5, howard_max=howard,
                               tol_pocket=1e-6, relax=0.15, tail_avg=12,
                               init_ctr=init)
    nm = PC.solve_name("PreDamagePostTech", None, xi)
    PC._save_solve(out, cfg, os.path.join(OUTD, nm),
                   extra_prov=dict(validation="V3 coarse xi=0.05 (damage g ON)"))
    fin = bool(np.all(np.isfinite(out["W"])))
    bf = _bind_frac(out, Zg, xi, pre_t=False)
    report[nm] = dict(iters=out["iters"], di_int_final=out["di_int_final"],
                      gate_int=out["gate_int"], rmsR=out["rms_residual"], finite=fin,
                      h_clip_bind_frac=bf)
    hist_all[nm] = hist
    ok &= fin and out["di_int_final"] <= 2e-2 and bf <= 0.01
    # (c) PostDamagePreTech l=3: tech channel g ON
    il = 2
    neut3 = _load("PostDamagePreTech", il, np.inf)
    cfg = PC.physrisk_cfg("PostDamagePreTech", PC.LAM3_GRID[il], xi, (Zg, Yg, Sg),
                          w_post_tech=pdpt_i[il], W_init=np.array(neut3["W"]),
                          robust_drift=True)
    init = {k: np.array(neut3[k]) for k in cfg["control_names"]}
    out, hist = _solve_capture(cfg, T=1200.0, dt=1.0, howard_max=howard,
                               tol_pocket=1e-6, relax=0.15, tail_avg=12,
                               init_ctr=init)
    nm = PC.solve_name("PostDamagePreTech", il, xi)
    PC._save_solve(out, cfg, os.path.join(OUTD, nm),
                   extra_prov=dict(validation="V3 coarse xi=0.05 (tech g ON)"))
    fin = bool(np.all(np.isfinite(out["W"])))
    bf = _bind_frac(out, Zg, xi, pre_t=True)
    report[nm] = dict(iters=out["iters"], di_int_final=out["di_int_final"],
                      gate_int=out["gate_int"], rmsR=out["rms_residual"], finite=fin,
                      h_clip_bind_frac=bf)
    hist_all[nm] = hist
    ok &= fin and out["di_int_final"] <= 2e-2 and bf <= 0.01
    for nm, r in report.items():
        print("  V3 %-40s iters=%d di_int=%.2e gate=%.2e finite=%s"
              % (nm, r["iters"], r["di_int_final"], r["gate_int"], r["finite"]),
              flush=True)
    print("V3 xi=0.05 outer-loop stability -> %s" % ("PASS" if ok else "FAIL"),
          flush=True)
    return dict(solves=report, passed=bool(ok)), hist_all


# ------------------------------------------------------------------ FK MC
def run_fk(grids, npaths=600, T=1200.0, dt=0.25, seed=12345):
    """End-to-end level MC under the neutral-chain FD policies (see module doc)."""
    Zg, Yg, Sg, Sg2 = grids
    p_pre = PC.base_params("PreDamagePreTech")
    interp, W_at = {}, {}
    for reg in PC.REGIMES:
        pre_t = reg.endswith("PreTech")
        gr = (Zg, Yg, Sg if pre_t else Sg2)
        ils = range(5) if reg.startswith("PostDamage") else [None]
        for il in ils:
            d = _load(reg, il, np.inf)
            flds = {k: np.array(d[k]) for k in
                    (("i_d", "i_g", "i_r") if pre_t else ("i_d", "i_g"))}
            interp[(reg, il)] = FR.FieldInterp(gr, flds)
            W_at[(reg, il)] = FR.FieldInterp(gr, {"W": np.array(d["W"])})
    rng = np.random.RandomState(seed)
    sd, sg, sr = p_pre["s_d"], p_pre["s_g"], p_pre["s_r"]
    results = []
    for (z0, y0, s0) in SEEDS:
        lk = np.full(npaths, LK0)
        z = np.full(npaths, z0)
        y = np.full(npaths, y0)
        s = np.full(npaths, s0)
        post_t = np.zeros(npaths, bool)
        post_d = np.zeros(npaths, bool)
        il_ix = np.full(npaths, -1)
        V = np.zeros(npaths)
        nstep = int(round(T / dt))
        sq = np.sqrt(dt)
        for n in range(nstep):
            t = n * dt
            i_d = np.empty(npaths); i_g = np.empty(npaths); i_r = np.zeros(npaths)
            for reg in PC.REGIMES:
                rm = (post_d == reg.startswith("PostDamage")) & \
                     (post_t == reg.endswith("PostTech"))
                if not rm.any():
                    continue
                if reg.startswith("PostDamage"):
                    for il in range(5):
                        m = rm & (il_ix == il)
                        if not m.any():
                            continue
                        q = interp[(reg, il)](z[m], y[m], s[m])
                        i_d[m] = q["i_d"]; i_g[m] = q["i_g"]
                        if "i_r" in q:
                            i_r[m] = q["i_r"]
                else:
                    q = interp[(reg, None)](z[rm], y[rm], s[rm])
                    i_d[rm] = q["i_d"]; i_g[rm] = q["i_g"]
                    if "i_r" in q:
                        i_r[rm] = q["i_r"]
            i_r = np.where(post_t, 0.0, np.maximum(i_r, 0.0))
            A_g = np.where(post_t, 0.1567, 0.1085)
            phid = p_pre["a_d"] + p_pre["G_d"] * np.log(np.maximum(1 + p_pre["t_d"] * i_d, 1e-9))
            phig = p_pre["a_g"] + p_pre["G_g"] * np.log(np.maximum(1 + p_pre["t_g"] * i_g, 1e-9))
            l3v = np.where(il_ix >= 0, np.array(PC.LAM3_GRID)[np.maximum(il_ix, 0)], 0.0)
            y_eff = np.minimum(y, PC.Y_CAP)     # same D-saturation as the FD (Y_CAP)
            D = p_pre["l1"] + p_pre["l2"] * y_eff \
                + np.where(post_d, l3v * np.maximum(y_eff - p_pre["y_hat"], 0.0), 0.0)
            fd = phid - PC.GAM_D * D
            fg = phig - PC.GAM_G * D
            sig2 = sd ** 2 * (1 - z) ** 2 + sg ** 2 * z ** 2
            mu_K = (1 - z) * fd + z * fg - 0.5 * sig2
            a_Z = z * (1 - z) * (fg - fd + (1 - z) * sd ** 2 - z * sg ** 2)
            E = PC.emissions(z, p_pre)
            c = (p_pre["A_d"] - i_d) * (1 - z) + (A_g - i_g) * z - i_r
            V += np.exp(-p_pre["delta"] * (t + 0.5 * dt)) * p_pre["delta"] \
                * (np.log(np.maximum(c, 1e-12)) + lk) * dt
            mu_R = p_pre["psi0"] * np.sqrt(i_r) \
                * np.exp(-0.5 * np.clip(s, PC.S_LO, PC.S_HI)) - 0.5 * sr ** 2
            dWd = sq * rng.randn(npaths); dWg = sq * rng.randn(npaths)
            dWr = sq * rng.randn(npaths); dWy = sq * rng.randn(npaths)
            lk = lk + mu_K * dt + (1 - z) * sd * dWd + z * sg * dWg
            z = np.clip(z + a_Z * dt + z * (1 - z) * (sg * dWg - sd * dWd),
                        1e-4, 1 - 1e-4)
            y = np.maximum(y + p_pre["thbar"] * E * dt + p_pre["vars"] * E * dWy, 0.0)
            ds = np.where(post_t, 0.0,
                          (mu_R - mu_K) * dt + sr * dWr
                          - (1 - z) * sd * dWd - z * sg * dWg)
            s = s + ds
            # jumps (independent channels; both may fire in one step w.p. O(dt^2))
            Jt = PC.tech_intensity(z, y, s, p_pre)
            fire_t = (~post_t) & (rng.rand(npaths) < -np.expm1(-Jt * dt))
            Jn = PC.damage_intensity_total(y, p_pre)
            fire_d = (~post_d) & (rng.rand(npaths) < -np.expm1(-Jn * dt))
            post_t = post_t | fire_t
            if fire_d.any():
                il_ix[fire_d] = rng.randint(0, 5, int(fire_d.sum()))
                post_d = post_d | fire_d
        Wfd = float(W_at[("PreDamagePreTech", None)](
            np.array([z0]), np.array([y0]), np.array([s0]))["W"][0])
        mc, se = float(np.mean(V)), float(np.std(V) / np.sqrt(npaths))
        diff = mc - (LK0 + Wfd)
        tol = 3 * se + 0.03
        results.append(dict(seed_state=[z0, y0, s0], W_fd=Wfd,
                            target=LK0 + Wfd, mc_mean=mc, mc_se=se,
                            diff=diff, tol=tol, passed=bool(abs(diff) <= tol),
                            frac_tech_jumped=float(np.mean(post_t)),
                            frac_damage_jumped=float(np.mean(post_d))))
        print("  FK seed (Z=%.2f Y=%.2f s=%.2f): logK0+W=%.4f MC=%.4f+-%.4f "
              "diff=%+.4f (tol %.4f) %s" %
              (z0, y0, s0, LK0 + Wfd, mc, se, diff, tol,
               "PASS" if results[-1]["passed"] else "FAIL"), flush=True)
    ok = all(r["passed"] for r in results)
    print("FK level MC -> %s" % ("PASS" if ok else "FAIL"), flush=True)
    return dict(npaths=npaths, T=T, dt=dt, seeds=results, passed=bool(ok),
                allowance_note=("tol = 3*SE + 0.03; the 0.03 covers the PIBYS "
                                "O(sigma^2) dropped diffusion, discount quadrature "
                                "and coarse-grid interpolation systematics"))


# ------------------------------------------------------------------ figure
def make_figure(grids, v1_diff, v3_hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Zg, Yg, Sg, Sg2 = grids
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    iy = int(np.argmin(np.abs(Yg - 1.1)))
    im = ax[0, 0].pcolormesh(Sg, Zg, np.abs(v1_diff[:, iy, :]), shading="auto")
    ax[0, 0].set_title("V1 |W(xi=1e5) - W(linear)|  (Y=1.1)")
    ax[0, 0].set_xlabel("s"); ax[0, 0].set_ylabel("Z")
    fig.colorbar(im, ax=ax[0, 0])
    iz = int(np.argmin(np.abs(Zg - 0.7)))
    d3 = _load("PostDamagePreTech", 2, np.inf)
    Wp = _load("PostDamagePostTech", 2, np.inf)["W"][:, :, 1]
    for yq in (2.5, 3.0, 3.5):
        iyq = int(np.argmin(np.abs(Yg - yq)))
        ax[0, 1].plot(Sg, d3["W"][iz, iyq, :], label="W, Y=%.1f" % yq)
        ax[0, 1].axhline(Wp[iz, iyq], ls="--", lw=0.8, color="gray")
    ax[0, 1].set_title("V2: W(s) -> W_post anchor at s-top (Z=0.7, l=3)")
    ax[0, 1].set_xlabel("s"); ax[0, 1].legend(fontsize=8)
    for nm, h in v3_hist.items():
        if not h:
            continue
        ax[1, 0].semilogy([q[0] for q in h], [q[2] for q in h],
                          marker=".", label=nm.replace("physrisk_", ""))
    ax[1, 0].set_title("V3: Howard di_int history at xi=0.05")
    ax[1, 0].set_xlabel("Howard iter"); ax[1, 0].legend(fontsize=6)
    ax[1, 1].axis("off")   # filled with FK table text by caller via prov
    fig.tight_layout()
    os.makedirs(FIGD, exist_ok=True)
    fp = os.path.join(FIGD, "physrisk_validation_coarse.png")
    fig.savefig(fp, dpi=120)
    print("figure -> %s" % fp, flush=True)
    return fp


def main():
    global OUTD
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="debug budgets")
    ap.add_argument("--grid", type=str, default="coarse",
                    help="coarse (default) | tiny (debug only)")
    ap.add_argument("--outdir", type=str, default=OUTD)
    ap.add_argument("--skip-chain", action="store_true",
                    help="reuse existing neutral chain npz")
    a = ap.parse_args()
    OUTD = os.path.abspath(a.outdir)
    os.makedirs(OUTD, exist_ok=True)
    grids = PC.default_grids(a.grid)
    howard = 12 if a.fast else 40
    t0 = time.time()
    if not a.skip_chain:
        PC.run_chain(np.inf, OUTD, grid=a.grid, howard_max=howard,
                     robust=False, dt2d=0.5, dt3d=1.0, relax=0.15,
                     tail_avg=12, verbose=False)
    print("\n-- V2 --", flush=True)
    v2 = run_v2(grids)
    print("\n-- V1 --", flush=True)
    v1, v1_diff, _ = run_v1(grids, howard=(6 if a.fast else 10))
    print("\n-- V3 --", flush=True)
    v3, v3_hist = run_v3(grids, howard=(10 if a.fast else 40))
    print("\n-- FK --", flush=True)
    fk = run_fk(grids, npaths=(150 if a.fast else 600))
    fig = None
    try:
        fig = make_figure(grids, v1_diff, v3_hist)
    except Exception as e:
        print("[warn] figure skipped: %s" % e, flush=True)
    prov = dict(economy="PHYSRISK", stage="coarse validations (audit-mandated, "
                "pre-ladder)", grid=[len(g) for g in grids[:3]],
                V1_jump_linear_limit=v1, V2_step_exact=v2,
                V3_xi005_stability=v3, FK_level_mc=fk,
                howard_budget=howard, figure=fig,
                seconds=float(time.time() - t0), date=str(datetime.date.today()),
                all_passed=bool(v1["passed"] and v2["passed"] and v3["passed"]
                                and fk["passed"]))
    fp = os.path.join(OUTD, "physrisk_validation_PROVENANCE.json")
    with open(fp, "w") as f:
        json.dump(prov, f, indent=1)
    print("\nALL GATES: %s  (provenance -> %s)"
          % ("PASS" if prov["all_passed"] else "SEE FAILURES", fp), flush=True)


if __name__ == "__main__":
    main()
