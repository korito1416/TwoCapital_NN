"""physrisk_callbacks.py -- PHYSRISK economy configuration for the shared reduced-FD
harness solvers/fd_reduced.py (economy zoo, build 5; design_physrisk.json).

ECONOMY (physical-risk green-Solow; the MINIMAL PAIR with JONES):
  JONES's scale-free skeleton with ONE coupling swapped -- the utility damage
  -delta logN(Y) is DELETED; temperature instead physically DESTROYS capital at
  sector-specific rates subtracted from the capital drifts:
      dK_j = K_j [phi_j(i_j) - Lambda_j(Y)] dt + sigma_j K_j dW_j,
      Lambda_j(Y) = gamma_j * D(Y),   gamma_d = 1.0 > gamma_g = 0.25,
      D(Y) = lambda1 + lambda2*Y                      (pre-damage)
           = lambda1 + lambda2*Y + lambda3(l)*(Y-yhat)  (post-damage)
  Same intensity emissions E(Z) = eta*A_d*(1-Z)*K0, same intensity tech hazard
  K0*e^{s}/varrho (s = logR - logK), same damage-jump J_n(Y) and lambda3 grid,
  same production R&D technology, full robustness (all h channels + jump g's).

REDUCED HJB (exact reduction V = logK + W(Z,Y,s); design sympy-verified):
  JONES's reduced HJB with phi_d -> phi_d - Lambda_d(Y), phi_g -> phi_g - Lambda_g(Y)
  inside mu_K (coefficient of the standalone flow term = V_logK*mu_K with V_logK=1,
  and of -W_s) and inside mu_Z, and the -delta logN(Y) utility term dropped:
    0 = max { delta log c - delta W + mu_K^Lam
              + [psi0 sqrt(i_r) e^{-s/2} - s_r^2/2 - mu_K^Lam] W_s + mu_Z^Lam W_Z
              + thbar E(Z) W_Y + diffusion - (1/2xi)(L_d^2+L_g^2+L_r^2+L_y^2) }
        + xi (e^s K0/varrho)[1 - e^{-(W^postT - W)/xi}]
        + xi sum_l (1/5) J_n(Y) [1 - e^{-(W^l - W)/xi}]        (pre-damage only)
  Loadings: L_d = (1 - W_s - Z W_Z)(1-Z)s_d, L_g = (1 - W_s + (1-Z)W_Z) Z s_g,
  L_r = W_s s_r, L_y = W_Y E varsigma.  FOCs identical in form to JONES/production
  (Lambda_j additive in drift, no i-derivative): fd_reduced.foc_controls.

ROBUST EVALUATION FORM: at finite xi the harness costate feedback (needs_costates)
hands the LAGGED (W_Z, W_Y, W_S) to drift/flow; we implement the distorted-measure
form: drifts use mu + sigma*h with h_j = -(1/xi) L_j, the flow carries the unit
V_logK-loading distortion (1-Z)s_d h_d + Z s_g h_g and the penalty (xi/2)|h|^2.
At h = h* this equals the substituted -(1/2xi)|L|^2 form; the true-residual
monitor evaluates the same expressions with CURRENT costates.

SHARED-WITH-JONES: emissions(), tech_intensity(), damage_intensity(), the
diffusion coefficient block and the FOC wiring are the JONES scale-free skeleton
verbatim.  jones_callbacks.py did not exist when this file was written (JONES is
build 4, unbuilt); the shared pieces live HERE and jones_callbacks.py should
`from physrisk_callbacks import emissions, tech_intensity, damage_intensity, ...`
and re-add its -delta*logN(Y) flow term / drop the Lambda_j drifts.  V2
(step_exact jump-sink stability at the s-top, hazard 8.7/yr) is run by
physrisk_validate.py on the SAME hazard and therefore certifies JONES's too.

DOCUMENTED CHOICES / GUARDS (deviations from the design text, all conservative):
  * Post-damage D(Y) uses lambda3*max(Y - yhat, 0) (production damage-slope
    convention, zoo_lift_common.dlogN_dY) instead of the design's raw
    lambda3*(Y-yhat): identical on the economically-visited post-damage slab
    Y >= yhat = 2.5; below yhat (lift-coverage-only region) it avoids a
    negative-destruction bonus.
  * tech_intensity clips the exponent s at +-35 (repo overflow convention):
    along evaluation characteristics s can leave the grid by O(100) when
    destruction turns mu_K strongly negative; the clip keeps J finite and the
    step_exact weight then correctly drives W -> W_post.
  * damage_intensity clips its exponent (r2/2)(y-y_lo)^2 at +35 for the same
    reason (y is unbounded along characteristics).
  * D(Y) saturates at Y_CAP = 4 (the box top), the fd_pdpt_v5 Y_CAP convention:
    the destruction rate keeps accruing at D(4) beyond the box instead of
    growing linearly forever along out-of-box characteristics. Without it the
    destroyed low-Z/high-Y post-damage corner value depends on far-out-of-box
    destruction growth and the Howard iteration limit-cycles there.
  * mu_R's e^{-s/2} factor evaluates s clipped to the grid range [-6, 2]
    (consistent with edge-clamped control interpolation; prevents the
    e^{-s/2} blow-up feedback for characteristics far below the grid).
  * TRUST-REGION clip |h_j| <= H_CLIP = 25 on each worst-case drift distortion
    (lagged-costate Howard feedback at xi = 0.05 can spiral through the h_y
    channel -- W_Y ~ -8 in the post-damage l=5 corner gives h_y ~ 11, well
    inside the clip; the clip only caps the transient spiral. If the clip
    binds at CONVERGENCE the solve is not trusted there -- the validation
    script reports the bind fraction, and it must be ~0 in the interior).

CLI:
  python physrisk_callbacks.py --smoke
  python physrisk_callbacks.py --chain --xi 148.4 --grid full \
      --outdir ../outputs/physrisk [--warm-xi 148.4] [--howard 60] [--robust 1]
The --chain command runs the 12-solve backward ladder for ONE xi:
  5x PostDamagePostTech(l) [2-D] -> PreDamagePostTech [2-D, 5 damage channels]
  -> 5x PostDamagePreTech(l) [3-D, tech channel] -> PreDamagePreTech [3-D, 6 channels]
Every solve writes <name>.npz + <name>_PROVENANCE.json.
"""
import os
import sys
import json
import time
import argparse
import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "maps")))
import fd_reduced as FR                    # noqa: E402  (validated shared harness)
import zoo_lift_common as ZC               # noqa: E402  (shared production constants)

# ----------------------------------------------------------------- calibration
K0 = 880.0                    # production K_0 (models/params.py)
VARRHO = 746.67               # production R&D scaling varrho
RHO_S = VARRHO / K0           # 0.848489...: hazard = e^{s}/RHO_S = K0 e^{s}/varrho
ETA = 0.291
THBAR = 1.86e-3
VARSIG = 1.2 * 1.86e-3
R1, R2, Y_LO = 1.5, 0.36, 1.5          # damage-jump intensity J_n(y)
GAM_D, GAM_G = 1.0, 0.25               # destruction exposure (Fried channel)
LAM3_GRID = (0.0, 1.0 / 12, 1.0 / 6, 1.0 / 4, 1.0 / 3)   # L = 5
XI_FAMILY = (0.05, 0.1, 148.4)
S_LO, S_HI = -6.0, 3.0                 # top extended +2->+3 (V2 boundary layer outside consumed box s<=2)
H_CLIP = 25.0                          # trust-region cap on each |h_j| (see module doc)

REGIMES = ("PostDamagePostTech", "PreDamagePostTech",
           "PostDamagePreTech", "PreDamagePreTech")


def xi_tag(xi):
    if not np.isfinite(xi):
        return "inf"
    return ("%g" % xi).replace(".", "p").replace("+", "").replace("-", "m")


def solve_name(regime, il=None, xi=148.4):
    l = "" if il is None else "_l%d" % (il + 1)
    return "physrisk_%s%s_xi%s" % (regime, l, xi_tag(xi))


def base_params(regime):
    post_t = regime.endswith("PostTech")
    return dict(
        delta=ZC.DELTA, A_d=ZC.A_D,
        A_g=(ZC.A_G_POST if post_t else ZC.A_G_PRE),
        a_d=ZC.ALPHA_D, G_d=ZC.GAMMA_D, t_d=ZC.THETA_D, s_d=ZC.SIGMA_D,
        a_g=ZC.ALPHA_G, G_g=ZC.GAMMA_G, t_g=ZC.THETA_G, s_g=ZC.SIGMA_G,
        psi0=ZC.PSI_0, psi1=ZC.PSI_1, s_r=ZC.SIGMA_R,
        l1=ZC.LAMBDA_1, l2=ZC.LAMBDA_2, y_hat=ZC.Y_HAT,
        gam_d=GAM_D, gam_g=GAM_G, eta=ETA, thbar=THBAR, vars=VARSIG,
        K0=K0, varrho=VARRHO, r1=R1, r2=R2, y_lo=Y_LO, lam3=0.0)


# ---------------------------------------------- scale-free skeleton (JONES-shared)
def emissions(z, p):
    """Intensity emissions E(Z) = eta*A_d*(1-Z)*K0 (scale-free; E0 = 10.01)."""
    return p["eta"] * p["A_d"] * (1.0 - z) * p["K0"]


def tech_intensity(z, y, s, p):
    """Scale-free tech hazard K0*e^{s}/varrho (= production's 0.0150/yr at s0).
    Exponent clipped at +-35 (repo convention; s unbounded along characteristics)."""
    return np.exp(np.clip(s, -FR.EXP_CLIP, FR.EXP_CLIP)) * p["K0"] / p["varrho"]


def damage_intensity_total(y, p):
    """TOTAL damage-jump intensity J_n(y) = r1*(exp((r2/2)(y-y_lo)^2)-1), y>=y_lo.
    Per-channel intensity is J_n/5. Exponent clipped at +35."""
    dy = np.maximum(np.asarray(y, dtype=float) - p["y_lo"], 0.0)
    arg = np.clip(0.5 * p["r2"] * dy ** 2, 0.0, FR.EXP_CLIP)
    return p["r1"] * np.expm1(arg)


# ------------------------------------------------------- PHYSRISK destruction
Y_CAP = 4.0     # destruction-rate saturation horizon = box top (fd_pdpt_v5 Y_CAP
                # convention: D keeps accruing at D(Y_CAP) beyond it, no linear
                # blow-up along out-of-box characteristics; in-box values unchanged
                # except through path segments beyond Y=4)


def D_of(y, lam3, post_damage, p):
    """Destruction driver D(Y): the production damage SLOPE reused as a rate.
    Pre-damage l1 + l2*Y; post-damage adds lam3*max(Y - yhat, 0) (see module doc).
    Y saturates at Y_CAP = 4 (fd_pdpt_v5 convention, see above)."""
    y = np.minimum(np.asarray(y, dtype=float), Y_CAP)
    D = p["l1"] + p["l2"] * y
    if post_damage:
        D = D + lam3 * np.maximum(y - p["y_hat"], 0.0)
    return D


# --------------------------------------------------------------- cfg factory
def physrisk_cfg(regime, lam3, xi, grids, w_post_tech=None, w_post_damage=None,
                 W_init=None, robust_drift=True, step_exact=True, box=None):
    """Build the fd_reduced cfg dict for one PHYSRISK regime solve.

    regime        : one of REGIMES
    lam3          : lambda3 value (post-damage regimes; ignored pre-damage)
    xi            : robustness parameter (np.inf = uncertainty-neutral)
    grids         : (Zg, Yg, Sg); post-tech regimes use a dummy 3-point s-axis
    w_post_tech   : f(z,y,s) -> W of the SAME-damage post-tech regime (pre-tech only)
    w_post_damage : list of 5 f(z,y,s) -> W of the post-damage regime at lambda3(l)
                    (pre-damage regimes only)
    """
    p = base_params(regime)
    post_d = regime.startswith("PostDamage")
    pre_t = regime.endswith("PreTech")
    p["lam3"] = float(lam3) if post_d else 0.0
    xi_f = float(xi) if np.isfinite(xi) else np.inf
    finite_xi = np.isfinite(xi_f) and xi_f < FR.NEUTRAL_XI
    use_h = bool(robust_drift) and finite_xi

    def _phis(ctr):
        phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * ctr["i_d"], 1e-9))
        phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * ctr["i_g"], 1e-9))
        return phid, phig

    def _h(cost, z, E):
        """Worst-case drift distortions from the LAGGED costates (h = -(1/xi)L),
        each clipped to +-H_CLIP (trust region; see module doc)."""
        W_Z, W_Y, W_S = cost["W_Z"], cost["W_Y"], cost["W_S"]
        hd = -((1.0 - W_S - z * W_Z) * (1.0 - z) * p["s_d"]) / xi_f
        hg = -((1.0 - W_S + (1.0 - z) * W_Z) * z * p["s_g"]) / xi_f
        hr = (-(W_S * p["s_r"]) / xi_f) if pre_t else np.zeros_like(z)
        hy = -(W_Y * E * p["vars"]) / xi_f
        return (np.clip(hd, -H_CLIP, H_CLIP), np.clip(hg, -H_CLIP, H_CLIP),
                np.clip(hr, -H_CLIP, H_CLIP), np.clip(hy, -H_CLIP, H_CLIP))

    def _destruction(y):
        D = D_of(y, p["lam3"], post_d, p)
        return p["gam_d"] * D, p["gam_g"] * D

    def _mu_K(z, phid, phig, Ld, Lg):
        sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
        sig2 = sd2 * (1 - z) ** 2 + sg2 * z ** 2
        return (1 - z) * (phid - Ld) + z * (phig - Lg) - 0.5 * sig2

    def drift(z, y, s, ctr, cost, p_):
        phid, phig = _phis(ctr)
        Ld, Lg = _destruction(y)
        fd = phid - Ld
        fg = phig - Lg
        sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
        a_Z = z * (1 - z) * (fg - fd + (1 - z) * sd2 - z * sg2)
        E = emissions(z, p)
        a_Y = p["thbar"] * E
        if pre_t:
            mu_K = _mu_K(z, phid, phig, Ld, Lg)
            ir = np.maximum(ctr["i_r"], 0.0)
            mu_R = (p["psi0"] * np.sqrt(ir)
                    * np.exp(-0.5 * np.clip(s, S_LO, S_HI)) - 0.5 * p["s_r"] ** 2)
            a_S = mu_R - mu_K
        else:
            a_S = np.zeros_like(z)
        if use_h and cost is not None:
            hd, hg, hr, hy = _h(cost, z, E)
            a_Z = a_Z + z * (1 - z) * (p["s_g"] * hg - p["s_d"] * hd)
            a_Y = a_Y + p["vars"] * E * hy
            if pre_t:
                a_S = a_S + p["s_r"] * hr - (1 - z) * p["s_d"] * hd - z * p["s_g"] * hg
        return a_Z, a_Y, a_S

    def flow(z, y, s, ctr, cost, p_):
        c = (p["A_d"] - ctr["i_d"]) * (1 - z) + (p["A_g"] - ctr["i_g"]) * z
        if pre_t:
            c = c - np.maximum(ctr["i_r"], 0.0)
        phid, phig = _phis(ctr)
        Ld, Lg = _destruction(y)
        fl = p["delta"] * np.log(np.maximum(c, 1e-12)) + _mu_K(z, phid, phig, Ld, Lg)
        # NO -delta*logN(Y) term: PHYSRISK deletes the utility damage.
        if use_h and cost is not None:
            E = emissions(z, p)
            hd, hg, hr, hy = _h(cost, z, E)
            fl = fl + (1 - z) * p["s_d"] * hd + z * p["s_g"] * hg   # unit V_logK loading
            fl = fl + 0.5 * xi_f * (hd ** 2 + hg ** 2 + hr ** 2 + hy ** 2)
        return fl

    def controls(W_Z, W_Y, W_S, Z, Y, S, p_):
        qd = 1.0 - W_S - Z * W_Z
        qg = 1.0 - W_S + (1.0 - Z) * W_Z
        if pre_t:
            return FR.foc_controls(qd, qg, Z, p, qr=W_S, S=S)
        return FR.foc_controls(qd, qg, Z, p)

    def diff(Z, Y, S, ctr, p_):
        sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
        sig2 = sd2 * (1 - Z) ** 2 + sg2 * Z ** 2
        E = emissions(Z, p)
        d = {"ZZ": 0.5 * Z ** 2 * (1 - Z) ** 2 * (sd2 + sg2),
             "YY": 0.5 * p["vars"] ** 2 * E ** 2}
        if pre_t:
            d["SS"] = 0.5 * (sig2 + p["s_r"] ** 2)
            d["SZ"] = Z * (1 - Z) ** 2 * sd2 - Z ** 2 * (1 - Z) * sg2
        return d

    def init_controls(Z, Y, S, p_):
        out = {"i_d": np.full_like(Z, 0.05), "i_g": np.full_like(Z, 0.06)}
        if pre_t:
            out["i_r"] = np.full_like(Z, 0.003)
        return out

    jumps = []
    if pre_t:
        assert w_post_tech is not None, "%s needs w_post_tech" % regime
        jumps.append({"name": "tech", "intensity": tech_intensity,
                      "w_post": w_post_tech})
    if not post_d:
        assert w_post_damage is not None and len(w_post_damage) == 5, \
            "%s needs 5 w_post_damage interpolators" % regime
        for il in range(5):
            jumps.append({
                "name": "damage_l%d" % (il + 1),
                "intensity": (lambda z, y, s, p_: damage_intensity_total(y, p_) / 5.0),
                "w_post": w_post_damage[il]})

    names = ["i_d", "i_g"] + (["i_r"] if pre_t else [])
    cfg = dict(name="PHYSRISK-%s(lam3=%.4f,xi=%s)" % (regime, p["lam3"], xi_tag(xi)),
               params=p, grids=grids, controls=controls, control_names=names,
               drift=drift, flow=flow, diff=diff, init_controls=init_controls,
               xi=xi_f, jumps=jumps, needs_costates=bool(robust_drift),
               step_exact=bool(step_exact))
    if W_init is not None:
        cfg["W_init"] = W_init
    if box is not None:
        cfg["box"] = box
    return cfg


# ------------------------------------------------------------- chain driver
def default_grids(kind="full"):
    if kind == "full":
        nZ, nY, nS = 61, 41, 69   # nS 61->69 keeps ds when s-top extends +2->+3
    elif kind == "coarse":
        nZ, nY, nS = 31, 21, 31
    elif kind == "tiny":
        nZ, nY, nS = 13, 9, 11
    else:
        raise ValueError(kind)
    Zg = np.linspace(0.01, 0.99, nZ)
    Yg = np.linspace(0.0, 4.0, nY)
    Sg = np.linspace(S_LO, S_HI, nS)
    Sg2 = np.linspace(S_LO, S_HI, 3)        # dummy s-axis for the 2-D post-tech regimes
    return Zg, Yg, Sg, Sg2


def w_interp_of(npz_or_out, grids):
    """W interpolator f(z,y,s) from a solved output dict/npz (edge-clamped)."""
    W = np.asarray(npz_or_out["W"])
    it = FR.FieldInterp(grids, {"W": W})
    return lambda z, y, s: it(z, y, s)["W"]


def _save_solve(out, cfg, path_base, extra_prov=None):
    fields = dict(Z=out["Z"], Y=out["Y"], S=out["S"], W=out["W"],
                  W_Z=out["W_Z"], W_Y=out["W_Y"], W_S=out["W_S"],
                  i_d=out["i_d"], i_g=out["i_g"], c=out["c"])
    if "i_r" in out:
        fields["i_r"] = out["i_r"]
    np.savez_compressed(path_base + ".npz", **fields)
    prov = dict(economy="PHYSRISK (physical-risk green-Solow, differential capital "
                        "destruction; minimal pair with JONES)",
                design="benchmarks/economy_zoo/design_physrisk.json",
                harness="benchmarks/economy_zoo/solvers/fd_reduced.py",
                callbacks="benchmarks/economy_zoo/solvers/physrisk_callbacks.py",
                cfg_name=cfg["name"],
                grid=[int(len(g)) for g in cfg["grids"]],
                xi=(float(cfg["xi"]) if np.isfinite(cfg["xi"]) else "inf"),
                lam3=float(cfg["params"]["lam3"]),
                A_g=float(cfg["params"]["A_g"]),
                gam_d=GAM_D, gam_g=GAM_G,
                step_exact=bool(cfg.get("step_exact", False)),
                robust_drift=bool(cfg.get("needs_costates", False)),
                gates=dict(iters=int(out["iters"]),
                           di_int_final=float(out["di_int_final"]),
                           gate_int=float(out["gate_int"]),
                           true_residual_max=float(out["max_abs_residual"]),
                           true_residual_rms=float(out["rms_residual"])),
                seconds=float(out["time"]),
                date=str(datetime.date.today()))
    if extra_prov:
        prov.update(extra_prov)
    with open(path_base + "_PROVENANCE.json", "w") as f:
        json.dump(prov, f, indent=1)
    return prov


def _load_warm(outdir, regime, il, warm_xi, names):
    """(init_ctr, W_init) from a previously solved npz at another xi (same grid)."""
    if warm_xi is None:
        return None, None
    fp = os.path.join(outdir, solve_name(regime, il, warm_xi) + ".npz")
    if not os.path.exists(fp):
        return None, None
    d = np.load(fp)
    init_ctr = {k: d[k] for k in names if k in d}
    return init_ctr, np.array(d["W"])


def run_chain(xi, outdir, grid="full", warm_xi=None, howard_max=60, robust=True,
              T=1200.0, dt2d=0.5, dt3d=1.0, relax=0.2, tail_avg=10, tol=1e-6,
              verbose=True):
    """Backward 12-solve ladder for ONE xi. Returns dict of provenance per solve.

    dt2d / dt3d: evaluation time steps for the 2-D (post-tech) and 3-D (pre-tech)
    solves. DEVIATION from the harness-validated reference dt = 2.5, REQUIRED for
    PHYSRISK: the destruction dynamics (Lambda_d up to ~0.52/yr post-damage l=5)
    are ~10x faster than the reference economy's, and at dt = 2.5 the Howard
    iteration limit-cycles in the low-Z destroyed strip (measured: l=5 coarse 2-D
    mutual-consistency gate 0.28 at dt=2.5 -> 5e-2 at dt=1.0 -> 3.9e-3 at dt=0.5).
    relax/tail_avg strengthened accordingly (0.3/6 -> 0.2/10 default)."""
    os.makedirs(outdir, exist_ok=True)
    Zg, Yg, Sg, Sg2 = default_grids(grid)
    provs = {}
    t0 = time.time()

    def solve_and_save(regime, il, cfg, init_ctr):
        nm = solve_name(regime, il, xi)
        dt = dt3d if regime.endswith("PreTech") else dt2d
        print("== [%s] solving %s (dt=%.2f) ==" % (time.strftime("%H:%M:%S"), nm, dt),
              flush=True)
        out = FR.solve_reduced(cfg, T=T, dt=dt, howard_max=howard_max,
                               tol_pocket=tol, relax=relax, tail_avg=tail_avg,
                               init_ctr=init_ctr, verbose=verbose,
                               ref_pt=(0.7, 1.1, -4.364))
        provs[nm] = _save_solve(out, cfg, os.path.join(outdir, nm))
        print("   iters=%d gate_int=%.2e rmsR=%.2e maxR=%.2e (%.0fs)"
              % (out["iters"], out["gate_int"], out["rms_residual"],
                 out["max_abs_residual"], out["time"]), flush=True)
        return out

    # ---- stage 1: PostDamagePostTech(l) -- 2-D, no jumps
    pdpt = []
    prev_ctr = None
    for il, l3 in enumerate(LAM3_GRID):
        cfg = physrisk_cfg("PostDamagePostTech", l3, xi, (Zg, Yg, Sg2),
                           robust_drift=robust)
        ic, wi = _load_warm(outdir, "PostDamagePostTech", il, warm_xi,
                            cfg["control_names"])
        if wi is not None:
            cfg["W_init"] = wi
        out = solve_and_save("PostDamagePostTech", il, cfg, ic or prev_ctr)
        prev_ctr = {k: out[k] for k in cfg["control_names"]}
        pdpt.append(out)
    pdpt_interp = [w_interp_of(o, (Zg, Yg, Sg2)) for o in pdpt]

    # ---- stage 2: PreDamagePostTech -- 2-D, 5 damage channels
    cfg = physrisk_cfg("PreDamagePostTech", 0.0, xi, (Zg, Yg, Sg2),
                       w_post_damage=pdpt_interp, robust_drift=robust)
    ic, wi = _load_warm(outdir, "PreDamagePostTech", None, warm_xi,
                        cfg["control_names"])
    if wi is not None:
        cfg["W_init"] = wi
    elif np.isfinite(xi):
        cfg["W_init"] = pdpt[0]["W"]      # warm g-lag from the mildest post regime
    predpost = solve_and_save("PreDamagePostTech", None, cfg, ic)
    predpost_interp = w_interp_of(predpost, (Zg, Yg, Sg2))

    # ---- stage 3: PostDamagePreTech(l) -- 3-D, tech channel
    pdpre = []
    prev_ctr = None
    for il, l3 in enumerate(LAM3_GRID):
        cfg = physrisk_cfg("PostDamagePreTech", l3, xi, (Zg, Yg, Sg),
                           w_post_tech=pdpt_interp[il], robust_drift=robust)
        ic, wi = _load_warm(outdir, "PostDamagePreTech", il, warm_xi,
                            cfg["control_names"])
        if wi is not None:
            cfg["W_init"] = wi
        elif np.isfinite(xi):
            cfg["W_init"] = np.repeat(pdpt[il]["W"][:, :, 1:2], Sg.size, axis=2)
        out = solve_and_save("PostDamagePreTech", il, cfg, ic or prev_ctr)
        prev_ctr = {k: out[k] for k in cfg["control_names"]}
        pdpre.append(out)
    pdpre_interp = [w_interp_of(o, (Zg, Yg, Sg)) for o in pdpre]

    # ---- stage 4: PreDamagePreTech -- 3-D, tech + 5 damage channels
    cfg = physrisk_cfg("PreDamagePreTech", 0.0, xi, (Zg, Yg, Sg),
                       w_post_tech=predpost_interp, w_post_damage=pdpre_interp,
                       robust_drift=robust)
    ic, wi = _load_warm(outdir, "PreDamagePreTech", None, warm_xi,
                        cfg["control_names"])
    if wi is not None:
        cfg["W_init"] = wi
    elif np.isfinite(xi):
        cfg["W_init"] = pdpre[0]["W"]
    solve_and_save("PreDamagePreTech", None, cfg, ic)

    print("== chain xi=%s done in %.0fs ==" % (xi_tag(xi), time.time() - t0),
          flush=True)
    with open(os.path.join(outdir, "physrisk_chain_xi%s_SUMMARY.json"
                           % xi_tag(xi)), "w") as f:
        json.dump(provs, f, indent=1)
    return provs


# ------------------------------------------------------------------- smoke
def _smoke():
    """Tiny-grid neutral chain: shape/finiteness/economics sanity in ~1 min."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        provs = run_chain(np.inf, td, grid="tiny", howard_max=6, robust=False,
                          T=400.0, dt2d=2.5, dt3d=2.5, verbose=False)
        for nm, pr in provs.items():
            g = pr["gates"]
            assert np.isfinite(g["true_residual_rms"]), nm
            print("  %-42s rmsR=%.2e gate=%.2e" % (nm, g["true_residual_rms"],
                                                   g["gate_int"]), flush=True)
    print("SMOKE PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--xi", type=str, default="148.4")
    ap.add_argument("--grid", type=str, default="full")
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(HERE, "..", "outputs", "physrisk"))
    ap.add_argument("--warm-xi", type=str, default=None)
    ap.add_argument("--howard", type=int, default=60)
    ap.add_argument("--robust", type=int, default=1)
    ap.add_argument("--dt2d", type=float, default=0.5)
    ap.add_argument("--dt3d", type=float, default=1.0)
    ap.add_argument("--relax", type=float, default=0.2)
    ap.add_argument("--tail", type=int, default=10)
    a = ap.parse_args()
    if a.smoke:
        _smoke()
        return
    if a.chain:
        xi = np.inf if a.xi.lower() == "inf" else float(a.xi)
        warm = None if a.warm_xi is None else (
            np.inf if a.warm_xi.lower() == "inf" else float(a.warm_xi))
        run_chain(xi, os.path.abspath(a.outdir), grid=a.grid, warm_xi=warm,
                  howard_max=a.howard, robust=bool(a.robust),
                  dt2d=a.dt2d, dt3d=a.dt3d, relax=a.relax, tail_avg=a.tail)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
