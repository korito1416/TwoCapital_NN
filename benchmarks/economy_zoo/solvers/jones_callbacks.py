"""
jones_callbacks.py -- JONES economy (scale-free semi-endogenous intensity economy,
the FD-3D flagship) per-regime callback configs for the shared reduced-FD harness
solvers/fd_reduced.py, plus the ladder runner and the audit-mandated coarse
validations V1/V2/V3.

ECONOMY (design_jones.json + PORTFOLIO.json innovation-theory pair, critique-fixed):
the production model with EXACTLY TWO surgical scale-effect removals
  (1) tech-jump hazard J_g = e^{s}/rho_s,  s = logR - logK,  rho_s = varrho/K0
      = 746.67/880 = 0.848489  (Jones semi-endogenous: knowledge INTENSITY);
      equals production's 0.0150/yr at (R0, K0) = (11.2, 880) exactly.
  (2) intensity emissions E(Z) = eta*A_d*(1-Z)*K0 = 10.01 at t=0 (Kaya composition:
      only the dirty mix warms; growth is emissions-neutral).
Everything else IS production: perfect-substitutes AK, log utility, phi_j, the
psi0*sqrt(i_r)*e^{psi1(logK-logR)} R&D technology, FULL nonlinear logN(Y; lambda3)
with the damage jump J_n(Y), and full robustness (h incl. h_y, jump distortions g,
exp-clip 35).

EXACT REDUCTION (sympy-verified by both design teams): V = logK + W(Z, Y, s).
Chain rule: V_logK = 1 - W_s, V_logR = W_s, V_{logK,logK} = V_{logR,logR} = W_ss,
V_{logK,Z} = -W_sZ; the delta*logK utility term cancels -delta*logK identically.

VALUE OBJECT / TRANSFORM (matches the repo net-space convention that
make_map_anchor.py consumes, see maps/zoo_lift_common.py):
  W here is the reduced NET-SPACE value:  v = V + logN(Y; lambda3) = logK + W.
  Hence the flow carries the v5 damage-drift form  -(lNy*a_Y + lNyy*b_Y)  with the
  REGIME-CORRECT slope (pre: l1 + l2*Y; post: + lambda3*max(Y - yhat, 0), the
  value+slope-continuous repo form of zoo_lift_common.dlogN_dY) instead of an
  explicit -delta*logN term, and h_y uses the true V_Y = W_Y - lNy.
  Slope saturation: lNy, lNyy evaluated at y_eff = min(Y, Y_CAP=4) -- the v5
  temperature-damage horizon (identical treatment to fd_pdpt_v5).
  Damage-JUMP gap in net space: V^l - V = (W^l - W) - (lambda3_l/2)*
  max(min(Y,Y_CAP) - yhat, 0)^2 (the logN books differ across the jump; the
  correction saturates at Y_CAP consistently with the slope). Tech-jump gap
  needs no correction (same damage state on both sides).

REGIME LADDER (OneJump 4-regime layout; solve backward):
  PostDamagePostTech (per lambda3, 5x): 2-D W^l(Z,Y), no jumps, no i_r,
      A_g = A_g'' = 0.1567.  Solved s-degenerate on a 5-point s-grid (a_S = 0).
  PreDamagePostTech: 2-D W(Z,Y), 5 damage channels (1/5)J_n(Y) -> W^l, no i_r.
  PostDamagePreTech (per lambda3, 5x): 3-D W^l(Z,Y,s), tech channel e^s/rho_s ->
      the SAME-lambda3 PostDamagePostTech W^l(Z,Y); i_r active; A_g = 0.1085.
  PreDamagePreTech: 3-D W(Z,Y,s), tech channel -> PreDamagePostTech W(Z,Y) AND
      5 damage channels -> PostDamagePreTech W^l(Z,Y,s); i_r active.
  xi ladder per (regime, lambda3): 148.4 (cold) -> 0.1 -> 0.05, warm-chained
      (controls + W_init), downstream solves read at the MATCHING xi.

ROBUSTNESS: evaluated under the worst-case measure along characteristics --
drifts get +sigma.h with h from the FROZEN previous costates (harness
needs_costates), the flow gets the penalty +xi|h|^2/2 (equivalent at the fixed
point to the -(1/2xi)|L|^2 form in design_jones.json), and jump channels get the
lagged distortion g = exp(clip(-(W_post - W_lag)/xi, +-35)) with the semi-implicit
exact-exponential sink (fd_reduced step_exact=True -- REQUIRED here: J*dt up to
8.7/yr * 2.5y ~ 22 at the s-top).

NUMERICAL GUARDS (documented deviations from the raw formulas):
  * exp-clip +-35 on the jump-distortion exponent (harness), on e^{-psi1*s}
    (harness foc_controls), and on the J_n(Y) exponent (r2/2)(Y-y_lo)^2 -- the
    last binds only on far-field characteristics (Y >~ 21), where the pre-damage
    state is an instantly-jumping transient; step_exact makes W -> W_post there.
  * damage-gap correction saturates at Y_CAP (see VALUE OBJECT).

CLI (run from anywhere; paths are absolute):
  python jones_callbacks.py ladder --regime <Regime> [--l3idx k] \
         [--nz 61 --ny 31 --ns 41] [--outdir .../outputs/jones]
      -> the 3-xi warm-chained solves for one (regime, lambda3) ladder cell;
         this is what jones_ladder.sbatch calls.
  python jones_callbacks.py solve --regime <Regime> [--l3idx k] --xi <v> ...
  python jones_callbacks.py v1 | v2 | v3      (coarse login-budget validations)
  python jones_callbacks.py smoke             (tiny-grid 4-regime chain sanity)
Every solve writes <name>.npz + <name>_PROVENANCE.json.
"""
import os
import sys
import io
import json
import time
import argparse
import contextlib
import datetime
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "maps")))
from fd_reduced import (solve_reduced, simulate_W, true_residual,  # noqa: E402
                        foc_controls, FieldInterp, EXP_CLIP)
import zoo_lift_common as ZL                                       # noqa: E402

OUTD = os.path.join(HERE, "outputs", "jones")
VALD = os.path.join(HERE, "outputs", "jones_validation")
FIGD = os.path.abspath(os.path.join(HERE, "..", "figures"))

# ------------------------------------------------------------------ calibration
# Production values (models/params.py; cross-checked below). Shared sector/damage
# constants come from zoo_lift_common so the solver and the lift can never drift.
K0 = 880.0                       # calibration-scale total capital
VARRHO = 746.67                  # production R&D arrival scaling
RHO_S = VARRHO / K0              # = 0.848489: hazard = e^s / rho_s
THBAR = 1.86 / 1000              # mean TCRE
ETA = 0.291                      # emissions intensity of dirty output
VARSIG = 1.2 * 1.86 / 1000       # temperature vol loading
R1, R2 = 1.5, 0.36               # damage-jump intensity
Y_LO = 1.5                       # damage-jump lower threshold y_lower
Y_CAP = 4.0                      # v5 temperature-damage horizon (slope saturation)
NL = 5
LAM3_GRID = [0.0, 1.0 / 12, 1.0 / 6, 1.0 / 4, 1.0 / 3]
XI_LADDER = [148.4, 0.1, 0.05]   # solve order (warm-chained)
S_MIN, S_MAX = -6.0, 3.0         # top extended +2->+3: s-top boundary layer (hazard~8.7/yr, V2 finding) pushed OUTSIDE the consumed map box s<=2
NS_POSTTECH = 5                  # s-degenerate grid for the 2-D regimes

REGIMES = ["PostDamagePostTech", "PreDamagePostTech",
           "PostDamagePreTech", "PreDamagePreTech"]


def xi_tag(xi):
    if not np.isfinite(xi):
        return "xiinf"
    return "xi" + (f"{xi:g}").replace(".", "p")


def fname(regime, l3idx, xi):
    lt = f"l3{l3idx}" if l3idx is not None else "l3x"
    return f"jones_{regime}_{lt}_{xi_tag(xi)}.npz"


def check_params():
    """Cross-check hard-coded calibration vs models/params.py (read-only)."""
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                                        "models")))
        from params import PARAMS as PP
        pairs = [(THBAR, PP["θ_bar"]), (ETA, PP["η"]), (VARSIG, PP["ϛ"]),
                 (R1, PP["r1"]), (R2, PP["r2"]), (Y_LO, PP["y_lower"]),
                 (ZL.Y_HAT, PP["y_upper"]), (VARRHO, PP["varrho"]),
                 (ZL.LAMBDA_1, PP["λ1"]), (ZL.LAMBDA_2, PP["λ2"]),
                 (ZL.PSI_0, PP["ψ0"]), (ZL.PSI_1, PP["ψ1"]),
                 (ZL.SIGMA_R, PP["σ_κ"]), (ZL.A_D, PP["A_d"]),
                 (ZL.A_G_PRE, PP["A_g"]), (ZL.A_G_POST, PP["A_g_prime_prime"])]
        bad = [i for i, (a, b) in enumerate(pairs) if abs(a - b) > 1e-12]
        assert not bad, f"calibration mismatch vs models/params.py at {bad}"
        print("[check_params] all constants match models/params.py", flush=True)
    except ImportError as e:
        print(f"[check_params] models/params.py unavailable ({e}); "
              "using hard-coded values", flush=True)


# ------------------------------------------------------------------ parameters
def base_params(regime, lam3, xi):
    pretech = regime.endswith("PreTech")
    postdam = regime.startswith("PostDamage")
    return dict(
        delta=ZL.DELTA, A_d=ZL.A_D,
        A_g=(ZL.A_G_PRE if pretech else ZL.A_G_POST),
        a_d=ZL.ALPHA_D, G_d=ZL.GAMMA_D, t_d=ZL.THETA_D, s_d=ZL.SIGMA_D,
        a_g=ZL.ALPHA_G, G_g=ZL.GAMMA_G, t_g=ZL.THETA_G, s_g=ZL.SIGMA_G,
        psi0=ZL.PSI_0, psi1=ZL.PSI_1, s_r=ZL.SIGMA_R,
        l1=ZL.LAMBDA_1, l2=ZL.LAMBDA_2, y_up=ZL.Y_HAT,
        thbar=THBAR, eta=ETA, vars=VARSIG, E0=ETA * ZL.A_D * K0,
        rho_s=RHO_S, r1=R1, r2=R2, y_lo=Y_LO, Y_CAP=Y_CAP,
        lam3=(float(lam3) if postdam else 0.0),
        postdam=bool(postdam), pretech=bool(pretech), xi=float(xi))


# ------------------------------------------------------------------ callbacks
def _E(z, p):
    return p["E0"] * (1.0 - z)


def _phis(ctr, p):
    phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * ctr["i_d"], 1e-9))
    phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * ctr["i_g"], 1e-9))
    return phid, phig


def _lN_slopes(y, p):
    """Regime-correct damage slope/curvature, saturated at Y_CAP (v5 horizon)."""
    y_eff = np.minimum(y, p["Y_CAP"])
    lNy = p["l1"] + p["l2"] * y_eff
    lNyy = np.full_like(np.asarray(y_eff, dtype=float), p["l2"])
    if p["postdam"]:
        exc = np.maximum(y_eff - p["y_up"], 0.0)
        lNy = lNy + p["lam3"] * exc
        lNyy = lNyy + p["lam3"] * (exc > 0.0)
    return lNy, lNyy


def _h_channels(z, y, cost, p):
    """Closed-form Brownian drift distortions from the FROZEN costates.
    cost=None (first Howard iteration) or xi=inf => zero distortion."""
    zr = np.zeros_like(np.asarray(z, dtype=float))
    if cost is None or not np.isfinite(p["xi"]):
        return zr, zr, zr, zr
    xi = p["xi"]
    W_Z, W_Y, W_S = cost["W_Z"], cost["W_Y"], cost["W_S"]
    lNy, _ = _lN_slopes(y, p)
    qd = (1.0 - W_S) - z * W_Z            # V_logK - Z V_Z  (V_logK = 1 - W_s)
    qg = (1.0 - W_S) + (1.0 - z) * W_Z
    H_CLIP = 25.0   # trust-region cap on each |h_j| (ports PHYSRISK's fix for the
    # lagged-costate h_y feedback spiral at xi=0.05; caps only the transient --
    # if it binds at convergence the solve is not trusted there, checked in V3)
    h_d = np.clip(-qd * (1.0 - z) * p["s_d"] / xi, -H_CLIP, H_CLIP)
    h_g = np.clip(-qg * z * p["s_g"] / xi, -H_CLIP, H_CLIP)
    h_r = np.clip((-W_S * p["s_r"] / xi), -H_CLIP, H_CLIP) if p["pretech"] else zr
    h_y = np.clip(-(W_Y - lNy) * _E(z, p) * p["vars"] / xi, -H_CLIP, H_CLIP)
    return h_d, h_g, h_r, h_y


def _alk(z, ctr, p, h_d, h_g):
    phid, phig = _phis(ctr, p)
    Dc = p["s_d"] ** 2 * (1 - z) ** 2 + p["s_g"] ** 2 * z ** 2
    return ((1 - z) * phid + z * phig - 0.5 * Dc
            + (1 - z) * p["s_d"] * h_d + z * p["s_g"] * h_g), phid, phig


def drift(z, y, s, ctr, cost, p):
    h_d, h_g, h_r, h_y = _h_channels(z, y, cost, p)
    a_lK, phid, phig = _alk(z, ctr, p, h_d, h_g)
    a_Z = z * (1 - z) * (phig - phid + (1 - z) * p["s_d"] ** 2
                         - z * p["s_g"] ** 2) \
        + z * (1 - z) * (p["s_g"] * h_g - p["s_d"] * h_d)
    a_Y = (p["thbar"] + p["vars"] * h_y) * _E(z, p)
    if p["pretech"]:
        ir = np.maximum(ctr["i_r"], 0.0)
        a_R = p["psi0"] * np.sqrt(ir) \
            * np.exp(np.clip(-p["psi1"] * s, -EXP_CLIP, EXP_CLIP)) \
            - 0.5 * p["s_r"] ** 2 + p["s_r"] * h_r
        a_S = a_R - a_lK
    else:
        a_S = np.zeros_like(np.asarray(z, dtype=float))
    return a_Z, a_Y, a_S


def flow(z, y, s, ctr, cost, p):
    h_d, h_g, h_r, h_y = _h_channels(z, y, cost, p)
    a_lK, _, _ = _alk(z, ctr, p, h_d, h_g)
    E = _E(z, p)
    a_Y = (p["thbar"] + p["vars"] * h_y) * E
    c = (p["A_d"] - ctr["i_d"]) * (1 - z) + (p["A_g"] - ctr["i_g"]) * z
    if p["pretech"]:
        c = c - ctr["i_r"]
    lNy, lNyy = _lN_slopes(y, p)
    b_Y = 0.5 * p["vars"] ** 2 * E ** 2
    fl = p["delta"] * np.log(np.maximum(c, 1e-12)) + a_lK \
        - (lNy * a_Y + lNyy * b_Y)
    if np.isfinite(p["xi"]) and cost is not None:
        fl = fl + 0.5 * p["xi"] * (h_d ** 2 + h_g ** 2 + h_r ** 2 + h_y ** 2)
    return fl


def controls(W_Z, W_Y, W_S, Z, Y, S, p):
    if p["pretech"]:
        qd = (1.0 - W_S) - Z * W_Z
        qg = (1.0 - W_S) + (1.0 - Z) * W_Z
        return foc_controls(qd, qg, Z, p, qr=W_S, S=S)
    qd = 1.0 - Z * W_Z          # post-tech: W is s-independent, V_logK = 1
    qg = 1.0 + (1.0 - Z) * W_Z
    return foc_controls(qd, qg, Z, p)


def diff(Z, Y, S, ctr, p):
    sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
    E = _E(Z, p)
    D = {"ZZ": 0.5 * Z ** 2 * (1 - Z) ** 2 * (sd2 + sg2),
         "YY": 0.5 * p["vars"] ** 2 * E ** 2}
    if p["pretech"]:
        Dc = sd2 * (1 - Z) ** 2 + sg2 * Z ** 2
        D["SS"] = 0.5 * (Dc + p["s_r"] ** 2)
        D["SZ"] = Z * (1 - Z) ** 2 * sd2 - Z ** 2 * (1 - Z) * sg2
    return D


def init_controls(Z, Y, S, p):
    out = {"i_d": np.zeros_like(Z), "i_g": np.full_like(Z, 0.05)}
    if p["pretech"]:
        out["i_r"] = np.full_like(Z, 0.002)
    return out


# ------------------------------------------------------------------ jumps
def Jn(y, p):
    """Damage-jump intensity r1*(exp((r2/2)(y-y_lo)^2)-1)*1{y>=y_lo}; exponent
    clipped at +35 (binds only on far-field characteristics, Y >~ 21)."""
    ex = np.clip(0.5 * p["r2"] * (y - p["y_lo"]) ** 2, 0.0, EXP_CLIP)
    return p["r1"] * (np.exp(ex) - 1.0) * (y >= p["y_lo"])


def tech_intensity(s, p):
    return np.exp(np.clip(s, -EXP_CLIP, EXP_CLIP)) / p["rho_s"]


def dmg_gap(y, l3, p):
    """logN book difference across the damage jump (net space), saturated at
    Y_CAP: (lambda3/2) * max(min(Y,Y_CAP) - yhat, 0)^2."""
    exc = np.maximum(np.minimum(y, p["Y_CAP"]) - p["y_up"], 0.0)
    return 0.5 * l3 * exc ** 2


class W2Interp:
    """Edge-clamped bilinear (Z,Y) interpolation of an s-independent field,
    exposed with the harness w_post signature f(z, y, s)."""

    def __init__(self, Zg, Yg, W2):
        self.f = RGI((Zg, Yg), W2, method="linear", bounds_error=False,
                     fill_value=None)
        self.lo = (Zg[0], Yg[0]); self.hi = (Zg[-1], Yg[-1])

    def __call__(self, z, y, s=None):
        q = np.empty((np.size(z), 2))
        q[:, 0] = np.clip(z, self.lo[0], self.hi[0])
        q[:, 1] = np.clip(y, self.lo[1], self.hi[1])
        return self.f(q)


class W3Interp:
    """Edge-clamped trilinear (Z,Y,s) interpolation, harness w_post signature."""

    def __init__(self, Zg, Yg, Sg, W):
        self.fi = FieldInterp((Zg, Yg, Sg), {"W": W})

    def __call__(self, z, y, s):
        return self.fi(z, y, s)["W"]


def downstream_spec(regime, l3idx):
    """(regime, l3idx) pairs whose SAME-xi solves this regime's jumps read."""
    if regime == "PostDamagePostTech":
        return []
    if regime == "PreDamagePostTech":
        return [("PostDamagePostTech", k) for k in range(NL)]
    if regime == "PostDamagePreTech":
        return [("PostDamagePostTech", l3idx)]
    if regime == "PreDamagePreTech":
        return [("PreDamagePostTech", None)] \
            + [("PostDamagePreTech", k) for k in range(NL)]
    raise ValueError(regime)


def load_downstream(regime, l3idx, xi, outdir):
    """Load the post-jump interpolators; FAIL LOUDLY on missing files."""
    spec = downstream_spec(regime, l3idx)
    interps, files = [], []
    for (rg, k) in spec:
        path = os.path.join(outdir, fname(rg, k, xi))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"downstream solve missing for {regime}: {path} "
                f"(dependency order: post regimes before pre, same xi)")
        d = np.load(path)
        files.append(path)
        if rg.endswith("PostTech"):
            interps.append(((rg, k), W2Interp(d["Z"], d["Y"], d["W2"])))
        else:
            interps.append(((rg, k), W3Interp(d["Z"], d["Y"], d["S"], d["W"])))
    return interps, files


def build_jumps(regime, l3idx, xi, interps, p):
    jumps = []
    by_key = dict(interps)
    if regime == "PreDamagePostTech":
        for k in range(NL):
            I = by_key[("PostDamagePostTech", k)]
            l3v = LAM3_GRID[k]
            jumps.append({
                "name": f"damage_l3{k}",
                "intensity": (lambda z, y, s, pp: Jn(y, pp) / NL),
                "w_post": (lambda z, y, s, I=I, l3v=l3v, pp=p:
                           I(z, y, s) - dmg_gap(y, l3v, pp))})
    elif regime == "PostDamagePreTech":
        I = by_key[("PostDamagePostTech", l3idx)]
        jumps.append({"name": "tech",
                      "intensity": (lambda z, y, s, pp: tech_intensity(s, pp)),
                      "w_post": (lambda z, y, s, I=I: I(z, y, s))})
    elif regime == "PreDamagePreTech":
        I = by_key[("PreDamagePostTech", None)]
        jumps.append({"name": "tech",
                      "intensity": (lambda z, y, s, pp: tech_intensity(s, pp)),
                      "w_post": (lambda z, y, s, I=I: I(z, y, s))})
        for k in range(NL):
            I = by_key[("PostDamagePreTech", k)]
            l3v = LAM3_GRID[k]
            jumps.append({
                "name": f"damage_l3{k}",
                "intensity": (lambda z, y, s, pp: Jn(y, pp) / NL),
                "w_post": (lambda z, y, s, I=I, l3v=l3v, pp=p:
                           I(z, y, s) - dmg_gap(y, l3v, pp))})
    return jumps


# ------------------------------------------------------------------ config
def make_grids(regime, nz, ny, ns):
    Zg = np.linspace(0.01, 0.99, nz)
    Yg = np.linspace(0.0, 4.0, ny)
    n_s = ns if regime.endswith("PreTech") else NS_POSTTECH
    Sg = np.linspace(S_MIN, S_MAX, n_s)
    return Zg, Yg, Sg


def build_cfg(regime, l3idx, xi, grids, interps, W_init=None):
    lam3 = LAM3_GRID[l3idx] if l3idx is not None else 0.0
    p = base_params(regime, lam3, xi)
    cfg = dict(name=f"JONES-{regime}"
                    + (f"-l3{l3idx}" if l3idx is not None else "")
                    + f"-{xi_tag(xi)}",
               params=p, grids=grids, controls=controls,
               control_names=(["i_d", "i_g", "i_r"] if p["pretech"]
                              else ["i_d", "i_g"]),
               drift=drift, flow=flow, diff=diff, init_controls=init_controls,
               xi=xi, jumps=build_jumps(regime, l3idx, xi, interps, p),
               needs_costates=True, step_exact=True)
    if W_init is not None:
        cfg["W_init"] = W_init
    return cfg


# ------------------------------------------------------------------ solve+save
def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def monotonicity_stats(out, pretech):
    """Sanity fractions: W should (weakly) decrease in Y; pre-tech W should
    (weakly) increase in s (knowledge is good news)."""
    W = out["W"]
    dY = np.diff(W, axis=1)
    st = {"frac_W_increasing_in_Y": float(np.mean(dY > 1e-10))}
    if pretech:
        dS = np.diff(W, axis=2)
        st["frac_W_decreasing_in_s"] = float(np.mean(dS < -1e-10))
        st["min_dW_ds"] = float(np.min(dS / (out["S"][1] - out["S"][0])))
    return st


_R = float(os.environ.get("JONES_RELAX", "0.3"))
_RIR = os.environ.get("JONES_RELAX_IR")            # i_r-specific damping (Howard limit-cycle fix)
RELAX = ({"i_d": _R, "i_g": _R, "i_r": float(_RIR)} if _RIR else _R)       # Howard damping; tech-channel
TAIL_AVG = int(os.environ.get("JONES_TAIL_AVG", "6"))     # i_r<->hazard loop needs ~0.1/12

def solve_one(regime, l3idx, xi, outdir, nz, ny, ns, howard=100, T=1200.0,
              dt=2.5, init_ctr=None, W_init=None, tol=1e-6, tag_extra=None):
    grids = make_grids(regime, nz, ny, ns)
    interps, dfiles = load_downstream(regime, l3idx, xi, outdir)
    cfg = build_cfg(regime, l3idx, xi, grids, interps, W_init=W_init)
    print(f"== solve {cfg['name']} grid=({grids[0].size},{grids[1].size},"
          f"{grids[2].size}) howard<={howard} ==", flush=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = solve_reduced(cfg, T=T, dt=dt, howard_max=howard,
                            tol_pocket=tol, relax=RELAX, tail_avg=TAIL_AVG,
                            init_ctr=init_ctr, verbose=True,
                            ref_pt=(0.7, 1.1, -4.364))
    log = buf.getvalue()
    print(log, flush=True)
    assert np.all(np.isfinite(out["W"])), "non-finite W"
    st = monotonicity_stats(out, cfg["params"]["pretech"])

    os.makedirs(outdir, exist_ok=True)
    name = fname(regime, l3idx, xi)
    if tag_extra:
        name = name.replace(".npz", f"_{tag_extra}.npz")
    path = os.path.join(outdir, name)
    save = dict(Z=out["Z"], Y=out["Y"], S=out["S"], W=out["W"],
                W_Z=out["W_Z"], W_Y=out["W_Y"], W_S=out["W_S"],
                i_d=out["i_d"], i_g=out["i_g"], c=out["c"],
                lam3=np.float64(cfg["params"]["lam3"]), xi=np.float64(xi))
    if "i_r" in out:
        save["i_r"] = out["i_r"]
    if not regime.endswith("PreTech"):
        s_spread = float(np.max(np.ptp(out["W"], axis=2)))
        assert s_spread < 1e-8, f"post-tech W not s-independent: {s_spread}"
        save["W2"] = out["W"][:, :, 0]
    np.savez_compressed(path, **save)
    prov = dict(
        economy="JONES (scale-free semi-endogenous intensity economy)",
        design="benchmarks/economy_zoo/design_jones.json",
        harness="benchmarks/economy_zoo/solvers/fd_reduced.py",
        callbacks="benchmarks/economy_zoo/solvers/jones_callbacks.py",
        value_object="net-space reduced value: v = V + logN(Y;lambda3) = logK + W",
        regime=regime, l3idx=l3idx,
        lam3=cfg["params"]["lam3"], xi=(xi if np.isfinite(xi) else "inf"),
        grid=[int(g.size) for g in grids], T=T, dt=dt,
        howard_max=howard, relax=RELAX, tail_avg=TAIL_AVG, tol_pocket=tol,
        step_exact=True, needs_costates=True, exp_clip=EXP_CLIP,
        warm_start=bool(init_ctr is not None),
        downstream_files=dfiles,
        gates=dict(iters=out["iters"], di_int_final=out["di_int_final"],
                   mutual_consistency_gate_int=out["gate_int"],
                   true_residual_max=out["max_abs_residual"],
                   true_residual_rms=out["rms_residual"]),
        monotonicity=st, seconds=out["time"],
        howard_log=log.strip().splitlines(),
        date=str(datetime.date.today()))
    with open(path.replace(".npz", "_PROVENANCE.json"), "w") as f:
        json.dump(_json_safe(prov), f, indent=1)
    print(f"  -> {path}\n     iters={out['iters']} gate_int={out['gate_int']:.2e}"
          f" rmsR={out['rms_residual']:.2e} maxR={out['max_abs_residual']:.2e}"
          f" {st}", flush=True)
    return out


def run_ladder(regime, l3idx, outdir, nz, ny, ns, howard_cold=100,
               howard_warm=60, T=1200.0, dt=2.5):
    """The 3-xi warm-chained ladder cell for one (regime, lambda3)."""
    check_params()
    prev = None
    for xi in XI_LADDER:
        if prev is None:
            out = solve_one(regime, l3idx, xi, outdir, nz, ny, ns,
                            howard=howard_cold, T=T, dt=dt)
        else:
            names = ["i_d", "i_g"] + (["i_r"] if regime.endswith("PreTech")
                                      else [])
            init = {k: prev[k] for k in names}
            out = solve_one(regime, l3idx, xi, outdir, nz, ny, ns,
                            howard=howard_warm, T=T, dt=dt,
                            init_ctr=init, W_init=prev["W"])
        prev = out
    return prev


# ================================================================== validations
def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def _prep_coarse_terminal(outdir, nz, ny, l3set=range(NL), xi_list=(np.inf,),
                          howard=40):
    """Coarse terminal (PostDamagePostTech) solves; within a lambda3, later
    xi's warm-chain from the first (cold) one."""
    for k in l3set:
        prev = None
        for xi in xi_list:
            path = os.path.join(outdir, fname("PostDamagePostTech", k, xi))
            if os.path.exists(path):
                prev = {kk: np.load(path)[kk] for kk in ("i_d", "i_g", "W")}
                continue
            init = ({"i_d": prev["i_d"], "i_g": prev["i_g"]}
                    if prev is not None else None)
            Wi = prev["W"] if prev is not None else None
            out = solve_one("PostDamagePostTech", k, xi, outdir, nz, ny,
                            NS_POSTTECH, howard=(howard if prev is None
                                                 else max(15, howard // 2)),
                            init_ctr=init, W_init=Wi)
            prev = out


def validate_v1(nz=31, ny=21, ns=25, howard_base=40, howard_arm=15):
    """V1: jump-bearing correctness. A finite-but-huge-xi solve (nonlinear
    xi*J*(1-e^{-(W'-W)/xi}) source, xi=1e5) must match the explicitly-LINEAR
    jump-expectation solve (harness neutral branch, xi=inf: source J*(W'-W)).
    Gate: field RMS(W) <= 1e-4.  Run on BOTH jump types:
      V1a  PreDamagePostTech  (5 damage channels, 2-D)
      V1b  PostDamagePreTech l3=2  (tech channel e^s/rho_s, 3-D)."""
    os.makedirs(VALD, exist_ok=True)
    check_params()
    XI_BIG = 1.0e5
    res = {}

    print("\n[V1] preparing coarse terminal solves (xi=inf)", flush=True)
    _prep_coarse_terminal(VALD, nz, ny, xi_list=(np.inf,), howard=howard_base)
    # arms read downstream at their own xi -> alias the xi=inf terminals
    for k in range(NL):
        src = os.path.join(VALD, fname("PostDamagePostTech", k, np.inf))
        for xi in (XI_BIG,):
            dst = os.path.join(VALD, fname("PostDamagePostTech", k, xi))
            if not os.path.exists(dst):
                import shutil
                shutil.copyfile(src, dst)
                shutil.copyfile(src.replace(".npz", "_PROVENANCE.json"),
                                dst.replace(".npz", "_PROVENANCE.json"))

    for label, reg, l3idx, ns_ in (("V1a_damage", "PreDamagePostTech", None,
                                    NS_POSTTECH),
                                   ("V1b_tech", "PostDamagePreTech", 2, ns)):
        print(f"\n[{label}] baseline xi=inf cold", flush=True)
        base = solve_one(reg, l3idx, np.inf, VALD, nz, ny, ns_,
                         howard=howard_base)
        names = ["i_d", "i_g"] + (["i_r"] if reg.endswith("PreTech") else [])
        init = {k: base[k] for k in names}
        print(f"[{label}] arm LINEAR (xi=inf, warm, {howard_arm} it)",
              flush=True)
        armL = solve_one(reg, l3idx, np.inf, VALD, nz, ny, ns_,
                         howard=howard_arm, init_ctr=init, W_init=base["W"],
                         tag_extra="armL")
        print(f"[{label}] arm NONLINEAR (xi=1e5, warm, {howard_arm} it)",
              flush=True)
        armX = solve_one(reg, l3idx, XI_BIG, VALD, nz, ny, ns_,
                         howard=howard_arm, init_ctr=init, W_init=base["W"],
                         tag_extra="armX")
        rms = _rms(armX["W"] - armL["W"])
        mx = float(np.max(np.abs(armX["W"] - armL["W"])))
        ok = rms <= 1e-4
        res[label] = dict(regime=reg, l3idx=l3idx, xi_big=XI_BIG,
                          W_rms=rms, W_max=mx, gate_1e_4=bool(ok))
        print(f"[{label}] RMS(W_nonlin - W_linear) = {rms:.3e}  "
              f"max = {mx:.3e}  GATE<=1e-4: {'PASS' if ok else 'FAIL'}",
              flush=True)

    with open(os.path.join(VALD, "V1_REPORT.json"), "w") as f:
        json.dump(_json_safe(res), f, indent=1)
    return res


def validate_v2(nz=31, ny=21, ns=25):
    """V2: step_exact stability where J*dt is large (tech hazard 8.7/yr at
    s=+2, dt=2.5 -> J*dt ~ 21.8). Uses the converged V1b baseline policy;
    shows (a) no overflow/NaN, (b) W monotone increasing in s, (c) W at the
    s-top pinned near the post-tech target, (d) the left-rectangle
    (step_exact=False) quadrature corrupts the s-top while step_exact does not."""
    os.makedirs(VALD, exist_ok=True)
    reg, l3idx = "PostDamagePreTech", 2
    base_path = os.path.join(VALD, fname(reg, l3idx, np.inf))
    if not os.path.exists(base_path):
        _prep_coarse_terminal(VALD, nz, ny, xi_list=(np.inf,))
        solve_one(reg, l3idx, np.inf, VALD, nz, ny, ns, howard=40)
    d = np.load(base_path)
    grids = (d["Z"], d["Y"], d["S"])
    interps, _ = load_downstream(reg, l3idx, np.inf, VALD)
    cfg = build_cfg(reg, l3idx, np.inf, grids, interps)
    ctr = {k: d[k] for k in ("i_d", "i_g", "i_r")}
    cost = {"W_Z": d["W_Z"], "W_Y": d["W_Y"], "W_S": d["W_S"]}

    W_ex = simulate_W(cfg, ctr, W_lag=d["W"], cost_fields=cost)
    cfg_lr = dict(cfg); cfg_lr["step_exact"] = False
    W_lr = simulate_W(cfg_lr, ctr, W_lag=d["W"], cost_fields=cost)

    Wp = interps[0][1](grids[0].repeat(len(grids[1])),
                       np.tile(grids[1], len(grids[0])), None) \
        .reshape(len(grids[0]), len(grids[1]))
    top_gap_ex = float(np.max(np.abs(W_ex[:, :, -1] - Wp)))
    top_gap_lr = float(np.max(np.abs(W_lr[:, :, -1] - Wp)))
    dS = np.diff(W_ex, axis=2) / (grids[2][1] - grids[2][0])
    res = dict(
        finite=bool(np.all(np.isfinite(W_ex))),
        J_dt_max=float(tech_intensity(grids[2][-1], cfg["params"]) * 2.5),
        min_dW_ds=float(np.min(dS)),
        frac_dW_ds_negative=float(np.mean(dS < -1e-10)),
        s_top_gap_step_exact=top_gap_ex,
        s_top_gap_left_rectangle=top_gap_lr,
        s_top_mean_W_minus_Wpost_exact=float(np.mean(W_ex[:, :, -1] - Wp)),
        rms_step_exact_vs_leftrect=_rms(W_ex - W_lr))
    ok = res["finite"] and res["min_dW_ds"] > -1e-6
    res["gate_stable_monotone"] = bool(ok)
    print("[V2]", json.dumps(_json_safe(res), indent=1), flush=True)
    print(f"[V2] GATE (finite + monotone W in s): {'PASS' if ok else 'FAIL'}",
          flush=True)
    with open(os.path.join(VALD, "V2_REPORT.json"), "w") as f:
        json.dump(_json_safe(res), f, indent=1)
    return res


def validate_v3(nz=31, ny=21, ns=25, howard=30):
    """V3: xi=0.05 outer-loop stability with the g-distortion ON (exp-clip 35
    in the harness). Ladder 148.4 -> 0.1 -> 0.05 on
      V3a PostDamagePreTech l3=2  (tech channel: good-news jump, g < 1)
      V3b PreDamagePostTech       (damage channels: bad-news jump, g > 1)
    Reports the Howard di_int history and the mutual-consistency gate."""
    os.makedirs(VALD, exist_ok=True)
    check_params()
    res = {}
    # terminals at every ladder xi (cheap 2-D solves, warm-chained in xi)
    print("\n[V3] terminal solves at ladder xi", flush=True)
    _prep_coarse_terminal(VALD, nz, ny, xi_list=tuple(XI_LADDER), howard=40)

    for label, reg, l3idx, ns_ in (("V3a_tech", "PostDamagePreTech", 2, ns),
                                   ("V3b_damage", "PreDamagePostTech", None,
                                    NS_POSTTECH)):
        prev, hist = None, {}
        # reuse the xi=inf V1/V2 baseline as the cold start where available
        b = os.path.join(VALD, fname(reg, l3idx, np.inf))
        if os.path.exists(b):
            d = np.load(b)
            names = ["i_d", "i_g"] + (["i_r"] if reg.endswith("PreTech")
                                      else [])
            prev = {k: d[k] for k in names}
            prev["W"] = d["W"]
        for xi in XI_LADDER:
            init = None; Wi = None
            if prev is not None:
                names = ["i_d", "i_g"] + (["i_r"] if reg.endswith("PreTech")
                                          else [])
                init = {k: prev[k] for k in names}
                Wi = prev["W"]
            out = solve_one(reg, l3idx, xi, VALD, nz, ny, ns_,
                            howard=(40 if init is None else howard),
                            init_ctr=init, W_init=Wi)
            hist[xi_tag(xi)] = dict(
                iters=out["iters"], di_int_final=out["di_int_final"],
                gate_int=out["gate_int"],
                rms_residual=out["rms_residual"],
                max_abs_residual=out["max_abs_residual"],
                finite=bool(np.all(np.isfinite(out["W"]))))
            prev = out
        g05 = hist["xi0p05"]
        ok = g05["finite"] and g05["gate_int"] < 5e-3
        res[label] = dict(regime=reg, ladder=hist, gate_xi005=bool(ok))
        print(f"[{label}] xi=0.05: iters={g05['iters']} "
              f"di_int={g05['di_int_final']:.2e} gate_int={g05['gate_int']:.2e}"
              f" rmsR={g05['rms_residual']:.2e} "
              f"GATE: {'PASS' if ok else 'FAIL'}", flush=True)

    with open(os.path.join(VALD, "V3_REPORT.json"), "w") as f:
        json.dump(_json_safe(res), f, indent=1)
    return res


def smoke(nz=13, ny=9, ns=9):
    """Tiny-grid full 4-regime chain at xi=148.4 then 0.05: pipeline sanity."""
    sd = os.path.join(VALD, "smoke")
    os.makedirs(sd, exist_ok=True)
    for k in range(NL):
        run_ladder("PostDamagePostTech", k, sd, nz, ny, ns,
                   howard_cold=25, howard_warm=12)
    run_ladder("PreDamagePostTech", None, sd, nz, ny, ns,
               howard_cold=25, howard_warm=12)
    for k in range(NL):
        run_ladder("PostDamagePreTech", k, sd, nz, ny, ns,
                   howard_cold=25, howard_warm=12)
    run_ladder("PreDamagePreTech", None, sd, nz, ny, ns,
               howard_cold=25, howard_warm=12)
    print("[smoke] full 4-regime x 3-xi chain complete", flush=True)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("cmd", choices=["solve", "ladder", "v1", "v2", "v3",
                                    "smoke", "check-params"])
    ap.add_argument("--regime", choices=REGIMES)
    ap.add_argument("--l3idx", type=int, default=None)
    ap.add_argument("--xi", type=float, default=None)
    ap.add_argument("--nz", type=int, default=61)
    ap.add_argument("--ny", type=int, default=31)
    ap.add_argument("--ns", type=int, default=41)
    ap.add_argument("--outdir", default=OUTD)
    ap.add_argument("--howard", type=int, default=100)
    ap.add_argument("--howard-warm", type=int, default=60)
    a = ap.parse_args()

    if a.cmd == "check-params":
        check_params(); return
    if a.cmd == "smoke":
        smoke(); return
    if a.cmd == "v1":
        validate_v1(); return
    if a.cmd == "v2":
        validate_v2(); return
    if a.cmd == "v3":
        validate_v3(howard=a.howard); return

    assert a.regime, "--regime required"
    l3idx = a.l3idx
    if a.regime.startswith("PostDamage"):
        assert l3idx is not None and 0 <= l3idx < NL, \
            "--l3idx 0..4 required for post-damage regimes"
    else:
        l3idx = None
    if a.cmd == "ladder":
        run_ladder(a.regime, l3idx, a.outdir, a.nz, a.ny, a.ns,
                   howard_cold=a.howard, howard_warm=a.howard_warm)
    else:
        assert a.xi is not None, "--xi required for solve"
        check_params()
        solve_one(a.regime, l3idx, a.xi, a.outdir, a.nz, a.ny, a.ns,
                  howard=a.howard)


if __name__ == "__main__":
    main()
