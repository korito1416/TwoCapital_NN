"""
fd_reduced.py -- shared reduced-FD PIBYS harness for the economy_zoo FD-3D members
(JONES, PHYSRISK), generalizing benchmarks/post_damage_post_tech/fd_pdpt_v5_stable.py
from states (logK, Z, Y) to a generic reduced box (Z, Y, s) with pluggable per-economy
callbacks.

WHAT IS KEPT from the verified v5/v5_stable PIBYS scheme (byte-parity where possible):
  * EVALUATE (policy fixed) by characteristics: from EVERY grid node integrate the
    deterministic drift ODE forward (RK2 midpoint, controls edge-clamped trilinear along
    the path) and ACCUMULATE the discounted running flow. No Eulerian transport of W ->
    no artificial diffusion; O(sigma^2) diffusion is dropped from the evaluation (it is
    ~1e-4 here, exactly the v5 approximation) and re-enters only the residual monitor.
  * IMPROVE: central-difference costates (W_Z, W_Y, W_s) of the SMOOTH integrated W ->
    per-economy FOC/Newton control block (closed form for this parameter family).
  * OUTER Howard loop (v5_stable pattern): control under-relaxation (relax), convergence
    measured on an economically relevant INTERIOR BOX (pocket tolerance, not the whole-grid
    sup-norm which is dominated by feasibility-floor corners), tail-averaging of the last
    K policies to kill floor-corner limit cycles, and a FINAL MUTUAL-CONSISTENCY GATE:
    re-evaluate W at the tail-averaged policy, recompute FOC controls, require the interior
    control delta to be small, and REPORT it.
  * TRUE-residual monitor: central differences including the tiny diffusion terms and the
    full nonlinear jump brackets, evaluated with the CURRENT W (not the lagged one).

WHAT IS NEW (needed by the zoo economies):
  * Generic state names (Z, Y, s). s is the economy's third state (JONES/PHYSRISK:
    s = logR - logK; the degenerate validation below instead sets s = logK).
  * JUMP CHANNELS with robust xi-distortion, treated SEMI-IMPLICITLY along
    characteristics: the jump sink -J*g*W enters the EXACT exponential integrating factor
    exp(-int (delta + sum J*g) dt)  (unconditionally stable/monotone even where J*dt >> 1,
    e.g. the JONES s-top where the hazard is 8.7/yr), while the downstream gain
    J*g*W_post + xi*J*(1 - g + g log g) is an explicit flow source from the already-solved
    post-jump interpolators. The distortion g = exp(-(W_post - W_lag)/xi) uses the PREVIOUS
    outer iterate W_lag (lagged fixed point; exact at convergence) with exponent clipped to
    +-EXP_CLIP = +-35 (float32-safe guard, per repo convention).
  * Optional costate feedback into drift/flow (needs_costates=True): the frozen costate
    grids of the previous iterate are interpolated along paths and handed to the economy's
    drift/flow callbacks, so finite-xi worst-case drift terms sigma*h (O(sigma^2/xi)) and
    the -(1/2 xi)|L|^2 penalty can be included in the evaluation at small xi.
  * A reusable FOC control helper `foc_controls` for the shared FOC family
      delta/c = phi_d'(i_d) qd = phi_g'(i_g) qg [= psi0 psi1 i_r^{psi1-1} e^{-psi1 s} qr]
    with the feasibility costate floor (QFLOOR: keeps 1 + theta*i > 0, i.e. K >= 0; NOT a
    bias hack -- see fd_pdpt_v5.py) and, for psi1 = 1/2, the exact closed form:
    c solves  kappa c^2 + (1+Q) c - B = 0 with
      Q = [(1-Z) G_d qd + Z G_g qg]/delta,  B = (1-Z)(A_d + 1/theta_d) + Z(A_g + 1/theta_g),
      kappa = (psi0 psi1 e^{-psi1 s} qr / delta)^2   (i_r = kappa c^2),
    which reduces to the v5 two-control formula when kappa = 0.

ECONOMY CONFIG (a plain dict `cfg`):
  required:
    'name'    : str
    'params'  : dict p, passed verbatim to all callbacks
    'grids'   : (Zg, Yg, Sg) 1-D arrays
    'controls': f(W_Z, W_Y, W_S, Z, Y, S, p) -> dict of (nZ,nY,nS) control fields, must
                include the fields named in 'control_names' plus 'c' (consumption, derived,
                never interpolated)
    'control_names': list of control field names interpolated along paths, e.g.
                ['i_d','i_g'] or ['i_d','i_g','i_r']
    'drift'   : f(z, y, s, ctr, cost, p) -> (a_Z, a_Y, a_S) on flat arrays
    'flow'    : f(z, y, s, ctr, cost, p) -> flat flow array. Must contain the FULL running
                reward (delta log c + any delta*state utility terms + deterministic damage
                terms incl. any flow-embedded second-order pieces, exactly as
                fd_pdpt_v5.simulate_v does) but NOT the -delta*W term and NOT jump terms.
    'init_controls': f(Z, Y, S, p) -> dict of initial control fields (cold start), or the
                caller passes warm-start fields via solve_reduced(init_ctr=...).
  optional:
    'xi'      : scalar (np.inf or >= ~148 => neutral jump/drift treatment). Default inf.
    'jumps'   : list of channels, each {'name': str,
                 'intensity': f(z, y, s, p) -> J >= 0,
                 'w_post':    f(z, y, s)   -> post-jump value W' at these points}
    'diff'    : f(Z, Y, S, ctr, p) -> dict of second-order coefficient arrays for the
                residual monitor, keys among {'ZZ','YY','SS','SZ','ZY','SY'} multiplying
                W_ZZ, W_YY, W_SS, W_sZ, W_ZY, W_sY (coefficients INCLUDE the 1/2).
    'needs_costates': bool (default False) -- interpolate frozen (W_Z, W_Y, W_S) along
                paths into the `cost` dict for drift/flow. NOTE: on the FIRST Howard
                iteration cost is None (no previous iterate) -- callbacks must treat
                cost=None as zero distortion.
    'W_init'  : (nZ,nY,nS) array, first-iteration W_lag for the jump distortion g and
                costate feedback (default zeros; for small xi ALWAYS pass a warm W, e.g.
                the neutral-xi solve, or g = exp(+W_post/xi) explodes on iteration 0).
    'z_clip'  : (lo, hi) path clip for Z (default (1e-4, 1-1e-4), as v5).
    'step_exact': bool (default False). False = v5's left-rectangle discount quadrature
                V += exp(-Lam)*flow*dt (byte-parity with the validated reference solve).
                True = exact per-step integral for piecewise-constant flow/rate,
                weight (1 - exp(-lam*dt))/lam with lam = delta + sum J*g. REQUIRED for
                jump-heavy economies: where (delta + J)*dt >> 1 (JONES s-top, hazard
                8.7/yr at dt = 2.5) the left rectangle overweights the first step by
                ~J*dt and corrupts W near the edge; the exact weight gives the correct
                limit W -> W_post there. The two differ by an O(delta*dt) quadrature
                factor elsewhere; the true-residual monitor is the arbiter.
    'box'     : ((z0,z1),(y0,y1),(s0,s1)) interior box for the pocket tolerance
                (default: trim 10% of the range on each side).

RETURNS (solve_reduced): dict with grids Z, Y, S; W; every control field + c; costates
W_Z, W_Y, W_S; iters, time, di_int_final, gate_int, max_abs_residual, rms_residual
(interior true residual incl. jumps/diffusion).

VALIDATION (run `python fd_reduced.py`): degenerate configuration that EXACTLY embeds the
verified post-damage post-tech solve (fd_pdpt_v5_stable, lam3 = 1/6, xi = 148.4) into the
reduced harness by taking s == logK: same drift/flow/FOC formulas, no jumps, grids equal to
the reference (Z 61, Y 31, s = logK 31). The harness W(Z, Y, s) must reproduce the reference
v(logK, Z, Y) (transposed) -- field RMS target <= 5e-3. Writes
outputs/harness_validation_degenerate.npz + _PROVENANCE.json + a comparison figure.
"""
import os
import sys
import time
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

EXP_CLIP = 35.0     # repo-wide guard: clip any exponent that can exceed ~35
QFLOOR = 2e-3       # feasibility costate floor, identical to fd_pdpt_v5 (keeps 1+theta*i>0)
NEUTRAL_XI = 1e6    # xi at/above this (or inf) => uncertainty-neutral jump treatment

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "outputs")
FIGD = os.path.abspath(os.path.join(HERE, "..", "figures"))


# ----------------------------------------------------------------------------- derivatives
def _grad(v, axis, dx):
    """Central interior, one-sided at faces (identical to fd_pdpt_v5._grad)."""
    g = np.zeros_like(v)
    lo = [slice(None)] * 3; hi = [slice(None)] * 3; md = [slice(None)] * 3
    lo[axis] = slice(2, None); hi[axis] = slice(0, -2); md[axis] = slice(1, -1)
    g[tuple(md)] = (v[tuple(lo)] - v[tuple(hi)]) / (2 * dx)
    e0 = [slice(None)] * 3; e1 = [slice(None)] * 3; en = [slice(None)] * 3; en1 = [slice(None)] * 3
    e0[axis] = 0; e1[axis] = 1; en[axis] = -1; en1[axis] = -2
    g[tuple(e0)] = (v[tuple(e1)] - v[tuple(e0)]) / dx
    g[tuple(en)] = (v[tuple(en)] - v[tuple(en1)]) / dx
    return g


def _second(v, axis, dx):
    s = np.zeros_like(v)
    lo = [slice(None)] * 3; hi = [slice(None)] * 3; md = [slice(None)] * 3
    lo[axis] = slice(2, None); hi[axis] = slice(0, -2); md[axis] = slice(1, -1)
    s[tuple(md)] = (v[tuple(lo)] - 2 * v[tuple(md)] + v[tuple(hi)]) / dx ** 2
    return s


def _cross(v, ax1, ax2, d1, d2):
    return _grad(_grad(v, ax1, d1), ax2, d2)


# ------------------------------------------------------------------------- interpolation
class FieldInterp:
    """Edge-clamped trilinear interpolation of a STACK of fields on (Zg,Yg,Sg); one RGI
    call evaluates all fields (v5's ctrl() pattern, vectorized across fields)."""

    def __init__(self, grids, fields):
        # fields: dict name -> (nZ,nY,nS) array
        self.names = list(fields.keys())
        vals = np.stack([fields[k] for k in self.names], axis=-1)
        self.lo = np.array([g[0] for g in grids]); self.hi = np.array([g[-1] for g in grids])
        self.f = RGI(tuple(grids), vals, method="linear", bounds_error=False, fill_value=None)

    def __call__(self, z, y, s):
        q = np.empty((z.size, 3))
        q[:, 0] = np.clip(z, self.lo[0], self.hi[0])
        q[:, 1] = np.clip(y, self.lo[1], self.hi[1])
        q[:, 2] = np.clip(s, self.lo[2], self.hi[2])
        out = self.f(q)
        return {k: out[:, i] for i, k in enumerate(self.names)}


# --------------------------------------------------------------------- shared FOC helper
def foc_controls(qd, qg, Z, p, qr=None, S=None):
    """Closed-form FOC control block for the shared zoo FOC family (see module docstring).

    qd, qg : costates multiplying phi_d'/phi_g' (floored at QFLOOR = feasibility K>=0).
    qr     : R&D costate (W_s); if None -> two-control economy (i_r absent). Floored at 0
             (no R&D when knowledge has non-positive marginal value).
    S      : s array, required when qr is given (the e^{-psi1 s} factor). psi1 must be 1/2
             (exact quadratic-in-c closed form); other psi1 would need a Newton loop.
    Returns dict(i_d, i_g, c[, i_r]).
    """
    qd = np.maximum(qd, QFLOOR); qg = np.maximum(qg, QFLOOR)
    B = (1 - Z) * (p["A_d"] + 1.0 / p["t_d"]) + Z * (p["A_g"] + 1.0 / p["t_g"])
    Q = ((1 - Z) * p["G_d"] * qd + Z * p["G_g"] * qg) / p["delta"]
    if qr is None:
        c = B / (1.0 + Q)
        out = {}
    else:
        assert abs(p["psi1"] - 0.5) < 1e-12, "closed form requires psi1 = 1/2"
        qr = np.maximum(qr, 0.0)
        kap = (p["psi0"] * p["psi1"]
               * np.exp(np.clip(-p["psi1"] * S, -EXP_CLIP, EXP_CLIP)) * qr / p["delta"]) ** 2
        disc = (1.0 + Q) ** 2 + 4.0 * kap * B
        c = np.where(kap > 1e-30,
                     (-(1.0 + Q) + np.sqrt(disc)) / np.maximum(2.0 * kap, 1e-30),
                     B / (1.0 + Q))
        out = {"i_r": kap * c ** 2}
    out["i_d"] = p["G_d"] * qd * c / p["delta"] - 1.0 / p["t_d"]
    out["i_g"] = p["G_g"] * qg * c / p["delta"] - 1.0 / p["t_g"]
    out["c"] = c
    return out


# ------------------------------------------------------------------- policy EVALUATION
def simulate_W(cfg, ctr_fields, W_lag=None, cost_fields=None, T=1200.0, dt=2.5):
    """PIBYS policy evaluation on the reduced (Z,Y,s) box: integrate W from EVERY node
    under the fixed control fields. RK2 characteristics; discount by the exact exponential
    integrating factor exp(-int (delta + sum_j J_j g_j) dt)  (semi-implicit jump sink);
    explicit jump gains from the post-jump interpolators; g from the LAGGED W."""
    p = cfg["params"]; Zg, Yg, Sg = cfg["grids"]
    zclip = cfg.get("z_clip", (1e-4, 1.0 - 1e-4))
    xi = cfg.get("xi", np.inf)
    neutral = (not np.isfinite(xi)) or xi >= NEUTRAL_XI
    jumps = cfg.get("jumps", [])
    ctrI = FieldInterp((Zg, Yg, Sg), {k: ctr_fields[k] for k in cfg["control_names"]})
    costI = None
    if cfg.get("needs_costates", False) and cost_fields is not None:
        costI = FieldInterp((Zg, Yg, Sg), cost_fields)
    WlagI = None
    if jumps and not neutral:
        Wl = W_lag if W_lag is not None else np.zeros((Zg.size, Yg.size, Sg.size))
        WlagI = FieldInterp((Zg, Yg, Sg), {"W": Wl})

    ZZ, YY, SS = np.meshgrid(Zg, Yg, Sg, indexing="ij")
    z = ZZ.ravel().copy(); y = YY.ravel().copy(); s = SS.ravel().copy()
    V = np.zeros_like(z); Lam = np.zeros_like(z)   # Lam = accumulated discount integral
    nstep = int(round(T / dt))
    for _ in range(nstep):
        ctr = ctrI(z, y, s)
        cost = costI(z, y, s) if costI is not None else None
        a_Z, a_Y, a_S = cfg["drift"](z, y, s, ctr, cost, p)
        flow = cfg["flow"](z, y, s, ctr, cost, p)
        sink = 0.0
        for jp in jumps:
            J = jp["intensity"](z, y, s, p)
            Wp = jp["w_post"](z, y, s)
            if neutral:
                # neutral limit: g = 1, bracket -> J (W' - W): gain J*W', sink J
                flow = flow + J * Wp
                sink = sink + J
            else:
                Wl = WlagI(z, y, s)["W"]
                g = np.exp(np.clip(-(Wp - Wl) / xi, -EXP_CLIP, EXP_CLIP))
                flow = flow + J * g * Wp + xi * J * (1.0 - g + g * np.log(np.maximum(g, 1e-300)))
                sink = sink + J * g
        if cfg.get("step_exact", False):
            lam_step = p["delta"] + sink
            wgt = -np.expm1(-np.clip(lam_step * dt, 0.0, 700.0)) / np.maximum(lam_step, 1e-300)
            V += np.exp(-np.clip(Lam, 0.0, 700.0)) * flow * wgt
        else:
            V += np.exp(-np.clip(Lam, 0.0, 700.0)) * flow * dt
        # RK2 midpoint state update (controls re-read at midpoint, as v5)
        zm = np.clip(z + 0.5 * dt * a_Z, *zclip); ym = y + 0.5 * dt * a_Y; sm = s + 0.5 * dt * a_S
        ctr_m = ctrI(zm, ym, sm)
        cost_m = costI(zm, ym, sm) if costI is not None else None
        a2_Z, a2_Y, a2_S = cfg["drift"](zm, ym, sm, ctr_m, cost_m, p)
        z = np.clip(z + dt * a2_Z, *zclip); y = y + dt * a2_Y; s = s + dt * a2_S
        Lam = Lam + (p["delta"] + sink) * dt
    return V.reshape(ZZ.shape)


# --------------------------------------------------------------------- residual monitor
def true_residual(cfg, W, ctr):
    """TRUE HJB residual at interior nodes (monitor only): central differences including
    the tiny diffusion terms (cfg['diff']) and the FULL nonlinear jump brackets with the
    CURRENT W. Returns the interior residual array."""
    p = cfg["params"]; Zg, Yg, Sg = cfg["grids"]
    dZ = Zg[1] - Zg[0]; dY = Yg[1] - Yg[0]; dS = Sg[1] - Sg[0]
    ZZ, YY, SS = np.meshgrid(Zg, Yg, Sg, indexing="ij")
    zf, yf, sf = ZZ.ravel(), YY.ravel(), SS.ravel()
    ctr_flat = {k: v.ravel() for k, v in ctr.items()}
    W_Z = _grad(W, 0, dZ); W_Y = _grad(W, 1, dY); W_S = _grad(W, 2, dS)
    cost_flat = {"W_Z": W_Z.ravel(), "W_Y": W_Y.ravel(), "W_S": W_S.ravel()} \
        if cfg.get("needs_costates", False) else None
    a_Z, a_Y, a_S = cfg["drift"](zf, yf, sf, ctr_flat, cost_flat, p)
    flow = cfg["flow"](zf, yf, sf, ctr_flat, cost_flat, p)
    R = (flow.reshape(W.shape) - p["delta"] * W
         + a_Z.reshape(W.shape) * W_Z + a_Y.reshape(W.shape) * W_Y + a_S.reshape(W.shape) * W_S)
    if "diff" in cfg:
        D = cfg["diff"](ZZ, YY, SS, ctr, p)
        sec = {"ZZ": (0, dZ), "YY": (1, dY), "SS": (2, dS)}
        for k, (ax, dx) in sec.items():
            if k in D:
                R += D[k] * _second(W, ax, dx)
        crx = {"SZ": (2, 0, dS, dZ), "ZY": (0, 1, dZ, dY), "SY": (2, 1, dS, dY)}
        for k, (a1, a2, d1, d2) in crx.items():
            if k in D:
                R += D[k] * _cross(W, a1, a2, d1, d2)
    xi = cfg.get("xi", np.inf)
    neutral = (not np.isfinite(xi)) or xi >= NEUTRAL_XI
    for jp in cfg.get("jumps", []):
        J = jp["intensity"](zf, yf, sf, p).reshape(W.shape)
        Wp = jp["w_post"](zf, yf, sf).reshape(W.shape)
        if neutral:
            R += J * (Wp - W)
        else:
            g = np.exp(np.clip(-(Wp - W) / xi, -EXP_CLIP, EXP_CLIP))
            R += J * g * (Wp - W) + xi * J * (1.0 - g + g * np.log(np.maximum(g, 1e-300)))
    return R[1:-1, 1:-1, 1:-1]


# ------------------------------------------------------------------------- Howard loop
def _default_box(Zg, Yg, Sg):
    def trim(g):
        span = g[-1] - g[0]
        return (g[0] + 0.1 * span, g[-1] - 0.1 * span)
    return (trim(Zg), trim(Yg), trim(Sg))


def _box_ix(Zg, Yg, Sg, box):
    (z0, z1), (y0, y1), (s0, s1) = box
    return np.ix_((Zg >= z0) & (Zg <= z1), (Yg >= y0) & (Yg <= y1), (Sg >= s0) & (Sg <= s1))


def solve_reduced(cfg, T=1200.0, dt=2.5, howard_max=200, tol_pocket=1e-6, relax=0.3,
                  tail_avg=6, init_ctr=None, verbose=True, ref_pt=None):
    """Howard outer loop (v5_stable pattern) on the reduced (Z,Y,s) box. See module doc."""
    p = cfg["params"]; Zg, Yg, Sg = cfg["grids"]
    dZ = Zg[1] - Zg[0]; dY = Yg[1] - Yg[0]; dS = Sg[1] - Sg[0]
    ZZ, YY, SS = np.meshgrid(Zg, Yg, Sg, indexing="ij")
    names = cfg["control_names"]
    box = _box_ix(Zg, Yg, Sg, cfg.get("box") or _default_box(Zg, Yg, Sg))

    if init_ctr is not None:
        ctr = {k: np.array(init_ctr[k], dtype=float) for k in names}
    else:
        ctr = {k: np.array(v, dtype=float) for k, v in cfg["init_controls"](ZZ, YY, SS, p).items()
               if k in names}
    W_prev = cfg.get("W_init")
    cost_prev = None
    if ref_pt is not None:
        iz = int(np.argmin(np.abs(Zg - ref_pt[0]))); iy = int(np.argmin(np.abs(Yg - ref_pt[1])))
        isx = int(np.argmin(np.abs(Sg - ref_pt[2])))

    t0 = time.time(); hist = {k: [] for k in names}
    di_int = np.inf; it = 0
    for it in range(howard_max):
        W = simulate_W(cfg, ctr, W_lag=W_prev, cost_fields=cost_prev, T=T, dt=dt)      # EVALUATE
        W_Z = _grad(W, 0, dZ); W_Y = _grad(W, 1, dY); W_S = _grad(W, 2, dS)            # IMPROVE
        new = cfg["controls"](W_Z, W_Y, W_S, ZZ, YY, SS, p)
        di_full = max(np.max(np.abs(new[k] - ctr[k])) for k in names)
        di_int = max(np.max(np.abs((new[k] - ctr[k])[box])) for k in names)
        for k in names:
            rk = relax[k] if isinstance(relax, dict) else relax   # per-control damping
            ctr[k] = (1 - rk) * ctr[k] + rk * new[k]
            hist[k].append(ctr[k].copy())
            if len(hist[k]) > tail_avg:
                hist[k].pop(0)
        W_prev = W
        cost_prev = {"W_Z": W_Z, "W_Y": W_Y, "W_S": W_S}
        if verbose and (it % 5 == 0 or it < 5):
            msg = f"  [howard {it:3d}] di_full={di_full:.2e} di_int={di_int:.2e}"
            if ref_pt is not None:
                msg += " |" + "".join(f" {k}={ctr[k][iz, iy, isx]:+.5f}" for k in names)
            print(msg, flush=True)
        if di_int < tol_pocket and it > tail_avg:
            break

    # tail-average the policy, then ONE final mutual-consistency gate
    ctr = {k: np.mean(hist[k], axis=0) for k in names}
    W = simulate_W(cfg, ctr, W_lag=W_prev, cost_fields=cost_prev, T=T, dt=dt)
    W_Z = _grad(W, 0, dZ); W_Y = _grad(W, 1, dY); W_S = _grad(W, 2, dS)
    gate = cfg["controls"](W_Z, W_Y, W_S, ZZ, YY, SS, p)
    gate_int = max(np.max(np.abs((gate[k] - ctr[k])[box])) for k in names)
    ctr = {k: gate[k] for k in gate}          # adopt gate-consistent controls (incl. c)
    R = true_residual(cfg, W, {k: v for k, v in ctr.items()})
    out = dict(Z=Zg, Y=Yg, S=Sg, W=W, W_Z=W_Z, W_Y=W_Y, W_S=W_S,
               iters=it + 1, time=time.time() - t0, di_int_final=float(di_int),
               gate_int=float(gate_int), max_abs_residual=float(np.max(np.abs(R))),
               rms_residual=float(np.sqrt(np.mean(R ** 2))))
    out.update(ctr)
    return out


# =============================================================================
# VALIDATION: degenerate embedding of the verified post-damage post-tech solve.
# s == logK exactly reproduces fd_pdpt_v5_stable's (logK, Z, Y) problem inside the
# reduced (Z, Y, s) harness: same drift, same flow (incl. the Y_CAP damage-slope
# saturation and LK_CAP emissions cap), same FOC block, no jumps.
# =============================================================================
def _degenerate_cfg(lam3=1.0 / 6.0):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "post_damage_post_tech")))
    import fd_pdpt_v5 as FDV5   # read-only reuse: byte-identical params/constants
    p = dict(FDV5.P)
    p.update(delta=p["delta"], A_g=p["A_gpp"], lam3=lam3,
             LK_CAP=FDV5.LK_CAP, Y_CAP=FDV5.Y_CAP)

    def _phis(ctr, p):
        phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * ctr["i_d"], 1e-9))
        phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * ctr["i_g"], 1e-9))
        return phid, phig

    def drift(z, y, s, ctr, cost, p):
        phid, phig = _phis(ctr, p)
        Dc = p["s_d"] ** 2 * (1 - z) ** 2 + p["s_g"] ** 2 * z ** 2
        a_S = (1 - z) * phid + z * phig - Dc / 2.0            # a_S = a_logK
        a_Z = z * (1 - z) * (phig - phid + (1 - z) * p["s_d"] ** 2 - z * p["s_g"] ** 2)
        E = p["eta"] * p["A_d"] * (1 - z) * np.exp(np.minimum(s, p["LK_CAP"]))
        a_Y = p["thbar"] * E
        return a_Z, a_Y, a_S

    def flow(z, y, s, ctr, cost, p):
        c = (p["A_d"] - ctr["i_d"]) * (1 - z) + (p["A_g"] - ctr["i_g"]) * z
        E = p["eta"] * p["A_d"] * (1 - z) * np.exp(np.minimum(s, p["LK_CAP"]))
        a_Y = p["thbar"] * E
        y_eff = np.minimum(y, p["Y_CAP"])
        lNy = p["l1"] + p["l2"] * y_eff + p["lam3"] * (y_eff - p["y_up"])
        lNyy = p["l2"] + p["lam3"]
        b_Y = 0.5 * p["vars"] ** 2 * E ** 2
        return p["delta"] * (np.log(np.maximum(c, 1e-12)) + s) - (lNy * a_Y + lNyy * b_Y)

    def controls(W_Z, W_Y, W_S, Z, Y, S, p):
        qd = W_S - Z * W_Z            # V_logK - Z V_Z with V = W, logK = s
        qg = W_S + (1 - Z) * W_Z
        return foc_controls(qd, qg, Z, p)   # two-control closed form (== FDV5.controls)

    def diff(Z, Y, S, ctr, p):
        sd2, sg2 = p["s_d"] ** 2, p["s_g"] ** 2
        Dc = sd2 * (1 - Z) ** 2 + sg2 * Z ** 2
        E = p["eta"] * p["A_d"] * (1 - Z) * np.exp(np.minimum(S, p["LK_CAP"]))
        return {"SS": Dc / 2.0,
                "ZZ": 0.5 * Z ** 2 * (1 - Z) ** 2 * (sd2 + sg2),
                "YY": 0.5 * p["vars"] ** 2 * E ** 2}

    def init_controls(Z, Y, S, p):
        return {"i_d": np.zeros_like(Z), "i_g": np.full_like(Z, 0.05)}

    Zg = np.linspace(0.02, 0.98, 61)
    Yg = np.linspace(0.0, 4.0, 31)
    Sg = np.linspace(4.0, 7.0, 31)     # s == logK grid of the reference solve
    return dict(name="DEGENERATE-PDPT(s=logK)", params=p, grids=(Zg, Yg, Sg),
                controls=controls, control_names=["i_d", "i_g"], drift=drift, flow=flow,
                diff=diff, init_controls=init_controls, xi=np.inf, jumps=[],
                box=((0.3, 0.95), (0.5, 4.0), (4.3, 6.7)))


def validate(warm=True, howard_max=40, quick=False):
    ref_file = os.path.abspath(os.path.join(
        HERE, "..", "..", "post_damage_post_tech", "outputs",
        "fd_pdpt_v5_stable_lam3_0167_xi148.npz"))
    ref = np.load(ref_file)
    cfg = _degenerate_cfg(lam3=1.0 / 6.0)
    if quick:  # fallback grid if the full grid exceeds the login-node budget
        cfg["grids"] = (np.linspace(0.02, 0.98, 41), np.linspace(0.0, 4.0, 21),
                        np.linspace(4.0, 7.0, 25))
    Zg, Yg, Sg = cfg["grids"]

    init_ctr = None
    if warm:
        # warm-start from the reference's own converged controls: (lk,z,y) -> (z,y,s)
        def to_zys(a):
            return np.ascontiguousarray(np.transpose(a, (1, 2, 0)))
        if quick:
            gi = FieldInterp((ref["Z"], ref["Y"], ref["logK"]),
                             {"i_d": to_zys(ref["i_d"]), "i_g": to_zys(ref["i_g"])})
            ZZ, YY, SS = np.meshgrid(Zg, Yg, Sg, indexing="ij")
            q = gi(ZZ.ravel(), YY.ravel(), SS.ravel())
            init_ctr = {k: q[k].reshape(ZZ.shape) for k in ("i_d", "i_g")}
        else:
            init_ctr = {"i_d": to_zys(ref["i_d"]), "i_g": to_zys(ref["i_g"])}

    print(f"[fd_reduced validate] degenerate s=logK vs {os.path.basename(ref_file)} "
          f"grid=({Zg.size},{Yg.size},{Sg.size}) warm={warm}", flush=True)
    out = solve_reduced(cfg, T=1200.0, dt=2.5, howard_max=howard_max, tol_pocket=1e-6,
                        relax=0.3, tail_avg=6, init_ctr=init_ctr, verbose=True,
                        ref_pt=(0.7, 3.0, np.log(880.0)))

    # compare on the shared grid (full grid if not quick; else interpolate reference)
    def ref_on_mine(name):
        gi = FieldInterp((ref["Z"], ref["Y"], ref["logK"]),
                         {name: np.ascontiguousarray(np.transpose(ref[name], (1, 2, 0)))})
        ZZ, YY, SS = np.meshgrid(Zg, Yg, Sg, indexing="ij")
        return gi(ZZ.ravel(), YY.ravel(), SS.ravel())[name].reshape(ZZ.shape)

    box = _box_ix(Zg, Yg, Sg, cfg["box"])
    stats = {}
    for mine, theirs in (("W", "v"), ("i_d", "i_d"), ("i_g", "i_g")):
        rf = (np.transpose(ref[theirs], (1, 2, 0)) if not quick else ref_on_mine(theirs))
        d = out[mine] - rf
        stats[mine] = dict(rms=float(np.sqrt(np.mean(d ** 2))),
                           rms_box=float(np.sqrt(np.mean(d[box] ** 2))),
                           max_abs=float(np.max(np.abs(d))))
    gate_pass = stats["W"]["rms"] <= 5e-3
    print("\n[validate] field discrepancy vs fd_pdpt_v5_stable (transposed):")
    for k, s in stats.items():
        print(f"  {k:4s}: RMS={s['rms']:.3e}  RMS(box)={s['rms_box']:.3e}  max={s['max_abs']:.3e}")
    print(f"  harness: iters={out['iters']} gate_int={out['gate_int']:.2e} "
          f"maxR={out['max_abs_residual']:.2e} rmsR={out['rms_residual']:.2e} "
          f"time={out['time']:.0f}s")
    print(f"  GATE W-RMS <= 5e-3: {'PASS' if gate_pass else 'FAIL'}", flush=True)

    os.makedirs(OUTD, exist_ok=True)
    np.savez_compressed(os.path.join(OUTD, "harness_validation_degenerate.npz"),
                        Z=Zg, Y=Yg, S=Sg, W=out["W"], i_d=out["i_d"], i_g=out["i_g"],
                        c=out["c"], W_Z=out["W_Z"], W_Y=out["W_Y"], W_S=out["W_S"])
    prov = dict(
        economy="HARNESS-VALIDATION (degenerate s=logK embedding of post-damage post-tech)",
        harness="benchmarks/economy_zoo/solvers/fd_reduced.py",
        reference=ref_file, lam3=1.0 / 6.0, xi="neutral (148.4-equivalent; no jumps)",
        grid=[int(Zg.size), int(Yg.size), int(Sg.size)], warm_start=bool(warm),
        T=1200.0, dt=2.5, relax=0.3, tail_avg=6, tol_pocket=1e-6,
        gates=dict(W_rms_le_5e_3=dict(value=stats["W"]["rms"], target=5e-3,
                                      passed=bool(gate_pass)),
                   mutual_consistency_gate_int=out["gate_int"],
                   true_residual_max=out["max_abs_residual"],
                   true_residual_rms=out["rms_residual"]),
        field_discrepancy=stats, iters=out["iters"], seconds=out["time"],
        date="2026-07-19")
    with open(os.path.join(OUTD, "harness_validation_PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        iy = int(np.argmin(np.abs(Yg - 3.0)))
        rfv = (np.transpose(ref["v"], (1, 2, 0)) if not quick else ref_on_mine("v"))
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for a, fld, ttl in ((ax[0], out["W"][:, iy, :], "harness W"),
                            (ax[1], rfv[:, iy, :], "reference v"),
                            (ax[2], (out["W"] - rfv)[:, iy, :], "difference")):
            im = a.pcolormesh(Sg, Zg, fld, shading="auto")
            a.set_xlabel("s = logK"); a.set_ylabel("Z"); a.set_title(f"{ttl}  (Y=3.0)")
            fig.colorbar(im, ax=a)
        fig.suptitle("fd_reduced degenerate validation vs fd_pdpt_v5_stable (lam3=1/6, xi=148.4)")
        fig.tight_layout()
        os.makedirs(FIGD, exist_ok=True)
        fig.savefig(os.path.join(FIGD, "harness_validation_degenerate.png"), dpi=120)
        print(f"  figure -> {os.path.join(FIGD, 'harness_validation_degenerate.png')}")
    except Exception as e:   # figure is best-effort; validation numbers already reported
        print(f"  [warn] figure skipped: {e}")
    return stats, out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    cold = "--cold" in sys.argv
    validate(warm=not cold, howard_max=(60 if cold else 40), quick=quick)
