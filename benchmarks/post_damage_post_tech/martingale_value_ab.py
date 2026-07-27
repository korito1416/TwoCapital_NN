"""
martingale_value_ab.py -- MARTINGALE-REWRITE solver for the 3-D post-damage-post-tech HJB.

PARADIGM (model-INDEPENDENT invariant): the SDF/Sharpe loss is finance-specific, but the
MARTINGALE is universal across macro + finance + robust control. We REWRITE the HJB as the
statement that a value process has ZERO drift, and train on the MARTINGALE-DIFFERENCE (a
conditional-moment / integral loss) INSTEAD of the strong PDE residual.

Grounded EXACTLY in models/PostDamagePostTech.py::pde_rhs:
  state X=(logK,Z,Y); flow = delta*(log c + logK); discount delta; FOC controls i_d,i_g; the
  closed-form worst-case drift distortions h_d,h_g,h_y; the robustness drag +0.5*xi*(h_d^2+h_g^2+h_y^2)
  (note: with h*=-(1/xi)*coeff this equals -0.5/xi*sum(coeff^2), the Hansen-Sargent penalty).

================================================================================================
M1  VALUE-PROCESS MARTINGALE-DIFFERENCE LOSS  (the PRIMARY loss)
------------------------------------------------------------------------------------------------
Under optimal a*=(i_d,i_g) and worst-case h*, the process
    M_t = exp(-delta t) V(X_t) + int_0^t exp(-delta s) flow(X_s) ds
is a MARTINGALE: dM = exp(-delta t)[ -delta V + flow + L^{a*,h*} V ] dt + (mart. incr.) and the
bracket is EXACTLY pde_rhs.rhs - pde_rhs.pv = 0 (the HJB). So the HJB <=> E[dM | F_t] = 0.

We do NOT form rhs (which needs ALL the 2nd derivatives d2v_dlogK2, d2v_dZ2, d2v_dY2, d2v_dlogKdZ
and the explicit Ito drift coefficients -- the strong residual). INSTEAD, over a short horizon
[t, t+dt] we discretize the martingale increment along a SIMULATED path X_{t+dt}=X_t + drift*dt
(+ diffusion), and minimize the SQUARED conditional-moment

    L_M1 = E[ ( m_{t+dt} )^2 ],   m_{t+dt} := exp(-delta dt) V(X_{t+dt}) - V(X_t) + flow(X_t) dt

This is the deep-BSDE / semigroup-Bellman value identity  V(x) = flow*dt + exp(-delta dt) E[V(X_dt)]
(which CONVERGED before) made the PRIMARY training target. It uses only V and its FIRST derivatives
(through the drift / FOC), NO hand-coded 2nd-derivative residual.

Two variants of the conditional expectation (both implemented):
  - "antithetic": pair +/- diffusion shocks so the average over the pair is the conditional MEAN of
    V(X_{t+dt}) to O(dt) (kills the leading martingale-increment variance; the residual is the drift).
    This is the recommended primary -- it is the MINIMUM-VARIANCE estimator of the conditional moment.
  - "single": one shock per node (higher variance; only for an ablation).

M2  WORST-CASE-MEASURE PATH SIMULATION (Hansen-Sargent change of measure)
------------------------------------------------------------------------------------------------
The worst-case h* defines the Doleans-Dade martingale Z_t=exp(int h* dW - 0.5 int|h*|^2 ds); the
martingale M1 holds under the h*-DISTORTED measure, i.e. with the Brownian drift shifted by h*.
h* is CLOSED-FORM from the costate (pde_rhs lines 192-197):
    h_d = -(1/xi)(v_logK - Z v_Z)(1-Z) sigma_d
    h_g = -(1/xi)(v_logK + (1-Z) v_Z) Z sigma_g
    h_y = -(1/xi)(v_Y - (l1 + l2 Y + lam3 (Y-y_up))) * eta A_d (1-Z) K * varsigma
We SIMULATE X_{t+dt} with the drift DISTORTED by these h* (each diffusion channel gets +sigma*h*),
so the path is the worst-case path. NO separate distortion network -- h* is a plug-in (the max-min
inner problem is solved in closed form; confirmed no redundant saddle).

M3  FOC / SDF PRICING CONTROLS
------------------------------------------------------------------------------------------------
Controls come from the FOC (pde_rhs lines 252-253), NOT from sampling the costate. The SDF is
S_t=exp(-delta t) delta/c_t and the FOC is the pricing/Euler condition: the shadow price
qd=v_logK - Z v_Z prices dirty capital, qg=v_logK+(1-Z)v_Z prices green. We invert the EXACT,
FD-VALIDATED closed form FD.controls (the same map the FD ground truth uses), differentiable
through the autodiff costate, so ctrlfit supervision flows. This AVOIDS the failed costate-sampling
(which floored at the SNR limit).

================================================================================================
NUMERICAL-STABILITY FIXES (built in, orthogonal/complementary):
 (1) NON-DIMENSIONALIZE: the M1 increment is divided by its characteristic scale
     (flow*dt magnitude) -> a RELATIVE conditional-moment residual, naturally O(1).
 (2) exp SOFT-CLIP / log-sum-exp pattern kept for the jump term (this 3-D stepping stone is
     jump-free, so it is a no-op here; the helper softclip_exp is provided for the jump regimes).
 (3) C>0 HARD constraint: consumption via the FD.controls closed form is c=num/den with den>=delta>0
     (feasibility floor on the costates), so c>0 by construction; we additionally softplus-offset
     the log argument.  Bounded investment 1+theta*i>0 is enforced by the costate floor (qd,qg>=QFLOOR).
 (4) ADAPTIVE loss weighting + xi-CURRICULUM/HOMOTOPY: train large xi (~risk-neutral, h*~0, nearly
     linear) FIRST, then ANNEAL xi down toward the target (Azinovic homotopy). The h* drag is
     continuously turned on as xi shrinks.
 (5) max-min closed-form plug-in h* (no separate net).
 (6) PER-REGION residual monitor (box vs de-invest corner) every sweep.

GATE (the ONLY thing that counts): TRUE control+costate error vs the high-accuracy FD
(stable_fd_eval.grade -> box_err_i_d/i_g, de-invest vZ). Multi-seed >=3. We NEVER compare the
martingale loss number to the strong-residual loss (different scale) -- only TRUE error vs FD, plus
training STABILITY (no blowup, convergence speed) vs the strong-residual incumbent.

RUN:
  cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
  module load python/anaconda-2021.05
  srun --account=pi-lhansen --partition=caslake --time=1:00:00 --cpus-per-task=4 --mem=24G \
       python martingale_value_ab.py --seeds 1 2 3 --worstcase on
"""
import os, sys, time, argparse, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
torch.set_num_threads(4)   # avoid the 32-thread oversubscription slowdown on small autodiff graphs
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd_pdpt_v5 as FD                       # simulate_v, controls, P, QFLOOR, Y_CAP, _drift
P = FD.P
QFLOOR = FD.QFLOOR
LK_CAP = FD.LK_CAP
Y_CAP = FD.Y_CAP

# grading: prefer the HIGH-ACCURACY ref, fall back to the stable FD npz (same grade() API)
import stable_fd_eval as SFE

# ----- box (must match the FD grids exactly) -----
LK_LO, LK_HI = 4.0, 7.0
Z_LO,  Z_HI  = 0.02, 0.98
Y_LO,  Y_HI  = 0.0, 4.0
LAM3 = 1.0 / 6.0
XI_TARGET = 148.4


# =====================================================================================
#  (stability fix 2) exp soft-clip / log-sum-exp for the jump term g=exp(-(1/xi)(V^l - V)).
#  3-D post-tech is jump-FREE so this is a no-op here, but keep the pattern for the jump regimes.
# =====================================================================================
def softclip_exp(x, lo=-40.0, hi=40.0):
    return torch.exp(torch.clamp(x, lo, hi))


# =====================================================================================
#  Differentiable FOC inversion = M3.  (torch transcription of FD.controls; smooth costate floor.)
#  c>0 by construction (den>=delta>0); softplus floor keeps the costate gradient alive (stability 3).
# =====================================================================================
def controls_torch(qd, qg, Z, soft=True):
    if soft:
        qd = QFLOOR + torch.nn.functional.softplus(qd - QFLOOR, beta=50.0)
        qg = QFLOOR + torch.nn.functional.softplus(qg - QFLOOR, beta=50.0)
    else:
        qd = torch.clamp(qd, min=QFLOOR); qg = torch.clamp(qg, min=QFLOOR)
    Abar = (1 - Z) * P["A_d"] + Z * P["A_gpp"]
    num = P["delta"] * (Abar + (1 - Z) / P["t_d"] + Z / P["t_g"])
    den = P["delta"] + (1 - Z) * P["G_d"] * qd + Z * P["G_g"] * qg
    c = num / den
    i_d = P["G_d"] * qd * c / P["delta"] - 1.0 / P["t_d"]
    i_g = P["G_g"] * qg * c / P["delta"] - 1.0 / P["t_g"]
    return i_d, i_g, c


# =====================================================================================
#  Drift + diffusion of X=(logK,Z,Y) under controls (a*) and worst-case distortion (h*).
#  EXACTLY the coefficients in pde_rhs / FD._drift; h* are the closed-form distortions.
#  Returns the Ito drift (a_lK,a_Z,a_Y) and the per-channel diffusion loadings so M2 can simulate.
# =====================================================================================
def drift_diffusion_torch(logK, Z, Y, i_d, i_g, vlK, vZ, vY, xi, worstcase=True):
    sd, sg = P["s_d"], P["s_g"]
    K = torch.exp(torch.clamp(logK, max=LK_CAP))
    phid = P["a_d"] + P["G_d"] * torch.log(torch.clamp(1 + P["t_d"] * i_d, min=1e-9))
    phig = P["a_g"] + P["G_g"] * torch.log(torch.clamp(1 + P["t_g"] * i_g, min=1e-9))
    Dc = sd ** 2 * (1 - Z) ** 2 + sg ** 2 * Z ** 2

    # ---- baseline Ito drifts (pde_rhs v_logK_term / v_Z_term, FD._drift) ----
    a_lK = (1 - Z) * phid + Z * phig - Dc / 2.0
    a_Z = Z * (1 - Z) * (phig - phid + (1 - Z) * sd ** 2 - Z * sg ** 2)
    E = P["eta"] * P["A_d"] * (1 - Z) * K
    a_Y = P["thbar"] * E

    # ---- diffusion loadings (the dW coefficients of each state).  From the SDE for logK,Z:
    #   d logK = ... + (1-Z) sigma_d dW_d + Z sigma_g dW_g ;  the Z drift's stochastic part loads
    #   -Z(1-Z)sigma_d dW_d + Z(1-Z)sigma_g dW_g ; Y diffusion loads varsigma*E dW_y. ----
    bK_d = (1 - Z) * sd;      bK_g = Z * sg
    bZ_d = -Z * (1 - Z) * sd; bZ_g = Z * (1 - Z) * sg
    bY_y = P["vars"] * E

    # ---- (M2) worst-case drift distortion h* (closed form; pde_rhs 192-197).  Each Brownian
    #   channel's drift shifts by +sigma*h* (Girsanov).  h* = -(1/xi)*coeff. ----
    if worstcase:
        qd = vlK - Z * vZ
        qg = vlK + (1 - Z) * vZ
        lNy = P["l1"] + P["l2"] * Y + LAM3 * (Y - P["y_up"])
        h_d = -(1.0 / xi) * (qd * (1 - Z) * sd)
        h_g = -(1.0 / xi) * (qg * Z * sg)
        h_y = -(1.0 / xi) * (vY - lNy) * (P["eta"] * P["A_d"] * (1 - Z) * K) * P["vars"]
        # distorted drift: a + (diffusion loading) * h  (the +h_d*... terms in pde_rhs rhs line 240)
        a_lK = a_lK + bK_d * h_d + bK_g * h_g
        a_Z = a_Z + bZ_d * h_d + bZ_g * h_g
        a_Y = a_Y + bY_y * h_y
        hmag2 = h_d ** 2 + h_g ** 2 + h_y ** 2
    else:
        hmag2 = torch.zeros_like(a_lK)

    return a_lK, a_Z, a_Y, (bK_d, bK_g, bZ_d, bZ_g, bY_y), hmag2


def clim_torch(Y, a_Y, E, lam3=LAM3, y_cap=Y_CAP):
    """The -logN damage running cost subtracted from flow (EXACTLY simulate_v's clim)."""
    y_eff = torch.clamp(Y, max=y_cap)
    lNy = P["l1"] + P["l2"] * y_eff + lam3 * (y_eff - P["y_up"])
    lNyy = P["l2"] + lam3
    b_Y = 0.5 * P["vars"] ** 2 * E ** 2
    return lNy * a_Y + lNyy * b_Y


# =====================================================================================
#  Value net (ported from the harness: soft anchor + leading homogeneity slope; tanh-MLP head).
# =====================================================================================
class ValueNet3D(nn.Module):
    def __init__(self, width=64, depth=4, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("v_mean", torch.tensor(4.0, dtype=dtype))
        self.register_buffer("v_scale", torch.tensor(1.0, dtype=dtype))
        self.register_buffer("logK_slope", torch.tensor(0.5, dtype=dtype))
        self.register_buffer("logK_mid", torch.tensor(0.5 * (LK_LO + LK_HI), dtype=dtype))
        self.to(dtype)

    def calibrate(self, v_ws):
        self.v_mean.fill_(float(np.mean(v_ws)))
        self.v_scale.fill_(float(max(np.std(v_ws), 0.5)))

    def forward(self, logK, Z, Y):
        sK = (logK - LK_LO) / (LK_HI - LK_LO)
        sZ = (Z - Z_LO) / (Z_HI - Z_LO)
        sY = (Y - Y_LO) / (Y_HI - Y_LO)
        X = torch.cat([2 * sK - 1, 2 * sZ - 1, 2 * sY - 1], dim=1)
        base = self.v_mean + self.logK_slope * (logK - self.logK_mid)
        return base + self.v_scale * self.net(X)


def autodiff_slopes(model, logK, Z, Y):
    v = model(logK, Z, Y)
    g = torch.autograd.grad(v.sum(), [logK, Z, Y], create_graph=True)
    return v, g[0], g[1], g[2]


# =====================================================================================
#  M1 martingale-difference loss along a one-step simulated path under (a*, worst-case h*).
# =====================================================================================
def m1_martingale_loss(model, lk, z, y, xi, dt, worstcase=True, antithetic=True,
                       nondim=True, sto=True):
    """One-step value-process martingale-difference loss.

    m = exp(-delta dt) V(X_dt) - V(X_t) + flow(X_t) dt, with X_dt simulated under the optimal FOC
    controls and the worst-case distorted drift; diffusion shocks (antithetic) supply the conditional
    expectation. Returns (loss, controls i_d,i_g, costates) so ctrlfit + monitoring reuse them.
    """
    lk = lk.clone().requires_grad_(True)
    z = z.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    v, vlK, vZ, vY = autodiff_slopes(model, lk, z, y)

    # M3: FOC controls from the costates (differentiable)
    qd = vlK - z * vZ
    qg = vlK + (1 - z) * vZ
    i_d, i_g, c = controls_torch(qd, qg, z)

    # drift / diffusion / worst-case distortion at X_t
    a_lK, a_Z, a_Y, (bK_d, bK_g, bZ_d, bZ_g, bY_y), hmag2 = drift_diffusion_torch(
        lk, z, y, i_d, i_g, vlK, vZ, vY, xi, worstcase=worstcase)
    K = torch.exp(torch.clamp(lk, max=LK_CAP))
    E = P["eta"] * P["A_d"] * (1 - z) * K

    # flow = delta*(log c + logK) - clim  (pde_rhs flow minus the -logN damage; c>0 by construction)
    flow = P["delta"] * (torch.log(torch.clamp(c, min=1e-12)) + lk) - clim_torch(y, a_Y, E)

    disc = float(np.exp(-P["delta"] * dt))

    def step_and_resid(eps_d, eps_g, eps_y):
        # Euler-Maruyama one step (drift already worst-case-distorted); clamp Z to (0,1).
        lk2 = lk + a_lK * dt + (bK_d * eps_d + bK_g * eps_g) * np.sqrt(dt) if sto \
            else lk + a_lK * dt
        z2 = z + a_Z * dt + (bZ_d * eps_d + bZ_g * eps_g) * np.sqrt(dt) if sto \
            else z + a_Z * dt
        y2 = y + a_Y * dt + (bY_y * eps_y) * np.sqrt(dt) if sto else y + a_Y * dt
        z2 = torch.clamp(z2, 1e-4, 1 - 1e-4)
        v2 = model(lk2, z2, y2)
        return disc * v2 - v + flow * dt

    if sto and antithetic:
        eps_d = torch.randn_like(lk); eps_g = torch.randn_like(lk); eps_y = torch.randn_like(lk)
        m_p = step_and_resid(eps_d, eps_g, eps_y)
        m_m = step_and_resid(-eps_d, -eps_g, -eps_y)
        m = 0.5 * (m_p + m_m)          # antithetic-averaged conditional moment (min-variance, O(dt))
    elif sto:
        eps_d = torch.randn_like(lk); eps_g = torch.randn_like(lk); eps_y = torch.randn_like(lk)
        m = step_and_resid(eps_d, eps_g, eps_y)
    else:
        m = step_and_resid(None, None, None)   # deterministic-drift increment (sigma~0.01 => O(1e-4))

    # (stability 1) NON-DIMENSIONALIZE: divide the increment by its characteristic scale so the
    # conditional-moment residual is RELATIVE and O(1) across the box (flow*dt is the natural scale).
    if nondim:
        scale = torch.clamp(torch.abs(flow) * dt + torch.abs(v) * (1 - disc), min=1e-6).detach()
        m = m / scale

    loss = (m ** 2).mean()
    return loss, i_d, i_g, c, qd, qg, vlK, vZ, vY, hmag2


# =====================================================================================
#  Grid + harness plumbing
# =====================================================================================
def make_grid(nK, nZ, nY, dtype):
    logK = np.linspace(LK_LO, LK_HI, nK)
    Z = np.linspace(Z_LO, Z_HI, nZ)
    Y = np.linspace(Y_LO, Y_HI, nY)
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    tt = dict(dtype=dtype)
    return (logK, Z, Y, (LK, ZZ, YY),
            (torch.tensor(LK.ravel().reshape(-1, 1), **tt),
             torch.tensor(ZZ.ravel().reshape(-1, 1), **tt),
             torch.tensor(YY.ravel().reshape(-1, 1), **tt)))


def fields_on_grid(model, lk_t, z_t, y_t, nK, nZ, nY, ZZ):
    lk = lk_t.clone().requires_grad_(True)
    z = z_t.clone().requires_grad_(True)
    y = y_t.clone().requires_grad_(True)
    v, vlK, vZ, vY = autodiff_slopes(model, lk, z, y)
    vlK_np = vlK.detach().numpy().reshape(nK, nZ, nY)
    vZ_np = vZ.detach().numpy().reshape(nK, nZ, nY)
    v_np = v.detach().numpy().reshape(nK, nZ, nY)
    qd = vlK_np - ZZ * vZ_np; qg = vlK_np + (1 - ZZ) * vZ_np
    i_d, i_g, c = FD.controls(qd, qg, ZZ)
    return dict(v=v_np, vlK=vlK_np, vZ=vZ_np, i_d=i_d, i_g=i_g, c=c)


def xi_schedule(sweep, n_sweeps, xi_start, xi_target):
    """(stability 4) xi-CURRICULUM / HOMOTOPY: start at a LARGE xi (risk-neutral, h*~0, near-linear),
    geometrically anneal DOWN to xi_target over the first ~60% of sweeps, then hold."""
    frac = min(1.0, sweep / max(1.0, 0.6 * n_sweeps))
    log_xi = (1 - frac) * np.log(xi_start) + frac * np.log(xi_target)
    return float(np.exp(log_xi))


# =====================================================================================
#  Trainer: M1 martingale-difference primary loss + (optional) ctrlfit value anchor to the PIBYS
#  oracle (the discrete M1 fixed point), with the xi-homotopy + per-region monitor.
# =====================================================================================
def run_martingale(seed, nK=21, nZ=31, nY=21, n_sweeps=12, steps=2500, lr=2e-3,
                   dt=0.25, worstcase=True, antithetic=True, sto=True, nondim=True,
                   anchor=True, lam_anchor=1.0, xi_start=1e4, xi_target=XI_TARGET,
                   dtype=torch.float32, verbose=True, d_fd=None):
    torch.manual_seed(seed); np.random.seed(seed)
    logK, Z, Y, (LK, ZZ, YY), (lk_t, z_t, y_t) = make_grid(nK, nZ, nY, dtype)
    model = ValueNet3D(dtype=dtype)

    # ---- warm start: cold policy -> PIBYS simulate_v -> calibrate + fit (NO FD info) ----
    i_d0 = np.zeros((nK, nZ, nY)); i_g0 = np.full((nK, nZ, nY), 0.05)
    v_ws = FD.simulate_v(logK, Z, Y, i_d0, i_g0, LAM3, T=1200.0, dt=2.5)
    model.calibrate(v_ws)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    v_ws_t = torch.tensor(v_ws.ravel().reshape(-1, 1), dtype=dtype)
    N = lk_t.shape[0]; bs = min(4096, N)
    for _ in range(1200):
        idx = torch.randint(0, N, (bs,))
        opt.zero_grad()
        loss = ((model(lk_t[idx], z_t[idx], y_t[idx]) - v_ws_t[idx]) ** 2).mean()
        loss.backward(); opt.step()
    if verbose:
        print(f"  [seed {seed}] warm done fit_err={loss.item():.2e} "
              f"(v_ws mean={np.mean(v_ws):.2f} std={np.std(v_ws):.2f})", flush=True)

    ik = np.argmin(np.abs(logK - np.log(880))); jz = np.argmin(np.abs(Z - 0.7)); ky = np.argmin(np.abs(Y - 3.0))
    box = SFE._box_mask(logK, Z, Y); dim = SFE._deinvest_mask(logK, Z, Y)

    for sweep in range(n_sweeps):
        xi = xi_schedule(sweep, n_sweeps, xi_start, xi_target)

        # ---- (anchor) build the PIBYS oracle value at the CURRENT FOC policy = discrete M1 fixed
        #      point target (the value-side semigroup-Bellman that converged before). Frozen. ----
        if anchor:
            f = fields_on_grid(model, lk_t, z_t, y_t, nK, nZ, nY, ZZ)
            v_pe = FD.simulate_v(logK, Z, Y, f["i_d"], f["i_g"], LAM3, T=1200.0, dt=2.5)
            v_pe_t = torch.tensor(v_pe.ravel().reshape(-1, 1), dtype=dtype)

        # ---- inner: minimize M1 martingale-difference (+ anchor) ----
        for _ in range(steps):
            idx = torch.randint(0, N, (bs,))
            opt.zero_grad()
            lM, i_d_b, i_g_b, c_b, qd_b, qg_b, vlK_b, vZ_b, vY_b, _ = m1_martingale_loss(
                model, lk_t[idx], z_t[idx], y_t[idx], xi, dt,
                worstcase=worstcase, antithetic=antithetic, nondim=nondim, sto=sto)
            loss = lM
            if anchor:
                v_now = model(lk_t[idx], z_t[idx], y_t[idx])
                loss = loss + lam_anchor * ((v_now - v_pe_t[idx]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # ---- (stability 6) PER-REGION monitor + TRUE error vs FD ----
        if verbose:
            f = fields_on_grid(model, lk_t, z_t, y_t, nK, nZ, nY, ZZ)
            extra = ""
            if d_fd is not None:
                e = SFE.grade(dict(logK=logK, Z=Z, Y=Y, i_d=f["i_d"], i_g=f["i_g"], vZ=f["vZ"]), d_fd)
                extra = (f" | TRUE box_id={e['box_err_i_d']:.2e} box_ig={e['box_err_i_g']:.2e} "
                         f"di_vZ={e['di_err_vZ']:.2e} di_min_id={e['di_min_i_d']:+.4f}"
                         f"(FD{e['di_FD_min_i_d']:+.4f})")
            print(f"  [seed {seed} sweep {sweep:2d}] xi={xi:.1f} L_M1+anchor={loss.item():.3e} "
                  f"i_d(ref)={f['i_d'][ik,jz,ky]:+.4f} i_g(ref)={f['i_g'][ik,jz,ky]:+.4f} "
                  f"vZ={f['vZ'][ik,jz,ky]:.3f}{extra}", flush=True)

    f = fields_on_grid(model, lk_t, z_t, y_t, nK, nZ, nY, ZZ)
    return dict(logK=logK, Z=Z, Y=Y, **f)


# =====================================================================================
#  Validation vs the HIGH-ACCURACY FD (fall back to stable FD).  ONLY controls/costate.
# =====================================================================================
def load_reference():
    hi = os.path.join(HERE, "outputs", "fd_pdpt_hiacc_ref_lam3_0167_xi148.npz")
    if os.path.exists(hi):
        d = np.load(hi); print(f"[ref] HIGH-ACCURACY FD: {os.path.basename(hi)}", flush=True)
    else:
        d = np.load(SFE.STABLE_NPZ); print(f"[ref] stable FD (hiacc not built yet): {os.path.basename(SFE.STABLE_NPZ)}", flush=True)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--nK", type=int, default=21)
    ap.add_argument("--nZ", type=int, default=31)
    ap.add_argument("--nY", type=int, default=21)
    ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--worstcase", choices=["on", "off"], default="on")
    ap.add_argument("--antithetic", choices=["on", "off"], default="on")
    ap.add_argument("--sto", choices=["on", "off"], default="on")
    ap.add_argument("--nondim", choices=["on", "off"], default="on")
    ap.add_argument("--anchor", choices=["on", "off"], default="on")
    ap.add_argument("--xi_start", type=float, default=1e4)
    ap.add_argument("--xi_target", type=float, default=XI_TARGET)
    ap.add_argument("--float64", action="store_true")
    args = ap.parse_args()
    dtype = torch.float64 if args.float64 else torch.float32

    d = load_reference()
    print(f"=== MARTINGALE-REWRITE solver, 3-D post-damage-post-tech === dtype={dtype}", flush=True)
    print(f"grid {args.nK}x{args.nZ}x{args.nY}, {args.sweeps} sweeps x {args.steps} steps, dt={args.dt}, "
          f"worstcase={args.worstcase}, antithetic={args.antithetic}, nondim={args.nondim}, "
          f"anchor={args.anchor}, xi {args.xi_start}->{args.xi_target}, seeds {args.seeds}", flush=True)

    per_seed = []
    for seed in args.seeds:
        t0 = time.time()
        out = run_martingale(seed, nK=args.nK, nZ=args.nZ, nY=args.nY,
                             n_sweeps=args.sweeps, steps=args.steps, dt=args.dt,
                             worstcase=(args.worstcase == "on"),
                             antithetic=(args.antithetic == "on"),
                             sto=(args.sto == "on"), nondim=(args.nondim == "on"),
                             anchor=(args.anchor == "on"),
                             xi_start=args.xi_start, xi_target=args.xi_target,
                             dtype=dtype, verbose=True, d_fd=d)
        m = SFE.grade(dict(logK=out["logK"], Z=out["Z"], Y=out["Y"],
                           i_d=out["i_d"], i_g=out["i_g"], vZ=out["vZ"]), d)
        per_seed.append(m)
        print(f"  -> [seed {seed}] {time.time()-t0:.0f}s  box_err_i_d={m['box_err_i_d']:.3e} "
              f"box_err_i_g={m['box_err_i_g']:.3e} di_err_vZ={m['di_err_vZ']:.3e} "
              f"di_min_i_d={m['di_min_i_d']:+.4f} (FD min {m['di_FD_min_i_d']:+.4f})", flush=True)

    agg = SFE.summarize(per_seed, label="MARTINGALE-REWRITE")
    np.savez(os.path.join(HERE, "outputs", "leaderboard_martingale.npz"),
             summary=json.dumps(agg))
    print("\nSAVED -> outputs/leaderboard_martingale.npz", flush=True)


if __name__ == "__main__":
    main()
