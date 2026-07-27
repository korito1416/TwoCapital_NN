"""
operator_backward_ab.py -- OPERATOR-BACKWARD (martingale-representation / BSDE-consistent) NN solver
on the 3-D post-damage-post-tech stepping stone (state = logK, Z, Y; terminal/jump-free), with a
forward-autodiff A/B baseline, validated against the fd_pdpt_v5 FD ground truth.

THE DIAGNOSIS BEING TESTED
--------------------------
The HJB is (rho - L) v = f, L = drift.grad + 0.5 diff:Hess.
  FORWARD direction: residual R=(rho-L)v-f APPLIES L to v => must DIFFERENTIATE a fitted v
    (autodiff grad). Ill-conditioned in the weak-id Z axis (mu_Z->0); ~1e-3 slope floor.
  BACKWARD direction: v=(rho-L)^{-1} f is the resolvent = Feynman-Kac = a SMOOTHING/integrating
    operator. The costate sigma^T grad(v) is read off the MARTINGALE REPRESENTATION as a
    least-squares regression of the value increment on the Brownian increment along sampled
    controlled transitions -- NEVER autodiff of a fitted v.

WHAT THIS SCRIPT DOES
---------------------
A) OPERATOR-BACKWARD: value net V + costate head Q (outputs qd,qg directly, the FOC inputs) +
   a v_Y head. Costate trained by martingale-regression of the sampled value increment on the
   Brownian increments (no autodiff in the FOC path). Value trained by the semigroup-Bellman
   one-step fixed point v(x)=f*dt+exp(-rho*dt)E[v(X_dt)] under the worst-case controlled transition.
   Howard outer loop over frozen policies.
B) FORWARD-AUTODIFF baseline: the same value net, costate = autograd.grad(V). Identical Howard /
   FOC / FD-validation harness. (This is the architecture egm_howard_ctrlfit_ab.py embodies.)
C) DGM-PIA: read from fd_vs_nn.py's printout (run separately; needs TF) -- reported, not re-run here.

DECISIVE METRIC: TRUE error vs fd_pdpt_v5 npz -- max|i_d-FD|, max|i_g-FD|, max|v-FD|, max|vZ-FD|,
max|qd-FD| over the 3-D INTERIOR box (slice(2,-2),slice(8,-8),slice(2,-2)). Multi-seed.
We ALSO report the SNR / sampling variance of the martingale regression (the central honesty check).

Reuses: fd_pdpt_v5 (FD.controls, FD.simulate_v, FD._drift, P, QFLOOR, true_error harness pieces).
"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd_pdpt_v5 as FD
import egm_howard_ctrlfit_ab as EGM   # reuse ValueNet3D, controls_torch, true_error, run_egm_howard

P = FD.P
QFLOOR = FD.QFLOOR
LK_LO, LK_HI = EGM.LK_LO, EGM.LK_HI
Z_LO, Z_HI = EGM.Z_LO, EGM.Z_HI
Y_LO, Y_HI = EGM.Y_LO, EGM.Y_HI
LAM3 = EGM.LAM3
XI = EGM.XI
LK_CAP = FD.LK_CAP
Y_CAP = FD.Y_CAP

s_d = P["s_d"]; s_g = P["s_g"]; varsig = P["vars"]; eta = P["eta"]; A_d = P["A_d"]
delta = P["delta"]; l1 = P["l1"]; l2 = P["l2"]; y_up = P["y_up"]; thbar = P["thbar"]


# =====================================================================================
#  Networks: shared-trunk value head + DIRECT costate head (qd,qg) + v_Y head
# =====================================================================================
class ValueCostateNet(nn.Module):
    """Shared tanh trunk; head V -> value (soft anchor like ValueNet3D); head Q -> (qd,qg,vY).
    qd,qg are the EXACT FOC costates (= v_logK - Z v_Z, v_logK + (1-Z) v_Z); vY = v_Y for h_y.
    The costate head is supervised by the martingale regression, NEVER by autodiff of V."""

    def __init__(self, width=64, depth=4, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.head_v = nn.Linear(width, 1)
        self.head_q = nn.Linear(width, 3)   # qd, qg, vY
        self.register_buffer("v_mean", torch.tensor(4.0, dtype=dtype))
        self.register_buffer("v_scale", torch.tensor(1.0, dtype=dtype))
        self.register_buffer("logK_slope", torch.tensor(0.5, dtype=dtype))
        self.register_buffer("logK_mid", torch.tensor(0.5 * (LK_LO + LK_HI), dtype=dtype))
        # costate head defaults: small positive bias so qd,qg start feasible (>QFLOOR)
        with torch.no_grad():
            self.head_q.bias[:] = torch.tensor([0.1, 0.3, 0.0], dtype=dtype)
            self.head_q.weight *= 0.1
        self.to(dtype)

    def calibrate(self, v_ws):
        self.v_mean.fill_(float(np.mean(v_ws)))
        self.v_scale.fill_(float(max(np.std(v_ws), 0.5)))

    def _feat(self, logK, Z, Y):
        sK = (logK - LK_LO) / (LK_HI - LK_LO)
        sZ = (Z - Z_LO) / (Z_HI - Z_LO)
        sY = (Y - Y_LO) / (Y_HI - Y_LO)
        X = torch.cat([2 * sK - 1, 2 * sZ - 1, 2 * sY - 1], dim=1)
        return self.trunk(X)

    def value(self, logK, Z, Y):
        h = self._feat(logK, Z, Y)
        base = self.v_mean + self.logK_slope * (logK - self.logK_mid)
        return base + self.v_scale * self.head_v(h)

    def costate(self, logK, Z, Y):
        """Return (qd, qg, vY) directly from the costate head."""
        h = self._feat(logK, Z, Y)
        q = self.head_q(h)
        return q[:, 0:1], q[:, 1:2], q[:, 2:3]

    def forward(self, logK, Z, Y):
        return self.value(logK, Z, Y)


# =====================================================================================
#  Generator: worst-case-controlled one-step transition + flow (torch, batched)
# =====================================================================================
def drift_and_loadings(logK, Z, Y, i_d, i_g, qd, qg, vY):
    """Worst-case-controlled drift (a_lK,a_Z,a_Y) and the 3 Brownian loading vectors, all torch.
    Returns drift components, the per-channel loadings, h_d,h_g,h_y, E."""
    phid = P["a_d"] + P["G_d"] * torch.log(torch.clamp(1 + P["t_d"] * i_d, min=1e-9))
    phig = P["a_g"] + P["G_g"] * torch.log(torch.clamp(1 + P["t_g"] * i_g, min=1e-9))
    Dc = s_d ** 2 * (1 - Z) ** 2 + s_g ** 2 * Z ** 2
    K = torch.exp(torch.clamp(logK, max=LK_CAP))
    E = eta * A_d * (1 - Z) * K
    # worst-case h (closed form from the costate; O(1e-6) at xi=148.4, kept for completeness)
    h_d = -(1.0 / XI) * qd * (1 - Z) * s_d
    h_g = -(1.0 / XI) * qg * Z * s_g
    lNy_full = l1 + l2 * Y + LAM3 * (Y - y_up)
    h_y = -(1.0 / XI) * (vY - lNy_full) * eta * A_d * (1 - Z) * K * varsig
    # drift (worst-case-corrected: h shifts the Brownian drift)
    a_lK = (1 - Z) * phid + Z * phig - Dc / 2.0 + (1 - Z) * s_d * h_d + Z * s_g * h_g
    a_Z = Z * (1 - Z) * (phig - phid + (1 - Z) * s_d ** 2 - Z * s_g ** 2) \
        + (-Z * (1 - Z) * s_d * h_d + Z * (1 - Z) * s_g * h_g)
    a_Y = (thbar + h_y * varsig) * E
    # Brownian loadings on (dW_d, dW_g, dW_Y) channels for the state (logK, Z, Y)
    # logK: (1-Z)s_d dW_d + Z s_g dW_g ; Z: -Z(1-Z)s_d dW_d + Z(1-Z)s_g dW_g ; Y: varsig*E dW_Y
    L_K = (( 1 - Z) * s_d, Z * s_g, torch.zeros_like(Z))
    L_Z = (-Z * (1 - Z) * s_d, Z * (1 - Z) * s_g, torch.zeros_like(Z))
    L_Y = (torch.zeros_like(Z), torch.zeros_like(Z), varsig * E)
    return a_lK, a_Z, a_Y, (L_K, L_Z, L_Y), h_d, h_g, h_y, E


def flow_fn(logK, Z, Y, i_d, i_g, c, a_Y, E, h_d, h_g, h_y):
    """Semigroup flow f(x,a*) = delta*(log c + logK) - clim + robustness penalty."""
    y_eff = torch.clamp(Y, max=Y_CAP)
    lNy = l1 + l2 * y_eff + LAM3 * (y_eff - y_up)
    lNyy = l2 + LAM3
    b_Y = 0.5 * varsig ** 2 * E ** 2
    clim = lNy * a_Y + lNyy * b_Y
    pen = 0.5 * XI * (h_d ** 2 + h_g ** 2 + h_y ** 2)
    return delta * (torch.log(torch.clamp(c, min=1e-12)) + logK) - clim + pen


def step_children(logK, Z, Y, a_lK, a_Z, a_Y, loadings, xi, dt):
    """Euler-Maruyama one-step transition for M antithetic shock draws.
    xi: (B, M, 3) standard normals. Returns child states each (B, M, 1)."""
    L_K, L_Z, L_Y = loadings
    sdt = np.sqrt(dt)
    xd = xi[..., 0:1]; xg = xi[..., 1:2]; xy = xi[..., 2:3]   # (B,M,1)
    # broadcast drift/loadings (B,1,1)
    def b(t):
        return t.unsqueeze(1)
    lk_c = b(logK) + b(a_lK) * dt + sdt * (b(L_K[0]) * xd + b(L_K[1]) * xg)
    z_c = b(Z) + b(a_Z) * dt + sdt * (b(L_Z[0]) * xd + b(L_Z[1]) * xg)
    y_c = b(Y) + b(a_Y) * dt + sdt * (b(L_Y[2]) * xy)
    # boundary handling: clip Z exactly as simulate_v; reflect logK,Y into box
    lk_c = torch.clamp(lk_c, LK_LO, LK_HI)
    z_c = torch.clamp(z_c, 1e-4, 1 - 1e-4)
    y_c = torch.clamp(y_c, Y_LO, Y_HI)
    return lk_c, z_c, y_c


# =====================================================================================
#  Operator-backward training (one Howard sweep)
# =====================================================================================
def sample_box(B, dtype, gen):
    lk = LK_LO + (LK_HI - LK_LO) * torch.rand(B, 1, generator=gen, dtype=dtype)
    z = Z_LO + (Z_HI - Z_LO) * torch.rand(B, 1, generator=gen, dtype=dtype)
    y = Y_LO + (Y_HI - Y_LO) * torch.rand(B, 1, generator=gen, dtype=dtype)
    return lk, z, y


def antithetic_normals(B, M, dtype, gen):
    """M antithetic standard-normal draws: (B, M, 3). M must be even."""
    half = M // 2
    z = torch.randn(B, half, 3, generator=gen, dtype=dtype)
    return torch.cat([z, -z], dim=1)


def run_operator_backward(seed, nK=21, nZ=31, nY=21, n_howard=12, fit_steps=1500,
                          warm_steps=1500, lr=2e-3, dt=2.5, T=1200.0, B=2048, M=64,
                          lam_q=1.0, dtype=torch.float32, verbose=True, d_fd=None,
                          measure_snr=True):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    model = ValueCostateNet(dtype=dtype)

    # grid for FOC/oracle (numpy) + tensors for final field extraction
    logK, Z, Y, (LK, ZZ, YY), (lk_t, z_t, y_t) = EGM.make_grid(nK, nZ, nY, dtype)

    # ---- warm start: cold policy -> oracle v_ws -> fit V head (NO FD info) ----
    i_d0 = np.zeros((nK, nZ, nY)); i_g0 = np.full((nK, nZ, nY), 0.05)
    v_ws = FD.simulate_v(logK, Z, Y, i_d0, i_g0, LAM3, T=T, dt=dt)
    model.calibrate(v_ws)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    v_ws_t = torch.tensor(v_ws.ravel().reshape(-1, 1), dtype=dtype)
    Nw = lk_t.shape[0]; bsw = min(2048, Nw)
    for _ in range(warm_steps):
        idx = torch.randint(0, Nw, (bsw,))
        opt.zero_grad()
        v = model.value(lk_t[idx], z_t[idx], y_t[idx])
        loss = ((v - v_ws_t[idx]) ** 2).mean()
        loss.backward(); opt.step()
    if verbose:
        print(f"  [seed {seed}] OPBACK warm-start done, fit_err={loss.item():.2e} "
              f"(v_ws mean={np.mean(v_ws):.2f} std={np.std(v_ws):.2f})", flush=True)

    ik = np.argmin(np.abs(logK - np.log(880)))
    jz = np.argmin(np.abs(Z - 0.7)); ky = np.argmin(np.abs(Y - 3.0))

    snr_report = None
    for sweep in range(n_howard):
        # (1) read current costate -> FOC controls on the GRID -> oracle policy-eval value
        with torch.no_grad():
            qd_g, qg_g, vY_g = model.costate(lk_t, z_t, y_t)
        qd_g = qd_g.detach().numpy().reshape(nK, nZ, nY)
        qg_g = qg_g.detach().numpy().reshape(nK, nZ, nY)
        i_d_g, i_g_g, c_g = FD.controls(qd_g, qg_g, ZZ)
        v_pe = FD.simulate_v(logK, Z, Y, i_d_g, i_g_g, LAM3, T=T, dt=dt)
        v_pe_t = torch.tensor(v_pe.ravel().reshape(-1, 1), dtype=dtype)
        N = lk_t.shape[0]

        # (2) fit V head to the oracle value AND costate head by martingale regression
        for step in range(fit_steps):
            # (a) value fit toward oracle policy-eval value (the resolvent/backward target,
            #     mesh-bound but removes EMA-target variance during de-risk)
            idx = torch.randint(0, N, (min(B, N),))
            opt.zero_grad()
            v_fit = model.value(lk_t[idx], z_t[idx], y_t[idx])
            loss_v = ((v_fit - v_pe_t[idx]) ** 2).mean()

            # (b) mesh-free semigroup-Bellman + martingale-regression costate on fresh collocation
            lk, z, y = sample_box(min(B, N), dtype, gen)
            qd, qg, vY = model.costate(lk, z, y)
            qd_f = QFLOOR + torch.nn.functional.softplus(qd - QFLOOR, beta=50.0)
            qg_f = QFLOOR + torch.nn.functional.softplus(qg - QFLOOR, beta=50.0)
            i_d, i_g, c = EGM.controls_torch(qd, qg, z)
            a_lK, a_Z, a_Y, loadings, h_d, h_g, h_y, E = drift_and_loadings(
                lk, z, y, i_d, i_g, qd_f, qg_f, vY)
            xi = antithetic_normals(lk.shape[0], M, dtype, gen)  # (B,M,3)
            lk_c, z_c, y_c = step_children(lk, z, y, a_lK, a_Z, a_Y, loadings, xi, dt)
            Bc = lk.shape[0]
            with torch.no_grad():
                vC = model.value(lk_c.reshape(-1, 1), z_c.reshape(-1, 1),
                                 y_c.reshape(-1, 1)).reshape(Bc, M, 1)
            vbar = vC.mean(dim=1, keepdim=True)            # E[v(X_dt)|x]
            dV = (vC - vbar)                               # (B,M,1) martingale increment

            # Semigroup-Bellman value target (mesh-free) -- contraction exp(-delta dt)<1
            f = flow_fn(lk, z, y, i_d, i_g, c, a_Y, E, h_d, h_g, h_y)
            v_self = model.value(lk, z, y)
            tgt = (f * dt + np.exp(-delta * dt) * vbar.squeeze(1)).detach()
            loss_bellman = ((v_self - tgt) ** 2).mean()

            # MARTINGALE-REGRESSION costate target (NO autodiff of v).
            # dV ~ sqrt(dt) [ Z_d xi_d + Z_g xi_g + Z_Y xi_Y ], OLS slope = mean(dV*xi)/dt^(0.5).
            # Antithetic + zero-mean Gaussian => E[xi_a xi_b]=delta_ab, so per-channel OLS:
            #   Z_b_hat = (1/M) sum_m dV_m xi_b_m / sqrt(dt)
            sdt = np.sqrt(dt)
            xd = xi[..., 0:1]; xg = xi[..., 1:2]; xy = xi[..., 2:3]
            Zd_hat = (dV * xd).mean(dim=1) / sdt           # (B,1)
            Zg_hat = (dV * xg).mean(dim=1) / sdt
            ZY_hat = (dV * xy).mean(dim=1) / sdt
            # divide out the KNOWN analytic loadings -> (qd,qg,vY) targets
            load_d = (1 - z) * s_d                          # >0 in interior
            load_g = z * s_g
            load_y = varsig * eta * A_d * (1 - z) * torch.exp(torch.clamp(lk, max=LK_CAP))
            qd_tgt = (Zd_hat / load_d).detach()
            qg_tgt = (Zg_hat / load_g).detach()
            vY_tgt = (ZY_hat / load_y).detach()
            loss_q = ((qd - qd_tgt) ** 2 + (qg - qg_tgt) ** 2 + (vY - vY_tgt) ** 2).mean()

            loss = loss_v + loss_bellman + lam_q * loss_q
            loss.backward(); opt.step()

        if verbose:
            extra = ""
            if d_fd is not None and (sweep % 3 == 0 or sweep == n_howard - 1):
                cur = dict(logK=logK, Z=Z, Y=Y)
                with torch.no_grad():
                    qd_e, qg_e, _ = model.costate(lk_t, z_t, y_t)
                    v_e = model.value(lk_t, z_t, y_t)
                qd_e = qd_e.numpy().reshape(nK, nZ, nY)
                qg_e = qg_e.numpy().reshape(nK, nZ, nY)
                i_de, i_ge, _ = FD.controls(qd_e, qg_e, ZZ)
                # reconstruct vlK,vZ from qd,qg for the vZ error metric
                vZ_e = qg_e - qd_e
                vlK_e = qd_e + ZZ * vZ_e
                cur["i_d"] = i_de; cur["i_g"] = i_ge; cur["vlK"] = vlK_e; cur["vZ"] = vZ_e
                cur["v"] = v_e.numpy().reshape(nK, nZ, nY)
                e = EGM.true_error(cur, d_fd)
                extra = (f" | TRUEerr i_d={e['err_i_d']:.2e} i_g={e['err_i_g']:.2e} "
                         f"v={e['err_v']:.2e} vZ={e['err_vZ']:.2e}")
            print(f"  [seed {seed} sw {sweep:2d}] Lv={loss_v.item():.2e} Lbell={loss_bellman.item():.2e} "
                  f"Lq={loss_q.item():.2e} i_d(ref)={i_d_g[ik,jz,ky]:+.4f} i_g(ref)={i_g_g[ik,jz,ky]:+.4f} "
                  f"qd(ref)={qd_g[ik,jz,ky]:.3f}{extra}", flush=True)

    # --- SNR diagnostic of the martingale regression at the reference & interior (honesty check) ---
    if measure_snr:
        snr_report = martingale_snr(model, d_fd, dt, M, dtype, gen)

    # final fields on the grid (costate from the Q head, NOT autodiff)
    with torch.no_grad():
        qd_e, qg_e, vY_e = model.costate(lk_t, z_t, y_t)
        v_e = model.value(lk_t, z_t, y_t)
    qd_np = qd_e.numpy().reshape(nK, nZ, nY)
    qg_np = qg_e.numpy().reshape(nK, nZ, nY)
    v_np = v_e.numpy().reshape(nK, nZ, nY)
    i_d_g, i_g_g, c_g = FD.controls(qd_np, qg_np, ZZ)
    vZ_np = qg_np - qd_np
    vlK_np = qd_np + ZZ * vZ_np
    out = dict(logK=logK, Z=Z, Y=Y, v=v_np, i_d=i_d_g, i_g=i_g_g, c=c_g,
               vlK=vlK_np, vZ=vZ_np, qd=qd_np, qg=qg_np)
    return out, snr_report


def martingale_snr(model, d_fd, dt, M, dtype, gen, n_pts=2000, M_big=4096):
    """Compare the SINGLE-batch martingale-regression costate (the training estimator) against a
    HIGH-M reference of the SAME estimator, AND against the FD truth, on interior points.
    Reports: (1) regression std at training M (sampling noise), (2) bias vs FD costate qd."""
    from scipy.interpolate import RegularGridInterpolator as RGI
    # interior sample
    lk = LK_LO + (LK_HI - LK_LO) * torch.rand(n_pts, 1, generator=gen, dtype=dtype)
    z = (Z_LO + 0.13) + (Z_HI - 0.13 - (Z_LO + 0.13)) * torch.rand(n_pts, 1, generator=gen, dtype=dtype)
    y = Y_LO + (Y_HI - Y_LO) * torch.rand(n_pts, 1, generator=gen, dtype=dtype)
    with torch.no_grad():
        qd, qg, vY = model.costate(lk, z, y)
        i_d, i_g, c = EGM.controls_torch(qd, qg, z)
        a_lK, a_Z, a_Y, loadings, h_d, h_g, h_y, E = drift_and_loadings(lk, z, y, i_d, i_g, qd, qg, vY)

    def regress(Mn):
        with torch.no_grad():
            xi = antithetic_normals(n_pts, Mn, dtype, gen)
            lk_c, z_c, y_c = step_children(lk, z, y, a_lK, a_Z, a_Y, loadings, xi, dt)
            vC = model.value(lk_c.reshape(-1, 1), z_c.reshape(-1, 1),
                             y_c.reshape(-1, 1)).reshape(n_pts, Mn, 1)
            dV = vC - vC.mean(dim=1, keepdim=True)
            sdt = np.sqrt(dt)
            Zd = (dV * xi[..., 0:1]).mean(1) / sdt
            qd_reg = Zd / ((1 - z) * s_d)
        return qd_reg.squeeze(1).numpy()

    # repeat the training-M estimator a few times to get its sampling std
    reps = np.stack([regress(M) for _ in range(8)], axis=0)
    qd_M_mean = reps.mean(0); qd_M_std = reps.std(0)
    qd_ref = regress(M_big)   # low-variance reference
    qd_net = qd.squeeze(1).numpy()

    # FD truth at these points
    fZ = RGI((d_fd["logK"], d_fd["Z"], d_fd["Y"]),
             d_fd["vlK"] - d_fd["Z"][None, :, None] * d_fd["vZ"],  # qd_FD = vlK - Z vZ
             bounds_error=False, fill_value=None)
    pts = np.stack([lk.squeeze(1).numpy(), z.squeeze(1).numpy(), y.squeeze(1).numpy()], axis=1)
    qd_FD = fZ(pts)

    return dict(
        reg_std_at_M=float(np.median(qd_M_std)),
        reg_std_at_M_max=float(np.percentile(qd_M_std, 95)),
        qnet_minus_FD_med=float(np.median(np.abs(qd_net - qd_FD))),
        qnet_minus_FD_max=float(np.percentile(np.abs(qd_net - qd_FD), 95)),
        regbigM_minus_FD_med=float(np.median(np.abs(qd_ref - qd_FD))),
        regM_minus_regbigM_med=float(np.median(np.abs(qd_M_mean - qd_ref))),
    )


# =====================================================================================
#  main: run operator-backward + forward-autodiff baseline + report
# =====================================================================================
def summarize(label, errs):
    print(f"  === {label} SUMMARY (median [min,max] over seeds) ===", flush=True)
    for k in ("err_i_d", "err_i_g", "err_v", "err_vlK", "err_vZ"):
        a = np.array(errs[k])
        print(f"    {k}: median={np.median(a):.3e}  [{a.min():.3e}, {a.max():.3e}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--nK", type=int, default=21)
    ap.add_argument("--nZ", type=int, default=31)
    ap.add_argument("--nY", type=int, default=21)
    ap.add_argument("--howard", type=int, default=10)
    ap.add_argument("--fit", type=int, default=1200)
    ap.add_argument("--warm", type=int, default=1500)
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--B", type=int, default=2048)
    ap.add_argument("--lam_q", type=float, default=1.0)
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--skip_baseline", action="store_true")
    args = ap.parse_args()
    dtype = torch.float64 if args.float64 else torch.float32

    d = np.load(os.path.join(HERE, "outputs", "fd_pdpt_v5_lam3_0167_xi148.npz"))
    d = {k: d[k] for k in d.files}

    print(f"=== OPERATOR-BACKWARD vs FORWARD-AUTODIFF, 3-D post-damage-post-tech === dtype={dtype}", flush=True)
    print(f"grid {args.nK}x{args.nZ}x{args.nY}, {args.howard} Howard sweeps, M={args.M}, B={args.B}, "
          f"dt={args.dt}, seeds {args.seeds}", flush=True)

    # ---------- (A) OPERATOR-BACKWARD ----------
    print("\n######## (A) OPERATOR-BACKWARD (martingale-regression costate, NO autodiff in FOC) ########", flush=True)
    errsA = {k: [] for k in ("err_i_d", "err_i_g", "err_v", "err_vlK", "err_vZ")}
    snrs = []
    for seed in args.seeds:
        t0 = time.time()
        out, snr = run_operator_backward(
            seed, nK=args.nK, nZ=args.nZ, nY=args.nY, n_howard=args.howard,
            fit_steps=args.fit, warm_steps=args.warm, dt=args.dt, M=args.M, B=args.B,
            lam_q=args.lam_q, dtype=dtype, verbose=True, d_fd=d)
        e = EGM.true_error(out, d)
        for k in errsA:
            errsA[k].append(e[k])
        snrs.append(snr)
        # also report qd true-error explicitly
        from scipy.interpolate import RegularGridInterpolator as RGI
        qd_FD = d["vlK"] - d["Z"][None, :, None] * d["vZ"]
        out_qd = EGM.interp_to_fd({**out}, "qd", d)
        box = (slice(2, -2), slice(8, -8), slice(2, -2))
        err_qd = float(np.max(np.abs(out_qd - qd_FD)[box]))
        print(f"  -> [seed {seed}] {time.time()-t0:.0f}s  max|i_d-FD|={e['err_i_d']:.3e} "
              f"max|i_g-FD|={e['err_i_g']:.3e} max|v-FD|={e['err_v']:.3e} "
              f"max|vZ-FD|={e['err_vZ']:.3e} max|qd-FD|={err_qd:.3e}", flush=True)
        if snr is not None:
            print(f"     SNR/variance: reg_std@M={snr['reg_std_at_M']:.2e} (95p {snr['reg_std_at_M_max']:.2e}) "
                  f"| qnet-FD med={snr['qnet_minus_FD_med']:.2e} max95={snr['qnet_minus_FD_max']:.2e} "
                  f"| regBigM-FD med={snr['regbigM_minus_FD_med']:.2e} "
                  f"| regM-regBigM med={snr['regM_minus_regbigM_med']:.2e}", flush=True)
    summarize("(A) OPERATOR-BACKWARD", errsA)

    # ---------- (B) FORWARD-AUTODIFF baseline (reuse egm_howard) ----------
    if not args.skip_baseline:
        print("\n######## (B) FORWARD-AUTODIFF baseline (costate = autograd.grad(v)) ########", flush=True)
        errsB = {k: [] for k in ("err_i_d", "err_i_g", "err_v", "err_vlK", "err_vZ")}
        for seed in args.seeds:
            t0 = time.time()
            out = EGM.run_egm_howard(
                seed, nK=args.nK, nZ=args.nZ, nY=args.nY, n_howard=args.howard,
                fit_steps=args.fit, warm_steps=args.warm, ctrlfit=True, dt=args.dt,
                dtype=dtype, verbose=False, d_fd=d)
            e = EGM.true_error(out, d)
            for k in errsB:
                errsB[k].append(e[k])
            print(f"  -> [seed {seed}] {time.time()-t0:.0f}s  max|i_d-FD|={e['err_i_d']:.3e} "
                  f"max|i_g-FD|={e['err_i_g']:.3e} max|v-FD|={e['err_v']:.3e} "
                  f"max|vZ-FD|={e['err_vZ']:.3e}", flush=True)
        summarize("(B) FORWARD-AUTODIFF", errsB)

        # ---------- A/B verdict ----------
        print("\n######## A/B VERDICT (median over seeds) ########", flush=True)
        for k, lbl in [("err_i_d", "max|i_d-FD|"), ("err_i_g", "max|i_g-FD|"),
                       ("err_v", "max|v-FD|"), ("err_vZ", "max|vZ-FD|")]:
            a = np.median(errsA[k]); b = np.median(errsB[k])
            verdict = "OPBACK better" if a < b else ("autodiff better" if b < a else "tie")
            print(f"    {lbl:14s}: opback={a:.3e}  autodiff={b:.3e}  -> {verdict}", flush=True)

    print("\n(C) DGM-PIA: run fd_vs_nn.py separately (needs TF) for the third baseline.", flush=True)


if __name__ == "__main__":
    main()
