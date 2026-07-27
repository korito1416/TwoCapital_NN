"""
egm_howard_ctrlfit_ab.py -- Phase-A de-risk of the EGM/Howard FOC-transport NN method on the
3-D post-damage-post-tech stepping stone (state = logK, Z, Y; terminal/no-jumps).

RECIPE per Howard sweep (the 1-D winner generalized to 3-D):
  (1) value net v(logK,Z,Y) with a SOFT boundary ansatz; read v_logK,v_Z,v_Y by AUTODIFF.
  (2) FOC inversion -> controls i_d,i_g via the EXACT, FD-VALIDATED closed form
      (fd_pdpt_v5.controls; verified to reproduce FD controls to 0.0 when fed FD's own slopes).
  (3) FREEZE controls -> the linear policy-evaluation ORACLE = fd_pdpt_v5.simulate_v (PIBYS,
      the grid-converged no-artificial-diffusion policy evaluation -- the 3-D analog of the
      1-D Thomas solve). Returns v_pe on the grid.
  (4) CTRLFIT supervise: L = mean((i_d(p_nn)-i_d_pe)^2 + (i_g(p_nn)-i_g_pe)^2) + lam_v*mean((v-v_pe)^2),
      lam_v=1.0. i_d(p_nn) = autodiff slope pushed through the differentiable FOC map.
      GUARDS: no (di/dvp)^2 on a strong residual (there is none here); no mu_Z reweighting; lam_v>0.

ABLATION: ctrlfit (control-space + value anchor) vs PLAIN value-L2 only (lam_v=1, control term off).

VALIDATION (DECISIVE): true error vs fd_pdpt_v5 npz -- max|i_d-FD|, max|i_g-FD|, max|v-FD| over the
3-D INTERIOR box. Multi-seed. NEVER compare loss numbers across formulations. The existing DGM-PIA
NN baseline (Q4) is read from fd_vs_nn.py's own printout (run separately; requires TF).
"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
torch.set_num_threads(4)   # CRITICAL: the 32-thread default oversubscribes on the small minibatch
                           # autodiff graph and is ~10x slower (340ms/step vs 30ms/step at 4 threads).
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd_pdpt_v5 as FD   # simulate_v, controls, P, QFLOOR, Y_CAP

P = FD.P
QFLOOR = FD.QFLOOR

# ----- FD box (must match fd_pdpt_v5.solve grids exactly so v_pe is comparable to FD truth) -----
LK_LO, LK_HI = 4.0, 7.0
Z_LO,  Z_HI  = 0.02, 0.98
Y_LO,  Y_HI  = 0.0, 4.0
LAM3 = 1.0 / 6.0
XI   = 148.4


# =====================================================================================
#  Differentiable FOC inversion (torch transcription of fd_pdpt_v5.controls; smooth floor)
# =====================================================================================
def controls_torch(qd, qg, Z, soft=True):
    """Exact closed-form FOC controls, differentiable. Smooth softplus floor on costates
    (matches the hard np.maximum(.,QFLOOR) in fd_pdpt_v5 but keeps gradients alive)."""
    if soft:
        # softplus floor approaching the hard QFLOOR floor; sharp (beta large) so interior is identity
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
#  3-D value net with SOFT boundary ansatz
# =====================================================================================
def damage_offset(Y, lam3=LAM3):
    """Analytic integrated -logN running-cost offset (the leading Y-dependence of v).
    -(1/delta)*(l1*Y + 0.5*l2*Y^2 + 0.5*lam3*(Y-y_up)^2). Matches the clim term in simulate_v."""
    l1, l2, y_up, dl = P["l1"], P["l2"], P["y_up"], P["delta"]
    return -(1.0 / dl) * (l1 * Y + 0.5 * l2 * Y ** 2 + 0.5 * lam3 * (Y - y_up) ** 2)


class ValueNet3D(nn.Module):
    """v(logK,Z,Y) = v_mean + v_scale * b(sZ)*net([2sK-1,2sZ-1,2sY-1]) + logK_slope*(logK-logK_mid).

    LESSON FROM SMOKE TEST: the analytic -(1/delta) damage_offset over-states the Y-running-cost by
    ~10x (FD v is only 2.5-5.4, the analytic offset swings ~22 over Y) and DOMINATES, pinning vlK->1
    and breaking the warm-start fit. So we DROP the analytic v_base and instead use a correctly-scaled
    SOFT anchor: a learnable level/scale (calibrated to the v_ws statistics, set in calibrate()) plus a
    fixed leading logK slope (~vlK~0.5, the homogeneity slope) and let the net + value-L2 anchor learn
    the rest. b(sZ)=sZ(1-sZ) still softly de-weights the net on the Z-faces. This is the design's
    'lower-risk: drop the hard ansatz and rely on the value-L2 anchor' branch.
    """

    def __init__(self, width=32, depth=4, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        # calibrated level/scale + leading logK slope (set by calibrate(); defaults are FD-scale sane)
        self.register_buffer("v_mean", torch.tensor(4.0, dtype=dtype))
        self.register_buffer("v_scale", torch.tensor(1.0, dtype=dtype))
        self.register_buffer("logK_slope", torch.tensor(0.5, dtype=dtype))  # ~vlK homogeneity
        self.register_buffer("logK_mid", torch.tensor(0.5 * (LK_LO + LK_HI), dtype=dtype))
        self.to(dtype)

    def calibrate(self, v_ws):
        """Set the level/scale from the warm-start value so the net starts near the right magnitude."""
        self.v_mean.fill_(float(np.mean(v_ws)))
        self.v_scale.fill_(float(max(np.std(v_ws), 0.5)))

    def forward(self, logK, Z, Y):
        sK = (logK - LK_LO) / (LK_HI - LK_LO)
        sZ = (Z - Z_LO) / (Z_HI - Z_LO)
        sY = (Y - Y_LO) / (Y_HI - Y_LO)
        X = torch.cat([2 * sK - 1, 2 * sZ - 1, 2 * sY - 1], dim=1)
        # NO b(sZ) suppression: vZ ranges to ~3.5 and is the weakly-identified direction (Q1);
        # suppressing the net there would cripple exactly the slope we must resolve. Pure soft anchor.
        base = self.v_mean + self.logK_slope * (logK - self.logK_mid)
        return base + self.v_scale * self.net(X)


def autodiff_slopes(model, logK, Z, Y):
    """Return v, v_logK, v_Z, v_Y (create_graph=True so the FOC map stays differentiable)."""
    v = model(logK, Z, Y)
    g = torch.autograd.grad(v.sum(), [logK, Z, Y], create_graph=True)
    return v, g[0], g[1], g[2]


# =====================================================================================
#  Howard loop
# =====================================================================================
def make_grid(nK, nZ, nY, dtype):
    logK = np.linspace(LK_LO, LK_HI, nK)
    Z = np.linspace(Z_LO, Z_HI, nZ)
    Y = np.linspace(Y_LO, Y_HI, nY)
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    tt = dict(dtype=dtype)
    lk_t = torch.tensor(LK.ravel().reshape(-1, 1), **tt)
    z_t = torch.tensor(ZZ.ravel().reshape(-1, 1), **tt)
    y_t = torch.tensor(YY.ravel().reshape(-1, 1), **tt)
    return logK, Z, Y, (LK, ZZ, YY), (lk_t, z_t, y_t)


def slopes_to_controls_np(vlK, vZ, ZZ):
    qd = vlK - ZZ * vZ
    qg = vlK + (1 - ZZ) * vZ
    i_d, i_g, c = FD.controls(qd, qg, ZZ)
    return i_d, i_g, c, qd, qg


def run_egm_howard(seed, nK=21, nZ=31, nY=21, n_howard=12, fit_steps=1500,
                   warm_steps=1500, lr=2e-3, lam_v=1.0, ctrlfit=True,
                   dt=2.5, T=1200.0, dtype=torch.float32, verbose=True, d_fd=None):
    torch.manual_seed(seed); np.random.seed(seed)
    logK, Z, Y, (LK, ZZ, YY), (lk_t, z_t, y_t) = make_grid(nK, nZ, nY, dtype)
    ZZf = ZZ  # numpy meshgrid for oracle/controls
    model = ValueNet3D(dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def read_slopes():
        lk = lk_t.clone().requires_grad_(True)
        z = z_t.clone().requires_grad_(True)
        y = y_t.clone().requires_grad_(True)
        v, vlK, vZ, vY = autodiff_slopes(model, lk, z, y)
        return v, vlK, vZ, vY, lk, z, y

    # ---- warm start: cold policy (i_d=0, i_g=0.05) -> simulate_v -> fit net to v_ws (NO FD info) ----
    i_d0 = np.zeros((nK, nZ, nY)); i_g0 = np.full((nK, nZ, nY), 0.05)
    v_ws = FD.simulate_v(logK, Z, Y, i_d0, i_g0, LAM3, T=T, dt=dt)
    model.calibrate(v_ws)                      # set level/scale to the v_ws magnitude FIRST
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    v_ws_t = torch.tensor(v_ws.ravel().reshape(-1, 1), dtype=dtype)
    Nw = lk_t.shape[0]; bsw = min(2048, Nw)
    for _ in range(warm_steps):
        idx = torch.randint(0, Nw, (bsw,))
        opt.zero_grad()
        v = model(lk_t[idx], z_t[idx], y_t[idx])
        loss = ((v - v_ws_t[idx]) ** 2).mean()
        loss.backward(); opt.step()
    if verbose:
        print(f"  [seed {seed}] warm-start done, fit_err={loss.item():.2e} "
              f"(v_ws mean={np.mean(v_ws):.2f} std={np.std(v_ws):.2f})", flush=True)

    ik = np.argmin(np.abs(logK - np.log(880))); jz = np.argmin(np.abs(Z - 0.7)); ky = np.argmin(np.abs(Y - 3.0))

    for sweep in range(n_howard):
        # (1)+(2) read autodiff slopes -> FOC controls on the grid
        with torch.no_grad():
            pass
        v_d, vlK_d, vZ_d, vY_d, _, _, _ = read_slopes()
        vlK_np = vlK_d.detach().numpy().reshape(nK, nZ, nY)
        vZ_np = vZ_d.detach().numpy().reshape(nK, nZ, nY)
        i_d_g, i_g_g, c_g, qd_g, qg_g = slopes_to_controls_np(vlK_np, vZ_np, ZZf)
        # (3) FREEZE -> oracle policy-eval value v_pe
        v_pe = FD.simulate_v(logK, Z, Y, i_d_g, i_g_g, LAM3, T=T, dt=dt)
        v_pe_t = torch.tensor(v_pe.ravel().reshape(-1, 1), dtype=dtype)
        # frozen oracle controls (targets) -- recomputed from the SAME slopes via the exact map
        i_d_pe_t = torch.tensor(i_d_g.ravel().reshape(-1, 1), dtype=dtype)
        i_g_pe_t = torch.tensor(i_g_g.ravel().reshape(-1, 1), dtype=dtype)

        # (4) ctrlfit / value-L2 supervise toward v_pe (minibatched over collocation for CPU speed)
        N = lk_t.shape[0]
        bs = min(2048, N)
        for _ in range(fit_steps):
            idx = torch.randint(0, N, (bs,))
            opt.zero_grad()
            lk = lk_t[idx].clone().requires_grad_(True)
            z = z_t[idx].clone().requires_grad_(True)
            y = y_t[idx].clone().requires_grad_(True)
            v, vlK, vZ, vY = autodiff_slopes(model, lk, z, y)
            loss = lam_v * ((v - v_pe_t[idx]) ** 2).mean()
            if ctrlfit:
                qd = vlK - z * vZ
                qg = vlK + (1 - z) * vZ
                i_d_nn, i_g_nn, _ = controls_torch(qd, qg, z)
                loss = loss + ((i_d_nn - i_d_pe_t[idx]) ** 2).mean() + ((i_g_nn - i_g_pe_t[idx]) ** 2).mean()
            loss.backward(); opt.step()

        if verbose:
            extra = ""
            if d_fd is not None and (sweep % 3 == 0 or sweep == n_howard - 1):
                cur = dict(logK=logK, Z=Z, Y=Y)
                cur["i_d"] = i_d_g; cur["i_g"] = i_g_g; cur["vlK"] = vlK_np; cur["vZ"] = vZ_np
                cur["v"] = model(lk_t, z_t, y_t).detach().numpy().reshape(nK, nZ, nY)
                e = true_error(cur, d_fd)
                extra = (f" | TRUEerr i_d={e['err_i_d']:.2e} i_g={e['err_i_g']:.2e} "
                         f"v={e['err_v']:.2e} vlK={e['err_vlK']:.2e} vZ={e['err_vZ']:.2e}")
            print(f"  [seed {seed} sweep {sweep:2d}] loss={loss.item():.3e} "
                  f"i_d(ref)={i_d_g[ik,jz,ky]:+.4f} i_g(ref)={i_g_g[ik,jz,ky]:+.4f} "
                  f"vlK={vlK_np[ik,jz,ky]:.3f} vZ={vZ_np[ik,jz,ky]:.3f} "
                  f"frac_qd<floor={np.mean(qd_g<QFLOOR):.3f}{extra}", flush=True)

    # final fields on the grid
    v_d, vlK_d, vZ_d, vY_d, _, _, _ = read_slopes()
    vlK_np = vlK_d.detach().numpy().reshape(nK, nZ, nY)
    vZ_np = vZ_d.detach().numpy().reshape(nK, nZ, nY)
    v_np = v_d.detach().numpy().reshape(nK, nZ, nY)
    i_d_g, i_g_g, c_g, qd_g, qg_g = slopes_to_controls_np(vlK_np, vZ_np, ZZf)
    return dict(logK=logK, Z=Z, Y=Y, v=v_np, i_d=i_d_g, i_g=i_g_g, c=c_g,
                vlK=vlK_np, vZ=vZ_np)


# =====================================================================================
#  Validation vs FD truth
# =====================================================================================
def interp_to_fd(out, key, d):
    """Interpolate an EGM field (on its own coarse grid) to the FD grid for a clean comparison."""
    from scipy.interpolate import RegularGridInterpolator as RGI
    f = RGI((out["logK"], out["Z"], out["Y"]), out[key],
            bounds_error=False, fill_value=None)
    LK, ZZ, YY = np.meshgrid(d["logK"], d["Z"], d["Y"], indexing="ij")
    q = np.stack([LK.ravel(), ZZ.ravel(), YY.ravel()], axis=1)
    return f(q).reshape(LK.shape)


def true_error(out, d, box=None):
    """max|.-FD| over the interior box. Returns dict of errors."""
    if box is None:
        # interior: drop boundary layers; restrict to the calibrated, non-corner box
        box = (slice(2, -2), slice(8, -8), slice(2, -2))  # Z~[0.15,0.85], logK,Y interior
    res = {}
    for k in ("i_d", "i_g", "v", "vlK", "vZ"):
        a = interp_to_fd(out, k, d)
        res["err_" + k] = float(np.max(np.abs(a - d[k])[box]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--nK", type=int, default=21)
    ap.add_argument("--nZ", type=int, default=31)
    ap.add_argument("--nY", type=int, default=21)
    ap.add_argument("--howard", type=int, default=12)
    ap.add_argument("--fit", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=1500)
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--only", choices=["ctrlfit", "plain", "both"], default="both")
    args = ap.parse_args()
    dtype = torch.float64 if args.float64 else torch.float32

    d = np.load(os.path.join(HERE, "outputs", "fd_pdpt_v5_lam3_0167_xi148.npz"))
    d = {k: d[k] for k in d.files}

    print(f"=== EGM-Howard NN, 3-D post-damage-post-tech === dtype={dtype}", flush=True)
    print(f"grid {args.nK}x{args.nZ}x{args.nY}, {args.howard} Howard sweeps, seeds {args.seeds}", flush=True)

    ablations = [("CTRLFIT (ctrl-space + value anchor)", True),
                 ("PLAIN value-L2 only", False)]
    if args.only == "ctrlfit":
        ablations = ablations[:1]
    elif args.only == "plain":
        ablations = ablations[1:]
    for label, ctrlfit in ablations:
        print(f"\n######## {label} ########", flush=True)
        errs = {k: [] for k in ("err_i_d", "err_i_g", "err_v", "err_vlK", "err_vZ")}
        for seed in args.seeds:
            t0 = time.time()
            out = run_egm_howard(seed, nK=args.nK, nZ=args.nZ, nY=args.nY,
                                 n_howard=args.howard, fit_steps=args.fit,
                                 warm_steps=args.warm, ctrlfit=ctrlfit, dt=args.dt,
                                 dtype=dtype, verbose=True, d_fd=d)
            e = true_error(out, d)
            for k in errs:
                errs[k].append(e[k])
            print(f"  -> [seed {seed}] {time.time()-t0:.0f}s  "
                  f"max|i_d-FD|={e['err_i_d']:.3e} max|i_g-FD|={e['err_i_g']:.3e} "
                  f"max|v-FD|={e['err_v']:.3e} max|vlK-FD|={e['err_vlK']:.3e} "
                  f"max|vZ-FD|={e['err_vZ']:.3e}", flush=True)
        print(f"  === {label} SUMMARY (median [min,max] over seeds) ===", flush=True)
        for k in ("err_i_d", "err_i_g", "err_v", "err_vlK", "err_vZ"):
            a = np.array(errs[k])
            print(f"    {k}: median={np.median(a):.3e}  [{a.min():.3e}, {a.max():.3e}]", flush=True)


if __name__ == "__main__":
    main()
