"""
spectral_value_ab.py -- P2 paradigm: SPECTRAL / random-Fourier-feature VALUE REPRESENTATION on the
3-D post-damage-post-tech stepping stone (state = logK, Z, Y; terminal/no-jumps).

WHY (the paradigm bet): the ~1e-3 floor is NOT a fixed-point error -- the operator is uniformly
elliptic, so the value AND its costates are well-posed. The floor is the AUTODIFF-SLOPE NOISE of a
tanh-MLP: a deep tanh net has SPECTRAL BIAS (Rahaman 2019) -- it fits v to ~1e-3 but its first
derivative carries ~1e-3 high-frequency ripple, and the FOC map amplifies that ripple into the
controls. A SPECTRAL value representation removes that ripple by construction: v is a finite sum of
smooth basis functions whose derivatives are EXACT (RFF: d/dx[a*cos(w.x+b)] = -a*w*sin(...); Chebyshev:
T_n' is a closed recurrence), so the costate is as clean as the value -- there is no "differentiate-a-
noisy-net" amplification.

This file is a DROP-IN replacement for ValueNet3D in egm_howard_ctrlfit_ab.py: it reuses the SAME
Howard+ctrlfit pipeline (FD.simulate_v oracle, FD.controls FOC map, stable_fd_eval.grade). Only the
value PARAMETERIZATION changes. Two representations are provided behind one interface:

  REP = "rff"  : Random Fourier Features (Tancik 2020). v(x) = base(x) + s * W2 @ phi(x),
                 phi = [cos(B x), sin(B x)] / sqrt(m), B ~ N(0, diag(sigma_band)^2). The bandwidth
                 sigma_band is CALIBRATED per-axis (not isotropic): logK,Z,Y have very different
                 effective frequencies (Z is the weakly-identified, advection-dominated direction with
                 the sharpest v_Z; Y carries the smooth damage trend). Calibrated bandwidth + value
                 anchor is the design's headline -- it avoids the two known traps:
                   * SIREN over-deepening (the leaderboard SIREN arm lost): we use a SHALLOW linear
                     read-out over fixed features (a least-squares-like head), NOT a deep sinusoidal
                     net, so there is no depth to over-fit and the derivative stays a clean finite sum.
                   * isotropic bandwidth: too low -> can't resolve v_Z; too high -> derivative noise
                     returns. Per-axis calibration from the warm-start v_ws sidesteps both.

  REP = "cheb" : Chebyshev tensor-product spectral basis. v(x) = base(x) + sum_{ijk} c_{ijk}
                 T_i(2sK-1) T_j(2sZ-1) T_k(2sY-1). T_n and T_n' via the standard recurrences (NO
                 autodiff through a recurrence is needed for the value; we still use torch autograd for
                 the costate so the ctrlfit FOC map stays differentiable, but the graph is a smooth
                 polynomial -> machine-clean derivative). Spectral (exponential) convergence on a
                 smooth v; orders ~ (12,16,10) over (logK,Z,Y) resolve the box with ~2000 coeffs.

BOUNDARY ANSATZ + VALUE ANCHOR (shared, ported from ValueNet3D):
  base(x) = v_mean + logK_slope*(logK - logK_mid)   [the homogeneity slope v_logK~0.5 carried exactly;
  the spectral head learns the residual]. v_mean,v_scale,logK_slope are calibrated from v_ws statistics
  (NO FD info). The value-L2 anchor to the frozen-policy oracle v_pe is kept (lam_v=1.0) -- it is what
  pins the LEVEL; the spectral basis then resolves the SLOPE cleanly.

GATE (unchanged, the only thing that counts): stable_fd_eval.grade -> box_err_i_d/i_g and de-invest
vZ error vs the STABLE FD npz. Multi-seed >=3 (seed = RFF frequency draw B; Chebyshev is
deterministic so its "seeds" perturb only the init + minibatch order, a genuine robustness check).
NEVER v / loss.

RUN:
  cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/post_damage_post_tech
  module load python/anaconda-2021.05
  srun --account=pi-lhansen --partition=caslake --time=1:00:00 --cpus-per-task=4 --mem=24G \
       python spectral_value_ab.py --rep rff  --seeds 1 2 3
  srun ... python spectral_value_ab.py --rep cheb --seeds 1 2 3
"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd_pdpt_v5 as FD                       # simulate_v, controls, P, QFLOOR  (the SAME oracle/FOC map)
from egm_howard_ctrlfit_ab import (           # reuse the proven pipeline verbatim
    controls_torch, make_grid, slopes_to_controls_np,
    LK_LO, LK_HI, Z_LO, Z_HI, Y_LO, Y_HI, LAM3,
)
from stable_fd_eval import grade, load_stable_fd, summarize

QFLOOR = FD.QFLOOR


# =====================================================================================
#  SPECTRAL value representations (the ONLY change vs the incumbent)
# =====================================================================================
class _SpectralBase(nn.Module):
    """Shared boundary ansatz + value anchor; subclasses add the spectral head."""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        self.register_buffer("v_mean", torch.tensor(4.0, dtype=dtype))
        self.register_buffer("v_scale", torch.tensor(1.0, dtype=dtype))
        self.register_buffer("logK_slope", torch.tensor(0.5, dtype=dtype))  # homogeneity slope
        self.register_buffer("logK_mid", torch.tensor(0.5 * (LK_LO + LK_HI), dtype=dtype))

    def calibrate(self, v_ws):
        self.v_mean.fill_(float(np.mean(v_ws)))
        self.v_scale.fill_(float(max(np.std(v_ws), 0.5)))

    def _scaled(self, logK, Z, Y):
        sK = (logK - LK_LO) / (LK_HI - LK_LO)
        sZ = (Z - Z_LO) / (Z_HI - Z_LO)
        sY = (Y - Y_LO) / (Y_HI - Y_LO)
        return sK, sZ, sY

    def _base(self, logK):
        return self.v_mean + self.logK_slope * (logK - self.logK_mid)


class RFFValue(_SpectralBase):
    """Random Fourier Features with PER-AXIS calibrated bandwidth + shallow linear read-out.

    v = base + v_scale * head( [cos(X@B^T + b), sin(X@B^T + b)] / sqrt(m) ),  head = single Linear(2m,1).
    B is FIXED (frequency draw = the multi-seed knob). Derivatives are exact finite sums:
       dv/dx_j = v_scale/sqrt(m) * sum_k head_k * (-sin/+cos)(...) * B_{k,j}
    -> the costate has the SAME smoothness as v; no autodiff-noise amplification. We still call
    torch.autograd for the costate (to keep the ctrlfit FOC map differentiable end-to-end), but the
    graph is a single smooth sinusoid layer, so the slope is machine-clean.
    """
    def __init__(self, m=512, band=(2.5, 6.0, 2.0), seed=0, dtype=torch.float64):
        super().__init__(dtype)
        # band = per-axis std of the (scaled-to-[0,1]) frequencies. Z gets the LARGEST band:
        # it is the weakly-identified, sharp-v_Z direction (cell-Peclet 47-139 in FD). Y the smallest
        # (smooth damage trend). Calibrated below from v_ws spectral content; these are sane defaults.
        g = torch.Generator().manual_seed(seed)
        band_t = torch.tensor(band, dtype=dtype)
        B = torch.randn(m, 3, generator=g, dtype=dtype) * band_t  # (m,3) frequencies, per-axis scaled
        b = 2 * np.pi * torch.rand(m, 1, generator=g, dtype=dtype).squeeze(1)
        self.register_buffer("B", B)
        self.register_buffer("phase", b)
        self.m = m
        self.head = nn.Linear(2 * m, 1, bias=True).to(dtype)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)  # start at base() then learn

    def calibrate_band(self, v_ws_grid, axes):
        """OPTIONAL: rescale per-axis bandwidth from the warm-start value's empirical spectrum so the
        top resolved frequency matches v_ws's roll-off per axis (keeps Z rich, Y modest). Uses only
        v_ws (NO FD info). Recomputes B in place keeping the same random directions."""
        # crude per-axis dominant-frequency estimate via FFT of the mean profile along each axis
        prof = [v_ws_grid.mean(axis=tuple(j for j in range(3) if j != a)) for a in range(3)]
        scales = []
        for pr in prof:
            sp = np.abs(np.fft.rfft(pr - pr.mean()))
            k = np.argmax(sp) + 1 if sp.size > 1 else 1
            scales.append(max(1.0, float(k)))
        s = torch.tensor(scales, dtype=self.dtype)
        # renormalize columns of B to unit per-axis std, then apply the calibrated scale
        col_std = self.B.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.B.copy_((self.B / col_std) * s)

    def forward(self, logK, Z, Y):
        sK, sZ, sY = self._scaled(logK, Z, Y)
        X = torch.cat([sK, sZ, sY], dim=1)            # (N,3) in [0,1]
        proj = X @ self.B.t() + self.phase            # (N,m)
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=1) / np.sqrt(self.m)
        return self._base(logK) + self.v_scale * self.head(phi)


def _cheb_basis(x, n):
    """Chebyshev T_0..T_{n-1} evaluated at x in [-1,1], differentiable (recurrence in torch).
    Returns (N, n). T_0=1, T_1=x, T_{k+1}=2x T_k - T_{k-1}. Autograd through the recurrence gives the
    EXACT T_n' (a smooth polynomial), so the costate is machine-clean."""
    cols = [torch.ones_like(x), x]
    for k in range(2, n):
        cols.append(2 * x * cols[-1] - cols[-2])
    return torch.cat(cols, dim=1)


class ChebValue(_SpectralBase):
    """Tensor-product Chebyshev spectral value. Coeffs are the ONLY trainable head; the basis is a
    smooth polynomial -> exact derivatives -> clean costate. Orders (oK,oZ,oY); oZ largest for the
    sharp-v_Z direction."""
    def __init__(self, orders=(12, 16, 10), seed=0, dtype=torch.float64):
        super().__init__(dtype)
        self.oK, self.oZ, self.oY = orders
        torch.manual_seed(seed)
        ncoef = self.oK * self.oZ * self.oY
        self.coef = nn.Parameter(1e-3 * torch.randn(ncoef, 1, dtype=dtype))  # tiny init -> start at base()

    def forward(self, logK, Z, Y):
        sK, sZ, sY = self._scaled(logK, Z, Y)
        xK, xZ, xY = 2 * sK - 1, 2 * sZ - 1, 2 * sY - 1
        TK = _cheb_basis(xK, self.oK)                 # (N,oK)
        TZ = _cheb_basis(xZ, self.oZ)
        TY = _cheb_basis(xY, self.oY)
        # tensor product features (N, oK*oZ*oY) via broadcasting outer products
        feat = (TK.unsqueeze(2).unsqueeze(3) *
                TZ.unsqueeze(1).unsqueeze(3) *
                TY.unsqueeze(1).unsqueeze(2)).reshape(logK.shape[0], -1)
        return self._base(logK) + self.v_scale * (feat @ self.coef)


def build_value(rep, seed, dtype):
    if rep == "rff":
        return RFFValue(m=512, seed=seed, dtype=dtype)
    if rep == "cheb":
        return ChebValue(orders=(12, 16, 10), seed=seed, dtype=dtype)
    raise ValueError(rep)


# =====================================================================================
#  Howard + ctrlfit loop  (IDENTICAL to egm_howard_ctrlfit_ab; only build_value differs)
# =====================================================================================
def autodiff_slopes(model, logK, Z, Y):
    v = model(logK, Z, Y)
    g = torch.autograd.grad(v.sum(), [logK, Z, Y], create_graph=True)
    return v, g[0], g[1], g[2]


def run_spectral_howard(seed, rep="rff", nK=21, nZ=31, nY=21, n_howard=12,
                        fit_steps=1500, warm_steps=1500, lr=2e-3, lam_v=1.0,
                        ctrlfit=True, dt=2.5, T=1200.0, dtype=torch.float64, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    logK, Z, Y, (LK, ZZ, YY), (lk_t, z_t, y_t) = make_grid(nK, nZ, nY, dtype)
    ZZf = ZZ
    model = build_value(rep, seed, dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- warm start: cold policy -> simulate_v -> fit spectral head to v_ws (NO FD info) ----
    i_d0 = np.zeros((nK, nZ, nY)); i_g0 = np.full((nK, nZ, nY), 0.05)
    v_ws = FD.simulate_v(logK, Z, Y, i_d0, i_g0, LAM3, T=T, dt=dt)
    model.calibrate(v_ws)
    if rep == "rff":
        model.calibrate_band(v_ws, (logK, Z, Y))      # per-axis bandwidth from v_ws spectrum
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
        print(f"  [seed {seed} rep={rep}] warm fit_err={loss.item():.2e} "
              f"(v_ws mean={np.mean(v_ws):.2f} std={np.std(v_ws):.2f})", flush=True)

    for sweep in range(n_howard):
        lk = lk_t.clone().requires_grad_(True); z = z_t.clone().requires_grad_(True); y = y_t.clone().requires_grad_(True)
        v_d, vlK_d, vZ_d, _ = autodiff_slopes(model, lk, z, y)
        vlK_np = vlK_d.detach().numpy().reshape(nK, nZ, nY)
        vZ_np = vZ_d.detach().numpy().reshape(nK, nZ, nY)
        i_d_g, i_g_g, c_g, qd_g, qg_g = slopes_to_controls_np(vlK_np, vZ_np, ZZf)
        v_pe = FD.simulate_v(logK, Z, Y, i_d_g, i_g_g, LAM3, T=T, dt=dt)
        v_pe_t = torch.tensor(v_pe.ravel().reshape(-1, 1), dtype=dtype)
        i_d_pe_t = torch.tensor(i_d_g.ravel().reshape(-1, 1), dtype=dtype)
        i_g_pe_t = torch.tensor(i_g_g.ravel().reshape(-1, 1), dtype=dtype)

        N = lk_t.shape[0]; bs = min(2048, N)
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
            print(f"  [seed {seed} sweep {sweep:2d}] loss={loss.item():.3e} "
                  f"frac_qd<floor={np.mean(qd_g < QFLOOR):.3f}", flush=True)

    lk = lk_t.clone().requires_grad_(True); z = z_t.clone().requires_grad_(True); y = y_t.clone().requires_grad_(True)
    v_d, vlK_d, vZ_d, _ = autodiff_slopes(model, lk, z, y)
    vlK_np = vlK_d.detach().numpy().reshape(nK, nZ, nY)
    vZ_np = vZ_d.detach().numpy().reshape(nK, nZ, nY)
    v_np = v_d.detach().numpy().reshape(nK, nZ, nY)
    i_d_g, i_g_g, c_g, qd_g, qg_g = slopes_to_controls_np(vlK_np, vZ_np, ZZf)
    return dict(logK=logK, Z=Z, Y=Y, v=v_np, i_d=i_d_g, i_g=i_g_g, c=c_g, vlK=vlK_np, vZ=vZ_np)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["rff", "cheb"], default="rff")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--nK", type=int, default=21)
    ap.add_argument("--nZ", type=int, default=31)
    ap.add_argument("--nY", type=int, default=21)
    ap.add_argument("--howard", type=int, default=12)
    ap.add_argument("--fit", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=1500)
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()
    dtype = torch.float32 if args.float32 else torch.float64  # spectral basis benefits from float64

    d = load_stable_fd()
    print(f"=== P2 SPECTRAL value ({args.rep}) vs STABLE FD === dtype={dtype}", flush=True)
    print(f"grid {args.nK}x{args.nZ}x{args.nY}, {args.howard} Howard sweeps, seeds {args.seeds}", flush=True)
    per_seed = []
    for seed in args.seeds:
        t0 = time.time()
        out = run_spectral_howard(seed, rep=args.rep, nK=args.nK, nZ=args.nZ, nY=args.nY,
                                  n_howard=args.howard, fit_steps=args.fit, warm_steps=args.warm,
                                  dtype=dtype, verbose=True)
        m = grade(out, d)
        per_seed.append(m)
        print(f"  -> [seed {seed}] {time.time()-t0:.0f}s  box_i_d={m['box_err_i_d']:.3e} "
              f"box_i_g={m['box_err_i_g']:.3e} di_vZ={m['di_err_vZ']:.3e} "
              f"depth_gap={m['di_match_depth']:.3e}", flush=True)
    agg = summarize(per_seed, label=f"P2-spectral-{args.rep}")
    np.savez(os.path.join(HERE, "outputs", f"leaderboard_spectral_{args.rep}.npz"),
             **{f"{k}_{stat}": v for k, sv in agg.items() for stat, v in sv.items()})
    print(f"\nsaved -> outputs/leaderboard_spectral_{args.rep}.npz", flush=True)


if __name__ == "__main__":
    main()
