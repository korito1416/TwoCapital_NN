"""
costate_trial.py  --  Q5 (the user's priority): COSTATE / EGM-style level recovery.

IDEA (weak-identification of the value LEVEL):
  The baseline value net predicts the TRANSFORMED scalar v = V - logN(Y) directly.
  The HJB only pins v's LEVEL through the -delta*v term (delta=0.01), so the level is
  identified only to ~residual/delta and wanders by init (level_spread ~ O(1)).

  Q5 instead has the net predict the DIFFERENTIALS (costates p = (dv/dlogK, dv/dZ,
  dv/dY)) and RECOVERS the level by integrating the costate field along a straight
  path from a single fixed anchor x0 (EGM-style "solve the ODE to back out the level"):

        v(s) = v0 + INT_0^1  p(x0 + t*(s-x0)) . (s-x0)  dt

  with v0 = the TF surrogate's transformed-v at x0 (THE single pinned constant).
  The line integral is path-well-defined only if p is conservative (symmetric
  Jacobian), so we ADD an integrability (curl) penalty.

  Because the level is now (anchor constant) + (a deterministic integral of the
  network's OWN derivative field), the LEVEL should be pinned: any two seeds that
  learn the same costate field recover the same level. We test whether level_spread
  over 3 seeds collapses WITHOUT inflating loss_v / FOC.

CONSISTENCY WITH THE EXISTING ECONOMICS:
  models_torch/PostDamagePostTech.py::pde_rhs is written in the TRANSFORMED v
  (its `v` = V - logN; it explicitly re-adds the logN slope inside h_y as
  dv_dY - (lam1 + lam2*Y + lam3*(Y - y_upper))). So the costates we model and
  integrate are the TRANSFORMED-v costates; we recover transformed v and feed
  the SAME economics. LEVEL (per the protocol, V = pv/delta with pv = delta*v) is
  therefore the recovered transformed-v level -- exactly the quantity that drifts.

DESIGN ACTUALLY IMPLEMENTED (see honest notes at bottom of file):
  * Costate net  C(X) -> (p_logK, p_Z, p_Y), FeedForwardSubNet backbone, out dim 3,
    LINEAR final activation, input width 7 (same X layout as the baseline).
  * Level recovery by 8-point Gauss-Legendre line integral from a fixed anchor x0
    (box center for logK,Z,Y), holding the NON-differentiated inputs (lam3, A_g'',
    logxi) fixed at the evaluation point's values along the path.
  * HJB residual = the SAME pde_rhs economics, but with the net costates substituted
    for (dv_dlogK, dv_dZ, dv_dY), the integrated v for the level term, and AUTOGRAD
    of the costate heads for the needed second derivatives (d p_logK/dlogK, etc.).
  * Integrability penalty lambda_curl * mean[ (dp_logK/dZ - dp_Z/dlogK)^2
                                            + (dp_logK/dY - dp_Y/dlogK)^2
                                            + (dp_Z/dY   - dp_Y/dZ)^2 ].
  * Controls i_g, i_d from their own nets via the SAME FOCs.
  * Loss: VALUE/COSTATE step (opt_v, costate net only) = rms(HJB residual)
    + lambda_curl*rms(curl). CONTROL step (opt_c, i_g/i_d only) = rms(FOC_d)
    + rms(FOC_g) + rms(dv_dY above y_upper). The costate net owns the residual and
    the integrability term; the controls own the FOCs. Gradients are clipped
    (max-norm) because the FOC = delta/c blows up early when the random-init
    controls push consumption to the clamp -- mirrors the baseline harness but the
    clip keeps the from-scratch trajectory finite.

STANDARD METRIC PROTOCOL: from scratch, 3 seeds {0,1,2}, N steps, batch 128, Adam,
WarmupCosine peak 4e-4. Probe = 256 fixed pts (gen seed 12345). LEVEL = mean over
probe of recovered v; level_spread = std over 3 seeds of LEVEL. Prints RESULT {dict}.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
HERE = os.path.join(ROOT, "models_torch_train")
MT = os.path.join(ROOT, "models_torch")
for _p in (HERE, MT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from params import PARAMS, investment_rate_activation  # noqa: E402
from tf_to_torch_loader import (  # noqa: E402
    build_root_torch_model, build_trainable_net, make_root_configs,
)
from train_root import sample_box, WarmupCosine  # noqa: E402

DELTA = float(PARAMS["δ"])
NPROBE = 256


# ---------------------------------------------------------------------------
#  Anchor + the single pinned constant v0 = surrogate transformed-v at x0
# ---------------------------------------------------------------------------
def anchor_point(params, dtype):
    """Fixed anchor x0 = box center for the 3 differentiated states."""
    x0_logK = 0.5 * (params["logK_min"] + params["logK_max"])
    x0_Z = 0.5 * (params["Z_min"] + params["Z_max"])
    x0_Y = 0.5 * (params["Y_min"] + params["Y_max"])
    return (torch.tensor([[x0_logK]], dtype=dtype),
            torch.tensor([[x0_Z]], dtype=dtype),
            torch.tensor([[x0_Y]], dtype=dtype))


def surrogate_v0(params, dtype, x0):
    """Read the TF-surrogate transformed-v at the anchor x0.

    The pseudo-states (lam3, logxi) are held at box centers for the anchor read;
    this single scalar is the ONLY pinned constant in the whole construction.
    """
    sur = build_root_torch_model(load_surrogate=True, dtype=dtype)
    sur.v_nn.eval()
    lam3 = torch.tensor([[0.5 * (params["λ3_min"] + params["λ3_max"])]], dtype=dtype)
    logxi = torch.tensor([[0.5 * (params["logξ_min"] + params["logξ_max"])]], dtype=dtype)
    A_g_pp = params["A_g_prime_prime"]
    X = torch.cat([x0[0], x0[1], x0[2], lam3,
                   A_g_pp * torch.ones_like(x0[2]), logxi, logxi], dim=1)
    with torch.no_grad():
        v0 = sur.v_nn(X).reshape(())
    return v0.detach()


# ---------------------------------------------------------------------------
#  Costate net wrapper + line-integral level recovery
# ---------------------------------------------------------------------------
class CostateModel:
    """Net predicts costates (p_logK,p_Z,p_Y); level recovered by line integral."""

    def __init__(self, params, dtype, seed):
        self.params = params
        self.dtype = dtype
        # costate net: same backbone, 3 linear outputs.
        v_cfg, ig_cfg, id_cfg = make_root_configs()
        c_cfg = dict(v_cfg)
        c_cfg["dim"] = 3
        c_cfg["activation"] = "tanh"          # smooth hidden act for derivative field
        c_cfg["final_activation"] = "linear"  # costates are unbounded reals
        c_cfg["nn_name"] = "costate_nn"
        self.c_nn = build_trainable_net(c_cfg, input_dim=7, seed=seed)

        # control nets (bounded custom activation, like baseline)
        ig = dict(ig_cfg); ig["final_activation"] = investment_rate_activation(params["θ_g"])
        idd = dict(id_cfg); idd["final_activation"] = investment_rate_activation(params["θ_d"])
        self.i_g_nn = build_trainable_net(ig, input_dim=7, seed=None if seed is None else seed + 101)
        self.i_d_nn = build_trainable_net(idd, input_dim=7, seed=None if seed is None else seed + 202)

        for net in (self.c_nn, self.i_g_nn, self.i_d_nn):
            net.to(dtype=dtype); net.train()

        # 8-point Gauss-Legendre nodes/weights on [0,1]
        nodes, weights = np.polynomial.legendre.leggauss(8)
        self.gl_t = torch.tensor(0.5 * (nodes + 1.0), dtype=dtype)      # (8,)
        self.gl_w = torch.tensor(0.5 * weights, dtype=dtype)           # (8,)

        self.x0 = anchor_point(params, dtype)
        self.v0 = surrogate_v0(params, dtype, self.x0)

    # ------------------------------------------------------------------
    def _costates(self, logK, Z, Y, lam3, A_g_pp, logxi):
        """Costate net forward at a batch of (logK,Z,Y) with given pseudo-states."""
        X = torch.cat([logK, Z, Y, lam3,
                       A_g_pp * torch.ones_like(Y), logxi, logxi], dim=1)
        p = self.c_nn(X)  # (N,3)
        return p[:, 0:1], p[:, 1:2], p[:, 2:3]

    def recover_v(self, logK, Z, Y, lam3, A_g_pp, logxi):
        """v(s) = v0 + INT_0^1 p(x0+t(s-x0)).(s-x0) dt  via 8-pt Gauss-Legendre.

        Vectorised over the batch; the line integral runs along the 3 differentiated
        states (logK,Z,Y). Pseudo-states (lam3, A_g'', logxi) are held at the
        evaluation point's values along the path.
        """
        N = logK.shape[0]
        x0K, x0Z, x0Y = self.x0[0], self.x0[1], self.x0[2]
        dK = logK - x0K   # (N,1)
        dZ = Z - x0Z
        dY = Y - x0Y
        acc = torch.zeros((N, 1), dtype=self.dtype)
        for t, w in zip(self.gl_t, self.gl_w):
            Kt = x0K + t * dK
            Zt = x0Z + t * dZ
            Yt = x0Y + t * dY
            pK, pZ, pY = self._costates(Kt, Zt, Yt, lam3, A_g_pp, logxi)
            integrand = pK * dK + pZ * dZ + pY * dY    # p . (s - x0)
            acc = acc + w * integrand
        return self.v0 + acc

    # ------------------------------------------------------------------
    def pde_terms(self, logK, Z, Y, logR, lam3, logxi, want_curl=False):
        """HJB residual (rhs - pv) + FOCs + optional integrability penalty.

        Mirrors models_torch/PostDamagePostTech.py::pde_rhs economics, but
        substitutes the net costates for first derivatives, the integrated v for
        the level, and autograd-of-costates for the needed second derivatives.
        """
        p = self.params
        A_d = p['A_d']; A_g_pp = p['A_g_prime_prime']
        δ = p['δ']
        α_d = p['α_d']; Γ_d = p['Γ_d']; θ_d = p['θ_d']; σ_d = p['σ_d']
        α_g = p['α_g']; Γ_g = p['Γ_g']; θ_g = p['θ_g']; σ_g = p['σ_g']
        θ_bar = p['θ_bar']; η = p['η']; ϛ = p['ϛ']
        λ1 = p['λ1']; λ2 = p['λ2']
        y_upper = p['y_upper']

        # leaf states (autograd targets for the second derivatives)
        logK = logK.detach().requires_grad_(True)
        Z = Z.detach().requires_grad_(True)
        Y = Y.detach().requires_grad_(True)

        # first derivatives  <-  NET COSTATES (with create_graph for 2nd derivs)
        dv_dlogK, dv_dZ, dv_dY = self._costates(logK, Z, Y, lam3, A_g_pp, logxi)

        def d(out, x):
            g = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out),
                                    create_graph=True, retain_graph=True,
                                    allow_unused=True)[0]
            return torch.zeros_like(x) if g is None else g.reshape(-1, 1)

        # second derivatives via autograd of the costate heads
        d2v_dlogK2 = d(dv_dlogK, logK)
        d2v_dlogKdZ = d(dv_dlogK, Z)
        d2v_dZ2 = d(dv_dZ, Z)
        d2v_dY2 = d(dv_dY, Y)

        # recovered (transformed) v for the level term -delta*v
        v = self.recover_v(logK, Z, Y, lam3, A_g_pp, logxi)

        # controls
        X = torch.cat([logK, Z, Y, lam3,
                       A_g_pp * torch.ones_like(Y), logxi, logxi], dim=1)
        i_g = self.i_g_nn(X)
        i_d = self.i_d_nn(X)

        ξ = torch.exp(logxi)
        K = torch.exp(logK)

        # ---- drift distortions (regime-correct V_y inside h_y) ----
        h_d = -1.0 / ξ * ((dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d)
        h_g = -1.0 / ξ * ((dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g)
        h_y = -1.0 / ξ * (dv_dY - (λ1 + λ2 * Y + lam3 * (Y - y_upper))) \
            * η * A_d * (1 - Z) * K * ϛ

        pv = δ * v

        c = (A_d - i_d) * (1 - Z) + (A_g_pp - i_g) * Z
        inside_log = torch.clamp(c, min=1e-8).reshape(-1, 1)
        flow = δ * (torch.log(inside_log) + logK)

        v_logKlogK_term = (σ_d ** 2 * (1 - Z) ** 2 + σ_g ** 2 * Z ** 2) / 2.0
        inside_log_i_d = torch.clamp(1.0 + θ_d * i_d, min=1e-8).reshape(-1, 1)
        inside_log_i_g = torch.clamp(1.0 + θ_g * i_g, min=1e-8).reshape(-1, 1)

        v_logK_term = (α_d + Γ_d * torch.log(inside_log_i_d)) * (1 - Z) \
            + (α_g + Γ_g * torch.log(inside_log_i_g)) * Z \
            - v_logKlogK_term
        v_Z_term = (α_g + Γ_g * torch.log(inside_log_i_g)
                    - (α_d + Γ_d * torch.log(inside_log_i_d))
                    - Z * σ_g ** 2 + (1 - Z) * σ_d ** 2) * Z * (1 - Z)
        v_ZZ_term = 0.5 * Z ** 2 * (1 - Z) ** 2 * (σ_g ** 2 + σ_d ** 2)
        v_logK_Z_term = -Z * (1 - Z) ** 2 * σ_d ** 2 + Z ** 2 * (1.0 - Z) * σ_g ** 2
        v_y_term = (θ_bar + h_y * ϛ) * η * A_d * (1 - Z) * K
        v_yy_term = 0.5 * ϛ ** 2 * (η * A_d * (1 - Z) * K) ** 2
        v_logN_term = (λ1 + λ2 * Y + lam3 * (Y - y_upper)) * v_y_term + (λ2 + lam3) * v_yy_term

        rhs = flow \
            + v_logK_term * dv_dlogK + v_logKlogK_term * d2v_dlogK2 \
            + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
            + h_d * (dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d \
            + h_g * (dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g \
            + v_logK_Z_term * d2v_dlogKdZ \
            + dv_dY * v_y_term + v_yy_term * d2v_dY2 \
            + 0.5 * ξ * (h_d ** 2 + h_g ** 2 + h_y ** 2) \
            + (-1.0) * v_logN_term

        resid = rhs - pv

        marginal_util_c = δ / inside_log
        FOC_d = -marginal_util_c + Γ_d * θ_d / inside_log_i_d * (dv_dlogK - Z * dv_dZ)
        FOC_g = -marginal_util_c + Γ_g * θ_g / inside_log_i_g * (dv_dlogK + (1.0 - Z) * dv_dZ)

        curl = None
        if want_curl:
            # antisymmetric Jacobian of the costate field (integrability defect)
            dpK_dZ = d(dv_dlogK, Z)
            dpZ_dlogK = d(dv_dZ, logK)
            dpK_dY = d(dv_dlogK, Y)
            dpY_dlogK = d(dv_dY, logK)
            dpZ_dY = d(dv_dZ, Y)
            dpY_dZ = d(dv_dY, Z)
            curl = ((dpK_dZ - dpZ_dlogK) ** 2
                    + (dpK_dY - dpY_dlogK) ** 2
                    + (dpZ_dY - dpY_dZ) ** 2)

        return resid, FOC_d, FOC_g, dv_dY, v, curl


# ---------------------------------------------------------------------------
#  Probe / metrics
# ---------------------------------------------------------------------------
def fixed_probe(dtype):
    g = torch.Generator().manual_seed(12345)
    def col(lo, hi):
        return (lo + (hi - lo) * torch.rand((NPROBE, 1), generator=g)).to(dtype)
    return [col(4, 7), col(0.01, 0.99), col(0, 4),
            col(1, 6), col(0, 1 / 3.), col(-3, 5)]


def _rms(x):
    return torch.sqrt(torch.mean(x ** 2))


def evaluate(model, probe):
    logK, Z, Y, logR, lam3, logxi = probe
    resid, FOC_d, FOC_g, dv_dY, v, curl = model.pde_terms(
        logK, Z, Y, logR, lam3, logxi, want_curl=True)
    level = float(v.mean().detach())
    loss_v = float(_rms(resid).detach())
    foc_d = float(_rms(FOC_d).detach())
    foc_g = float(_rms(FOC_g).detach())
    curl_v = float(_rms(curl).detach())
    return level, loss_v, foc_d, foc_g, curl_v


# ---------------------------------------------------------------------------
#  Training (one seed) -- DGM-PIA with costate parameterization
# ---------------------------------------------------------------------------
def train_one_seed(seed, steps, dtype, lr, lambda_curl, device=torch.device("cpu")):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CostateModel(PARAMS, dtype, seed)
    params = PARAMS

    # value/costate step trains the costate net (HJB residual + integrability);
    # control step trains ONLY the controls (FOCs). Keeping the costate net OUT of
    # the control optimizer prevents the early FOC=delta/c blow-up from corrupting
    # the costate field.
    opt_v = torch.optim.Adam(model.c_nn.parameters(), lr=lr)
    opt_c = torch.optim.Adam(
        list(model.i_g_nn.parameters()) + list(model.i_d_nn.parameters()), lr=lr)
    sch = WarmupCosine(lr, steps, max(1, steps // 100), 1e-5)
    CLIP = 1.0  # max grad-norm; FOC=delta/c spikes early from the consumption clamp

    for step in range(1, steps + 1):
        for grp in opt_v.param_groups: grp["lr"] = sch(step)
        for grp in opt_c.param_groups: grp["lr"] = sch(step)

        # ---- value/costate step: rms(HJB residual) + lambda_curl*rms(curl) ----
        s = sample_box(params, 128, dtype, device)
        resid, _, _, _, _, curl = model.pde_terms(*s, want_curl=True)
        vloss = _rms(resid) + lambda_curl * _rms(curl)
        opt_v.zero_grad(set_to_none=True); vloss.backward()
        torch.nn.utils.clip_grad_norm_(model.c_nn.parameters(), CLIP)
        opt_v.step()

        # ---- control step (fresh sample): FOCs only ----
        s2 = sample_box(params, 128, dtype, device)
        _, FOC_d, FOC_g, dv_dY, _, _ = model.pde_terms(*s2, want_curl=False)
        mask = ((s2[2] > params["y_upper"]).to(dtype)) * ((dv_dY > 0).to(dtype))
        closs = _rms(FOC_d) + _rms(FOC_g) + _rms(dv_dY * mask)
        opt_c.zero_grad(set_to_none=True); closs.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.i_g_nn.parameters()) + list(model.i_d_nn.parameters()), CLIP)
        opt_c.step()

    for net in (model.c_nn, model.i_g_nn, model.i_d_nn):
        net.eval()
    return model


def run_protocol(steps, seeds, dtype, lr, lambda_curl):
    probe = fixed_probe(dtype)
    levels, lvs, fds, fgs, curls = [], [], [], [], []
    for sd in seeds:
        t0 = time.time()
        model = train_one_seed(sd, steps, dtype, lr, lambda_curl)
        level, loss_v, foc_d, foc_g, curl_v = evaluate(model, probe)
        levels.append(level); lvs.append(loss_v)
        fds.append(foc_d); fgs.append(foc_g); curls.append(curl_v)
        print(f"  seed={sd}  level={level:+.4f}  loss_v={loss_v:.3e}  "
              f"FOC_d={foc_d:.3e} FOC_g={foc_g:.3e}  curl={curl_v:.3e}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    out = {
        "method": "costate_egm_level_recovery",
        "steps": steps, "seeds": list(seeds), "dtype": str(dtype).replace("torch.", ""),
        "lambda_curl": lambda_curl,
        "level_mean": float(np.mean(levels)),
        "level_spread": float(np.std(levels)),
        "loss_v": float(np.mean(lvs)),
        "FOC_d": float(np.mean(fds)),
        "FOC_g": float(np.mean(fgs)),
        "curl": float(np.mean(curls)),
        "levels": [round(x, 4) for x in levels],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--lambda_curl", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--smoke", action="store_true",
                    help="30-step single-seed smoke test (validation).")
    args = ap.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.smoke:
        print(f"[SMOKE] 30 steps, 1 seed, dtype={args.dtype}, lambda_curl={args.lambda_curl}",
              flush=True)
        out = run_protocol(30, [0], dtype, args.lr, args.lambda_curl)
        print("RESULT " + repr(out), flush=True)
        return

    print(f"[costate_trial] steps={args.steps} seeds={args.seeds} "
          f"dtype={args.dtype} lr={args.lr} lambda_curl={args.lambda_curl}", flush=True)
    out = run_protocol(args.steps, args.seeds, dtype, args.lr, args.lambda_curl)
    print("RESULT " + repr(out), flush=True)


if __name__ == "__main__":
    main()
