"""
method_trial.py -- weak-identification (value LEVEL) method comparison harness
for the torch root regime (PostDamagePostTech, 3-state terminal, no jumps).

Runs the STANDARD METRIC PROTOCOL for ONE architectural/parameterization variant
and prints a single line:

    RESULT {dict with keys method, loss_v, FOC_d, FOC_g, level_spread,
            level_mean, ran_ok, (note)}

Protocol (identical across variants so trials are comparable):
  * Train FROM SCRATCH (random init), seeds {0,1,2} (default), N steps (default
    3000), batch 128, Adam, WarmupCosine peak lr 4e-4, DGM-PIA two-step loss:
      value step   : rms(rhs - pv)        (+ anchor term for valuenet_anchor)
      control step : rms(FOC_d)+rms(FOC_g)+rms(dv_dY positive-part above y_upper)
    on a FRESH re-sample.
  * After training, evaluate on the FIXED probe (256 pts, generator seed 12345,
    box logK U(4,7), Z U(.01,.99), Y U(0,4), logR U(1,6), lam3 U(0,1/3),
    logxi U(-3,5)):
        LEVEL  = mean over probe of V (= pv/delta)
        loss_v = rms(rhs - pv) on probe
  * level_spread = std over the seeds of LEVEL  (KEY metric: weak-id of the level)
    level_mean   = mean over seeds of LEVEL
    loss_v/FOC_* = mean over seeds.

A method WINS if it lowers level_spread WITHOUT inflating loss_v / FOC vs baseline.

The 6 variants (each = root regime with ONE change):
  baseline        : current arch exactly (padded input width 7, frozen BN, softplus v).
  no_padding      : input width 5 = [logK,Z,Y,lam3,logxi] (drop dup logxi + const A_g'').
  no_BN           : baseline but BatchNorm layers bypassed (identity).
  linear_output   : baseline but value-net final activation = identity. note=frac v<0.
  structural_xi   : v = V_inf(s) - softplus(W(s,logxi))/xi, V_inf on non-xi inputs.
  valuenet_anchor : baseline + single-point LEVEL anchor (v_nn(x0)-v_target)^2 from TF surrogate.

All NEW; treats model/output dirs as READ-ONLY.  CPU, torch 1.12.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
HERE = os.path.join(ROOT, "models_torch_train")
MT = os.path.join(ROOT, "models_torch")
for p in (HERE, MT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tf_to_torch_loader import (  # noqa: E402
    build_root_torch_model,
    build_trainable_net,
    make_root_configs,
)
from feedforward_subnet import FeedForwardSubNet, _BatchNorm1dInference  # noqa: E402
from params import PARAMS, investment_rate_activation  # noqa: E402
from train_root import sample_box, WarmupCosine  # noqa: E402
from PostDamagePostTech import _grad  # noqa: E402

DELTA = float(PARAMS.get("δ", PARAMS.get("delta", 0.01)))
NPROBE = 256
DEVICE = torch.device("cpu")

# Fixed anchor state for valuenet_anchor (interior point).
ANCHOR_X0 = {"logK": 5.5, "Z": 0.5, "Y": 1.0, "lam3": 1.0 / 6.0, "logxi": 0.0}


# ---------------------------------------------------------------------------
#  Shared economics: given v, i_g, i_d (each [-1,1]) and the state tensors,
#  reproduce the PostDamagePostTech HJB residual + FOCs EXACTLY.
#  (Bit-faithful copy of models_torch/PostDamagePostTech.py::pde_rhs body,
#   refactored so the v/i nets can come from a custom assembly.)
# ---------------------------------------------------------------------------
def _economics(p, v, i_g, i_d, logK, Z, Y, logR, λ3, logξ):
    A_d = p['A_d']
    A_g_prime_prime = p['A_g_prime_prime']
    δ = p['δ']
    α_d = p['α_d']; Γ_d = p['Γ_d']; θ_d = p['θ_d']; σ_d = p['σ_d']
    α_g = p['α_g']; Γ_g = p['Γ_g']; θ_g = p['θ_g']; σ_g = p['σ_g']
    θ_bar = p['θ_bar']; η = p['η']; ϛ = p['ϛ']
    λ1 = p['λ1']; λ2 = p['λ2']
    y_upper = p['y_upper']

    ξ = torch.exp(logξ)
    K = torch.exp(logK)

    dv_dlogK = _grad(v, logK, create_graph=True)
    d2v_dlogK2 = _grad(dv_dlogK, logK, create_graph=False)
    d2v_dlogKdZ = _grad(dv_dlogK, Z, create_graph=False)

    dv_dZ = _grad(v, Z, create_graph=True)
    d2v_dZ2 = _grad(dv_dZ, Z, create_graph=False)

    dv_dY = _grad(v, Y, create_graph=True)
    d2v_dY2 = _grad(dv_dY, Y, create_graph=False)

    h_d = - 1.0 / ξ * ((dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d)
    h_g = - 1.0 / ξ * ((dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g)
    h_y = - 1.0 / ξ * (dv_dY - (λ1 + λ2 * Y + λ3 * (Y - y_upper))) * η * A_d * (1 - Z) * K * ϛ

    pv = δ * v

    c = (A_d - i_d) * (1 - Z) + (A_g_prime_prime - i_g) * Z
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
                - Z * σ_g ** 2
                + (1 - Z) * σ_d ** 2) * Z * (1 - Z)

    v_ZZ_term = 0.5 * Z ** 2 * (1 - Z) ** 2 * (σ_g ** 2 + σ_d ** 2)
    v_logK_Z_term = - Z * (1 - Z) ** 2 * σ_d ** 2 + Z ** 2 * (1.0 - Z) * σ_g ** 2
    v_y_term = (θ_bar + h_y * ϛ) * η * A_d * (1 - Z) * K
    v_yy_term = 0.5 * ϛ ** 2 * (η * A_d * (1 - Z) * K) ** 2
    v_logN_term = (λ1 + λ2 * Y + λ3 * (Y - y_upper)) * v_y_term + (λ2 + λ3) * v_yy_term

    rhs = flow \
        + v_logK_term * dv_dlogK + v_logKlogK_term * d2v_dlogK2 \
        + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
        + h_d * (dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d + h_g * (dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g \
        + v_logK_Z_term * d2v_dlogKdZ \
        + dv_dY * v_y_term + v_yy_term * d2v_dY2 \
        + 0.5 * ξ * (torch.pow(h_d, 2) + torch.pow(h_g, 2) + torch.pow(h_y, 2)) \
        + (-1.0) * v_logN_term

    marginal_util_c = δ / inside_log
    FOC_d = -marginal_util_c + Γ_d * θ_d / (inside_log_i_d) * (dv_dlogK - Z * dv_dZ)
    FOC_g = -marginal_util_c + Γ_g * θ_g / (inside_log_i_g) * (dv_dlogK + (1.0 - Z) * dv_dZ)

    return rhs, pv, dv_dY, c, 1.0 + θ_g * i_g, 1.0 + θ_d * i_d, FOC_d, FOC_g


# ---------------------------------------------------------------------------
#  Net builders (control configs are always tanh/custom; value config varies).
# ---------------------------------------------------------------------------
def _control_nets(seed, input_dim):
    """i_g_nn, i_d_nn (tanh hidden, bounded custom output) with given input width."""
    _, ig_cfg, id_cfg = make_root_configs()
    ig_cfg = dict(ig_cfg); ig_cfg["final_activation"] = investment_rate_activation(PARAMS["θ_g"])
    id_cfg = dict(id_cfg); id_cfg["final_activation"] = investment_rate_activation(PARAMS["θ_d"])
    i_g = build_trainable_net(ig_cfg, input_dim=input_dim, seed=seed + 101)
    i_d = build_trainable_net(id_cfg, input_dim=input_dim, seed=seed + 202)
    return i_g, i_d


def _value_net(seed, input_dim, final_activation="softplus"):
    v_cfg, _, _ = make_root_configs()
    v_cfg = dict(v_cfg); v_cfg["final_activation"] = final_activation
    return build_trainable_net(v_cfg, input_dim=input_dim, seed=seed + 7)


def _bypass_bn(net):
    """Replace every _BatchNorm1dInference in net with identity (frozen)."""
    for i in range(len(net.bn_layers)):
        nf = net.bn_layers[i].gamma.numel()
        ident = _BatchNorm1dInference(nf)  # gamma=1,beta=0,mean=0,var=1 -> identity
        net.bn_layers[i] = ident
    return net


# ---------------------------------------------------------------------------
#  Variant models.  Each exposes  pde_rhs(logK,Z,Y,logR,lam3,logxi) -> 8-tuple
#  and .v_params() (params trained by the value optimizer) and
#  .control_params() (params trained by the control optimizer).
# ---------------------------------------------------------------------------
class BaselineModel:
    """Variant 1 (also the carrier for 3,4,6 via flags). Padded width-7 input."""

    def __init__(self, seed, *, bypass_bn=False, value_final="softplus"):
        self.params = PARAMS
        self.v_nn = _value_net(seed, 7, final_activation=value_final)
        self.i_g_nn, self.i_d_nn = _control_nets(seed, 7)
        if bypass_bn:
            for net in (self.v_nn, self.i_g_nn, self.i_d_nn):
                _bypass_bn(net)

    def _X(self, logK, Z, Y, λ3, logξ):
        A_g_pp = self.params['A_g_prime_prime']
        return torch.cat(
            [logK, Z, Y, λ3, A_g_pp * torch.ones_like(Y), logξ, logξ], dim=1)

    def v_of(self, X):
        return self.v_nn(X)

    def pde_rhs(self, logK, Z, Y, logR, λ3, logξ):
        X = self._X(logK, Z, Y, λ3, logξ)
        v = self.v_of(X)
        i_g = self.i_g_nn(X)
        i_d = self.i_d_nn(X)
        return _economics(self.params, v, i_g, i_d, logK, Z, Y, logR, λ3, logξ)

    def v_params(self):
        return list(self.v_nn.parameters())

    def control_params(self):
        return list(self.i_g_nn.parameters()) + list(self.i_d_nn.parameters())

    def train(self):
        for net in (self.v_nn, self.i_g_nn, self.i_d_nn):
            net.train()


class NoPaddingModel(BaselineModel):
    """Variant 2: width-5 input [logK,Z,Y,lam3,logxi]."""

    def __init__(self, seed):
        self.params = PARAMS
        self.v_nn = _value_net(seed, 5, final_activation="softplus")
        self.i_g_nn, self.i_d_nn = _control_nets(seed, 5)

    def _X(self, logK, Z, Y, λ3, logξ):
        return torch.cat([logK, Z, Y, λ3, logξ], dim=1)


class StructuralXiModel:
    """Variant 5: v = V_inf(s_noxi) - softplus(W(s_all))/xi.

    V_inf : net on [logK,Z,Y,lam3]  (width 4), swish hidden, LINEAR output.
    W     : net on [logK,Z,Y,lam3,logxi] (width 5), swish hidden, softplus output.
    Controls: tanh/custom nets on the full width-5 input.
    """

    def __init__(self, seed):
        self.params = PARAMS
        vcfg, _, _ = make_root_configs()
        vinf_cfg = dict(vcfg); vinf_cfg["final_activation"] = "linear"; vinf_cfg["nn_name"] = "v_inf"
        w_cfg = dict(vcfg); w_cfg["final_activation"] = "softplus"; w_cfg["nn_name"] = "v_W"
        self.v_inf_nn = build_trainable_net(vinf_cfg, input_dim=4, seed=seed + 7)
        self.w_nn = build_trainable_net(w_cfg, input_dim=5, seed=seed + 13)
        self.i_g_nn, self.i_d_nn = _control_nets(seed, 5)

    def _Xfull(self, logK, Z, Y, λ3, logξ):
        return torch.cat([logK, Z, Y, λ3, logξ], dim=1)

    def _Xnoxi(self, logK, Z, Y, λ3):
        return torch.cat([logK, Z, Y, λ3], dim=1)

    def pde_rhs(self, logK, Z, Y, logR, λ3, logξ):
        Xfull = self._Xfull(logK, Z, Y, λ3, logξ)
        Xnoxi = self._Xnoxi(logK, Z, Y, λ3)
        ξ = torch.exp(logξ)
        v = self.v_inf_nn(Xnoxi) - self.w_nn(Xfull) / ξ
        i_g = self.i_g_nn(Xfull)
        i_d = self.i_d_nn(Xfull)
        return _economics(self.params, v, i_g, i_d, logK, Z, Y, logR, λ3, logξ)

    def v_params(self):
        return list(self.v_inf_nn.parameters()) + list(self.w_nn.parameters())

    def control_params(self):
        return list(self.i_g_nn.parameters()) + list(self.i_d_nn.parameters())

    def train(self):
        for net in (self.v_inf_nn, self.w_nn, self.i_g_nn, self.i_d_nn):
            net.train()


# ---------------------------------------------------------------------------
#  Probe + losses
# ---------------------------------------------------------------------------
def fixed_probe(dtype):
    g = torch.Generator().manual_seed(12345)

    def col(lo, hi):
        return (lo + (hi - lo) * torch.rand((NPROBE, 1), generator=g)).to(dtype)

    return [col(4, 7), col(0.01, 0.99), col(0, 4),
            col(1, 6), col(0, 1 / 3.), col(-3, 5)]


def _rms(x):
    return torch.sqrt(torch.mean(x ** 2))


def value_loss_terms(model, sample, anchor=None):
    """Returns (value_loss, diag_dict). Optionally adds anchor term."""
    logK = sample[0].detach().requires_grad_(True)
    Z = sample[1].detach().requires_grad_(True)
    Y = sample[2].detach().requires_grad_(True)
    logR, lam3, logxi = sample[3], sample[4], sample[5]
    rhs, pv, *_ = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    resid = rhs - pv
    vloss = _rms(resid)
    diag = {"loss_v": float(vloss.detach())}
    if anchor is not None:
        x0, v_target, lam = anchor
        v0 = model.v_of(x0) if hasattr(model, "v_of") else None
        anch = lam * (v0 - v_target) ** 2
        vloss = vloss + anch.reshape(())
        diag["anchor"] = float(anch.detach())
    return vloss, diag


def control_loss_terms(model, sample):
    logK = sample[0].detach().requires_grad_(True)
    Z = sample[1].detach().requires_grad_(True)
    Y = sample[2].detach().requires_grad_(True)
    logR, lam3, logxi = sample[3], sample[4], sample[5]
    rhs, pv, dv_dY, c, gig, gid, FOC_d, FOC_g = model.pde_rhs(
        logK, Z, Y, logR, lam3, logxi)
    y_upper = model.params["y_upper"]
    mask = ((Y > y_upper).to(dv_dY.dtype)) * ((dv_dY > 0).to(dv_dY.dtype))
    loss_dv_dY = dv_dY * mask
    closs = _rms(FOC_d) + _rms(FOC_g) + _rms(loss_dv_dY)
    diag = {"FOC_d": float(_rms(FOC_d).detach()), "FOC_g": float(_rms(FOC_g).detach())}
    return closs, diag


def probe_eval(model, probe):
    """Return (level_mean_over_probe, loss_v_on_probe, frac_v_neg)."""
    logK = probe[0].detach().requires_grad_(True)
    Z = probe[1].detach().requires_grad_(True)
    Y = probe[2].detach().requires_grad_(True)
    logR, lam3, logxi = probe[3], probe[4], probe[5]
    rhs, pv, *_ = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    V = (pv / DELTA).detach().cpu().numpy().reshape(-1)
    resid = (rhs - pv).detach().cpu().numpy().reshape(-1)
    frac_neg = float(np.mean(V < 0.0))
    return float(np.mean(V)), float(np.sqrt(np.mean(resid ** 2))), frac_neg


# ---------------------------------------------------------------------------
#  Surrogate v_target for valuenet_anchor (read ONCE).
# ---------------------------------------------------------------------------
def surrogate_v_target(dtype):
    """Build the TF surrogate, read its TRANSFORMED v at the fixed anchor x0.

    Returns (x0_tensor_width7, v_target_scalar_tensor).
    """
    m = build_root_torch_model(load_surrogate=True, dtype=dtype)
    A_g_pp = PARAMS['A_g_prime_prime']
    lk = torch.tensor([[ANCHOR_X0["logK"]]], dtype=dtype)
    z = torch.tensor([[ANCHOR_X0["Z"]]], dtype=dtype)
    y = torch.tensor([[ANCHOR_X0["Y"]]], dtype=dtype)
    l3 = torch.tensor([[ANCHOR_X0["lam3"]]], dtype=dtype)
    lx = torch.tensor([[ANCHOR_X0["logxi"]]], dtype=dtype)
    X = torch.cat([lk, z, y, l3, A_g_pp * torch.ones_like(y), lx, lx], dim=1)
    with torch.no_grad():
        v_t = m.v_nn(X).reshape(()).clone()
    return X, v_t


def anchor_x0_width7(dtype):
    A_g_pp = PARAMS['A_g_prime_prime']
    lk = torch.tensor([[ANCHOR_X0["logK"]]], dtype=dtype)
    z = torch.tensor([[ANCHOR_X0["Z"]]], dtype=dtype)
    y = torch.tensor([[ANCHOR_X0["Y"]]], dtype=dtype)
    l3 = torch.tensor([[ANCHOR_X0["lam3"]]], dtype=dtype)
    lx = torch.tensor([[ANCHOR_X0["logxi"]]], dtype=dtype)
    return torch.cat([lk, z, y, l3, A_g_pp * torch.ones_like(y), lx, lx], dim=1)


# ---------------------------------------------------------------------------
#  Build one model for a given variant + seed.
# ---------------------------------------------------------------------------
def build_variant(method, seed, dtype):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if method == "baseline":
        m = BaselineModel(seed)
    elif method == "no_padding":
        m = NoPaddingModel(seed)
    elif method == "no_BN":
        m = BaselineModel(seed, bypass_bn=True)
    elif method == "linear_output":
        m = BaselineModel(seed, value_final="linear")
    elif method == "structural_xi":
        m = StructuralXiModel(seed)
    elif method == "valuenet_anchor":
        m = BaselineModel(seed)
    else:
        raise ValueError(f"unknown method {method!r}")
    for net in _all_nets(m):
        net.to(dtype=dtype)
    m.train()
    return m


def _all_nets(m):
    nets = []
    for attr in ("v_nn", "i_g_nn", "i_d_nn", "v_inf_nn", "w_nn"):
        if hasattr(m, attr):
            nets.append(getattr(m, attr))
    return nets


# ---------------------------------------------------------------------------
#  Train one seed (DGM-PIA), return probe LEVEL, loss_v, FOCs, frac_neg.
# ---------------------------------------------------------------------------
def train_one(method, seed, steps, dtype, anchor=None):
    m = build_variant(method, seed, dtype)
    opt_v = torch.optim.Adam(m.v_params(), lr=4e-4)
    opt_c = torch.optim.Adam(m.control_params(), lr=4e-4)
    sch = WarmupCosine(4e-4, steps, max(1, steps // 100), 1e-5)
    params = m.params

    for step in range(1, steps + 1):
        lr = sch(step)
        for g in opt_v.param_groups:
            g["lr"] = lr
        for g in opt_c.param_groups:
            g["lr"] = lr

        s = sample_box(params, 128, dtype, DEVICE)
        vloss, _ = value_loss_terms(m, s, anchor=anchor)
        opt_v.zero_grad(set_to_none=True)
        vloss.backward()
        opt_v.step()

        s2 = sample_box(params, 128, dtype, DEVICE)
        closs, _ = control_loss_terms(m, s2)
        opt_c.zero_grad(set_to_none=True)
        closs.backward()
        opt_c.step()

    probe = fixed_probe(dtype)
    level, loss_v, frac_neg = probe_eval(m, probe)
    # final FOC diag on a fresh sample
    s3 = sample_box(params, 128, dtype, DEVICE)
    _, cdiag = control_loss_terms(m, s3)
    return {
        "level": level, "loss_v": loss_v,
        "FOC_d": cdiag["FOC_d"], "FOC_g": cdiag["FOC_g"],
        "frac_neg": frac_neg,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["baseline", "no_padding", "no_BN", "linear_output",
                             "structural_xi", "valuenet_anchor"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lambda_anchor", type=float, default=1.0)
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    t0 = time.time()
    ran_ok = True
    note = ""
    anchor = None

    try:
        if args.method == "valuenet_anchor":
            x0 = anchor_x0_width7(dtype)
            _, v_target = surrogate_v_target(dtype)
            anchor = (x0, v_target, args.lambda_anchor)
            note = f"v_target={float(v_target):.4f} lambda={args.lambda_anchor}"

        levels, loss_vs, focds, focgs, fracs = [], [], [], [], []
        for sd in seeds:
            r = train_one(args.method, sd, args.steps, dtype, anchor=anchor)
            levels.append(r["level"]); loss_vs.append(r["loss_v"])
            focds.append(r["FOC_d"]); focgs.append(r["FOC_g"]); fracs.append(r["frac_neg"])
            print(f"[seed {sd}] level={r['level']:.4f} loss_v={r['loss_v']:.4e} "
                  f"FOC_d={r['FOC_d']:.4e} FOC_g={r['FOC_g']:.4e} "
                  f"frac_v<0={r['frac_neg']:.3f}", flush=True)

        out = {
            "method": args.method,
            "loss_v": float(np.mean(loss_vs)),
            "FOC_d": float(np.mean(focds)),
            "FOC_g": float(np.mean(focgs)),
            "level_spread": float(np.std(levels)),
            "level_mean": float(np.mean(levels)),
            "ran_ok": ran_ok,
        }
        if args.method == "linear_output":
            note = f"frac_v<0={float(np.mean(fracs)):.4f}"
        if args.method == "no_BN":
            note = ("frozen-affine BN has NO trainable params and inits to identity "
                    "(gamma=1,beta=0,mean=0,var=1); from scratch it never updates, so "
                    "no_BN == baseline by construction in this port")
        if note:
            out["note"] = note
        out["steps"] = args.steps
        out["seeds"] = seeds
        out["dtype"] = args.dtype
        out["sec"] = round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        out = {"method": args.method, "loss_v": float("nan"), "FOC_d": float("nan"),
               "FOC_g": float("nan"), "level_spread": float("nan"),
               "level_mean": float("nan"), "ran_ok": False,
               "note": f"{type(e).__name__}: {e}"}

    print("RESULT " + repr(out), flush=True)


if __name__ == "__main__":
    main()
