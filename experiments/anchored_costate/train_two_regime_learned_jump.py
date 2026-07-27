"""Two-regime anchored trainer with a learned xi-conditioned jump minimizer.

This extends ``train_two_regime_joint.py`` only inside the experiment sandbox.
The pre-damage/post-tech damage-jump distortion is represented by a small
``JumpRobustNet`` instead of being hard-coded as

    log g_l = -(V_l^post - V^pre) / xi.

The robust net is trained adversarially/inner-loop style against the entropy
penalty objective, while the outer value/control updates use an envelope
stop-gradient through the current minimizer.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS_TORCH = ROOT / "models_torch"
MODELS_TORCH_TRAIN = ROOT / "models_torch_train"
for _path in (str(HERE), str(MODELS_TORCH_TRAIN), str(MODELS_TORCH)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from PreDamagePostTech import PreDamagePostTechModel, _grad  # noqa: E402
from anchored_nets import (  # noqa: E402
    PREDAMAGE_POSTTECH_BOUNDS_6,
    DetachedModule,
    InputNorm,
    RecenteredValueNet,
    WarmupCosine,
    boundary_gap_loss,
    build_control_net,
    build_value_phi,
    predamage_posttech_anchor_x,
    probe_level,
    rms,
    root_value_at_damage_anchor,
    sample_box,
    trainable_parameters,
    value_loss,
    control_loss,
)
from params import PARAMS  # noqa: E402
from tf_to_torch_loader import DEFAULT_CKPT_DIR, make_root_configs  # noqa: E402
from train_two_regime_joint import build_root, _dtype, _json_default  # noqa: E402


def set_module_grad(module, flag):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(flag)


class JumpRobustNet(nn.Module):
    """FiLM-style amortized minimizer for damage-jump log distortions."""

    def __init__(self, L, width=64, depth=3, seed=None, dtype=torch.float32, device=torch.device("cpu")):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.L = int(L)
        self.x_norm = InputNorm(PREDAMAGE_POSTTECH_BOUNDS_6)
        self.in_dim = 6 + 2 * self.L + 3
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        prev = self.in_dim
        for _ in range(depth):
            self.layers.append(nn.Linear(prev, width))
            self.films.append(nn.Linear(3, 2 * width))
            prev = width
        self.out = nn.Linear(prev, self.L)
        self.to(device=device, dtype=dtype)

    def forward(self, x_pre, diff, base_logg, logxi):
        invxi = torch.exp(-logxi).clamp(max=50.0)
        xi_features = torch.cat([logxi, invxi, torch.log1p(invxi)], dim=1)
        features = torch.cat(
            [
                self.x_norm(x_pre),
                diff,
                base_logg.clamp(min=-12.0, max=12.0),
                xi_features,
            ],
            dim=1,
        )
        h = features
        for layer, film in zip(self.layers, self.films):
            h = layer(h)
            gamma, beta = torch.chunk(film(xi_features), 2, dim=1)
            h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
            h = torch.tanh(h)
        return self.out(h)


class LearnedJumpPreDamagePostTechModel(PreDamagePostTechModel):
    """PreDamagePostTech HJB with a trainable jump minimizer."""

    def __init__(
        self,
        params=None,
        robust_mode="residual",
        jump_width=64,
        jump_depth=3,
        residual_scale=0.05,
        direct_scale=6.0,
        logg_clip=20.0,
        outer_envelope_detach=True,
        seed=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        super().__init__(params)
        self.robust_mode = robust_mode
        self.residual_scale = float(residual_scale)
        self.direct_scale = float(direct_scale)
        self.logg_clip = float(logg_clip)
        self.outer_envelope_detach = bool(outer_envelope_detach)
        self.jump_net = None
        if robust_mode != "closed":
            self.jump_net = JumpRobustNet(
                L=self.params["L"],
                width=jump_width,
                depth=jump_depth,
                seed=seed,
                dtype=dtype,
                device=device,
            )

    def _x_pre(self, logK, Z, Y, logxi):
        p = self.params
        return torch.cat(
            [logK, Z, Y, p["A_g_prime_prime"] * torch.ones_like(Y), logxi, logxi],
            dim=1,
        )

    def _post_values(self, logK, Z, Y, logxi):
        p = self.params
        vals = []
        for lam in p["λ3_values"]:
            x_jump = torch.cat(
                [
                    logK,
                    Z,
                    torch.ones_like(Y) * p["y_upper"],
                    float(lam) * torch.ones_like(Y),
                    p["A_g_prime_prime"] * torch.ones_like(Y),
                    logxi,
                    logxi,
                ],
                dim=1,
            )
            vals.append(self.v_PostDamagePostTech_nn(x_jump))
        return torch.cat(vals, dim=1)

    def _jump_logg(self, x_pre, diff, logxi, robust_input_detach=False, outer_detach=False):
        xi = torch.exp(logxi)
        base = -diff / xi
        if self.robust_mode == "closed":
            logg = base
        else:
            if robust_input_detach:
                x_in = x_pre.detach()
                diff_in = diff.detach()
                base_in = base.detach()
                logxi_in = logxi.detach()
            else:
                x_in = x_pre
                diff_in = diff
                base_in = base
                logxi_in = logxi
            raw = self.jump_net(x_in, diff_in, base_in, logxi_in)
            if self.robust_mode == "residual":
                logg = base_in + self.residual_scale * torch.tanh(raw)
            elif self.robust_mode == "direct":
                logg = self.direct_scale * torch.tanh(raw)
            else:
                raise ValueError(f"unknown robust_mode={self.robust_mode!r}")
        logg = logg.clamp(min=-self.logg_clip, max=self.logg_clip)
        return logg.detach() if outer_detach else logg, base.clamp(min=-self.logg_clip, max=self.logg_clip)

    def _jump_inner(self, logK, Z, Y, logxi, v, robust_input_detach=False, outer_detach=False):
        p = self.params
        xi = torch.exp(logxi)
        x_pre = self._x_pre(logK, Z, Y, logxi)
        v_post = self._post_values(logK, Z, Y, logxi)
        diff = v_post - v
        logg, base = self._jump_logg(
            x_pre, diff, logxi,
            robust_input_detach=robust_input_detach,
            outer_detach=outer_detach,
        )
        g = torch.exp(logg)
        inner = g * diff + xi * (1.0 - g + g * logg)
        closed = xi * (1.0 - torch.exp(base))
        gap = inner - closed
        return inner, gap, logg - base

    def jump_only_terms(self, sample):
        logK, Z, Y, logR, lam3, logxi = sample
        del logR, lam3
        logK = logK.detach()
        Z = Z.detach()
        Y = Y.detach()
        logxi = logxi.detach()
        x_pre = self._x_pre(logK, Z, Y, logxi)
        with torch.no_grad():
            v = self.v_nn(x_pre).detach()
            v_post = self._post_values(logK, Z, Y, logxi).detach()
            diff = v_post - v
            xi = torch.exp(logxi)
            base = (-diff / xi).clamp(min=-self.logg_clip, max=self.logg_clip)
            closed = xi * (1.0 - torch.exp(base))
        logg, base = self._jump_logg(
            x_pre.detach(), diff.detach(), logxi.detach(),
            robust_input_detach=True,
            outer_detach=False,
        )
        g = torch.exp(logg)
        inner = g * diff + torch.exp(logxi) * (1.0 - g + g * logg)
        gap = inner - closed
        logg_err = logg - base
        return {"jump_gap": gap, "logg_err": logg_err}

    def pde_rhs(self, logK, Z, Y, logR, λ3, logξ):
        """Same HJB as PreDamagePostTechModel, but with learned jump log g."""
        del logR, λ3
        p = self.params
        A_d = p["A_d"]
        A_g_prime_prime = p["A_g_prime_prime"]
        δ = p["δ"]
        α_d = p["α_d"]; Γ_d = p["Γ_d"]; θ_d = p["θ_d"]; σ_d = p["σ_d"]
        α_g = p["α_g"]; Γ_g = p["Γ_g"]; θ_g = p["θ_g"]; σ_g = p["σ_g"]
        θ_bar = p["θ_bar"]; η = p["η"]; ϛ = p["ϛ"]
        λ1 = p["λ1"]; λ2 = p["λ2"]; L = p["L"]
        r1 = p["r1"]; r2 = p["r2"]; y_lower = p["y_lower"]

        X = self._x_pre(logK, Z, Y, logξ)
        v = self.v_nn(X)
        i_g = self.i_g_nn(X)
        i_d = self.i_d_nn(X)
        ξ = torch.exp(logξ)
        K = torch.exp(logK)

        dv_dlogK = _grad(v, logK, create_graph=True)
        d2v_dlogK2 = _grad(dv_dlogK, logK, create_graph=False)
        d2v_dlogKdZ = _grad(dv_dlogK, Z, create_graph=False)
        dv_dZ = _grad(v, Z, create_graph=True)
        d2v_dZ2 = _grad(dv_dZ, Z, create_graph=False)
        dv_dY = _grad(v, Y, create_graph=True)
        d2v_dY2 = _grad(dv_dY, Y, create_graph=False)

        h_d = -1.0 / ξ * ((dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d)
        h_g = -1.0 / ξ * ((dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g)
        h_y = -1.0 / ξ * (dv_dY - (λ1 + λ2 * Y)) * η * A_d * (1 - Z) * K * ϛ

        pv = δ * v
        c = (A_d - i_d) * (1 - Z) + (A_g_prime_prime - i_g) * Z
        inside_log = torch.clamp(c, min=1e-8).reshape(-1, 1)
        flow = δ * (torch.log(inside_log) + logK)

        v_logKlogK_term = (σ_d ** 2 * (1 - Z) ** 2 + σ_g ** 2 * Z ** 2) / 2.0
        inside_log_i_d = torch.clamp(1.0 + θ_d * i_d, min=1e-8).reshape(-1, 1)
        inside_log_i_g = torch.clamp(1.0 + θ_g * i_g, min=1e-8).reshape(-1, 1)
        v_logK_term = (α_d + Γ_d * torch.log(inside_log_i_d)) * (1 - Z) \
            + (α_g + Γ_g * torch.log(inside_log_i_g)) * Z - v_logKlogK_term
        v_Z_term = (α_g + Γ_g * torch.log(inside_log_i_g)
                    - (α_d + Γ_d * torch.log(inside_log_i_d))
                    - Z * σ_g ** 2 + (1 - Z) * σ_d ** 2) * Z * (1 - Z)
        v_ZZ_term = 0.5 * Z ** 2 * (1 - Z) ** 2 * (σ_g ** 2 + σ_d ** 2)
        v_logK_Z_term = -Z * (1 - Z) ** 2 * σ_d ** 2 + Z ** 2 * (1.0 - Z) * σ_g ** 2
        v_y_term = (θ_bar + h_y * ϛ) * η * A_d * (1 - Z) * K
        v_yy_term = 0.5 * ϛ ** 2 * (η * A_d * (1 - Z) * K) ** 2
        v_logN_term = (λ1 + λ2 * Y) * v_y_term + λ2 * v_yy_term

        J_d = r1 * (torch.exp(r2 / 2 * torch.pow(Y - y_lower, 2)) - 1) \
            * (Y > y_lower).to(Y.dtype)
        inner, _gap, _logg_err = self._jump_inner(
            logK, Z, Y, logξ, v,
            robust_input_detach=self.outer_envelope_detach,
            outer_detach=self.outer_envelope_detach,
        )
        Jump_term = J_d / L * torch.sum(inner, dim=1, keepdim=True)

        rhs = flow \
            + v_logK_term * dv_dlogK + v_logKlogK_term * d2v_dlogK2 \
            + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
            + h_d * (dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d \
            + h_g * (dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g \
            + v_logK_Z_term * d2v_dlogKdZ \
            + dv_dY * v_y_term + v_yy_term * d2v_dY2 \
            + 0.5 * ξ * (torch.pow(h_d, 2) + torch.pow(h_g, 2) + torch.pow(h_y, 2)) \
            - v_logN_term + Jump_term

        marginal_util_c = δ / inside_log
        FOC_d = -marginal_util_c + Γ_d * θ_d / inside_log_i_d * (dv_dlogK - Z * dv_dZ)
        FOC_g = -marginal_util_c + Γ_g * θ_g / inside_log_i_g * (dv_dlogK + (1.0 - Z) * dv_dZ)
        return rhs, pv, dv_dY, c, 1.0 + θ_g * i_g, 1.0 + θ_d * i_d, FOC_d, FOC_g


def build_pre_jump(args, root_model, dtype, device):
    v_cfg, ig_cfg, id_cfg = make_root_configs()
    model = LearnedJumpPreDamagePostTechModel(
        {"v_nn_config": v_cfg, "i_g_nn_config": ig_cfg, "i_d_nn_config": id_cfg},
        robust_mode=args.jump_robust_mode,
        jump_width=args.jump_width,
        jump_depth=args.jump_depth,
        residual_scale=args.jump_residual_scale,
        direct_scale=args.jump_direct_scale,
        logg_clip=args.logg_clip,
        outer_envelope_detach=not args.no_outer_envelope_detach,
        seed=args.seed + 7000,
        dtype=dtype,
        device=device,
    )
    hidden = tuple([args.width] * args.layers)
    phi = build_value_phi(
        input_dim=6,
        bounds=PREDAMAGE_POSTTECH_BOUNDS_6,
        hidden=hidden,
        activation=args.activation,
        seed=args.seed + 4000,
        residual=not args.no_residual,
        value_arch=args.value_arch,
        dtype=dtype,
        device=device,
        ckpt_dir=args.ckpt_dir,
    )
    anchor_value = root_value_at_damage_anchor(root_model.v_nn, detach=not args.couple_anchor_grad)
    model.v_nn = RecenteredValueNet(phi, predamage_posttech_anchor_x, anchor_value)
    model.i_g_nn = build_control_net(
        6, PREDAMAGE_POSTTECH_BOUNDS_6, model.params["θ_g"], args.seed + 5000,
        activation="tanh", init_rate=args.pre_control_init_rate,
    )
    model.i_d_nn = build_control_net(
        6, PREDAMAGE_POSTTECH_BOUNDS_6, model.params["θ_d"], args.seed + 6000,
        activation="tanh", init_rate=args.pre_control_init_rate,
    )
    model.v_PostDamagePostTech_nn = (
        root_model.v_nn if args.couple_neighbor_grad else DetachedModule(root_model.v_nn)
    )
    for net in (model.v_nn, model.i_g_nn, model.i_d_nn, model.v_PostDamagePostTech_nn):
        net.to(device=device, dtype=dtype)
        net.train()
    if model.jump_net is not None:
        model.jump_net.to(device=device, dtype=dtype)
        model.jump_net.train()
        if args.jump_pretrained_path:
            state = torch.load(args.jump_pretrained_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.jump_net.load_state_dict(state, strict=True)
            print(f"[load] jump minimizer weights from {args.jump_pretrained_path}", flush=True)
        if args.freeze_jump_net:
            for param in model.jump_net.parameters():
                param.requires_grad_(False)
    return model


def jump_validation(pre_model, n, batches, dtype, device, seed):
    out = {"jump_gap": 0.0, "logg_err": 0.0}
    for i in range(batches):
        devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed) + 1777 * i)
            sample = sample_box(pre_model.params, n, dtype, device)
        terms = pre_model.jump_only_terms(sample)
        out["jump_gap"] += float(torch.mean(terms["jump_gap"]).detach().cpu())
        out["logg_err"] += float(rms(terms["logg_err"]).detach().cpu())
    for key in out:
        out[key] /= max(1, batches)
    return out


def validation(root_model, pre_model, n, batches, dtype, device, seed=None):
    accum = {
        "root_hjb": 0.0,
        "root_foc_d": 0.0,
        "root_foc_g": 0.0,
        "pre_hjb": 0.0,
        "pre_foc_d": 0.0,
        "pre_foc_g": 0.0,
        "root_level": 0.0,
        "pre_level": 0.0,
    }
    for i in range(batches):
        batch_seed = None if seed is None else int(seed) + 1009 * i
        root_sample = sample_box(root_model.params, n, dtype, device)
        pre_sample = sample_box(pre_model.params, n, dtype, device)
        if batch_seed is not None:
            devices = [device.index] if device.type == "cuda" and device.index is not None else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(batch_seed)
                root_sample = sample_box(root_model.params, n, dtype, device)
                torch.manual_seed(batch_seed + 17)
                pre_sample = sample_box(pre_model.params, n, dtype, device)
        root_loss, root_diag = value_loss(root_model, root_sample, include_foc=True)
        pre_loss, pre_diag = value_loss(pre_model, pre_sample, include_foc=True)
        del root_loss, pre_loss
        root_probe = probe_level(root_model, n, dtype, device)
        pre_probe = probe_level(pre_model, n, dtype, device)
        accum["root_hjb"] += root_diag["hjb"]
        accum["root_foc_d"] += root_diag["foc_d"]
        accum["root_foc_g"] += root_diag["foc_g"]
        accum["pre_hjb"] += pre_diag["hjb"]
        accum["pre_foc_d"] += pre_diag["foc_d"]
        accum["pre_foc_g"] += pre_diag["foc_g"]
        accum["root_level"] += root_probe["level_mean"]
        accum["pre_level"] += pre_probe["level_mean"]
    for key in accum:
        accum[key] /= max(1, batches)
    gap = boundary_gap_loss(root_model, pre_model, n, dtype, device, detach_root=True)
    accum["boundary_gap"] = float(gap.detach().cpu())
    accum.update(jump_validation(pre_model, n, batches, dtype, device, seed or 9876))
    return accum


def save_all(root_model, pre_model, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    torch.save(root_model.v_nn.state_dict(), os.path.join(export_dir, "root_v_recentered.pt"))
    torch.save(root_model.i_g_nn.state_dict(), os.path.join(export_dir, "root_i_g.pt"))
    torch.save(root_model.i_d_nn.state_dict(), os.path.join(export_dir, "root_i_d.pt"))
    torch.save(pre_model.v_nn.state_dict(), os.path.join(export_dir, "pre_damage_posttech_v_recentered.pt"))
    torch.save(pre_model.i_g_nn.state_dict(), os.path.join(export_dir, "pre_damage_posttech_i_g.pt"))
    torch.save(pre_model.i_d_nn.state_dict(), os.path.join(export_dir, "pre_damage_posttech_i_d.pt"))
    if pre_model.jump_net is not None:
        torch.save(pre_model.jump_net.state_dict(), os.path.join(export_dir, "jump_robust_net.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--valid_batch_size", type=int, default=512)
    ap.add_argument("--valid_batches", type=int, default=2)
    ap.add_argument("--valid_seed", type=int, default=12345)
    ap.add_argument("--lr_value", type=float, default=4e-4)
    ap.add_argument("--lr_control", type=float, default=4e-4)
    ap.add_argument("--lr_jump", type=float, default=4e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.01)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--activation", default="swish")
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--value_arch", choices=["clean", "frozen_middle"], default="clean")
    ap.add_argument("--root_anchor", choices=["surrogate", "constant"], default="surrogate")
    ap.add_argument("--root_anchor_value", type=float, default=0.0)
    ap.add_argument("--root_control_init", choices=["surrogate", "clean", "torch_scratch"], default="surrogate")
    ap.add_argument("--pre_control_init_rate", type=float, default=0.02)
    ap.add_argument("--freeze_root_controls", action="store_true")
    ap.add_argument("--couple_anchor_grad", action="store_true")
    ap.add_argument("--couple_neighbor_grad", action="store_true")
    ap.add_argument("--foc_in_value", action="store_true")
    ap.add_argument("--root_weight", type=float, default=1.0)
    ap.add_argument("--pre_weight", type=float, default=1.0)
    ap.add_argument("--root_control_weight", type=float, default=1.0)
    ap.add_argument("--pre_control_weight", type=float, default=1.0)
    ap.add_argument("--control_steps", type=int, default=1)
    ap.add_argument("--boundary_weight", type=float, default=0.0)
    ap.add_argument("--boundary_batch", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--export", default=None)
    ap.add_argument("--jump_robust_mode", choices=["closed", "residual", "direct"], default="residual")
    ap.add_argument("--jump_width", type=int, default=64)
    ap.add_argument("--jump_depth", type=int, default=3)
    ap.add_argument("--jump_residual_scale", type=float, default=0.05)
    ap.add_argument("--jump_direct_scale", type=float, default=6.0)
    ap.add_argument("--jump_steps", type=int, default=1)
    ap.add_argument("--jump_duality_weight", type=float, default=1.0)
    ap.add_argument("--logg_clip", type=float, default=20.0)
    ap.add_argument("--no_outer_envelope_detach", action="store_true")
    ap.add_argument("--jump_pretrained_path", default=None)
    ap.add_argument("--freeze_jump_net", action="store_true")
    args = ap.parse_args()

    dtype = _dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root_model = build_root(args, dtype, device)
    pre_model = build_pre_jump(args, root_model, dtype, device)

    if args.freeze_root_controls:
        for net in (root_model.i_g_nn, root_model.i_d_nn):
            for param in net.parameters():
                param.requires_grad_(False)

    value_modules = [root_model.v_nn, pre_model.v_nn]
    control_modules = [pre_model.i_g_nn, pre_model.i_d_nn]
    if not args.freeze_root_controls:
        control_modules.extend([root_model.i_g_nn, root_model.i_d_nn])
    jump_modules = [] if pre_model.jump_net is None else [pre_model.jump_net]

    opt_v = torch.optim.Adam(trainable_parameters(value_modules), lr=args.lr_value)
    opt_c = torch.optim.Adam(trainable_parameters(control_modules), lr=args.lr_control)
    jump_params = trainable_parameters(jump_modules)
    opt_j = None if not jump_params else torch.optim.Adam(jump_params, lr=args.lr_jump)
    warmup_steps = max(1, int(args.steps * args.warmup_frac))
    sched_v = WarmupCosine(args.lr_value, args.steps, warmup_steps, args.min_lr)
    sched_c = WarmupCosine(args.lr_control, args.steps, warmup_steps, args.min_lr)
    sched_j = WarmupCosine(args.lr_jump, args.steps, warmup_steps, args.min_lr)

    print(
        f"[setup] two-regime learned-jump mode={args.jump_robust_mode} dtype={args.dtype} "
        f"device={device} steps={args.steps} bs={args.batch_size} jump_steps={args.jump_steps}",
        flush=True,
    )
    history = []
    val0 = validation(root_model, pre_model, args.valid_batch_size, args.valid_batches, dtype, device, seed=args.valid_seed)
    print(
        f"[step      0] root_hjb={val0['root_hjb']:.5e} pre_hjb={val0['pre_hjb']:.5e} "
        f"jump_gap={val0['jump_gap']:.2e} logg_err={val0['logg_err']:.2e} "
        f"gap={val0['boundary_gap']:.3e} pre_foc=({val0['pre_foc_d']:.2e},{val0['pre_foc_g']:.2e})",
        flush=True,
    )
    history.append({"step": 0, **val0})
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for group in opt_v.param_groups:
            group["lr"] = sched_v(step)
        for group in opt_c.param_groups:
            group["lr"] = sched_c(step)
        if opt_j is not None:
            for group in opt_j.param_groups:
                group["lr"] = sched_j(step)

        if opt_j is not None:
            for module in value_modules + control_modules:
                set_module_grad(module, False)
            set_module_grad(pre_model.jump_net, True)
            for _ in range(max(1, args.jump_steps)):
                sample = sample_box(pre_model.params, args.batch_size, dtype, device)
                terms = pre_model.jump_only_terms(sample)
                jump_loss = torch.mean(terms["jump_gap"]) + args.jump_duality_weight * rms(terms["logg_err"])
                opt_j.zero_grad(set_to_none=True)
                jump_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters(jump_modules), args.grad_clip)
                opt_j.step()

        for module in value_modules:
            set_module_grad(module, True)
        for module in control_modules:
            set_module_grad(module, False)
        set_module_grad(pre_model.jump_net, False)
        root_sample = sample_box(root_model.params, args.batch_size, dtype, device)
        pre_sample = sample_box(pre_model.params, args.batch_size, dtype, device)
        root_loss, root_diag = value_loss(root_model, root_sample, include_foc=args.foc_in_value)
        pre_loss, pre_diag = value_loss(pre_model, pre_sample, include_foc=args.foc_in_value)
        gap = boundary_gap_loss(
            root_model, pre_model, args.boundary_batch, dtype, device,
            detach_root=not args.couple_anchor_grad,
        )
        value_obj = args.root_weight * root_loss + args.pre_weight * pre_loss
        if args.boundary_weight > 0:
            value_obj = value_obj + args.boundary_weight * gap
        opt_v.zero_grad(set_to_none=True)
        value_obj.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(value_modules), args.grad_clip)
        opt_v.step()

        for module in value_modules:
            set_module_grad(module, False)
        for module in control_modules:
            set_module_grad(module, True)
        set_module_grad(pre_model.jump_net, False)
        for _ in range(max(1, args.control_steps)):
            root_sample = sample_box(root_model.params, args.batch_size, dtype, device)
            pre_sample = sample_box(pre_model.params, args.batch_size, dtype, device)
            root_control, _ = control_loss(root_model, root_sample)
            pre_control, _ = control_loss(pre_model, pre_sample)
            control_obj = args.pre_control_weight * pre_control if args.freeze_root_controls else (
                args.root_control_weight * root_control + args.pre_control_weight * pre_control
            )
            opt_c.zero_grad(set_to_none=True)
            control_obj.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(control_modules), args.grad_clip)
            opt_c.step()

        if step % args.log_every == 0 or step == args.steps:
            for module in value_modules + control_modules:
                set_module_grad(module, True)
            if pre_model.jump_net is not None:
                set_module_grad(pre_model.jump_net, False)
            val = validation(root_model, pre_model, args.valid_batch_size, args.valid_batches, dtype, device, seed=args.valid_seed)
            dt = time.time() - t0
            print(
                f"[step {step:6d}] root_hjb={val['root_hjb']:.5e} pre_hjb={val['pre_hjb']:.5e} "
                f"jump_gap={val['jump_gap']:.2e} logg_err={val['logg_err']:.2e} "
                f"gap={val['boundary_gap']:.3e} pre_foc=({val['pre_foc_d']:.2e},{val['pre_foc_g']:.2e}) ({dt:.0f}s)",
                flush=True,
            )
            history.append({
                "step": step,
                **val,
                "train_root_hjb": root_diag["hjb"],
                "train_pre_hjb": pre_diag["hjb"],
            })

    final = history[-1]
    if args.export:
        save_all(root_model, pre_model, args.export)
        with open(os.path.join(args.export, "history.jsonl"), "w", encoding="utf-8") as f:
            for row in history:
                f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
        print(f"[export] wrote checkpoints/history to {args.export}", flush=True)

    result = {
        "mode": "two_regime_learned_jump",
        "jump_robust_mode": args.jump_robust_mode,
        "steps": args.steps,
        "root_hjb_final": final["root_hjb"],
        "pre_hjb_final": final["pre_hjb"],
        "jump_gap_final": final["jump_gap"],
        "logg_err_final": final["logg_err"],
        "boundary_gap_final": final["boundary_gap"],
    }
    print("RESULT_JSON " + json.dumps(result, default=_json_default, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
