"""Shared Torch utilities for anchored value/costate experiments.

This folder is intentionally separate from production TensorFlow/Torch ports.
The goal is to test the identification fix:

    v_rec(x) = phi(x) - phi(x_anchor; pseudo(x)) + anchor_value(x)

The subtraction removes the nearly free additive constant from the value net
while preserving derivatives with respect to true state variables.
"""

import math
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
MODELS_TORCH = ROOT / "models_torch"
MODELS_TORCH_TRAIN = ROOT / "models_torch_train"
for _path in (str(MODELS_TORCH_TRAIN), str(MODELS_TORCH)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from params import PARAMS, investment_rate_activation  # noqa: E402


ROOT_BOUNDS_7 = [
    (4.0, 7.0),       # logK
    (0.01, 0.99),    # Z
    (0.0, 4.0),      # Y
    (0.0, 1.0 / 3.0),# lambda3
    (0.0, 0.0),      # constant A_g''
    (-3.0, 5.0),     # logxi
    (-3.0, 5.0),     # duplicate logxi
]

PREDAMAGE_POSTTECH_BOUNDS_6 = [
    (4.0, 7.0),       # logK
    (0.01, 0.99),    # Z
    (0.0, 4.0),      # Y
    (0.0, 0.0),      # constant A_g''
    (-3.0, 5.0),     # logxi
    (-3.0, 5.0),     # duplicate logxi
]


class WarmupCosine:
    """Small Torch mirror of the repo's TF warmup-cosine schedule."""

    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int = 0, min_lr: float = 0.0):
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step: int) -> float:
        step_f = float(step)
        if self.warmup_steps > 0 and step_f < self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * step_f / max(1.0, self.warmup_steps)
        progress = min(
            max((step_f - self.warmup_steps) / max(1.0, self.total_steps - self.warmup_steps), 0.0),
            1.0,
        )
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))


class InputNorm(nn.Module):
    """Fixed affine normalization to roughly [-1, 1]."""

    def __init__(self, bounds: Sequence[Tuple[float, float]]):
        super().__init__()
        lows = np.array([b[0] for b in bounds], dtype=np.float32)
        highs = np.array([b[1] for b in bounds], dtype=np.float32)
        spans = highs - lows
        const = spans <= 0.0
        safe_spans = np.where(const, 1.0, spans)
        scale = np.where(const, 0.0, 2.0 / safe_spans).astype(np.float32)
        shift = np.where(const, 0.0, -(2.0 * lows / safe_spans + 1.0)).astype(np.float32)
        self.register_buffer("scale", torch.tensor(scale.reshape(1, -1)))
        self.register_buffer("shift", torch.tensor(shift.reshape(1, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.to(dtype=x.dtype, device=x.device) + self.shift.to(dtype=x.dtype, device=x.device)


def _activation(name: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]]):
    if callable(name):
        return name
    if name is None or str(name).lower() in {"linear", "none"}:
        return None
    key = str(name).lower()
    if key in {"swish", "silu"}:
        return lambda x: x * torch.sigmoid(x)
    if key == "tanh":
        return torch.tanh
    if key == "relu":
        return torch.relu
    if key == "softplus":
        return torch.nn.functional.softplus
    raise ValueError(f"Unsupported activation {name!r}")


class CleanMLP(nn.Module):
    """Plain normalized residual MLP used for both value phi and controls."""

    def __init__(
        self,
        input_dim: int,
        bounds: Sequence[Tuple[float, float]],
        hidden: Sequence[int] = (32, 32, 32, 32),
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = "swish",
        final_activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
        seed: Optional[int] = None,
        residual: bool = True,
        output_dim: int = 1,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.input_dim = int(input_dim)
        self.norm = InputNorm(bounds)
        self.activation = _activation(activation)
        self.final_activation = _activation(final_activation)
        self.residual = bool(residual)

        widths = list(hidden)
        layers = []
        prev = self.input_dim
        for width in widths:
            layers.append(nn.Linear(prev, int(width)))
            prev = int(width)
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(prev, int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        prev_h = None
        for layer in self.layers:
            y = layer(h if prev_h is None else prev_h)
            if self.activation is not None:
                y = self.activation(y)
            if self.residual and prev_h is not None and y.shape[-1] == prev_h.shape[-1]:
                prev_h = prev_h + y
            else:
                prev_h = y
        out = self.out(prev_h)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out


class FrozenMiddleMLP(CleanMLP):
    """4-layer MLP whose middle two hidden layers can be copied and frozen.

    This implements the transfer-learning architecture discussed in the notes:
    trainable input adapter -> frozen benchmark middle -> trainable output head.
    Gradients pass through the frozen layers, but their parameters do not update.
    """

    def freeze_middle_from(self, source_net: nn.Module, source_layers=(1, 2)) -> "FrozenMiddleMLP":
        if len(self.layers) < 4:
            raise ValueError("FrozenMiddleMLP expects at least four hidden layers")
        for idx in source_layers:
            src = source_net.dense_layers[idx]
            dst = self.layers[idx]
            if src.weight.shape != dst.weight.shape:
                raise ValueError(
                    f"Cannot copy source dense layer {idx}: "
                    f"{tuple(src.weight.shape)} != {tuple(dst.weight.shape)}"
                )
            with torch.no_grad():
                dst.weight.copy_(src.weight.to(dtype=dst.weight.dtype, device=dst.weight.device))
                if src.bias is not None and dst.bias is not None:
                    dst.bias.copy_(src.bias.to(dtype=dst.bias.dtype, device=dst.bias.device))
            for param in dst.parameters():
                param.requires_grad_(False)
        return self


def build_value_phi(
    input_dim: int,
    bounds,
    hidden,
    activation: str,
    seed: int,
    residual: bool = True,
    value_arch: str = "clean",
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    ckpt_dir: Optional[str] = None,
) -> nn.Module:
    if value_arch == "clean":
        return CleanMLP(
            input_dim=input_dim,
            bounds=bounds,
            hidden=hidden,
            activation=activation,
            final_activation=None,
            seed=seed,
            residual=residual,
        )
    if value_arch == "frozen_middle":
        if len(hidden) != 4 or any(int(width) != 32 for width in hidden):
            raise ValueError("frozen_middle requires exactly 4 hidden layers of width 32")
        phi = FrozenMiddleMLP(
            input_dim=input_dim,
            bounds=bounds,
            hidden=hidden,
            activation=activation,
            final_activation=None,
            seed=seed,
            residual=False,
        )
        from tf_to_torch_loader import build_root_torch_model

        source = build_root_torch_model(load_surrogate=True, ckpt_dir=ckpt_dir, dtype=dtype or torch.float32)
        source.v_nn.to(device=device or torch.device("cpu"), dtype=dtype or torch.float32)
        phi.to(device=device or torch.device("cpu"), dtype=dtype or torch.float32)
        phi.freeze_middle_from(source.v_nn, source_layers=(1, 2))
        return phi
    raise ValueError(f"Unknown value_arch={value_arch!r}")


AnchorFn = Callable[[torch.Tensor], torch.Tensor]


class RecenteredValueNet(nn.Module):
    """Value module with hard level pinning.

    phi_anchor_fn maps a batch X to an anchor-X batch with the same pseudo state
    columns but fixed true states.

    anchor_value_fn maps X to the desired value at the anchor point.  It may be a
    constant scalar, a detached post-jump value, or a live neighbor value.
    """

    def __init__(
        self,
        phi: CleanMLP,
        phi_anchor_fn: Callable[[torch.Tensor], torch.Tensor],
        anchor_value_fn: AnchorFn,
    ):
        super().__init__()
        self.phi = phi
        self.phi_anchor_fn = phi_anchor_fn
        self.anchor_value_fn = anchor_value_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi_x = self.phi(x)
        x0 = self.phi_anchor_fn(x)
        phi_0 = self.phi(x0)
        v0 = self.anchor_value_fn(x)
        if not torch.is_tensor(v0):
            v0 = torch.tensor(float(v0), dtype=x.dtype, device=x.device)
        if v0.ndim == 0:
            v0 = v0.reshape(1, 1).expand_as(phi_x)
        return phi_x - phi_0 + v0.to(dtype=x.dtype, device=x.device)


class DetachedModule(nn.Module):
    """Wrap a module and detach its output for stable teacher-style coupling."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x).detach()


def rms(x: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x) + eps)


def sample_box(params: dict, n: int, dtype: torch.dtype, device: torch.device):
    def col(lo: float, hi: float):
        return lo + (hi - lo) * torch.rand((n, 1), dtype=dtype, device=device)

    return (
        col(params["logK_min"], params["logK_max"]),
        col(params["Z_min"], params["Z_max"]),
        col(params["Y_min"], params["Y_max"]),
        col(params["logR_min"], params["logR_max"]),
        col(params["lambda3_min"] if "lambda3_min" in params else params["λ3_min"],
            params["lambda3_max"] if "lambda3_max" in params else params["λ3_max"]),
        col(params["logξ_min"], params["logξ_max"]),
    )


def set_requires_true_states(sample):
    logK, Z, Y, logR, lam3, logxi = sample
    return (
        logK.detach().requires_grad_(True),
        Z.detach().requires_grad_(True),
        Y.detach().requires_grad_(True),
        logR.detach().requires_grad_(True),
        lam3,
        logxi,
    )


def inverse_investment_raw(rate: float, theta: float) -> float:
    """Raw network output that maps to a target investment rate."""
    theta = float(theta)
    rate = min(max(float(rate), -1.0 / theta + 1e-6), 1.0 - 1e-6)
    exp_2x = (1.0 + 1.0 / theta) / (1.0 - rate) - 1.0
    return 0.5 * math.log(max(exp_2x, 1e-8))


def initialise_control_to_rate(net: nn.Module, theta: float, rate: float = 0.02) -> nn.Module:
    """Start controls from a feasible, low-investment policy."""
    raw = inverse_investment_raw(rate, theta)
    if hasattr(net, "out"):
        with torch.no_grad():
            net.out.weight.normal_(mean=0.0, std=1e-3)
            net.out.bias.fill_(raw)
    return net


def build_control_net(
    input_dim: int,
    bounds,
    theta: float,
    seed: int,
    activation: str = "tanh",
    init_rate: Optional[float] = 0.02,
) -> CleanMLP:
    net = CleanMLP(
        input_dim=input_dim,
        bounds=bounds,
        hidden=(32, 32, 32, 32),
        activation=activation,
        final_activation=investment_rate_activation(theta),
        seed=seed,
    )
    if init_rate is not None:
        initialise_control_to_rate(net, theta, init_rate)
    return net


def terminal_anchor_x_root(x: torch.Tensor) -> torch.Tensor:
    """Root input anchor: fixed true states, same lambda3/logxi columns."""
    out = x.clone()
    out[:, 0:1] = 0.5 * (PARAMS["logK_min"] + PARAMS["logK_max"])
    out[:, 1:2] = 0.5 * (PARAMS["Z_min"] + PARAMS["Z_max"])
    out[:, 2:3] = 0.5 * (PARAMS["Y_min"] + PARAMS["Y_max"])
    out[:, 4:5] = PARAMS["A_g_prime_prime"]
    return out


def predamage_posttech_anchor_x(x: torch.Tensor) -> torch.Tensor:
    """PreDamagePostTech input anchor: same logK/Z, Y at damage-jump target."""
    out = x.clone()
    out[:, 2:3] = PARAMS["y_upper"]
    out[:, 3:4] = PARAMS["A_g_prime_prime"]
    return out


def constant_anchor(value: float) -> AnchorFn:
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), float(value), dtype=x.dtype, device=x.device)

    return _fn


def root_surrogate_anchor_value(dtype: torch.dtype, device: torch.device, ckpt_dir: Optional[str] = None) -> float:
    """Read the existing TF-surrogate root value at the terminal anchor point."""
    from tf_to_torch_loader import build_root_torch_model

    model = build_root_torch_model(load_surrogate=True, ckpt_dir=ckpt_dir, dtype=dtype)
    model.v_nn.to(device=device, dtype=dtype)
    logxi_mid = 0.5 * (PARAMS["logξ_min"] + PARAMS["logξ_max"])
    x0 = torch.tensor(
        [[
            0.5 * (PARAMS["logK_min"] + PARAMS["logK_max"]),
            0.5 * (PARAMS["Z_min"] + PARAMS["Z_max"]),
            0.5 * (PARAMS["Y_min"] + PARAMS["Y_max"]),
            0.5 * (PARAMS["λ3_min"] + PARAMS["λ3_max"]),
            PARAMS["A_g_prime_prime"],
            logxi_mid,
            logxi_mid,
        ]],
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        return float(model.v_nn(x0).reshape(()).cpu())


def root_value_at_damage_anchor(root_value_net: nn.Module, detach: bool = True) -> AnchorFn:
    """Anchor pre-damage/post-tech to average terminal post-damage value.

    The existing TF jump operator evaluates post-damage value at Y=y_upper and
    averages over the discrete lambda3 grid.  This mirrors that operator.
    """

    lambda_grid = [float(v) for v in PARAMS["λ3_values"]]

    def _fn(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        logk = x[:, 0:1]
        z = x[:, 1:2]
        logxi = x[:, 4:5]
        chunks = []
        for lam in lambda_grid:
            x_root = torch.cat(
                [
                    logk,
                    z,
                    torch.full((n, 1), PARAMS["y_upper"], dtype=x.dtype, device=x.device),
                    torch.full((n, 1), lam, dtype=x.dtype, device=x.device),
                    torch.full((n, 1), PARAMS["A_g_prime_prime"], dtype=x.dtype, device=x.device),
                    logxi,
                    logxi,
                ],
                dim=1,
            )
            chunks.append(root_value_net(x_root))
        out = torch.stack(chunks, dim=0).mean(dim=0)
        return out.detach() if detach else out

    return _fn


def feasibility_penalty(c: torch.Tensor, gig: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
    return (
        torch.nn.functional.softplus(-100.0 * c).mean() / 100.0
        + torch.nn.functional.softplus(-100.0 * gig).mean() / 100.0
        + torch.nn.functional.softplus(-100.0 * gid).mean() / 100.0
    )


def value_loss(model, sample, include_foc: bool = False, feas_weight: float = 10.0):
    logK, Z, Y, logR, lam3, logxi = set_requires_true_states(sample)
    out = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    rhs, pv, dv_dY, c, gig, gid, foc_d, foc_g = out[:8]
    resid = rhs - pv
    loss = rms(resid) + feas_weight * feasibility_penalty(c, gig, gid)
    if include_foc:
        loss = loss + rms(foc_d) + rms(foc_g)
    diag = {
        "hjb": float(rms(resid).detach().cpu()),
        "foc_d": float(rms(foc_d).detach().cpu()),
        "foc_g": float(rms(foc_g).detach().cpu()),
        "c_min": float(c.min().detach().cpu()),
    }
    return loss, diag


def control_loss(model, sample, feas_weight: float = 10.0):
    logK, Z, Y, logR, lam3, logxi = set_requires_true_states(sample)
    out = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    _rhs, _pv, dv_dY, c, gig, gid, foc_d, foc_g = out[:8]
    mask = ((Y > model.params["y_upper"]).to(dv_dY.dtype)) * ((dv_dY > 0).to(dv_dY.dtype))
    loss_dvdy = dv_dY * mask
    loss = rms(foc_d) + rms(foc_g) + rms(loss_dvdy) + feas_weight * feasibility_penalty(c, gig, gid)
    diag = {
        "foc_d": float(rms(foc_d).detach().cpu()),
        "foc_g": float(rms(foc_g).detach().cpu()),
        "dvdy": float(rms(loss_dvdy).detach().cpu()),
        "c_min": float(c.min().detach().cpu()),
    }
    return loss, diag


def boundary_gap_loss(root_model, pre_model, n: int, dtype: torch.dtype, device: torch.device, detach_root: bool = True):
    """RMS gap at the damage-jump value-matching slice for PreDamagePostTech."""
    logk = PARAMS["logK_min"] + (PARAMS["logK_max"] - PARAMS["logK_min"]) * torch.rand((n, 1), dtype=dtype, device=device)
    z = PARAMS["Z_min"] + (PARAMS["Z_max"] - PARAMS["Z_min"]) * torch.rand((n, 1), dtype=dtype, device=device)
    y = torch.full((n, 1), PARAMS["y_upper"], dtype=dtype, device=device)
    logxi = PARAMS["logξ_min"] + (PARAMS["logξ_max"] - PARAMS["logξ_min"]) * torch.rand((n, 1), dtype=dtype, device=device)
    x_pre = torch.cat(
        [logk, z, y, torch.full_like(y, PARAMS["A_g_prime_prime"]), logxi, logxi],
        dim=1,
    )
    v_pre = pre_model.v_nn(x_pre)
    vals = []
    for lam in PARAMS["λ3_values"]:
        x_root = torch.cat(
            [
                logk,
                z,
                y,
                torch.full_like(y, float(lam)),
                torch.full_like(y, PARAMS["A_g_prime_prime"]),
                logxi,
                logxi,
            ],
            dim=1,
        )
        vals.append(root_model.v_nn(x_root))
    v_root = torch.stack(vals, dim=0).mean(dim=0)
    if detach_root:
        v_root = v_root.detach()
    return rms(v_pre - v_root)


def probe_level(model, n: int, dtype: torch.dtype, device: torch.device):
    sample = sample_box(model.params, n, dtype, device)
    logK, Z, Y, logR, lam3, logxi = set_requires_true_states(sample)
    rhs, pv, *_ = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    resid = rhs - pv
    return {
        "level_mean": float((pv / model.params["δ"]).mean().detach().cpu()),
        "hjb": float(rms(resid).detach().cpu()),
    }


def trainable_parameters(modules: Iterable[nn.Module]):
    params = []
    for module in modules:
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params
