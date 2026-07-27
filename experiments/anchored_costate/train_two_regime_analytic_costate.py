"""Warm-start two-regime training with analytical FOC-implied costate targets.

The HJB/low-mode pass fixes value-level residual offsets.  This trainer adds a
cheaper analytical correction for the remaining costate/FOC error:

    q_d = V_logK - Z V_Z       = MU / phi_d'(i_d)
    q_g = V_logK + (1-Z) V_Z   = MU / phi_g'(i_g)

so that

    V_Z      = q_g - q_d
    V_logK  = (1-Z) q_d + Z q_g.

The targets are computed from detached controls, following the envelope logic:
the value step corrects marginal prices, while the control step separately
updates policies.  Fixed slice grids are cached once and reused each step.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS_TORCH = ROOT / "models_torch"
MODELS_TORCH_TRAIN = ROOT / "models_torch_train"
for _path in (str(HERE), str(MODELS_TORCH_TRAIN), str(MODELS_TORCH)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from anchored_nets import (  # noqa: E402
    WarmupCosine,
    boundary_gap_loss,
    control_loss,
    rms,
    sample_box,
    trainable_parameters,
    value_loss,
)
from params import PARAMS  # noqa: E402
from tf_to_torch_loader import DEFAULT_CKPT_DIR  # noqa: E402
from train_two_regime_joint import (  # noqa: E402
    _dtype,
    _json_default,
    build_pre,
    build_root,
    save_all,
    validation,
)
from train_two_regime_lowmode import (  # noqa: E402
    DEFAULT_WARM_START as DEFAULT_BASE_WARM_START,
    dct_matrix_torch,
    load_joint_checkpoint,
    lowmode_mask,
    make_slice_grid,
    slice_specs,
)


DEFAULT_ANALYTIC_WARM_START = (
    "experiments/anchored_costate/results/lowmode_map_20260701_122130/"
    "dc1_rms025_foc025_seed0_task0"
)


def grad1(y, x, create_graph=True):
    g = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        g = torch.zeros_like(x)
    return g.reshape(-1, 1)


def root_input(logK, Z, Y, _logR, lam3, logxi, params):
    return torch.cat(
        [logK, Z, Y, lam3, params["A_g_prime_prime"] * torch.ones_like(Y), logxi, logxi],
        dim=1,
    )


def pre_input(logK, Z, Y, _logR, _lam3, logxi, params):
    return torch.cat(
        [logK, Z, Y, params["A_g_prime_prime"] * torch.ones_like(Y), logxi, logxi],
        dim=1,
    )


def set_state_grads(sample):
    logK, Z, Y, logR, lam3, logxi = sample
    return (
        logK.detach().requires_grad_(True),
        Z.detach().requires_grad_(True),
        Y.detach().requires_grad_(True),
        logR.detach(),
        lam3.detach(),
        logxi.detach(),
    )


def costate_projection_terms(model, sample, regime, relative=False, scale_floor=0.05):
    """Vectorized analytical target for V_logK and V_Z.

    Controls are evaluated under no_grad and detached.  The value derivative is
    differentiated with create_graph=True so this loss can train value params.
    """
    p = model.params
    logK, Z, Y, logR, lam3, logxi = set_state_grads(sample)
    input_fn = root_input if regime == "root" else pre_input
    X = input_fn(logK, Z, Y, logR, lam3, logxi, p)
    v = model.v_nn(X)
    dv_logK = grad1(v, logK, create_graph=True)
    dv_Z = grad1(v, Z, create_graph=True)
    qd_nn = dv_logK - Z * dv_Z
    qg_nn = dv_logK + (1.0 - Z) * dv_Z

    with torch.no_grad():
        Xc = input_fn(logK.detach(), Z.detach(), Y.detach(), logR, lam3, logxi, p)
        i_g = model.i_g_nn(Xc)
        i_d = model.i_d_nn(Xc)
        c = (p["A_d"] - i_d) * (1.0 - Z.detach()) + (p["A_g_prime_prime"] - i_g) * Z.detach()
        c = torch.clamp(c, min=1e-8).reshape(-1, 1)
        inside_d = torch.clamp(1.0 + p["θ_d"] * i_d, min=1e-8).reshape(-1, 1)
        inside_g = torch.clamp(1.0 + p["θ_g"] * i_g, min=1e-8).reshape(-1, 1)
        mu = p["δ"] / c
        qd_target = mu * inside_d / (p["Γ_d"] * p["θ_d"])
        qg_target = mu * inside_g / (p["Γ_g"] * p["θ_g"])
        pz_target = qg_target - qd_target
        pk_target = (1.0 - Z.detach()) * qd_target + Z.detach() * qg_target

    err_qd = qd_nn - qd_target
    err_qg = qg_nn - qg_target
    err_pk = dv_logK - pk_target
    err_pz = dv_Z - pz_target
    if relative:
        err_qd = err_qd / torch.clamp(torch.abs(qd_target), min=scale_floor)
        err_qg = err_qg / torch.clamp(torch.abs(qg_target), min=scale_floor)
        err_pk = err_pk / torch.clamp(torch.abs(pk_target), min=scale_floor)
        err_pz = err_pz / torch.clamp(torch.abs(pz_target), min=scale_floor)

    return {
        "err_qd": err_qd,
        "err_qg": err_qg,
        "err_pk": err_pk,
        "err_pz": err_pz,
        "dv_logK": dv_logK,
        "dv_Z": dv_Z,
        "qd_target": qd_target,
        "qg_target": qg_target,
        "c": c.detach(),
    }


def costate_projection_loss(model, sample, regime, mode="both", relative=False, scale_floor=0.05):
    terms = costate_projection_terms(model, sample, regime, relative=relative, scale_floor=scale_floor)
    q_loss = rms(terms["err_qd"]) + rms(terms["err_qg"])
    p_loss = rms(terms["err_pk"]) + rms(terms["err_pz"])
    if mode == "q":
        loss = q_loss
    elif mode == "p":
        loss = p_loss
    elif mode == "both":
        loss = q_loss + p_loss
    else:
        raise ValueError(f"Unknown costate mode {mode!r}")
    diag = {
        "costate_qd": float(rms(terms["err_qd"]).detach().cpu()),
        "costate_qg": float(rms(terms["err_qg"]).detach().cpu()),
        "costate_pk": float(rms(terms["err_pk"]).detach().cpu()),
        "costate_pz": float(rms(terms["err_pz"]).detach().cpu()),
        "costate_c_min": float(terms["c"].min().detach().cpu()),
    }
    return loss, diag, terms


def cache_slice_grids(args, specs, dtype, device):
    return [
        make_slice_grid(
            spec,
            args.slice_grid,
            args.slice_z_min,
            args.slice_z_max,
            args.slice_y_min,
            args.slice_y_max,
            dtype,
            device,
        )
        for spec in specs
    ]


def pde_chunk(model, chunk):
    logK, Z, Y, logR, lam3, logxi = chunk
    logK = logK.detach().requires_grad_(True)
    Z = Z.detach().requires_grad_(True)
    Y = Y.detach().requires_grad_(True)
    logR = logR.detach().requires_grad_(True)
    rhs, pv, _dvdy, c, gig, gid, foc_d, foc_g = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    return rhs - pv, foc_d, foc_g, c, gig, gid


def lowmode_loss_cached(pre_model, grids, args, D, mask, dtype, device):
    slice_losses = []
    diags = {key: [] for key in ("slice_rms", "slice_mean", "slice_dc", "slice_low", "slice_foc_d", "slice_foc_g")}
    n = args.slice_grid * args.slice_grid
    chunk_size = n if args.slice_chunk_size <= 0 else min(n, args.slice_chunk_size)

    for full in grids:
        resid_parts, foc_d_parts, foc_g_parts = [], [], []
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            chunk = tuple(col[start:stop] for col in full)
            resid, foc_d, foc_g, _c, _gig, _gid = pde_chunk(pre_model, chunk)
            resid_parts.append(resid)
            foc_d_parts.append(foc_d)
            foc_g_parts.append(foc_g)
        resid = torch.cat(resid_parts, dim=0).reshape(args.slice_grid, args.slice_grid)
        foc_d = torch.cat(foc_d_parts, dim=0)
        foc_g = torch.cat(foc_g_parts, dim=0)
        coeff = D @ resid @ D.T
        low_loss = rms(coeff[mask])
        dc_loss = torch.abs(coeff[0, 0]) / float(args.slice_grid)
        resid_rms = rms(resid)
        foc_loss = rms(foc_d) + rms(foc_g)
        slice_losses.append(
            args.slice_low_weight * low_loss
            + args.slice_dc_weight * dc_loss
            + args.slice_rms_weight * resid_rms
            + args.slice_foc_weight * foc_loss
        )
        with torch.no_grad():
            diags["slice_rms"].append(float(resid_rms.detach().cpu()))
            diags["slice_mean"].append(float(torch.mean(resid).detach().cpu()))
            diags["slice_dc"].append(float(dc_loss.detach().cpu()))
            diags["slice_low"].append(float(low_loss.detach().cpu()))
            diags["slice_foc_d"].append(float(rms(foc_d).detach().cpu()))
            diags["slice_foc_g"].append(float(rms(foc_g).detach().cpu()))

    loss = torch.stack(slice_losses).mean() if slice_losses else torch.zeros((), dtype=dtype, device=device)
    return loss, {key: float(np.mean(vals)) if vals else 0.0 for key, vals in diags.items()}


def slice_costate_loss_cached(pre_model, grids, args, D, mask):
    losses = []
    diags = {key: [] for key in ("slice_cost_qd", "slice_cost_qg", "slice_cost_pk", "slice_cost_pz", "slice_cost_low")}
    for full in grids:
        loss, diag, terms = costate_projection_loss(
            pre_model,
            full,
            "pre",
            mode=args.costate_mode,
            relative=args.costate_relative,
            scale_floor=args.costate_scale_floor,
        )
        qd_field = terms["err_qd"].reshape(args.slice_grid, args.slice_grid)
        qg_field = terms["err_qg"].reshape(args.slice_grid, args.slice_grid)
        low = rms((D @ qd_field @ D.T)[mask]) + rms((D @ qg_field @ D.T)[mask])
        losses.append(args.slice_costate_rms_weight * loss + args.slice_costate_low_weight * low)
        diags["slice_cost_qd"].append(diag["costate_qd"])
        diags["slice_cost_qg"].append(diag["costate_qg"])
        diags["slice_cost_pk"].append(diag["costate_pk"])
        diags["slice_cost_pz"].append(diag["costate_pz"])
        diags["slice_cost_low"].append(float(low.detach().cpu()))
    reduced = torch.stack(losses).mean() if losses else torch.zeros((), device=D.device, dtype=D.dtype)
    return reduced, {key: float(np.mean(vals)) if vals else 0.0 for key, vals in diags.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--costate_batch_size", type=int, default=256)
    ap.add_argument("--valid_batch_size", type=int, default=512)
    ap.add_argument("--valid_batches", type=int, default=2)
    ap.add_argument("--valid_seed", type=int, default=12345)
    ap.add_argument("--lr_value", type=float, default=3e-5)
    ap.add_argument("--lr_control", type=float, default=5e-5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_frac", type=float, default=0.02)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=250)
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
    ap.add_argument("--root_weight", type=float, default=0.10)
    ap.add_argument("--pre_weight", type=float, default=1.0)
    ap.add_argument("--root_control_weight", type=float, default=0.10)
    ap.add_argument("--pre_control_weight", type=float, default=1.0)
    ap.add_argument("--control_steps", type=int, default=2)
    ap.add_argument("--boundary_weight", type=float, default=0.0)
    ap.add_argument("--boundary_batch", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--warm_start_dir", default=DEFAULT_ANALYTIC_WARM_START)
    ap.add_argument("--fallback_warm_start_dir", default=DEFAULT_BASE_WARM_START)
    ap.add_argument("--export", default=None)

    ap.add_argument("--slice_grid", type=int, default=24)
    ap.add_argument("--slice_chunk_size", type=int, default=144)
    ap.add_argument("--slice_logxis", default="-2.995732273553991,-2.302585092994046")
    ap.add_argument("--slice_logks", default="5.5")
    ap.add_argument("--slice_logrs", default="3.5")
    ap.add_argument("--slice_lam3", type=float, default=1.0 / 6.0)
    ap.add_argument("--slice_z_min", type=float, default=0.05)
    ap.add_argument("--slice_z_max", type=float, default=0.95)
    ap.add_argument("--slice_y_min", type=float, default=0.2)
    ap.add_argument("--slice_y_max", type=float, default=3.8)
    ap.add_argument("--low_radius", type=float, default=4.0)
    ap.add_argument("--slice_low_weight", type=float, default=0.25)
    ap.add_argument("--slice_dc_weight", type=float, default=0.50)
    ap.add_argument("--slice_rms_weight", type=float, default=0.50)
    ap.add_argument("--slice_foc_weight", type=float, default=0.25)
    ap.add_argument("--slice_every", type=int, default=1)

    ap.add_argument("--costate_mode", choices=["q", "p", "both"], default="both")
    ap.add_argument("--costate_relative", action="store_true")
    ap.add_argument("--costate_scale_floor", type=float, default=0.05)
    ap.add_argument("--root_costate_weight", type=float, default=0.00)
    ap.add_argument("--pre_costate_weight", type=float, default=0.25)
    ap.add_argument("--slice_costate_weight", type=float, default=0.25)
    ap.add_argument("--slice_costate_rms_weight", type=float, default=1.0)
    ap.add_argument("--slice_costate_low_weight", type=float, default=0.25)
    ap.add_argument("--costate_every", type=int, default=1)
    args = ap.parse_args()

    dtype = _dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root_model = build_root(args, dtype, device)
    pre_model = build_pre(args, root_model, dtype, device)
    warm_start = args.warm_start_dir
    if warm_start and not Path(warm_start).exists() and args.fallback_warm_start_dir:
        warm_start = args.fallback_warm_start_dir
    if warm_start:
        load_joint_checkpoint(root_model, pre_model, warm_start, device)
        print(f"[warm_start] loaded {warm_start}", flush=True)

    if args.freeze_root_controls:
        for net in (root_model.i_g_nn, root_model.i_d_nn):
            for param in net.parameters():
                param.requires_grad_(False)

    value_modules = [root_model.v_nn, pre_model.v_nn]
    control_modules = [pre_model.i_g_nn, pre_model.i_d_nn]
    if not args.freeze_root_controls:
        control_modules.extend([root_model.i_g_nn, root_model.i_d_nn])

    opt_v = torch.optim.Adam(trainable_parameters(value_modules), lr=args.lr_value)
    opt_c = torch.optim.Adam(trainable_parameters(control_modules), lr=args.lr_control)
    warmup_steps = max(1, int(args.steps * args.warmup_frac))
    sched_v = WarmupCosine(args.lr_value, args.steps, warmup_steps, args.min_lr)
    sched_c = WarmupCosine(args.lr_control, args.steps, warmup_steps, args.min_lr)

    specs = slice_specs(args)
    grids = cache_slice_grids(args, specs, dtype, device)
    D = dct_matrix_torch(args.slice_grid, dtype, device)
    mask = lowmode_mask(args.slice_grid, args.low_radius, dtype, device)

    print(
        f"[setup] analytic-costate dtype={args.dtype} device={device} steps={args.steps} "
        f"bs={args.batch_size} costate_bs={args.costate_batch_size} slices={len(grids)} "
        f"grid={args.slice_grid} costate_mode={args.costate_mode} "
        f"weights=(pre_cost={args.pre_costate_weight}, slice_cost={args.slice_costate_weight})",
        flush=True,
    )

    history = []
    val0 = validation(root_model, pre_model, args.valid_batch_size, args.valid_batches, dtype, device, seed=args.valid_seed)
    slice0, slice_diag0 = lowmode_loss_cached(pre_model, grids, args, D, mask, dtype, device)
    del slice0
    cost0, cost_diag0 = slice_costate_loss_cached(pre_model, grids, args, D, mask)
    del cost0
    print(
        f"[step      0] root_hjb={val0['root_hjb']:.5e} pre_hjb={val0['pre_hjb']:.5e} "
        f"slice_rms={slice_diag0['slice_rms']:.3e} slice_mean={slice_diag0['slice_mean']:.3e} "
        f"slice_cost=({cost_diag0['slice_cost_qd']:.2e},{cost_diag0['slice_cost_qg']:.2e})",
        flush=True,
    )
    history.append({"step": 0, **val0, **slice_diag0, **cost_diag0})
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for group in opt_v.param_groups:
            group["lr"] = sched_v(step)
        for group in opt_c.param_groups:
            group["lr"] = sched_c(step)

        root_sample = sample_box(root_model.params, args.batch_size, dtype, device)
        pre_sample = sample_box(pre_model.params, args.batch_size, dtype, device)
        root_loss, root_diag = value_loss(root_model, root_sample, include_foc=False)
        pre_loss, pre_diag = value_loss(pre_model, pre_sample, include_foc=False)
        value_obj = args.root_weight * root_loss + args.pre_weight * pre_loss
        if args.boundary_weight > 0:
            value_obj = value_obj + args.boundary_weight * boundary_gap_loss(
                root_model,
                pre_model,
                args.boundary_batch,
                dtype,
                device,
                detach_root=not args.couple_anchor_grad,
            )
        if args.slice_every > 0 and step % args.slice_every == 0:
            slice_loss, train_slice_diag = lowmode_loss_cached(pre_model, grids, args, D, mask, dtype, device)
            value_obj = value_obj + slice_loss
        else:
            train_slice_diag = {}

        train_cost_diag = {}
        if args.costate_every > 0 and step % args.costate_every == 0:
            if args.root_costate_weight > 0:
                root_cost_sample = sample_box(root_model.params, args.costate_batch_size, dtype, device)
                root_cost_loss, root_cost_diag, _ = costate_projection_loss(
                    root_model,
                    root_cost_sample,
                    "root",
                    mode=args.costate_mode,
                    relative=args.costate_relative,
                    scale_floor=args.costate_scale_floor,
                )
                value_obj = value_obj + args.root_costate_weight * root_cost_loss
                train_cost_diag.update({f"root_{k}": v for k, v in root_cost_diag.items()})
            if args.pre_costate_weight > 0:
                pre_cost_sample = sample_box(pre_model.params, args.costate_batch_size, dtype, device)
                pre_cost_loss, pre_cost_diag, _ = costate_projection_loss(
                    pre_model,
                    pre_cost_sample,
                    "pre",
                    mode=args.costate_mode,
                    relative=args.costate_relative,
                    scale_floor=args.costate_scale_floor,
                )
                value_obj = value_obj + args.pre_costate_weight * pre_cost_loss
                train_cost_diag.update({f"pre_{k}": v for k, v in pre_cost_diag.items()})
            if args.slice_costate_weight > 0:
                slice_cost_loss, train_slice_cost_diag = slice_costate_loss_cached(pre_model, grids, args, D, mask)
                value_obj = value_obj + args.slice_costate_weight * slice_cost_loss
                train_cost_diag.update(train_slice_cost_diag)

        opt_v.zero_grad(set_to_none=True)
        value_obj.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(value_modules), args.grad_clip)
        opt_v.step()

        for _ in range(max(1, args.control_steps)):
            root_sample = sample_box(root_model.params, args.batch_size, dtype, device)
            pre_sample = sample_box(pre_model.params, args.batch_size, dtype, device)
            root_control, _ = control_loss(root_model, root_sample)
            pre_control, _ = control_loss(pre_model, pre_sample)
            control_obj = args.root_control_weight * root_control + args.pre_control_weight * pre_control
            opt_c.zero_grad(set_to_none=True)
            control_obj.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(control_modules), args.grad_clip)
            opt_c.step()

        if step % args.log_every == 0 or step == args.steps:
            val = validation(root_model, pre_model, args.valid_batch_size, args.valid_batches, dtype, device, seed=args.valid_seed)
            _slice_loss, slice_diag = lowmode_loss_cached(pre_model, grids, args, D, mask, dtype, device)
            del _slice_loss
            _slice_cost, slice_cost_diag = slice_costate_loss_cached(pre_model, grids, args, D, mask)
            del _slice_cost
            dt = time.time() - t0
            print(
                f"[step {step:6d}] root_hjb={val['root_hjb']:.5e} pre_hjb={val['pre_hjb']:.5e} "
                f"slice_rms={slice_diag['slice_rms']:.3e} slice_mean={slice_diag['slice_mean']:.3e} "
                f"slice_cost=({slice_cost_diag['slice_cost_qd']:.2e},{slice_cost_diag['slice_cost_qg']:.2e}) "
                f"foc=({val['pre_foc_d']:.2e},{val['pre_foc_g']:.2e}) ({dt:.0f}s)",
                flush=True,
            )
            history.append({
                "step": step,
                **val,
                **slice_diag,
                **slice_cost_diag,
                "train_root_hjb": root_diag["hjb"],
                "train_pre_hjb": pre_diag["hjb"],
                **{f"train_{k}": v for k, v in train_slice_diag.items()},
                **{f"train_{k}": v for k, v in train_cost_diag.items()},
            })

    final = history[-1]
    result = {
        "mode": "two_regime_analytic_costate",
        "steps": args.steps,
        "root_hjb_final": final["root_hjb"],
        "pre_hjb_final": final["pre_hjb"],
        "pre_foc_d_final": final["pre_foc_d"],
        "pre_foc_g_final": final["pre_foc_g"],
        "slice_rms_final": final["slice_rms"],
        "slice_mean_final": final["slice_mean"],
        "slice_cost_qd_final": final["slice_cost_qd"],
        "slice_cost_qg_final": final["slice_cost_qg"],
        "slice_cost_pk_final": final["slice_cost_pk"],
        "slice_cost_pz_final": final["slice_cost_pz"],
    }
    if args.export:
        save_all(root_model, pre_model, args.export)
        with open(os.path.join(args.export, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, default=_json_default, indent=2, sort_keys=True)
        with open(os.path.join(args.export, "history.jsonl"), "w", encoding="utf-8") as f:
            for row in history:
                f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
        with open(os.path.join(args.export, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, default=_json_default, indent=2, sort_keys=True)
        print(f"[export] wrote checkpoints/history to {args.export}", flush=True)

    print("RESULT_JSON " + json.dumps(result, default=_json_default, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
