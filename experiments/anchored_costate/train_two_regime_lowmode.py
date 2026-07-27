"""Warm-start two-regime training with map-reduced low-frequency slice losses.

The spectral diagnostics showed that the hard failure mode is not high-frequency
noise, but a low-frequency / DC HJB offset at fixed low ``xi`` slices.  This
trainer starts from an existing anchored two-regime checkpoint and adds a
map-reduce style slice loss:

  map:    evaluate HJB residual chunks on fixed (Z, Y) grids for several slices
  reduce: DCT-transform each residual field and penalize low modes + FOC RMS

Everything lives in the experiment sandbox; production model files are not
modified.
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
from tf_to_torch_loader import DEFAULT_CKPT_DIR  # noqa: E402
from train_two_regime_joint import (  # noqa: E402
    _dtype,
    _json_default,
    build_pre,
    build_root,
    save_all,
    validation,
)


DEFAULT_WARM_START = (
    "experiments/anchored_costate/results/precision_push_batch256/"
    "seed0_20000steps_20260630_210314/fedctrl_long_base"
)


def set_module_grad(module, flag):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(flag)


def load_joint_checkpoint(root_model, pre_model, checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    root_model.v_nn.load_state_dict(torch.load(checkpoint_dir / "root_v_recentered.pt", map_location=device))
    root_model.i_g_nn.load_state_dict(torch.load(checkpoint_dir / "root_i_g.pt", map_location=device))
    root_model.i_d_nn.load_state_dict(torch.load(checkpoint_dir / "root_i_d.pt", map_location=device))
    pre_model.v_nn.load_state_dict(torch.load(checkpoint_dir / "pre_damage_posttech_v_recentered.pt", map_location=device))
    pre_model.i_g_nn.load_state_dict(torch.load(checkpoint_dir / "pre_damage_posttech_i_g.pt", map_location=device))
    pre_model.i_d_nn.load_state_dict(torch.load(checkpoint_dir / "pre_damage_posttech_i_d.pt", map_location=device))


def dct_matrix_torch(n, dtype, device):
    x = torch.arange(n, dtype=dtype, device=device)
    k = torch.arange(n, dtype=dtype, device=device).reshape(-1, 1)
    mat = torch.cos(torch.pi / float(n) * (x + 0.5) * k)
    mat[0, :] *= (1.0 / float(n)) ** 0.5
    if n > 1:
        mat[1:, :] *= (2.0 / float(n)) ** 0.5
    return mat


def lowmode_mask(n, radius, dtype, device):
    k0 = torch.arange(n, dtype=dtype, device=device).reshape(-1, 1)
    k1 = torch.arange(n, dtype=dtype, device=device).reshape(1, -1)
    return (torch.sqrt(k0 * k0 + k1 * k1) <= float(radius))


def slice_specs(args):
    logxis = [float(x) for x in args.slice_logxis.split(",") if x.strip()]
    logks = [float(x) for x in args.slice_logks.split(",") if x.strip()]
    logrs = [float(x) for x in args.slice_logrs.split(",") if x.strip()]
    specs = []
    for logxi in logxis:
        for logk in logks:
            for logr in logrs:
                specs.append({
                    "logK": logk,
                    "logR": logr,
                    "logxi": logxi,
                    "lam3": args.slice_lam3,
                })
    return specs


def make_slice_grid(spec, grid, z_min, z_max, y_min, y_max, dtype, device):
    z = torch.linspace(z_min, z_max, grid, dtype=dtype, device=device)
    y = torch.linspace(y_min, y_max, grid, dtype=dtype, device=device)
    zz, yy = torch.meshgrid(z, y, indexing="ij")
    n = grid * grid
    logK = torch.full((n, 1), spec["logK"], dtype=dtype, device=device)
    Z = zz.reshape(n, 1)
    Y = yy.reshape(n, 1)
    logR = torch.full((n, 1), spec["logR"], dtype=dtype, device=device)
    lam3 = torch.full((n, 1), spec["lam3"], dtype=dtype, device=device)
    logxi = torch.full((n, 1), spec["logxi"], dtype=dtype, device=device)
    return logK, Z, Y, logR, lam3, logxi


def pde_chunk(model, chunk):
    logK, Z, Y, logR, lam3, logxi = chunk
    logK = logK.detach().requires_grad_(True)
    Z = Z.detach().requires_grad_(True)
    Y = Y.detach().requires_grad_(True)
    logR = logR.detach().requires_grad_(True)
    rhs, pv, _dvdy, c, gig, gid, foc_d, foc_g = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    return {
        "resid": rhs - pv,
        "foc_d": foc_d,
        "foc_g": foc_g,
        "c": c,
        "gig": gig,
        "gid": gid,
    }


def lowmode_map_reduce_loss(pre_model, specs, args, D, mask, dtype, device, train=True):
    """Map over slices/chunks and reduce residuals to low DCT modes."""
    slice_losses = []
    diags = {
        "slice_rms": [],
        "slice_mean": [],
        "slice_dc": [],
        "slice_low": [],
        "slice_foc_d": [],
        "slice_foc_g": [],
    }
    n = args.slice_grid * args.slice_grid
    chunk_size = n if args.slice_chunk_size <= 0 else min(n, args.slice_chunk_size)

    for spec in specs:
        full = make_slice_grid(
            spec,
            args.slice_grid,
            args.slice_z_min,
            args.slice_z_max,
            args.slice_y_min,
            args.slice_y_max,
            dtype,
            device,
        )
        resid_parts = []
        foc_d_parts = []
        foc_g_parts = []
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            chunk = tuple(col[start:stop] for col in full)
            out = pde_chunk(pre_model, chunk)
            resid_parts.append(out["resid"])
            foc_d_parts.append(out["foc_d"])
            foc_g_parts.append(out["foc_g"])
        resid = torch.cat(resid_parts, dim=0).reshape(args.slice_grid, args.slice_grid)
        foc_d = torch.cat(foc_d_parts, dim=0)
        foc_g = torch.cat(foc_g_parts, dim=0)
        coeff = D @ resid @ D.T
        low_coeff = coeff[mask]
        dc = coeff[0, 0]
        low_loss = rms(low_coeff)
        # Orthonormal 2D DCT has coeff[0, 0] = grid * mean(residual).
        # Normalize it back to residual units so the DC weight is interpretable.
        dc_loss = torch.abs(dc) / float(args.slice_grid)
        resid_rms = rms(resid)
        foc_loss = rms(foc_d) + rms(foc_g)
        loss = (
            args.slice_low_weight * low_loss
            + args.slice_dc_weight * dc_loss
            + args.slice_rms_weight * resid_rms
            + args.slice_foc_weight * foc_loss
        )
        slice_losses.append(loss)
        with torch.no_grad():
            diags["slice_rms"].append(float(resid_rms.detach().cpu()))
            diags["slice_mean"].append(float(torch.mean(resid).detach().cpu()))
            diags["slice_dc"].append(float(dc_loss.detach().cpu()))
            diags["slice_low"].append(float(low_loss.detach().cpu()))
            diags["slice_foc_d"].append(float(rms(foc_d).detach().cpu()))
            diags["slice_foc_g"].append(float(rms(foc_g).detach().cpu()))

    reduced = torch.stack(slice_losses).mean() if slice_losses else torch.zeros((), dtype=dtype, device=device)
    diag = {key: float(np.mean(vals)) if vals else 0.0 for key, vals in diags.items()}
    return reduced, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--valid_batch_size", type=int, default=512)
    ap.add_argument("--valid_batches", type=int, default=2)
    ap.add_argument("--valid_seed", type=int, default=12345)
    ap.add_argument("--lr_value", type=float, default=1e-4)
    ap.add_argument("--lr_control", type=float, default=1e-4)
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
    ap.add_argument("--root_weight", type=float, default=0.25)
    ap.add_argument("--pre_weight", type=float, default=1.0)
    ap.add_argument("--root_control_weight", type=float, default=0.25)
    ap.add_argument("--pre_control_weight", type=float, default=1.0)
    ap.add_argument("--control_steps", type=int, default=2)
    ap.add_argument("--boundary_weight", type=float, default=0.0)
    ap.add_argument("--boundary_batch", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--warm_start_dir", default=DEFAULT_WARM_START)
    ap.add_argument("--export", default=None)

    ap.add_argument("--slice_grid", type=int, default=16)
    ap.add_argument("--slice_chunk_size", type=int, default=128)
    ap.add_argument("--slice_logxis", default="-2.995732273553991,-2.302585092994046")
    ap.add_argument("--slice_logks", default="5.5")
    ap.add_argument("--slice_logrs", default="3.5")
    ap.add_argument("--slice_lam3", type=float, default=1.0 / 6.0)
    ap.add_argument("--slice_z_min", type=float, default=0.05)
    ap.add_argument("--slice_z_max", type=float, default=0.95)
    ap.add_argument("--slice_y_min", type=float, default=0.2)
    ap.add_argument("--slice_y_max", type=float, default=3.8)
    ap.add_argument("--low_radius", type=float, default=2.0)
    ap.add_argument("--slice_low_weight", type=float, default=0.0)
    ap.add_argument("--slice_dc_weight", type=float, default=1.0)
    ap.add_argument("--slice_rms_weight", type=float, default=0.25)
    ap.add_argument("--slice_foc_weight", type=float, default=0.25)
    ap.add_argument("--slice_every", type=int, default=1)
    args = ap.parse_args()

    dtype = _dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root_model = build_root(args, dtype, device)
    pre_model = build_pre(args, root_model, dtype, device)
    if args.warm_start_dir:
        load_joint_checkpoint(root_model, pre_model, args.warm_start_dir, device)
        print(f"[warm_start] loaded {args.warm_start_dir}", flush=True)

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

    D = dct_matrix_torch(args.slice_grid, dtype, device)
    mask = lowmode_mask(args.slice_grid, args.low_radius, dtype, device)
    specs = slice_specs(args)

    print(
        f"[setup] lowmode warm-start dtype={args.dtype} device={device} steps={args.steps} "
        f"bs={args.batch_size} slices={len(specs)} grid={args.slice_grid} "
        f"slice_weights=(dc={args.slice_dc_weight}, rms={args.slice_rms_weight}, "
        f"low={args.slice_low_weight}, foc={args.slice_foc_weight})",
        flush=True,
    )

    history = []
    val0 = validation(root_model, pre_model, args.valid_batch_size, args.valid_batches, dtype, device, seed=args.valid_seed)
    slice0, slice_diag0 = lowmode_map_reduce_loss(pre_model, specs, args, D, mask, dtype, device, train=False)
    del slice0
    print(
        f"[step      0] root_hjb={val0['root_hjb']:.5e} pre_hjb={val0['pre_hjb']:.5e} "
        f"slice_rms={slice_diag0['slice_rms']:.3e} slice_mean={slice_diag0['slice_mean']:.3e} "
        f"slice_foc=({slice_diag0['slice_foc_d']:.2e},{slice_diag0['slice_foc_g']:.2e})",
        flush=True,
    )
    history.append({"step": 0, **val0, **slice_diag0})
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
        gap = boundary_gap_loss(
            root_model,
            pre_model,
            args.boundary_batch,
            dtype,
            device,
            detach_root=not args.couple_anchor_grad,
        )
        value_obj = args.root_weight * root_loss + args.pre_weight * pre_loss
        if args.boundary_weight > 0:
            value_obj = value_obj + args.boundary_weight * gap
        if args.slice_every > 0 and step % args.slice_every == 0:
            slice_loss, train_slice_diag = lowmode_map_reduce_loss(pre_model, specs, args, D, mask, dtype, device)
            value_obj = value_obj + slice_loss
        else:
            train_slice_diag = {}

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
            _slice_loss, slice_diag = lowmode_map_reduce_loss(pre_model, specs, args, D, mask, dtype, device, train=False)
            del _slice_loss
            dt = time.time() - t0
            print(
                f"[step {step:6d}] root_hjb={val['root_hjb']:.5e} pre_hjb={val['pre_hjb']:.5e} "
                f"slice_rms={slice_diag['slice_rms']:.3e} slice_mean={slice_diag['slice_mean']:.3e} "
                f"slice_dc={slice_diag['slice_dc']:.3e} "
                f"slice_foc=({slice_diag['slice_foc_d']:.2e},{slice_diag['slice_foc_g']:.2e}) ({dt:.0f}s)",
                flush=True,
            )
            history.append({
                "step": step,
                **val,
                **slice_diag,
                "train_root_hjb": root_diag["hjb"],
                "train_pre_hjb": pre_diag["hjb"],
                **{f"train_{k}": v for k, v in train_slice_diag.items()},
            })

    final = history[-1]
    result = {
        "mode": "two_regime_lowmode",
        "steps": args.steps,
        "root_hjb_final": final["root_hjb"],
        "pre_hjb_final": final["pre_hjb"],
        "slice_rms_final": final["slice_rms"],
        "slice_mean_final": final["slice_mean"],
        "slice_dc_final": final["slice_dc"],
        "slice_foc_d_final": final["slice_foc_d"],
        "slice_foc_g_final": final["slice_foc_g"],
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
