"""Joint anchored training for PostDamagePostTech -> PreDamagePostTech.

This is the smallest cross-regime experiment:

* root terminal regime gets a hard recentered value anchor;
* pre-damage/post-tech reads the root value as its damage-jump target;
* pre value is hard-anchored at Y=y_upper to the root value averaged over λ3.

The economics still come from the validated torch PDE ports.  The experiment is
about identification/communication across regimes, not a new HJB discretization.
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

from PreDamagePostTech import PreDamagePostTechModel  # noqa: E402
from anchored_nets import (  # noqa: E402
    PREDAMAGE_POSTTECH_BOUNDS_6,
    ROOT_BOUNDS_7,
    CleanMLP,
    DetachedModule,
    RecenteredValueNet,
    WarmupCosine,
    boundary_gap_loss,
    build_value_phi,
    build_control_net,
    constant_anchor,
    predamage_posttech_anchor_x,
    probe_level,
    root_surrogate_anchor_value,
    root_value_at_damage_anchor,
    sample_box,
    terminal_anchor_x_root,
    trainable_parameters,
    value_loss,
    control_loss,
)
from params import PARAMS  # noqa: E402
from tf_to_torch_loader import (  # noqa: E402
    DEFAULT_CKPT_DIR,
    build_root_torch_model,
    make_root_configs,
)


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_root(args, dtype: torch.dtype, device: torch.device):
    model = build_root_torch_model(
        load_surrogate=(args.root_control_init == "surrogate"),
        ckpt_dir=args.ckpt_dir,
        dtype=dtype,
    )
    hidden = tuple([args.width] * args.layers)
    phi = build_value_phi(
        input_dim=7,
        bounds=ROOT_BOUNDS_7,
        hidden=hidden,
        activation=args.activation,
        seed=args.seed + 1000,
        residual=not args.no_residual,
        value_arch=args.value_arch,
        dtype=dtype,
        device=device,
        ckpt_dir=args.ckpt_dir,
    )
    if args.root_anchor == "surrogate":
        root_v0 = root_surrogate_anchor_value(dtype, device, ckpt_dir=args.ckpt_dir)
    else:
        root_v0 = args.root_anchor_value
    model.v_nn = RecenteredValueNet(phi, terminal_anchor_x_root, constant_anchor(root_v0))

    if args.root_control_init == "clean":
        model.i_g_nn = build_control_net(
            7, ROOT_BOUNDS_7, model.params["θ_g"], args.seed + 2000, activation="tanh")
        model.i_d_nn = build_control_net(
            7, ROOT_BOUNDS_7, model.params["θ_d"], args.seed + 3000, activation="tanh")

    for net in (model.v_nn, model.i_g_nn, model.i_d_nn):
        net.to(device=device, dtype=dtype)
        net.train()
    return model


def build_pre(args, root_model, dtype: torch.dtype, device: torch.device):
    v_cfg, ig_cfg, id_cfg = make_root_configs()
    model = PreDamagePostTechModel({
        "v_nn_config": v_cfg,
        "i_g_nn_config": ig_cfg,
        "i_d_nn_config": id_cfg,
    })
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
    anchor_value = root_value_at_damage_anchor(
        root_model.v_nn,
        detach=not args.couple_anchor_grad,
    )
    model.v_nn = RecenteredValueNet(phi, predamage_posttech_anchor_x, anchor_value)
    model.i_g_nn = build_control_net(
        6, PREDAMAGE_POSTTECH_BOUNDS_6, model.params["θ_g"], args.seed + 5000,
        activation="tanh", init_rate=args.pre_control_init_rate)
    model.i_d_nn = build_control_net(
        6, PREDAMAGE_POSTTECH_BOUNDS_6, model.params["θ_d"], args.seed + 6000,
        activation="tanh", init_rate=args.pre_control_init_rate)
    model.v_PostDamagePostTech_nn = (
        root_model.v_nn if args.couple_neighbor_grad else DetachedModule(root_model.v_nn)
    )

    for net in (model.v_nn, model.i_g_nn, model.i_d_nn, model.v_PostDamagePostTech_nn):
        net.to(device=device, dtype=dtype)
        net.train()
    return model


def _sample_for_validation(model, n, dtype, device, seed):
    if seed is None:
        return sample_box(model.params, n, dtype, device)
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        return sample_box(model.params, n, dtype, device)


def validation(root_model, pre_model, n: int, batches: int, dtype: torch.dtype, device: torch.device, seed=None):
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
    for _ in range(batches):
        batch_seed = None if seed is None else int(seed) + 1009 * _
        root_sample = _sample_for_validation(root_model, n, dtype, device, batch_seed)
        pre_sample = _sample_for_validation(pre_model, n, dtype, device, None if batch_seed is None else batch_seed + 17)
        with torch.enable_grad():
            _root_loss, root_diag = value_loss(root_model, root_sample, include_foc=True)
            _pre_loss, pre_diag = value_loss(pre_model, pre_sample, include_foc=True)
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
    with torch.enable_grad():
        gap = boundary_gap_loss(root_model, pre_model, n, dtype, device, detach_root=True)
    accum["boundary_gap"] = float(gap.detach().cpu())
    return accum


def save_all(root_model, pre_model, export_dir: str):
    os.makedirs(export_dir, exist_ok=True)
    torch.save(root_model.v_nn.state_dict(), os.path.join(export_dir, "root_v_recentered.pt"))
    torch.save(root_model.i_g_nn.state_dict(), os.path.join(export_dir, "root_i_g.pt"))
    torch.save(root_model.i_d_nn.state_dict(), os.path.join(export_dir, "root_i_d.pt"))
    torch.save(pre_model.v_nn.state_dict(), os.path.join(export_dir, "pre_damage_posttech_v_recentered.pt"))
    torch.save(pre_model.i_g_nn.state_dict(), os.path.join(export_dir, "pre_damage_posttech_i_g.pt"))
    torch.save(pre_model.i_d_nn.state_dict(), os.path.join(export_dir, "pre_damage_posttech_i_d.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--valid_batch_size", type=int, default=512)
    ap.add_argument("--valid_batches", type=int, default=2)
    ap.add_argument("--valid_seed", type=int, default=12345)
    ap.add_argument("--lr_value", type=float, default=4e-4)
    ap.add_argument("--lr_control", type=float, default=4e-4)
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
    ap.add_argument("--couple_anchor_grad", action="store_true",
                    help="Allow pre-regime anchor loss/PDE gradients to flow into root v.")
    ap.add_argument("--couple_neighbor_grad", action="store_true",
                    help="Allow pre-regime jump-target gradients to flow into root v.")
    ap.add_argument("--foc_in_value", action="store_true")
    ap.add_argument("--root_weight", type=float, default=1.0)
    ap.add_argument("--pre_weight", type=float, default=1.0)
    ap.add_argument("--root_control_weight", type=float, default=1.0)
    ap.add_argument("--pre_control_weight", type=float, default=1.0)
    ap.add_argument("--control_steps", type=int, default=1)
    ap.add_argument("--boundary_weight", type=float, default=0.0,
                    help="Usually unnecessary because the pre value has a hard boundary anchor.")
    ap.add_argument("--boundary_batch", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--export", default=None)
    args = ap.parse_args()

    dtype = _dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root_model = build_root(args, dtype, device)
    pre_model = build_pre(args, root_model, dtype, device)

    if args.freeze_root_controls:
        for net in (root_model.i_g_nn, root_model.i_d_nn):
            for p in net.parameters():
                p.requires_grad_(False)

    value_modules = [root_model.v_nn, pre_model.v_nn]
    control_modules = [pre_model.i_g_nn, pre_model.i_d_nn]
    if not args.freeze_root_controls:
        control_modules.extend([root_model.i_g_nn, root_model.i_d_nn])

    opt_v = torch.optim.Adam(trainable_parameters(value_modules), lr=args.lr_value)
    opt_c = torch.optim.Adam(trainable_parameters(control_modules), lr=args.lr_control)
    warmup_steps = max(1, int(args.steps * args.warmup_frac))
    sched_v = WarmupCosine(args.lr_value, args.steps, warmup_steps, args.min_lr)
    sched_c = WarmupCosine(args.lr_control, args.steps, warmup_steps, args.min_lr)

    print(
        f"[setup] two-regime joint dtype={args.dtype} device={device} steps={args.steps} "
        f"bs={args.batch_size} root_control={args.root_control_init} "
        f"couple_anchor_grad={args.couple_anchor_grad} "
        f"couple_neighbor_grad={args.couple_neighbor_grad} "
        f"warmup={warmup_steps}/{args.steps}"
    )

    history = []
    val0 = validation(root_model, pre_model, args.valid_batch_size, args.valid_batches, dtype, device, seed=args.valid_seed)
    print(
        f"[step      0] root_hjb={val0['root_hjb']:.5e} pre_hjb={val0['pre_hjb']:.5e} "
        f"gap={val0['boundary_gap']:.3e} root_foc=({val0['root_foc_d']:.2e},{val0['root_foc_g']:.2e}) "
        f"pre_foc=({val0['pre_foc_d']:.2e},{val0['pre_foc_g']:.2e})"
    )
    history.append({"step": 0, **val0})
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for group in opt_v.param_groups:
            group["lr"] = sched_v(step)
        for group in opt_c.param_groups:
            group["lr"] = sched_c(step)

        root_sample = sample_box(root_model.params, args.batch_size, dtype, device)
        pre_sample = sample_box(pre_model.params, args.batch_size, dtype, device)
        root_loss, root_diag = value_loss(root_model, root_sample, include_foc=args.foc_in_value)
        pre_loss, pre_diag = value_loss(pre_model, pre_sample, include_foc=args.foc_in_value)
        gap = boundary_gap_loss(
            root_model,
            pre_model,
            args.boundary_batch,
            dtype,
            device,
            detach_root=not args.couple_anchor_grad,
        )
        value_obj = args.root_weight * root_loss + args.pre_weight * pre_loss
        if args.boundary_weight > 0.0:
            value_obj = value_obj + args.boundary_weight * gap
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
            if args.freeze_root_controls:
                control_obj = args.pre_control_weight * pre_control
            else:
                control_obj = args.root_control_weight * root_control + args.pre_control_weight * pre_control
            opt_c.zero_grad(set_to_none=True)
            control_obj.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(control_modules), args.grad_clip)
            opt_c.step()

        if step % args.log_every == 0 or step == args.steps:
            val = validation(root_model, pre_model, args.valid_batch_size, args.valid_batches, dtype, device, seed=args.valid_seed)
            dt = time.time() - t0
            print(
                f"[step {step:6d}] root_hjb={val['root_hjb']:.5e} pre_hjb={val['pre_hjb']:.5e} "
                f"gap={val['boundary_gap']:.3e} root_foc=({val['root_foc_d']:.2e},{val['root_foc_g']:.2e}) "
                f"pre_foc=({val['pre_foc_d']:.2e},{val['pre_foc_g']:.2e}) ({dt:.0f}s)"
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
        print(f"[export] wrote checkpoints/history to {args.export}")

    result = {
        "mode": "two_regime_joint",
        "steps": args.steps,
        "dtype": args.dtype,
        "root_hjb_start": val0["root_hjb"],
        "root_hjb_final": final["root_hjb"],
        "pre_hjb_start": val0["pre_hjb"],
        "pre_hjb_final": final["pre_hjb"],
        "boundary_gap_final": final["boundary_gap"],
        "root_level_final": final["root_level"],
        "pre_level_final": final["pre_level"],
    }
    print("RESULT_JSON " + json.dumps(result, default=_json_default, sort_keys=True))


if __name__ == "__main__":
    main()
