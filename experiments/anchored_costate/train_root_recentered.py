"""Train a recentered PostDamagePostTech value network.

This is the smallest experiment for the level-identification problem:

    v(x) = phi(x) - phi(x_anchor; pseudo(x)) + v_anchor

The HJB economics are still supplied by ``models_torch/PostDamagePostTech.py``.
Only the value/control modules and optimizer loop are swapped out.
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
    ROOT_BOUNDS_7,
    CleanMLP,
    RecenteredValueNet,
    WarmupCosine,
    build_value_phi,
    build_control_net,
    constant_anchor,
    probe_level,
    root_surrogate_anchor_value,
    rms,
    sample_box,
    terminal_anchor_x_root,
    trainable_parameters,
    value_loss,
    control_loss,
)
from params import PARAMS  # noqa: E402
from tf_to_torch_loader import DEFAULT_CKPT_DIR, build_root_torch_model  # noqa: E402


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def assemble_root_x(sample, dtype, device):
    logk, z, y, _logr, lam3, logxi = sample
    return torch.cat(
        [
            logk.to(dtype=dtype, device=device),
            z.to(dtype=dtype, device=device),
            y.to(dtype=dtype, device=device),
            lam3.to(dtype=dtype, device=device),
            torch.full_like(y, PARAMS["A_g_prime_prime"]).to(dtype=dtype, device=device),
            logxi.to(dtype=dtype, device=device),
            logxi.to(dtype=dtype, device=device),
        ],
        dim=1,
    )


def anchor_gap(model, n: int, dtype: torch.dtype, device: torch.device) -> float:
    sample = sample_box(model.params, n, dtype, device)
    x = assemble_root_x(sample, dtype, device)
    x0 = terminal_anchor_x_root(x)
    with torch.no_grad():
        return float(rms(model.v_nn(x0) - model.v_nn.anchor_value_fn(x0)).cpu())


def validation(model, n: int, batches: int, dtype: torch.dtype, device: torch.device):
    accum = {"hjb": 0.0, "foc_d": 0.0, "foc_g": 0.0, "c_min": 0.0}
    level = 0.0
    for _ in range(batches):
        sample = sample_box(model.params, n, dtype, device)
        with torch.enable_grad():
            _loss, diag = value_loss(model, sample, include_foc=True)
            probe = probe_level(model, n, dtype, device)
        for key in accum:
            accum[key] += diag[key]
        level += probe["level_mean"]
    for key in accum:
        accum[key] /= max(1, batches)
    accum["level_mean"] = level / max(1, batches)
    accum["anchor_gap"] = anchor_gap(model, min(n, 512), dtype, device)
    return accum


def build_model(args, dtype: torch.dtype, device: torch.device):
    load_surrogate_controls = args.control_init == "surrogate"
    model = build_root_torch_model(
        load_surrogate=load_surrogate_controls,
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
    if args.anchor == "surrogate":
        v0 = root_surrogate_anchor_value(dtype, device, ckpt_dir=args.ckpt_dir)
    else:
        v0 = args.anchor_value
    model.v_nn = RecenteredValueNet(phi, terminal_anchor_x_root, constant_anchor(v0))

    if args.control_init == "clean":
        model.i_g_nn = build_control_net(
            7, ROOT_BOUNDS_7, model.params["θ_g"], args.seed + 2000, activation="tanh")
        model.i_d_nn = build_control_net(
            7, ROOT_BOUNDS_7, model.params["θ_d"], args.seed + 3000, activation="tanh")

    for net in (model.v_nn, model.i_g_nn, model.i_d_nn):
        net.to(device=device, dtype=dtype)
        net.train()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr_value", type=float, default=4e-4)
    ap.add_argument("--lr_control", type=float, default=4e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.01)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--valid_batch_size", type=int, default=512)
    ap.add_argument("--valid_batches", type=int, default=2)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--activation", default="swish")
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--value_arch", choices=["clean", "frozen_middle"], default="clean")
    ap.add_argument("--anchor", choices=["surrogate", "constant"], default="surrogate")
    ap.add_argument("--anchor_value", type=float, default=0.0)
    ap.add_argument("--control_init", choices=["surrogate", "clean", "torch_scratch"], default="surrogate")
    ap.add_argument("--freeze_controls", action="store_true")
    ap.add_argument("--foc_in_value", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--export", default=None)
    args = ap.parse_args()

    dtype = _dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args, dtype, device)
    if args.freeze_controls:
        for net in (model.i_g_nn, model.i_d_nn):
            for p in net.parameters():
                p.requires_grad_(False)

    warmup_steps = max(1, int(args.steps * args.warmup_frac))
    opt_v = torch.optim.Adam(trainable_parameters([model.v_nn]), lr=args.lr_value)
    opt_c = None
    if not args.freeze_controls:
        opt_c = torch.optim.Adam(
            trainable_parameters([model.i_g_nn, model.i_d_nn]),
            lr=args.lr_control,
        )
    sched_v = WarmupCosine(args.lr_value, args.steps, warmup_steps, args.min_lr)
    sched_c = WarmupCosine(args.lr_control, args.steps, warmup_steps, args.min_lr)

    print(
        f"[setup] root recentered dtype={args.dtype} device={device} steps={args.steps} "
        f"bs={args.batch_size} control_init={args.control_init} freeze_controls={args.freeze_controls} "
        f"anchor={args.anchor} warmup={warmup_steps}/{args.steps}"
    )
    print(f"[anchor] hard root anchor gap={anchor_gap(model, 256, dtype, device):.4e}")

    history = []
    val0 = validation(model, args.valid_batch_size, args.valid_batches, dtype, device)
    print(
        f"[step      0] hjb={val0['hjb']:.5e} FOC_d={val0['foc_d']:.5e} "
        f"FOC_g={val0['foc_g']:.5e} level={val0['level_mean']:.5e} "
        f"anchor_gap={val0['anchor_gap']:.2e}"
    )
    history.append({"step": 0, **val0})
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for group in opt_v.param_groups:
            group["lr"] = sched_v(step)
        if opt_c is not None:
            for group in opt_c.param_groups:
                group["lr"] = sched_c(step)

        sample = sample_box(model.params, args.batch_size, dtype, device)
        loss_v, diag_v = value_loss(model, sample, include_foc=args.foc_in_value)
        opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters([model.v_nn]), args.grad_clip)
        opt_v.step()

        if opt_c is not None:
            sample = sample_box(model.params, args.batch_size, dtype, device)
            loss_c, _diag_c = control_loss(model, sample)
            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters([model.i_g_nn, model.i_d_nn]), args.grad_clip)
            opt_c.step()

        if step % args.log_every == 0 or step == args.steps:
            val = validation(model, args.valid_batch_size, args.valid_batches, dtype, device)
            dt = time.time() - t0
            print(
                f"[step {step:6d}] hjb={val['hjb']:.5e} FOC_d={val['foc_d']:.5e} "
                f"FOC_g={val['foc_g']:.5e} level={val['level_mean']:.5e} "
                f"anchor_gap={val['anchor_gap']:.2e} ({dt:.0f}s)"
            )
            history.append({"step": step, **val, "train_hjb": diag_v["hjb"]})

    final = history[-1]
    if args.export:
        os.makedirs(args.export, exist_ok=True)
        torch.save(model.v_nn.state_dict(), os.path.join(args.export, "root_v_recentered.pt"))
        torch.save(model.i_g_nn.state_dict(), os.path.join(args.export, "root_i_g.pt"))
        torch.save(model.i_d_nn.state_dict(), os.path.join(args.export, "root_i_d.pt"))
        with open(os.path.join(args.export, "history.jsonl"), "w", encoding="utf-8") as f:
            for row in history:
                f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
        print(f"[export] wrote checkpoints/history to {args.export}")

    result = {
        "mode": "root_recentered",
        "steps": args.steps,
        "dtype": args.dtype,
        "hjb_start": val0["hjb"],
        "hjb_final": final["hjb"],
        "anchor_gap_final": final["anchor_gap"],
        "level_mean_final": final["level_mean"],
    }
    print("RESULT_JSON " + json.dumps(result, default=_json_default, sort_keys=True))


if __name__ == "__main__":
    main()
