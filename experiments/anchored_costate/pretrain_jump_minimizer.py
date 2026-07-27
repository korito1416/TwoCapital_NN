"""Pretrain a xi-conditioned jump minimizer on a simple closed-form benchmark.

This is a method test, not production training.  It teaches ``JumpRobustNet`` the
basic robust-control map

    (DeltaV, xi) -> log_g_closed = -DeltaV / xi

before transferring the network into the two-regime HJB experiment.  The point is
to test whether a pure NN-driven inner minimizer becomes stable after benchmark
distillation.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from anchored_nets import PREDAMAGE_POSTTECH_BOUNDS_6, WarmupCosine, rms  # noqa: E402
from params import PARAMS  # noqa: E402
from train_two_regime_learned_jump import JumpRobustNet  # noqa: E402


def dtype_from_name(name):
    return torch.float64 if name == "float64" else torch.float32


def sample_x_pre(n, dtype, device):
    cols = []
    for idx, (lo, hi) in enumerate(PREDAMAGE_POSTTECH_BOUNDS_6):
        if hi == lo:
            cols.append(torch.full((n, 1), float(lo), dtype=dtype, device=device))
        else:
            cols.append(lo + (hi - lo) * torch.rand((n, 1), dtype=dtype, device=device))
    # Keep the duplicated logxi column exactly equal.
    cols[5] = cols[4].clone()
    return torch.cat(cols, dim=1)


def sample_diff(n, L, args, dtype, device):
    if args.diff_distribution == "uniform":
        diff = (2.0 * torch.rand((n, L), dtype=dtype, device=device) - 1.0) * args.diff_scale
    elif args.diff_distribution == "normal":
        diff = torch.randn((n, L), dtype=dtype, device=device) * args.diff_scale
        diff = diff.clamp(min=-3.0 * args.diff_scale, max=3.0 * args.diff_scale)
    else:
        raise ValueError(f"unknown diff_distribution={args.diff_distribution!r}")

    if args.severity_slope != 0:
        y_grid = torch.linspace(0.0, 1.0, L, dtype=dtype, device=device).reshape(1, L)
        diff = diff + args.severity_slope * (y_grid - 0.5)
    return diff


def logg_from_raw(raw, mode, base, args):
    if mode == "direct":
        return args.direct_scale * torch.tanh(raw)
    if mode == "residual":
        return base + args.residual_scale * torch.tanh(raw)
    raise ValueError(f"unknown mode={mode!r}")


def validation(net, args, dtype, device, batches=4):
    vals = {"rmse": 0.0, "max_abs": 0.0}
    L = PARAMS["L"]
    with torch.no_grad():
        for _ in range(batches):
            x = sample_x_pre(args.valid_batch_size, dtype, device)
            logxi = x[:, 4:5]
            xi = torch.exp(logxi)
            diff = sample_diff(args.valid_batch_size, L, args, dtype, device)
            base = (-diff / xi).clamp(min=-args.logg_clip, max=args.logg_clip)
            raw = net(x, diff, base, logxi)
            pred = logg_from_raw(raw, args.mode, base, args).clamp(min=-args.logg_clip, max=args.logg_clip)
            err = pred - base
            vals["rmse"] += float(rms(err).cpu())
            vals["max_abs"] += float(torch.max(torch.abs(err)).cpu())
    for key in vals:
        vals[key] /= max(1, batches)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "residual"], default="direct")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--valid_batch_size", type=int, default=4096)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.02)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--diff_scale", type=float, default=0.25)
    ap.add_argument("--diff_distribution", choices=["uniform", "normal"], default="uniform")
    ap.add_argument("--severity_slope", type=float, default=0.05)
    ap.add_argument("--direct_scale", type=float, default=8.0)
    ap.add_argument("--residual_scale", type=float, default=0.05)
    ap.add_argument("--logg_clip", type=float, default=8.0)
    ap.add_argument("--export", required=True)
    args = ap.parse_args()

    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net = JumpRobustNet(
        L=PARAMS["L"],
        width=args.width,
        depth=args.depth,
        seed=args.seed + 1000,
        dtype=dtype,
        device=device,
    )
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    warmup = max(1, int(args.steps * args.warmup_frac))
    sched = WarmupCosine(args.lr, args.steps, warmup, args.min_lr)

    print(
        f"[setup] pretrain_jump mode={args.mode} steps={args.steps} bs={args.batch_size} "
        f"device={device} diff_scale={args.diff_scale}",
        flush=True,
    )
    t0 = time.time()
    history = []
    val0 = validation(net, args, dtype, device)
    print(f"[step      0] rmse={val0['rmse']:.5e} max={val0['max_abs']:.5e}", flush=True)
    history.append({"step": 0, **val0})

    L = PARAMS["L"]
    for step in range(1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = sched(step)
        x = sample_x_pre(args.batch_size, dtype, device)
        logxi = x[:, 4:5]
        xi = torch.exp(logxi)
        diff = sample_diff(args.batch_size, L, args, dtype, device)
        base = (-diff / xi).clamp(min=-args.logg_clip, max=args.logg_clip)
        raw = net(x, diff, base, logxi)
        pred = logg_from_raw(raw, args.mode, base, args).clamp(min=-args.logg_clip, max=args.logg_clip)
        loss = torch.mean((pred - base) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            val = validation(net, args, dtype, device)
            dt = time.time() - t0
            print(
                f"[step {step:6d}] rmse={val['rmse']:.5e} max={val['max_abs']:.5e} ({dt:.0f}s)",
                flush=True,
            )
            history.append({"step": step, **val})

    os.makedirs(args.export, exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "args": vars(args),
        "history": history,
    }
    torch.save(payload, os.path.join(args.export, "jump_minimizer_pretrained.pt"))
    with open(os.path.join(args.export, "history.jsonl"), "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    final = history[-1]
    print(f"[export] wrote {args.export}", flush=True)
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "mode": "pretrain_jump_minimizer",
                "minimizer_mode": args.mode,
                "steps": args.steps,
                "rmse_final": final["rmse"],
                "max_abs_final": final["max_abs"],
                "export": args.export,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
