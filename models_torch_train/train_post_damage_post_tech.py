"""
Torch DGM-PIA training loop for the ROOT regime (PostDamagePostTech).

Faithful mirror of models/PostDamagePostTech.py's training logic:

  * State sampling: stratified-uniform per column over the box (matches the TF
    ``stratified_uniform`` sampler).
  * 3-loss DGM-PIA:
      value loss   = sqrt(mean((rhs - pv)^2))
                     + sqrt(mean(FOC_g^2)) + sqrt(mean(FOC_d^2))
                     + sqrt(mean(loss_dv_dY^2))      [the TF "value" objective]
      control loss = -mean(rhs - pv)
                     + sqrt(mean(FOC_g^2)) + sqrt(mean(FOC_d^2))   [TF control obj]
    (Exactly as in TF objective_fn: value step trains v_nn on the full residual+
    FOC+dvdY loss; control step trains i_g/i_d on the -mean(residual)+FOC obj.)
  * Two SEPARATE Adam optimizers (value net vs the two control nets), each with a
    WarmupCosine LR schedule reproducing models/feedforward_subnet.py::WarmupCosine.
  * float32 by default; --dtype float64 toggles double precision.
  * Can warm-start from the TF surrogate (--init surrogate) or from scratch
    (--init scratch).

This is a SURROGATE-faithful torch replica, NOT a new method.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_TORCH = os.path.join(ROOT, "models_torch")
for p in (HERE, MODELS_TORCH):
    if p not in sys.path:
        sys.path.insert(0, p)

from tf_to_torch_loader import (  # noqa: E402
    build_root_torch_model, DEFAULT_CKPT_DIR,
)
from params import PARAMS  # noqa: E402  (torch params)


# ---------------------------------------------------------------------------
#  WarmupCosine LR (mirror of models/feedforward_subnet.py::WarmupCosine)
# ---------------------------------------------------------------------------
class WarmupCosine:
    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=0.0):
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = float(step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (
                step / max(1.0, self.warmup_steps))
        progress = min(max((step - self.warmup_steps)
                           / max(1.0, self.total_steps - self.warmup_steps), 0.0), 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
#  Stratified-uniform sampler (mirror of stratified_uniform in TF)
# ---------------------------------------------------------------------------
def stratified_uniform(lo, hi, n, dtype, device):
    edges = torch.linspace(lo, hi, n + 1, dtype=dtype, device=device)
    offsets = torch.rand((n, 1), dtype=dtype, device=device)
    draws = edges[:-1].reshape(-1, 1)
    widths = (edges[1:] - edges[:-1]).reshape(-1, 1)
    out = draws + widths * offsets
    return out[torch.randperm(n, device=device)]


def sample_box(params, n, dtype, device):
    logK = stratified_uniform(params["logK_min"], params["logK_max"], n, dtype, device)
    Z = stratified_uniform(params["Z_min"], params["Z_max"], n, dtype, device)
    Y = stratified_uniform(params["Y_min"], params["Y_max"], n, dtype, device)
    logR = stratified_uniform(params["logR_min"], params["logR_max"], n, dtype, device)
    lam3 = stratified_uniform(params["λ3_min"], params["λ3_max"], n, dtype, device)
    logxi = stratified_uniform(params["logξ_min"], params["logξ_max"], n, dtype, device)
    return logK, Z, Y, logR, lam3, logxi


# ---------------------------------------------------------------------------
#  Losses (mirror of TF objective_fn)
# ---------------------------------------------------------------------------
def _rms(x):
    return torch.sqrt(torch.mean(x ** 2))


def compute_losses(model, params, logK, Z, Y, logR, lam3, logxi):
    """Return (value_loss, control_loss, diag) mirroring TF objective_fn.

    value_loss  : trains v_nn   = rms(rhs-pv) + rms(FOC_g) + rms(FOC_d) + rms(loss_dv_dY)
    control_loss: trains i_g/i_d = -mean(rhs-pv) + rms(FOC_g) + rms(FOC_d)
    """
    logK = logK.detach().requires_grad_(True)
    Z = Z.detach().requires_grad_(True)
    Y = Y.detach().requires_grad_(True)

    rhs, pv, dv_dY, c, gig, gid, FOC_d, FOC_g = model.pde_rhs(
        logK, Z, Y, logR, lam3, logxi)

    resid = rhs - pv
    y_upper = params["y_upper"]
    mask = ((Y > y_upper).to(dv_dY.dtype)) * ((dv_dY > 0).to(dv_dY.dtype))
    loss_dv_dY = dv_dY * mask + 1e-7

    value_loss = _rms(resid) + _rms(FOC_g) + _rms(FOC_d) + _rms(loss_dv_dY)
    control_loss = -torch.mean(resid) + _rms(FOC_g) + _rms(FOC_d)

    diag = {
        "loss_v": float(_rms(resid).detach()),
        "loss_FOC_d": float(_rms(FOC_d).detach()),
        "loss_FOC_g": float(_rms(FOC_g).detach()),
        "loss_dv_dY": float(_rms(loss_dv_dY).detach()),
    }
    return value_loss, control_loss, diag


@torch.no_grad()
def validation(model, params, dtype, device, n=1024, batches=4, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    accum = {"loss_v": 0.0, "loss_FOC_d": 0.0, "loss_FOC_g": 0.0, "loss_dv_dY": 0.0}
    for _ in range(batches):
        with torch.enable_grad():
            sample = sample_box(params, n, dtype, device)
            _, _, diag = compute_losses(model, params, *sample)
        for k in accum:
            accum[k] += diag[k]
    for k in accum:
        accum[k] /= batches
    return accum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["surrogate", "scratch"], default="surrogate")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr_value", type=float, default=4e-4)   # 40e-5
    ap.add_argument("--lr_control", type=float, default=4e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)     # 10e-6
    ap.add_argument("--warmup_frac", type=float, default=0.01)
    ap.add_argument("--total_steps", type=int, default=None,
                    help="LR-schedule horizon (default = --steps).")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--export", default=None,
                    help="Folder to write torch checkpoints + history.")
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = PARAMS
    total_steps = args.total_steps or args.steps
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    # ---- build model ----
    load_surrogate = (args.init == "surrogate")
    model = build_root_torch_model(
        load_surrogate=load_surrogate, ckpt_dir=args.ckpt_dir, dtype=dtype)
    for net in (model.v_nn, model.i_g_nn, model.i_d_nn):
        net.to(device=device, dtype=dtype)
        net.train()

    # ---- optimizers: SEPARATE value vs control (mirrors TF) ----
    opt_v = torch.optim.Adam(model.v_nn.parameters(), lr=args.lr_value)
    opt_c = torch.optim.Adam(
        list(model.i_g_nn.parameters()) + list(model.i_d_nn.parameters()),
        lr=args.lr_control)
    sched_v = WarmupCosine(args.lr_value, total_steps, warmup_steps, args.min_lr)
    sched_c = WarmupCosine(args.lr_control, total_steps, warmup_steps, args.min_lr)

    print(f"[setup] init={args.init} dtype={args.dtype} device={device} "
          f"steps={args.steps} bs={args.batch_size} "
          f"lr_v={args.lr_value} lr_c={args.lr_control} min_lr={args.min_lr} "
          f"warmup={warmup_steps}/{total_steps}")

    # ---- baseline validation ----
    val0 = validation(model, params, dtype, device, seed=12345)
    print(f"[step 0] val loss_v={val0['loss_v']:.5e} "
          f"FOC_d={val0['loss_FOC_d']:.5e} FOC_g={val0['loss_FOC_g']:.5e} "
          f"dv_dY={val0['loss_dv_dY']:.5e}")

    history = [[0, val0["loss_v"], val0["loss_FOC_d"], val0["loss_FOC_g"], val0["loss_dv_dY"]]]
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for g in opt_v.param_groups:
            g["lr"] = sched_v(step)
        for g in opt_c.param_groups:
            g["lr"] = sched_c(step)

        # ---- value step ----
        sample = sample_box(params, args.batch_size, dtype, device)
        value_loss, _, _ = compute_losses(model, params, *sample)
        opt_v.zero_grad(set_to_none=True)
        value_loss.backward()
        opt_v.step()

        # ---- control step (fresh sample, mirrors TF re-sample) ----
        sample = sample_box(params, args.batch_size, dtype, device)
        _, control_loss, _ = compute_losses(model, params, *sample)
        opt_c.zero_grad(set_to_none=True)
        control_loss.backward()
        opt_c.step()

        if step % args.log_every == 0 or step == args.steps:
            val = validation(model, params, dtype, device, seed=12345)
            dt = time.time() - t0
            print(f"[step {step:6d}] val loss_v={val['loss_v']:.5e} "
                  f"FOC_d={val['loss_FOC_d']:.5e} FOC_g={val['loss_FOC_g']:.5e} "
                  f"dv_dY={val['loss_dv_dY']:.5e}  ({dt:.0f}s)")
            history.append([step, val["loss_v"], val["loss_FOC_d"],
                            val["loss_FOC_g"], val["loss_dv_dY"]])

    val_final = validation(model, params, dtype, device, seed=12345)
    print(f"\n[FINAL] init={args.init} steps={args.steps} "
          f"loss_v: {val0['loss_v']:.5e} -> {val_final['loss_v']:.5e}")

    if args.export:
        os.makedirs(args.export, exist_ok=True)
        torch.save(model.v_nn.state_dict(), os.path.join(args.export, "v_nn_torch.pt"))
        torch.save(model.i_g_nn.state_dict(), os.path.join(args.export, "i_g_nn_torch.pt"))
        torch.save(model.i_d_nn.state_dict(), os.path.join(args.export, "i_d_nn_torch.pt"))
        np.savetxt(os.path.join(args.export, "torch_training_history.csv"),
                   np.array(history),
                   header="step,loss_v,loss_FOC_d,loss_FOC_g,loss_dv_dY",
                   delimiter=",", comments="")
        print(f"[export] wrote torch checkpoints + history to {args.export}")

    print("RESULT_JSON " + repr({
        "init": args.init, "steps": args.steps,
        "loss_v_start": val0["loss_v"], "loss_v_final": val_final["loss_v"],
    }))


if __name__ == "__main__":
    main()
