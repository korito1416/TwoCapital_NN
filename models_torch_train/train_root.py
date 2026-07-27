"""
Torch DGM-PIA training loop for the ROOT regime (PostDamagePostTech).

Treats the TF-trained NN as a SOLVER SURROGATE (the NN is loaded / re-fit as a
function approximator of the HJB solution; this is NOT a new method).  The loop
faithfully mirrors models/PostDamagePostTech.py's DGM-PIA structure using the
already-validated torch pde_rhs in models_torch/PostDamagePostTech.py.

Per-step objective (this file, as specified):

  value   loss = sqrt(mean((rhs - pv)^2))                       -> trains v_nn
  control loss = sqrt(mean(FOC_d^2)) + sqrt(mean(FOC_g^2))
                 + sqrt(mean(dv_dY^2))                          -> trains i_g/i_d

with SEPARATE Adam optimizers for v_nn vs {i_g_nn, i_d_nn}, each driven by a
warmup_cosine LR schedule (peak ~4e-4).  State box is re-sampled each step
(batch 128) over:

    logK U(4,7), Z U(0.01,0.99), Y U(0,4), logR U(1,6),
    lam3 U(0,1/3), logxi U(-3,5).

float32 by default; set env TORCH_FLOAT64=1 (or --dtype float64) for double.
Checkpoints every --ckpt_every steps.

Run on Midway:
  srun --account=pi-lhansen --partition=caslake \
       python models_torch_train/train_root.py --steps 200 --init scratch
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

from tf_to_torch_loader import build_root_torch_model, DEFAULT_CKPT_DIR  # noqa: E402
from params import PARAMS  # noqa: E402


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
#  Sampler (uniform over each box column; re-sampled every step)
# ---------------------------------------------------------------------------
def _u(lo, hi, n, dtype, device):
    return lo + (hi - lo) * torch.rand((n, 1), dtype=dtype, device=device)


def sample_box(params, n, dtype, device):
    logK = _u(params["logK_min"], params["logK_max"], n, dtype, device)
    Z = _u(params["Z_min"], params["Z_max"], n, dtype, device)
    Y = _u(params["Y_min"], params["Y_max"], n, dtype, device)
    logR = _u(params["logR_min"], params["logR_max"], n, dtype, device)
    lam3 = _u(params["λ3_min"], params["λ3_max"], n, dtype, device)
    logxi = _u(params["logξ_min"], params["logξ_max"], n, dtype, device)
    return logK, Z, Y, logR, lam3, logxi


# ---------------------------------------------------------------------------
#  Losses
# ---------------------------------------------------------------------------
def _rms(x):
    return torch.sqrt(torch.mean(x ** 2))


def compute_terms(model, params, logK, Z, Y, logR, lam3, logxi):
    """Run torch pde_rhs and return the three loss components + diagnostics.

    pde_rhs returns: rhs, pv, dv_dY, c, (1+θ_g*i_g), (1+θ_d*i_d), FOC_d, FOC_g
    """
    logK = logK.detach().requires_grad_(True)
    Z = Z.detach().requires_grad_(True)
    Y = Y.detach().requires_grad_(True)

    rhs, pv, dv_dY, c, gig, gid, FOC_d, FOC_g = model.pde_rhs(
        logK, Z, Y, logR, lam3, logxi)

    resid = rhs - pv

    # dv_dY penalty: monotonicity guard for Y above the damage upper bound,
    # mirroring the TF objective (only the positive part above y_upper matters).
    y_upper = params["y_upper"]
    mask = ((Y > y_upper).to(dv_dY.dtype)) * ((dv_dY > 0).to(dv_dY.dtype))
    loss_dv_dY = dv_dY * mask

    value_loss = _rms(resid)
    control_loss = _rms(FOC_d) + _rms(FOC_g) + _rms(loss_dv_dY)

    diag = {
        "loss_v": float(_rms(resid).detach()),
        "loss_FOC_d": float(_rms(FOC_d).detach()),
        "loss_FOC_g": float(_rms(FOC_g).detach()),
        "loss_dv_dY": float(_rms(loss_dv_dY).detach()),
    }
    return value_loss, control_loss, diag


def save_ckpt(model, export_dir, tag):
    os.makedirs(export_dir, exist_ok=True)
    torch.save(model.v_nn.state_dict(),
               os.path.join(export_dir, f"v_nn_torch_{tag}.pt"))
    torch.save(model.i_g_nn.state_dict(),
               os.path.join(export_dir, f"i_g_nn_torch_{tag}.pt"))
    torch.save(model.i_d_nn.state_dict(),
               os.path.join(export_dir, f"i_d_nn_torch_{tag}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["surrogate", "scratch"], default="scratch")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-4, help="peak LR (value & control)")
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.01)
    ap.add_argument("--total_steps", type=int, default=None,
                    help="LR-schedule horizon (default = --steps).")
    ap.add_argument("--dtype", choices=["float32", "float64"], default=None,
                    help="overrides env TORCH_FLOAT64; default float32.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--export", default=None,
                    help="Folder for torch checkpoints + history.")
    args = ap.parse_args()

    # dtype: --dtype wins; else env TORCH_FLOAT64; else float32.
    if args.dtype is not None:
        use_f64 = (args.dtype == "float64")
    else:
        use_f64 = os.environ.get("TORCH_FLOAT64", "0") not in ("0", "", "false", "False")
    dtype = torch.float64 if use_f64 else torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = PARAMS
    total_steps = args.total_steps or args.steps
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    model = build_root_torch_model(
        load_surrogate=(args.init == "surrogate"),
        ckpt_dir=args.ckpt_dir, dtype=dtype)
    for net in (model.v_nn, model.i_g_nn, model.i_d_nn):
        net.to(device=device, dtype=dtype)
        net.train()

    opt_v = torch.optim.Adam(model.v_nn.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(
        list(model.i_g_nn.parameters()) + list(model.i_d_nn.parameters()), lr=args.lr)
    sched_v = WarmupCosine(args.lr, total_steps, warmup_steps, args.min_lr)
    sched_c = WarmupCosine(args.lr, total_steps, warmup_steps, args.min_lr)

    print(f"[setup] init={args.init} dtype={'float64' if use_f64 else 'float32'} "
          f"device={device} steps={args.steps} bs={args.batch_size} "
          f"lr={args.lr} min_lr={args.min_lr} warmup={warmup_steps}/{total_steps}")

    history = []
    t0 = time.time()
    first_diag = None

    for step in range(1, args.steps + 1):
        for g in opt_v.param_groups:
            g["lr"] = sched_v(step)
        for g in opt_c.param_groups:
            g["lr"] = sched_c(step)

        # ---- value step ----
        sample = sample_box(params, args.batch_size, dtype, device)
        value_loss, _, diag = compute_terms(model, params, *sample)
        opt_v.zero_grad(set_to_none=True)
        value_loss.backward()
        opt_v.step()

        # ---- control step (fresh re-sample, mirrors TF) ----
        sample = sample_box(params, args.batch_size, dtype, device)
        _, control_loss, _ = compute_terms(model, params, *sample)
        opt_c.zero_grad(set_to_none=True)
        control_loss.backward()
        opt_c.step()

        if first_diag is None:
            first_diag = diag

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            dt = time.time() - t0
            print(f"[step {step:6d}] loss_v={diag['loss_v']:.5e} "
                  f"FOC_d={diag['loss_FOC_d']:.5e} FOC_g={diag['loss_FOC_g']:.5e} "
                  f"dv_dY={diag['loss_dv_dY']:.5e}  ({dt:.0f}s)")
            history.append([step, diag["loss_v"], diag["loss_FOC_d"],
                            diag["loss_FOC_g"], diag["loss_dv_dY"]])

        if args.export and step % args.ckpt_every == 0:
            save_ckpt(model, args.export, f"step{step}")

    last_diag = history[-1] if history else [0, 0, 0, 0, 0]
    print(f"\n[FINAL] init={args.init} steps={args.steps} "
          f"loss_v: {first_diag['loss_v']:.5e} -> {last_diag[1]:.5e}")

    if args.export:
        save_ckpt(model, args.export, "final")
        np.savetxt(os.path.join(args.export, "torch_training_history.csv"),
                   np.array(history),
                   header="step,loss_v,loss_FOC_d,loss_FOC_g,loss_dv_dY",
                   delimiter=",", comments="")
        print(f"[export] wrote torch checkpoints + history to {args.export}")

    print("RESULT_JSON " + repr({
        "init": args.init, "steps": args.steps,
        "loss_v_start": first_diag["loss_v"], "loss_v_final": last_diag[1],
        "dtype": "float64" if use_f64 else "float32",
    }))


if __name__ == "__main__":
    main()
