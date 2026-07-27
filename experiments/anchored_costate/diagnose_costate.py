"""Diagnostics for analytical-costate losses vs FOC/PDE residuals."""

import argparse
import json
import sys
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

from anchored_nets import rms, sample_box  # noqa: E402
from train_two_regime_analytic_costate import costate_projection_terms  # noqa: E402
from train_two_regime_joint import _dtype, build_pre, build_root  # noqa: E402
from train_two_regime_lowmode import load_joint_checkpoint, make_slice_grid  # noqa: E402


LOWMODE = (
    "experiments/anchored_costate/results/lowmode_map_20260701_122130/"
    "dc1_rms025_foc025_seed0_task0"
)
ANALYTIC = (
    "experiments/anchored_costate/results/analytic_costate_cpu_20260701_130355/"
    "both_mild_cpu_seed0_task1"
)


def make_args(dtype_name, ckpt_dir):
    class Args:
        pass

    args = Args()
    args.width = 32
    args.layers = 4
    args.activation = "swish"
    args.no_residual = False
    args.value_arch = "clean"
    args.root_anchor = "surrogate"
    args.root_anchor_value = 0.0
    args.root_control_init = "surrogate"
    args.pre_control_init_rate = 0.02
    args.freeze_root_controls = False
    args.couple_anchor_grad = False
    args.couple_neighbor_grad = False
    args.ckpt_dir = ckpt_dir
    args.dtype = dtype_name
    args.seed = 0
    return args


def load_pair(path, dtype, dtype_name, device, ckpt_dir):
    args = make_args(dtype_name, ckpt_dir)
    root = build_root(args, dtype, device)
    pre = build_pre(args, root, dtype, device)
    load_joint_checkpoint(root, pre, path, device)
    for net in (root.v_nn, root.i_g_nn, root.i_d_nn, pre.v_nn, pre.i_g_nn, pre.i_d_nn):
        net.eval()
    return root, pre


def clone_sample(sample):
    return tuple(col.detach().clone() for col in sample)


def pde_foc_terms(model, sample):
    logK, Z, Y, logR, lam3, logxi = clone_sample(sample)
    logK.requires_grad_(True)
    Z.requires_grad_(True)
    Y.requires_grad_(True)
    logR.requires_grad_(True)
    rhs, pv, _dvdy, c, gig, gid, foc_d, foc_g = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    return {
        "resid": rhs - pv,
        "c": c,
        "gig": gig,
        "gid": gid,
        "foc_d": foc_d,
        "foc_g": foc_g,
    }


def sample_slice(args, dtype, device):
    spec = {"logK": args.logk, "logR": args.logr, "logxi": args.logxi, "lam3": args.lam3}
    return make_slice_grid(spec, args.grid, args.z_min, args.z_max, args.y_min, args.y_max, dtype, device)


def quantiles(x):
    arr = x.detach().cpu().numpy().reshape(-1)
    return {
        "mean": float(arr.mean()),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def summarize_model(label, model, sample, regime):
    p = model.params
    pde = pde_foc_terms(model, sample)
    terms = costate_projection_terms(model, sample, regime=regime, relative=False)
    qd_err = terms["err_qd"]
    qg_err = terms["err_qg"]
    # FOC identity check: FOC_j = phi_j'(i_j) * (q_j^NN - q_j^target).
    # Recover phi' from target relation and pde controls: gig/gid = 1 + theta*i.
    phi_d_prime = p["Γ_d"] * p["θ_d"] / torch.clamp(pde["gid"], min=1e-8)
    phi_g_prime = p["Γ_g"] * p["θ_g"] / torch.clamp(pde["gig"], min=1e-8)
    foc_d_from_costate = phi_d_prime * qd_err
    foc_g_from_costate = phi_g_prime * qg_err
    foc_d_gap = pde["foc_d"] - foc_d_from_costate
    foc_g_gap = pde["foc_g"] - foc_g_from_costate
    out = {
        "label": label,
        "regime": regime,
        "hjb": float(rms(pde["resid"]).detach().cpu()),
        "resid_mean": float(torch.mean(pde["resid"]).detach().cpu()),
        "foc_d": float(rms(pde["foc_d"]).detach().cpu()),
        "foc_g": float(rms(pde["foc_g"]).detach().cpu()),
        "qerr_d": float(rms(qd_err).detach().cpu()),
        "qerr_g": float(rms(qg_err).detach().cpu()),
        "perr_logK": float(rms(terms["err_pk"]).detach().cpu()),
        "perr_Z": float(rms(terms["err_pz"]).detach().cpu()),
        "foc_identity_gap_d": float(rms(foc_d_gap).detach().cpu()),
        "foc_identity_gap_g": float(rms(foc_g_gap).detach().cpu()),
        "c_min": float(pde["c"].min().detach().cpu()),
        "qd_target": quantiles(terms["qd_target"]),
        "qg_target": quantiles(terms["qg_target"]),
        "qd_err": quantiles(qd_err),
        "qg_err": quantiles(qg_err),
        "resid": quantiles(pde["resid"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--lowmode_dir", default=LOWMODE)
    ap.add_argument("--analytic_dir", default=ANALYTIC)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--logk", type=float, default=5.5)
    ap.add_argument("--logr", type=float, default=3.5)
    ap.add_argument("--lam3", type=float, default=1.0 / 6.0)
    ap.add_argument("--logxi", type=float, default=np.log(0.05))
    ap.add_argument("--z_min", type=float, default=0.05)
    ap.add_argument("--z_max", type=float, default=0.95)
    ap.add_argument("--y_min", type=float, default=0.2)
    ap.add_argument("--y_max", type=float, default=3.8)
    ap.add_argument("--random_n", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tf_to_torch_loader import DEFAULT_CKPT_DIR
    ckpt_dir = args.ckpt_dir or DEFAULT_CKPT_DIR
    dtype = _dtype(args.dtype)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(12345)
    np.random.seed(12345)

    rows = []
    for label, path in (("lowmode", args.lowmode_dir), ("analytic", args.analytic_dir)):
        _root, pre = load_pair(path, dtype, args.dtype, device, ckpt_dir)
        slice_sample = sample_slice(args, dtype, device)
        rows.append(summarize_model(f"{label}_slice", pre, slice_sample, "pre"))
        random_sample = sample_box(pre.params, args.random_n, dtype, device)
        rows.append(summarize_model(f"{label}_random", pre, random_sample, "pre"))

    text = json.dumps(rows, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    for row in rows:
        print(
            f"{row['label']:16s} hjb={row['hjb']:.4e} mean={row['resid_mean']:.2e} "
            f"foc=({row['foc_d']:.2e},{row['foc_g']:.2e}) "
            f"qerr=({row['qerr_d']:.2e},{row['qerr_g']:.2e}) "
            f"idgap=({row['foc_identity_gap_d']:.1e},{row['foc_identity_gap_g']:.1e})"
        )


if __name__ == "__main__":
    main()
