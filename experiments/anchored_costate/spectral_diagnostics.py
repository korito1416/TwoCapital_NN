"""Frequency-domain diagnostics for original and anchored HJB solvers.

This script evaluates HJB residual fields on a 2D (Z, Y) slice and decomposes
the fields with an orthonormal DCT-II.  DCT is preferable to a periodic FFT on
the finite state box because it avoids a fake periodic boundary discontinuity.

The diagnostic is intentionally read-only with respect to production models:
it loads original TF checkpoints through the validated torch ports and compares
them to experiment checkpoints in ``experiments/anchored_costate/results``.
"""

import argparse
import csv
import json
import os
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

from PreDamagePostTech import PreDamagePostTechModel  # noqa: E402
from anchored_nets import value_loss  # noqa: E402
from params import PARAMS  # noqa: E402
from tf_to_torch_loader import (  # noqa: E402
    DEFAULT_CKPT_DIR,
    build_root_torch_model,
    make_root_configs,
)
from tf_torch_harness import load_tf_weights_into_torch  # noqa: E402
from train_two_regime_joint import build_pre, build_root  # noqa: E402


def dtype_from_name(name):
    return torch.float64 if name == "float64" else torch.float32


def make_args(seed, dtype_name, ckpt_dir):
    class Args:
        pass

    args = Args()
    args.seed = int(seed)
    args.width = 32
    args.layers = 4
    args.activation = "swish"
    args.no_residual = False
    args.value_arch = "clean"
    args.root_anchor = "surrogate"
    args.root_anchor_value = 0.0
    args.root_control_init = "surrogate"
    args.pre_control_init_rate = 0.02
    args.couple_anchor_grad = False
    args.couple_neighbor_grad = False
    args.ckpt_dir = ckpt_dir
    args.dtype = dtype_name
    return args


def safe_load(path, device):
    return torch.load(str(path), map_location=device)


def build_original_pre(run_dir, dtype, device):
    v_cfg, ig_cfg, id_cfg = make_root_configs()
    model = PreDamagePostTechModel({
        "v_nn_config": v_cfg,
        "i_g_nn_config": ig_cfg,
        "i_d_nn_config": id_cfg,
    })
    ckpt_dir = Path(run_dir) / "PreDamagePostTech"
    root_ckpt = Path(run_dir) / "PostDamagePostTech" / "v_nn_checkpoint_PostDamagePostTech"
    load_tf_weights_into_torch(model.v_nn, str(ckpt_dir / "v_nn_checkpoint_PreDamagePostTech"))
    load_tf_weights_into_torch(model.i_g_nn, str(ckpt_dir / "i_g_nn_checkpoint_PreDamagePostTech"))
    load_tf_weights_into_torch(model.i_d_nn, str(ckpt_dir / "i_d_nn_checkpoint_PreDamagePostTech"))
    load_tf_weights_into_torch(model.v_PostDamagePostTech_nn, str(root_ckpt))
    for net in (model.v_nn, model.i_g_nn, model.i_d_nn, model.v_PostDamagePostTech_nn):
        net.to(device=device, dtype=dtype)
        net.eval()
    return model


def build_new_joint(checkpoint_dir, dtype, dtype_name, device, seed, ckpt_dir):
    args = make_args(seed=seed, dtype_name=dtype_name, ckpt_dir=ckpt_dir)
    root_model = build_root(args, dtype, device)
    pre_model = build_pre(args, root_model, dtype, device)
    checkpoint_dir = Path(checkpoint_dir)
    root_model.v_nn.load_state_dict(safe_load(checkpoint_dir / "root_v_recentered.pt", device))
    root_model.i_g_nn.load_state_dict(safe_load(checkpoint_dir / "root_i_g.pt", device))
    root_model.i_d_nn.load_state_dict(safe_load(checkpoint_dir / "root_i_d.pt", device))
    pre_model.v_nn.load_state_dict(safe_load(checkpoint_dir / "pre_damage_posttech_v_recentered.pt", device))
    pre_model.i_g_nn.load_state_dict(safe_load(checkpoint_dir / "pre_damage_posttech_i_g.pt", device))
    pre_model.i_d_nn.load_state_dict(safe_load(checkpoint_dir / "pre_damage_posttech_i_d.pt", device))
    for net in (root_model.v_nn, root_model.i_g_nn, root_model.i_d_nn,
                pre_model.v_nn, pre_model.i_g_nn, pre_model.i_d_nn,
                pre_model.v_PostDamagePostTech_nn):
        net.to(device=device, dtype=dtype)
        net.eval()
    return root_model, pre_model


def build_costate_root(checkpoint_dir, dtype, dtype_name, device, seed, ckpt_dir):
    from costate_robust_root import CostateRobustRoot

    class Args:
        pass

    args = Args()
    args.width = 32
    args.layers = 4
    args.activation = "tanh"
    args.no_residual = False
    args.seed = int(seed)
    args.control_init_rate = 0.02
    args.robust_mode = "residual"
    args.robust_width = 64
    args.robust_depth = 3
    args.robust_residual_scale = 0.05
    args.robust_direct_scale = 2.0
    args.p_logk_scale = 0.5
    args.p_other_scale = 0.2
    args.quad_points = 8
    args.ckpt_dir = ckpt_dir
    args.dtype = dtype_name

    model = CostateRobustRoot(args, dtype, device)
    checkpoint_dir = Path(checkpoint_dir)
    model.costate_nn.load_state_dict(safe_load(checkpoint_dir / "costate_nn.pt", device))
    model.i_g_nn.load_state_dict(safe_load(checkpoint_dir / "i_g_nn.pt", device))
    model.i_d_nn.load_state_dict(safe_load(checkpoint_dir / "i_d_nn.pt", device))
    robust_path = checkpoint_dir / "robust_nn.pt"
    if robust_path.exists() and model.robust_nn is not None:
        model.robust_nn.load_state_dict(safe_load(robust_path, device))
    model.costate_nn.eval()
    model.i_g_nn.eval()
    model.i_d_nn.eval()
    if model.robust_nn is not None:
        model.robust_nn.eval()
    return model


def grid_sample(args, dtype, device):
    z = torch.linspace(args.z_min, args.z_max, args.grid, dtype=dtype, device=device)
    y = torch.linspace(args.y_min, args.y_max, args.grid, dtype=dtype, device=device)
    zz, yy = torch.meshgrid(z, y, indexing="ij")
    n = args.grid * args.grid
    logk = torch.full((n, 1), args.logk, dtype=dtype, device=device)
    Z = zz.reshape(n, 1)
    Y = yy.reshape(n, 1)
    logR = torch.full((n, 1), args.logr, dtype=dtype, device=device)
    lam3 = torch.full((n, 1), args.lam3, dtype=dtype, device=device)
    logxi = torch.full((n, 1), args.logxi, dtype=dtype, device=device)
    return logk, Z, Y, logR, lam3, logxi


def residual_field(model, sample, grid, regime, costate=False):
    if costate:
        terms = model.pde_terms(sample, want_curl=True, robust_input_detach=True)
        resid = terms["resid"]
        aux = {
            "curl_rms": float(torch.sqrt(torch.mean(terms["curl"] ** 2) + 1e-16).detach().cpu()),
            "foc_d_rms": float(torch.sqrt(torch.mean(terms["FOC_d"] ** 2) + 1e-16).detach().cpu()),
            "foc_g_rms": float(torch.sqrt(torch.mean(terms["FOC_g"] ** 2) + 1e-16).detach().cpu()),
        }
    else:
        logK, Z, Y, logR, lam3, logxi = sample
        logK = logK.detach().requires_grad_(True)
        Z = Z.detach().requires_grad_(True)
        Y = Y.detach().requires_grad_(True)
        logR = logR.detach().requires_grad_(True)
        rhs, pv, dv_dY, c, gig, gid, foc_d, foc_g = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
        del dv_dY, c, gig, gid
        resid = rhs - pv
        aux = {
            "foc_d_rms": float(torch.sqrt(torch.mean(foc_d ** 2) + 1e-16).detach().cpu()),
            "foc_g_rms": float(torch.sqrt(torch.mean(foc_g ** 2) + 1e-16).detach().cpu()),
        }
    arr = resid.detach().cpu().numpy().reshape(grid, grid)
    return arr, aux


def dct_matrix(n):
    x = np.arange(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64).reshape(-1, 1)
    mat = np.cos(np.pi / n * (x + 0.5) * k)
    mat[0, :] *= np.sqrt(1.0 / n)
    mat[1:, :] *= np.sqrt(2.0 / n)
    return mat


def spectral_stats(field):
    field = np.asarray(field, dtype=np.float64)
    n = field.shape[0]
    centered = field - field.mean()
    D = dct_matrix(n)
    coeff = D @ field @ D.T
    coeff_centered = D @ centered @ D.T
    energy = coeff ** 2
    centered_energy = coeff_centered ** 2
    total = float(energy.sum()) + 1e-300
    centered_total = float(centered_energy.sum()) + 1e-300
    kk0, kk1 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    radius = np.sqrt(kk0 ** 2 + kk1 ** 2)

    def frac(mask, source=energy, denom=total):
        return float(source[mask].sum() / denom)

    high_mask = radius > max(4.0, n / 4.0)
    mid_mask = (radius > 2.0) & (~high_mask)
    low_mask = (radius <= 2.0)
    no_dc_low = low_mask.copy()
    no_dc_low[0, 0] = False

    flat = energy.reshape(-1)
    order = np.argsort(flat)[::-1]
    top = []
    for idx in order[:8]:
        i, j = np.unravel_index(int(idx), energy.shape)
        top.append({"k_z": int(i), "k_y": int(j), "energy_frac": float(energy[i, j] / total)})

    return {
        "rms": float(np.sqrt(np.mean(field ** 2))),
        "mean": float(field.mean()),
        "std": float(field.std()),
        "max_abs": float(np.max(np.abs(field))),
        "dc_frac": float(energy[0, 0] / total),
        "low_frac": frac(low_mask),
        "low_no_dc_frac": frac(no_dc_low),
        "mid_frac": frac(mid_mask),
        "high_frac": frac(high_mask),
        "centered_low_frac": frac(no_dc_low, centered_energy, centered_total),
        "centered_mid_frac": frac(mid_mask, centered_energy, centered_total),
        "centered_high_frac": frac(high_mask, centered_energy, centered_total),
        "top_modes": top,
        "coeff": coeff,
    }


def maybe_plot(fields, summaries, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        print(f"[plot] skipped: {exc}")
        return

    pdf_path = Path(out_dir) / "spectral_diagnostics.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        for name, field in fields.items():
            stats = summaries[name]
            coeff = np.log10(np.abs(stats["coeff"]) + 1e-16)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].imshow(field.T, origin="lower", aspect="auto")
            axes[0].set_title(f"{name}: residual")
            axes[0].set_xlabel("Z grid")
            axes[0].set_ylabel("Y grid")
            fig.colorbar(im0, ax=axes[0], fraction=0.046)
            im1 = axes[1].imshow(coeff.T, origin="lower", aspect="auto")
            axes[1].set_title("log10 |DCT coeff|")
            axes[1].set_xlabel("k_Z")
            axes[1].set_ylabel("k_Y")
            fig.colorbar(im1, ax=axes[1], fraction=0.046)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        labels = list(fields.keys())
        high = [summaries[k]["centered_high_frac"] for k in labels]
        low = [summaries[k]["centered_low_frac"] for k in labels]
        mid = [summaries[k]["centered_mid_frac"] for k in labels]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
        x = np.arange(len(labels))
        ax.bar(x, low, label="low")
        ax.bar(x, mid, bottom=low, label="mid")
        ax.bar(x, high, bottom=np.asarray(low) + np.asarray(mid), label="high")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("centered DCT energy fraction")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"[plot] wrote {pdf_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logk", type=float, default=5.5)
    ap.add_argument("--logr", type=float, default=3.5)
    ap.add_argument("--lam3", type=float, default=1.0 / 6.0)
    ap.add_argument("--logxi", type=float, default=np.log(0.05))
    ap.add_argument("--z_min", type=float, default=0.05)
    ap.add_argument("--z_max", type=float, default=0.95)
    ap.add_argument("--y_min", type=float, default=0.2)
    ap.add_argument("--y_max", type=float, default=3.8)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--orig_run_dir", default=str(Path(DEFAULT_CKPT_DIR).parent))
    ap.add_argument(
        "--new_dir",
        default="experiments/anchored_costate/results/precision_push_batch256/"
                "seed0_20000steps_20260630_210314/fedctrl_long_base",
    )
    ap.add_argument(
        "--costate_dir",
        default="experiments/anchored_costate/results/costate_robust_root/"
                "residual_seed11_20260701_101727_51345403",
    )
    ap.add_argument("--out_dir", default="experiments/anchored_costate/results/spectral_diagnostics")
    ap.add_argument("--skip_costate", action="store_true")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] grid={args.grid} dtype={args.dtype} device={device}")
    print(f"[slice] logK={args.logk:.4f} logR={args.logr:.4f} lam3={args.lam3:.4f} logxi={args.logxi:.4f}")

    sample = grid_sample(args, dtype, device)
    fields = {}
    aux_rows = {}

    print("[load] original root")
    orig_root = build_root_torch_model(load_surrogate=True, ckpt_dir=args.ckpt_dir, dtype=dtype)
    for net in (orig_root.v_nn, orig_root.i_g_nn, orig_root.i_d_nn):
        net.to(device=device, dtype=dtype)
        net.eval()
    fields["orig_root"], aux_rows["orig_root"] = residual_field(orig_root, sample, args.grid, "root")

    print("[load] original pre")
    orig_pre = build_original_pre(args.orig_run_dir, dtype, device)
    fields["orig_pre"], aux_rows["orig_pre"] = residual_field(orig_pre, sample, args.grid, "pre")

    print("[load] new anchored joint")
    new_root, new_pre = build_new_joint(args.new_dir, dtype, args.dtype, device, args.seed, args.ckpt_dir)
    fields["new_root"], aux_rows["new_root"] = residual_field(new_root, sample, args.grid, "root")
    fields["new_pre"], aux_rows["new_pre"] = residual_field(new_pre, sample, args.grid, "pre")

    if not args.skip_costate and Path(args.costate_dir).exists():
        print("[load] new costate root")
        costate_root = build_costate_root(args.costate_dir, dtype, args.dtype, device, 11, args.ckpt_dir)
        fields["costate_root"], aux_rows["costate_root"] = residual_field(
            costate_root, sample, args.grid, "root", costate=True
        )

    summaries = {}
    for name, field in fields.items():
        summaries[name] = spectral_stats(field)
        np.save(out_dir / f"{name}_residual.npy", field)
        coeff = summaries[name].pop("coeff")
        np.save(out_dir / f"{name}_dct_coeff.npy", coeff)

    rows = []
    for name, stats in summaries.items():
        row = {"model": name}
        for key, value in stats.items():
            if key != "top_modes":
                row[key] = value
        row.update(aux_rows.get(name, {}))
        row["top_modes_json"] = json.dumps(stats["top_modes"], sort_keys=True)
        rows.append(row)

    csv_path = out_dir / "spectral_summary.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "spectral_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    if not args.no_plot:
        plot_summaries = {}
        for name in fields:
            plot_summaries[name] = dict(summaries[name])
            plot_summaries[name]["coeff"] = np.load(out_dir / f"{name}_dct_coeff.npy")
        maybe_plot(fields, plot_summaries, out_dir)

    print(f"[write] {csv_path}")
    for row in rows:
        print(
            f"{row['model']:14s} rms={row['rms']:.4e} mean={row['mean']:.2e} "
            f"dc={row['dc_frac']:.2%} centered_high={row['centered_high_frac']:.2%} "
            f"foc=({row.get('foc_d_rms', float('nan')):.2e},{row.get('foc_g_rms', float('nan')):.2e})"
        )


if __name__ == "__main__":
    main()
