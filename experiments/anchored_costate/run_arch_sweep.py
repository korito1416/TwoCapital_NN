"""Run a small architecture sweep for the anchored two-regime sandbox."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


VARIANTS = {
    "anchor_clean": [],
    "deep_6x32": ["--layers", "6"],
    "wide_4x64": ["--width", "64"],
    "frozen_middle": ["--value_arch", "frozen_middle"],
    "federated_grad": ["--couple_anchor_grad", "--couple_neighbor_grad"],
    "federated_control": [
        "--couple_anchor_grad",
        "--couple_neighbor_grad",
        "--control_steps", "3",
        "--pre_control_weight", "3.0",
    ],
    "fedctrl_long_base": [
        "--couple_anchor_grad",
        "--couple_neighbor_grad",
        "--control_steps", "3",
        "--pre_control_weight", "3.0",
    ],
    "fedctrl_preweight3": [
        "--couple_anchor_grad",
        "--couple_neighbor_grad",
        "--control_steps", "3",
        "--pre_control_weight", "3.0",
        "--pre_weight", "3.0",
    ],
    "fedctrl_control5": [
        "--couple_anchor_grad",
        "--couple_neighbor_grad",
        "--control_steps", "5",
        "--pre_control_weight", "5.0",
    ],
    "fedctrl_low_lr": [
        "--couple_anchor_grad",
        "--couple_neighbor_grad",
        "--control_steps", "3",
        "--pre_control_weight", "3.0",
        "--lr_value", "1e-4",
        "--lr_control", "2e-4",
        "--min_lr", "1e-6",
    ],
    "fedctrl_foc_value": [
        "--couple_anchor_grad",
        "--couple_neighbor_grad",
        "--control_steps", "3",
        "--pre_control_weight", "3.0",
        "--foc_in_value",
    ],
    "fedctrl_focus_pre": [
        "--couple_anchor_grad",
        "--couple_neighbor_grad",
        "--control_steps", "3",
        "--pre_control_weight", "3.0",
        "--root_weight", "0.25",
        "--pre_weight", "5.0",
    ],
    "control_strong": ["--control_steps", "3", "--pre_control_weight", "3.0"],
    "frozen_middle_control": [
        "--value_arch", "frozen_middle",
        "--control_steps", "3",
        "--pre_control_weight", "3.0",
    ],
}


def parse_result(text):
    result = None
    for line in text.splitlines():
        if line.startswith("RESULT_JSON "):
            result = json.loads(line[len("RESULT_JSON "):])
    return result


def run_variant(name, extra, args, sweep_dir):
    out_dir = sweep_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(HERE / "train_two_regime_joint.py"),
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--valid_batch_size", str(args.valid_batch_size),
        "--valid_batches", str(args.valid_batches),
        "--valid_seed", str(args.valid_seed),
        "--log_every", str(args.log_every),
        "--seed", str(args.seed),
        "--dtype", args.dtype,
        "--lr_value", str(args.lr_value),
        "--lr_control", str(args.lr_control),
        "--root_control_init", args.root_control_init,
        "--export", str(out_dir),
    ] + list(extra)

    if args.foc_in_value:
        cmd.append("--foc_in_value")
    if args.freeze_root_controls:
        cmd.append("--freeze_root_controls")

    log_path = out_dir / "run.log"
    print(f"\n=== RUN {name}: {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, flush=True)
    result = parse_result(proc.stdout)
    if result is None:
        result = {"mode": "failed", "returncode": proc.returncode}
    result.update({
        "variant": name,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "log": str(log_path),
        "export": str(out_dir),
    })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--valid_batch_size", type=int, default=256)
    ap.add_argument("--valid_batches", type=int, default=2)
    ap.add_argument("--valid_seed", type=int, default=12345)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lr_value", type=float, default=4e-4)
    ap.add_argument("--lr_control", type=float, default=4e-4)
    ap.add_argument("--root_control_init", choices=["surrogate", "clean", "torch_scratch"], default="surrogate")
    ap.add_argument("--foc_in_value", action="store_true")
    ap.add_argument("--freeze_root_controls", action="store_true")
    ap.add_argument("--variants", default="all",
                    help="Comma list of variants or 'all'.")
    ap.add_argument("--output_root", default=str(HERE / "results" / "arch_sweep"))
    args = ap.parse_args()

    if args.variants == "all":
        names = list(VARIANTS)
    else:
        names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; valid={sorted(VARIANTS)}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_root) / f"seed{args.seed}_{args.steps}steps_{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        rows.append(run_variant(name, VARIANTS[name], args, sweep_dir))

    json_path = sweep_dir / "summary.json"
    csv_path = sweep_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    keys = [
        "variant", "returncode", "steps", "root_hjb_start", "root_hjb_final",
        "pre_hjb_start", "pre_hjb_final", "boundary_gap_final",
        "root_level_final", "pre_level_final", "elapsed_sec", "export", "log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n=== SWEEP SUMMARY ===")
    for row in rows:
        print(
            f"{row['variant']:>22s} "
            f"root={row.get('root_hjb_final', float('nan')):.4e} "
            f"pre={row.get('pre_hjb_final', float('nan')):.4e} "
            f"gap={row.get('boundary_gap_final', float('nan')):.1e} "
            f"rc={row['returncode']}"
        )
    print(f"SUMMARY_JSON {json_path}")
    print(f"SUMMARY_CSV {csv_path}")


if __name__ == "__main__":
    main()
