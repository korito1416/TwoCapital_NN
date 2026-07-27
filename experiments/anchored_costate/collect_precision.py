"""Collect precision-push results into a compact ranking table."""

import argparse
import csv
import glob
import json
import os
from pathlib import Path


def read_last_history(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[-1] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "experiments/anchored_costate/results/precision_push",
        "experiments/anchored_costate/results/precision_push_batch256",
    ])
    ap.add_argument("--out", default="experiments/anchored_costate/results/precision_push_summary.csv")
    args = ap.parse_args()

    records = []
    for root in args.roots:
        for hist in glob.glob(os.path.join(root, "**", "history.jsonl"), recursive=True):
            last = read_last_history(hist)
            if last is None:
                continue
            variant = Path(hist).parent.name
            run_dir = str(Path(hist).parent)
            records.append({
                "variant": variant,
                "step": last.get("step"),
                "root_hjb": last.get("root_hjb"),
                "pre_hjb": last.get("pre_hjb"),
                "boundary_gap": last.get("boundary_gap"),
                "root_foc_d": last.get("root_foc_d"),
                "root_foc_g": last.get("root_foc_g"),
                "pre_foc_d": last.get("pre_foc_d"),
                "pre_foc_g": last.get("pre_foc_g"),
                "root_level": last.get("root_level"),
                "pre_level": last.get("pre_level"),
                "run_dir": run_dir,
            })

    records.sort(key=lambda r: (
        float("inf") if r["pre_hjb"] is None else float(r["pre_hjb"]),
        float("inf") if r["root_hjb"] is None else float(r["root_hjb"]),
    ))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant", "step", "root_hjb", "pre_hjb", "boundary_gap",
        "root_foc_d", "root_foc_g", "pre_foc_d", "pre_foc_g",
        "root_level", "pre_level", "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"wrote {out_path}")
    for row in records[:20]:
        print(
            f"{row['variant']:>22s} step={row['step']} "
            f"root={float(row['root_hjb']):.4e} pre={float(row['pre_hjb']):.4e} "
            f"gap={float(row['boundary_gap']):.1e} "
            f"pre_foc=({float(row['pre_foc_d']):.3e},{float(row['pre_foc_g']):.3e})"
        )


if __name__ == "__main__":
    main()
