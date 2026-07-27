"""Collect map-reduce low-mode warm-start experiments."""

import argparse
import csv
import json
from pathlib import Path


def read_last_jsonl(path):
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--best_path", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for hist in sorted(root.glob("*/history.jsonl")):
        run_dir = hist.parent
        last = read_last_jsonl(hist)
        if not last:
            continue
        arg_path = run_dir / "args.json"
        run_args = json.loads(arg_path.read_text(encoding="utf-8")) if arg_path.exists() else {}
        row = {
            "run": run_dir.name,
            "path": str(run_dir),
            "step": last.get("step"),
            "root_hjb": last.get("root_hjb"),
            "pre_hjb": last.get("pre_hjb"),
            "slice_rms": last.get("slice_rms"),
            "slice_mean": last.get("slice_mean"),
            "slice_dc": last.get("slice_dc"),
            "slice_low": last.get("slice_low"),
            "slice_foc_d": last.get("slice_foc_d"),
            "slice_foc_g": last.get("slice_foc_g"),
            "root_weight": run_args.get("root_weight"),
            "pre_weight": run_args.get("pre_weight"),
            "slice_dc_weight": run_args.get("slice_dc_weight"),
            "slice_rms_weight": run_args.get("slice_rms_weight"),
            "slice_low_weight": run_args.get("slice_low_weight"),
            "slice_foc_weight": run_args.get("slice_foc_weight"),
        }
        missing = any(row.get(key) is None for key in ("root_hjb", "pre_hjb", "slice_rms", "slice_foc_d", "slice_foc_g"))
        if missing:
            row["score"] = float("inf")
        else:
            row["score"] = (
                float(row["slice_rms"])
                + 0.25 * abs(float(row.get("slice_mean") or 0.0))
                + 0.10 * float(row["pre_hjb"])
                + 0.05 * float(row["root_hjb"])
                + 0.05 * (float(row["slice_foc_d"]) + float(row["slice_foc_g"]))
            )
        rows.append(row)

    rows.sort(key=lambda row: row["score"])
    if args.out_csv is None:
        args.out_csv = str(root / "lowmode_summary.csv")
    if args.out_json is None:
        args.out_json = str(root / "lowmode_summary.json")
    if args.best_path is None:
        args.best_path = str(root / "best_path.txt")

    fields = [
        "run",
        "path",
        "score",
        "step",
        "root_hjb",
        "pre_hjb",
        "slice_rms",
        "slice_mean",
        "slice_dc",
        "slice_low",
        "slice_foc_d",
        "slice_foc_g",
        "root_weight",
        "pre_weight",
        "slice_dc_weight",
        "slice_rms_weight",
        "slice_low_weight",
        "slice_foc_weight",
    ]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    Path(args.out_json).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    best = rows[0]["path"] if rows else ""
    Path(args.best_path).write_text(best + "\n", encoding="utf-8")

    print(f"[collect] runs={len(rows)} root={root}")
    print(f"[collect] wrote {args.out_csv}")
    if best:
        print(f"[collect] best={best}")
        for row in rows[:8]:
            print(
                f"{row['run']:32s} score={row['score']:.4e} "
                f"slice={row['slice_rms']:.4e} mean={row['slice_mean']:.2e} "
                f"pre={row['pre_hjb']:.4e} root={row['root_hjb']:.4e} "
                f"foc=({row['slice_foc_d']:.2e},{row['slice_foc_g']:.2e})"
            )


if __name__ == "__main__":
    main()
