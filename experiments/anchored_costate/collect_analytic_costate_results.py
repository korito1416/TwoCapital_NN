"""Collect analytical-costate warm-start experiments."""

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


def score_row(row):
    keys = ("pre_hjb", "root_hjb", "pre_foc_d", "pre_foc_g", "slice_rms", "slice_cost_qd", "slice_cost_qg")
    if any(row.get(key) is None for key in keys):
        return float("inf")
    return (
        float(row["slice_rms"])
        + 0.25 * abs(float(row.get("slice_mean") or 0.0))
        + 0.10 * float(row["pre_hjb"])
        + 0.05 * float(row["root_hjb"])
        + 0.10 * (float(row["pre_foc_d"]) + float(row["pre_foc_g"]))
        + 0.50 * (float(row["slice_cost_qd"]) + float(row["slice_cost_qg"]))
    )


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
            "score": None,
            "step": last.get("step"),
            "root_hjb": last.get("root_hjb"),
            "pre_hjb": last.get("pre_hjb"),
            "pre_foc_d": last.get("pre_foc_d"),
            "pre_foc_g": last.get("pre_foc_g"),
            "slice_rms": last.get("slice_rms"),
            "slice_mean": last.get("slice_mean"),
            "slice_dc": last.get("slice_dc"),
            "slice_cost_qd": last.get("slice_cost_qd"),
            "slice_cost_qg": last.get("slice_cost_qg"),
            "slice_cost_pk": last.get("slice_cost_pk"),
            "slice_cost_pz": last.get("slice_cost_pz"),
            "slice_cost_low": last.get("slice_cost_low"),
            "costate_mode": run_args.get("costate_mode"),
            "pre_costate_weight": run_args.get("pre_costate_weight"),
            "slice_costate_weight": run_args.get("slice_costate_weight"),
            "slice_costate_low_weight": run_args.get("slice_costate_low_weight"),
            "slice_grid": run_args.get("slice_grid"),
            "lr_value": run_args.get("lr_value"),
            "lr_control": run_args.get("lr_control"),
        }
        row["score"] = score_row(row)
        rows.append(row)

    rows.sort(key=lambda row: row["score"])
    out_csv = Path(args.out_csv or root / "analytic_costate_summary.csv")
    out_json = Path(args.out_json or root / "analytic_costate_summary.json")
    best_path = Path(args.best_path or root / "best_path.txt")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    best = rows[0]["path"] if rows else ""
    best_path.write_text(best + "\n", encoding="utf-8")

    print(f"[collect] runs={len(rows)} root={root}")
    print(f"[collect] wrote {out_csv}")
    if best:
        print(f"[collect] best={best}")
        for row in rows[:8]:
            print(
                f"{row['run']:34s} score={row['score']:.4e} "
                f"slice={row['slice_rms']:.4e} cost=({row['slice_cost_qd']:.2e},{row['slice_cost_qg']:.2e}) "
                f"pre={row['pre_hjb']:.4e} foc=({row['pre_foc_d']:.2e},{row['pre_foc_g']:.2e})"
            )


if __name__ == "__main__":
    main()
