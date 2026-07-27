"""Collect RESULT_JSON lines from robust-minimizer experiment logs."""

import argparse
import csv
import json
from pathlib import Path


def parse_logs(root):
    rows = []
    for path in sorted(Path(root).rglob("*.out")):
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            marker = "RESULT_JSON "
            if marker in line:
                payload = line.split(marker, 1)[1].strip()
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                row["log_path"] = str(path)
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="experiments/anchored_costate/results",
        help="Directory to scan recursively for Slurm .out files.",
    )
    ap.add_argument(
        "--csv",
        default="experiments/anchored_costate/results/robust_minimizer_summary.csv",
    )
    args = ap.parse_args()

    rows = parse_logs(args.root)
    if not rows:
        print("No RESULT_JSON rows found.")
        return

    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    for row in rows:
        bits = [
            row.get("mode", "?"),
            row.get("robust_mode", row.get("jump_robust_mode", "?")),
            f"hjb={row.get('hjb_final', row.get('pre_hjb_final', ''))}",
            f"gap={row.get('robust_gap_final', row.get('jump_gap_final', ''))}",
            f"log={row['log_path']}",
        ]
        print(" | ".join(str(x) for x in bits))


if __name__ == "__main__":
    main()
