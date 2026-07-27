#!/usr/bin/env python3
"""Summarize initial, best, and final logged losses for model folders."""

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for raw_folder in args.folder:
        folder = Path(raw_folder).resolve()
        for history in sorted(folder.glob("*/training_history.csv")):
            data = np.genfromtxt(history, delimiter=",", names=True)
            if data.size == 0:
                continue
            data = np.atleast_1d(data)
            stage = history.parent.name
            for metric in data.dtype.names or ():
                if metric in {"step", "elapsed_time"}:
                    continue
                values = np.asarray(data[metric], dtype=float)
                rows.append(
                    {
                        "folder": str(folder),
                        "stage": stage,
                        "metric": metric,
                        "initial": float(values[0]),
                        "best": float(np.nanmin(values)),
                        "final": float(values[-1]),
                        "last_step": int(data["step"][-1]),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["folder", "stage", "metric", "initial", "best", "final", "last_step"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} ({len(rows)} metric rows)")


if __name__ == "__main__":
    main()
