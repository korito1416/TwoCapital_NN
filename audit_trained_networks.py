#!/usr/bin/env python3
"""Audit trained HJB networks on large independent state samples."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from SimulationStochasticJumps import RegimeModels, STAGE_SPECS  # noqa: E402


RESIDUAL_NAMES = {
    False: ["pde_rmse", "foc_d_rmse", "foc_g_rmse", "dv_dy_penalty_rmse"],
    True: [
        "pde_rmse",
        "foc_d_rmse",
        "foc_g_rmse",
        "foc_r_rmse",
        "dv_dy_penalty_rmse",
    ],
}


def stage_input(stage: str, sample, params: Dict[str, object]) -> tf.Tensor:
    logk, z, y, logr, lambda3, logxi = sample
    ones = tf.ones_like(y)
    if stage == "PostDamagePostTech":
        columns = [logk, z, y, lambda3, float(params["A_g_prime_prime"]) * ones, logxi, logxi]
    elif stage == "PostDamageIntermTech":
        columns = [logk, z, y, logr, lambda3, logxi, logxi, logxi]
    elif stage == "PostDamagePreTech":
        columns = [logk, z, y, logr, lambda3, logxi, logxi, logxi]
    elif stage == "PreDamagePostTech":
        columns = [logk, z, y, float(params["A_g_prime_prime"]) * ones, logxi, logxi]
    elif stage == "PreDamageIntermTech":
        columns = [logk, z, y, logr, logxi, logxi, logxi]
    elif stage == "PreDamagePreTech":
        columns = [logk, z, y, logr, logxi, logxi, logxi]
    else:
        raise ValueError(stage)
    return tf.concat(columns, axis=1)


def finite_stats(values: np.ndarray, prefix: str) -> Dict[str, object]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            f"{prefix}_finite_fraction": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p01": np.nan,
            f"{prefix}_p99": np.nan,
        }
    return {
        f"{prefix}_finite_fraction": float(finite.size / values.size),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p01": float(np.percentile(finite, 1)),
        f"{prefix}_p99": float(np.percentile(finite, 99)),
    }


def checkpoint_delta(model: tf.keras.Model, reference: tf.keras.Model) -> Dict[str, float]:
    weights = model.get_weights()
    reference_weights = reference.get_weights()
    if len(weights) != len(reference_weights):
        return {"delta_l2": np.nan, "relative_delta_l2": np.nan, "exact_match": 0}
    squared_delta = 0.0
    squared_reference = 0.0
    exact_match = True
    for current, previous in zip(weights, reference_weights):
        if current.shape != previous.shape:
            return {"delta_l2": np.nan, "relative_delta_l2": np.nan, "exact_match": 0}
        difference = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
        squared_delta += float(np.sum(difference**2))
        squared_reference += float(np.sum(np.asarray(previous, dtype=float) ** 2))
        exact_match = exact_match and np.array_equal(current, previous)
    delta_l2 = float(np.sqrt(squared_delta))
    return {
        "delta_l2": delta_l2,
        "relative_delta_l2": delta_l2 / max(float(np.sqrt(squared_reference)), 1e-16),
        "exact_match": int(exact_match),
    }


def training_history_stats(folder: Path, stage: str) -> Dict[str, object]:
    path = folder / stage / "training_history.csv"
    if not path.exists():
        return {}
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return {}
    data = np.atleast_1d(data)
    result: Dict[str, object] = {
        "history_rows": int(data.size),
        "history_last_step": int(data["step"][-1]),
    }
    for name in data.dtype.names or ():
        if name in {"step", "elapsed_time"}:
            continue
        values = np.asarray(data[name], dtype=float)
        result[f"history_initial_{name}"] = float(values[0])
        result[f"history_final_{name}"] = float(values[-1])
        result[f"history_best_{name}"] = float(np.nanmin(values))
    return result


def audit_stage(
    evaluator: RegimeModels,
    stage: str,
    sample_size: int,
    chunk_size: int,
    reference: Optional[RegimeModels],
) -> Dict[str, object]:
    model = evaluator.models[stage]
    has_ir = STAGE_SPECS[stage][2]
    residual_sums = np.zeros(len(RESIDUAL_NAMES[has_ir]), dtype=float)
    residual_weight = 0
    outputs: Dict[str, List[np.ndarray]] = {"V": [], "i_d": [], "i_g": []}
    if has_ir:
        outputs["i_r"] = []

    remaining = int(sample_size)
    while remaining:
        n = min(int(chunk_size), remaining)
        sample = model.sample(batch_size=n)
        losses = model.objective_fn(*sample, training=False)
        residual_sums += n * np.asarray([float(loss.numpy()) for loss in losses])
        residual_weight += n

        x = stage_input(stage, sample, evaluator.params)
        outputs["V"].append(np.asarray(model.v_nn(x, training=False)).reshape(-1))
        outputs["i_d"].append(np.asarray(model.i_d_nn(x, training=False)).reshape(-1))
        outputs["i_g"].append(np.asarray(model.i_g_nn(x, training=False)).reshape(-1))
        if has_ir:
            raw_ir = np.asarray(model.i_r_nn(x, training=False)).reshape(-1)
            outputs["i_r"].append(np.exp(-raw_ir))
        remaining -= n

    row: Dict[str, object] = {
        "folder": evaluator.export_folder,
        "stage": stage,
        "sample_size": sample_size,
        "chunk_size": chunk_size,
        "has_rd_control": int(has_ir),
    }
    for name, value in zip(RESIDUAL_NAMES[has_ir], residual_sums / residual_weight):
        row[name] = float(value)
    for name, batches in outputs.items():
        row.update(finite_stats(np.concatenate(batches), name))

    networks = {"V": model.v_nn, "i_d": model.i_d_nn, "i_g": model.i_g_nn}
    if has_ir:
        networks["i_r"] = model.i_r_nn
    for name, network in networks.items():
        weights = network.get_weights()
        row[f"{name}_parameter_count"] = int(sum(np.size(weight) for weight in weights))
        row[f"{name}_weights_finite"] = int(
            all(np.isfinite(weight).all() for weight in weights)
        )
        if reference is not None and stage in reference.models:
            reference_network = getattr(reference.models[stage], f"{name.lower()}_nn")
            for key, value in checkpoint_delta(network, reference_network).items():
                row[f"{name}_{key}_vs_reference"] = value
    row.update(training_history_stats(Path(evaluator.export_folder), stage))
    return row


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-folder", action="append", required=True)
    parser.add_argument("--reference-folder")
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stage", action="append")
    parser.add_argument(
        "--include-all-stages",
        action="store_true",
        help="Load intermediary regimes even for a one-jump folder.",
    )
    parser.add_argument("--output", default="network_training_audit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    reference = (
        RegimeModels(
            args.reference_folder,
            args.xi,
            batch_size=128,
            include_all_stages=args.include_all_stages,
        )
        if args.reference_folder
        else None
    )
    rows = []
    for raw_folder in args.export_folder:
        folder = Path(raw_folder).expanduser()
        folder = folder if folder.is_absolute() else ROOT / folder
        evaluator = RegimeModels(
            str(folder),
            args.xi,
            batch_size=128,
            include_all_stages=args.include_all_stages,
        )
        stages = args.stage or list(evaluator.models)
        for stage in stages:
            if stage not in evaluator.models:
                continue
            row = audit_stage(
                evaluator,
                stage,
                sample_size=args.sample_size,
                chunk_size=args.chunk_size,
                reference=reference,
            )
            rows.append(row)
            residuals = [
                f"{name}={row[name]:.3e}"
                for name in RESIDUAL_NAMES[STAGE_SPECS[stage][2]]
            ]
            print(f"{folder.name} | {stage} | " + " | ".join(residuals))
        tf.keras.backend.clear_session()

    output = Path(args.output)
    output = output if output.is_absolute() else ROOT / output
    write_csv(output, rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
