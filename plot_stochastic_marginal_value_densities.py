#!/usr/bin/env python3
"""Plot marginal-value densities around stochastic jump events.

For the transformed states

    logK = log(Kd + Kg),    Z = Kg / (Kd + Kg),

the marginal values of dirty and green capital are

    V_Kd = (V_logK - Z V_Z) / K,
    V_Kg = (V_logK + (1 - Z) V_Z) / K.

The value-network implementation reconstructs the temperature derivative as

    V_Y = v_Y - d log N / dY.

This script reports the positive marginal climate cost -V_Y.  Before damage,
d log N / dY = lambda1 + lambda2 Y.  After damage, the realized curvature adds
lambda3 (Y - y_upper).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from SimulationStochasticJumps import RegimeModels  # noqa: E402
from plot_stochastic_control_densities import (  # noqa: E402
    DEFAULT_MODELS,
    EVENTS,
    comparison_indices,
    comparison_labels,
    compute_range,
    discover_seeds,
    plot_density,
    summarize_values,
    transition_indices,
    write_csv,
)


MARGINALS = [
    ("dV_dKd", r"Marginal value of dirty capital $\partial V/\partial K^d$"),
    ("dV_dKg", r"Marginal value of green capital $\partial V/\partial K^g$"),
    ("climate_marginal_cost", r"Marginal climate cost $-\partial V/\partial Y$"),
]

STATE_ARRAYS = {
    "logK": "logK_sim",
    "Z": "Z_array",
    "Y": "Y_array",
    "logR": "logR_array",
    "A_g": "A_g_array",
    "lambda3": "gamma3_array",
    "tech_state": "tech_state_array",
    "damage_state": "damage_state_array",
}


def load_array(paths_folder: Path, prefix: str, seed: int) -> np.ndarray:
    return np.load(paths_folder / f"{prefix}_{seed}.npy", allow_pickle=True)


def state_at(arrays: Dict[str, np.ndarray], path_idx: int, time_idx: int) -> Dict[str, float]:
    return {
        key: float(array[path_idx, time_idx])
        for key, array in arrays.items()
    }


def xi_from_paths_folder(paths_folder: Path) -> float:
    raw = paths_folder.name
    prefix = "paths_ξ_"
    if not raw.startswith(prefix):
        raise ValueError(f"Cannot infer xi from {paths_folder}")
    return float(raw[len(prefix) :])


def collect_jump_states(
    paths_folder: Path,
    comparison: str,
    y0: Optional[float],
    seed_min: Optional[int],
    seed_max: Optional[int],
) -> Tuple[Dict[str, Dict[str, List[Dict[str, float]]]], List[int], List[Dict[str, object]]]:
    collected = {
        event_key: {"left": [], "right": []}
        for event_key in EVENTS
    }
    seeds = discover_seeds(paths_folder, y0=y0, seed_min=seed_min, seed_max=seed_max)
    missing_rows: List[Dict[str, object]] = []

    for seed in seeds:
        try:
            arrays = {
                key: load_array(paths_folder, prefix, seed)
                for key, prefix in STATE_ARRAYS.items()
            }
        except FileNotFoundError as exc:
            missing_rows.append({"seed": seed, "missing": str(exc)})
            continue

        n_paths, n_time = arrays["tech_state"].shape
        for path_idx in range(n_paths):
            for event_key, event_spec in EVENTS.items():
                state_path = arrays[
                    "tech_state" if event_spec["state_array"] == "tech_state_array" else "damage_state"
                ][path_idx]
                transitions = transition_indices(
                    state_path,
                    int(event_spec["from"]),
                    int(event_spec["to"]),
                )
                if transitions.size == 0:
                    continue
                indices = comparison_indices(int(transitions[0]), n_time, comparison)
                if indices is None:
                    continue
                left_idx, right_idx = indices
                collected[event_key]["left"].append(state_at(arrays, path_idx, left_idx))
                collected[event_key]["right"].append(state_at(arrays, path_idx, right_idx))

    return collected, seeds, missing_rows


def evaluate_states(
    evaluator: RegimeModels,
    states: List[Dict[str, float]],
    batch_size: int,
) -> Dict[str, List[float]]:
    output = {name: [] for name, _label in MARGINALS}
    if not states:
        return output

    states_by_stage: Dict[str, List[Dict[str, float]]] = {}
    for state in states:
        stage = evaluator.stage_for_state(int(state["tech_state"]), int(state["damage_state"]))
        states_by_stage.setdefault(stage, []).append(state)

    params = evaluator.params
    lambda1 = float(params["λ1"])
    lambda2 = float(params["λ2"])
    y_upper = float(params["y_upper"])

    for stage, stage_states in states_by_stage.items():
        model = evaluator.models[stage]
        for start in range(0, len(stage_states), batch_size):
            batch = stage_states[start : start + batch_size]
            x_np = np.asarray(
                [evaluator.state_vector(stage, state) for state in batch],
                dtype=np.float32,
            )
            x = tf.convert_to_tensor(x_np)
            with tf.GradientTape() as tape:
                tape.watch(x)
                value = model.v_nn(x, training=False)
            gradient = tape.gradient(
                value,
                x,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            ).numpy()

            logk = np.asarray([state["logK"] for state in batch], dtype=float)
            z = np.asarray([state["Z"] for state in batch], dtype=float)
            y = np.asarray([state["Y"] for state in batch], dtype=float)
            lambda3 = np.asarray([state["lambda3"] for state in batch], dtype=float)
            damage_state = np.asarray([state["damage_state"] for state in batch], dtype=float)
            capital = np.exp(logk)

            dv_dlogk = gradient[:, 0]
            dv_dz = gradient[:, 1]
            dv_dy = gradient[:, 2]

            dvalue_dkd = (dv_dlogk - z * dv_dz) / capital
            dvalue_dkg = (dv_dlogk + (1.0 - z) * dv_dz) / capital
            dlog_damage_dy = (
                lambda1
                + lambda2 * y
                + damage_state * lambda3 * (y - y_upper)
            )
            climate_marginal_cost = dlog_damage_dy - dv_dy

            output["dV_dKd"].extend(dvalue_dkd.astype(float).tolist())
            output["dV_dKg"].extend(dvalue_dkg.astype(float).tolist())
            output["climate_marginal_cost"].extend(climate_marginal_cost.astype(float).tolist())

    return output


def finite(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if np.isfinite(value)]


def plot_paths_folder(args: argparse.Namespace, paths_folder: Path) -> Dict[str, object]:
    export_folder = paths_folder.parents[1]
    xi = xi_from_paths_folder(paths_folder)
    evaluator = RegimeModels(str(export_folder), xi, batch_size=args.network_batch_size)
    collected_states, seeds, missing_rows = collect_jump_states(
        paths_folder,
        comparison=args.comparison,
        y0=args.y0,
        seed_min=args.seed_min,
        seed_max=args.seed_max,
    )

    evaluated: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for event_key in EVENTS:
        evaluated[event_key] = {}
        left_metrics = evaluate_states(
            evaluator,
            collected_states[event_key]["left"],
            batch_size=args.eval_batch_size,
        )
        right_metrics = evaluate_states(
            evaluator,
            collected_states[event_key]["right"],
            batch_size=args.eval_batch_size,
        )
        for metric, _label in MARGINALS:
            evaluated[event_key][metric] = {
                "left": finite(left_metrics[metric]),
                "right": finite(right_metrics[metric]),
            }

    out_dir = paths_folder / "ControlDensities" / f"marginal_{args.comparison}"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = comparison_labels(args.comparison)

    ranges: Dict[str, Optional[Tuple[float, float]]] = {}
    for metric, _label in MARGINALS:
        values: List[float] = []
        for event_key in EVENTS:
            values.extend(evaluated[event_key][metric]["left"])
            values.extend(evaluated[event_key][metric]["right"])
        ranges[metric] = compute_range(values, args.x_pctl_lo, args.x_pctl_hi)

    summary_rows: List[Dict[str, object]] = []
    figure_count = 0
    for event_key, event_spec in EVENTS.items():
        for metric, x_label in MARGINALS:
            left = evaluated[event_key][metric]["left"]
            right = evaluated[event_key][metric]["right"]
            row = {
                "model": export_folder.name,
                "xi": xi,
                "paths_folder": str(paths_folder),
                "comparison": args.comparison,
                "event": event_key,
                "event_title": event_spec["title"],
                "metric": metric,
                "seeds_used": len(seeds),
            }
            row.update(summarize_values(left, "left"))
            row.update(summarize_values(right, "right"))
            summary_rows.append(row)
            if not left and not right:
                continue
            output_path = out_dir / f"{metric}_kde_{event_key}_{args.comparison}.png"
            xi_title = "ξ = ∞" if np.isclose(xi, 148.6) else f"ξ = {xi:g}"
            title = f"{x_label} around {event_spec['title']}\n{xi_title}"
            if plot_density(
                left,
                right,
                x_label=x_label,
                title=title,
                output_path=output_path,
                x_range=ranges[metric],
                n_x=args.n_x,
                labels=labels,
            ):
                figure_count += 1

    write_csv(out_dir / "marginal_density_summary.csv", summary_rows)
    write_csv(out_dir / "missing_files.csv", missing_rows)
    metadata_path = out_dir / "metadata.txt"
    metadata_path.write_text(
        "\n".join(
            [
                f"paths_folder = {paths_folder}",
                f"comparison = {args.comparison}",
                f"Y0_filter = {args.y0}",
                f"seeds_used = {len(seeds)}",
                f"figures_written = {figure_count}",
                "dV_dKd = (V_logK - Z * V_Z) / K",
                "dV_dKg = (V_logK + (1 - Z) * V_Z) / K",
                "climate_marginal_cost = -V_Y = dlogN_dY - v_Y",
                "post_damage_dlogN_dY = lambda1 + lambda2 * Y + lambda3 * (Y - y_upper)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plot_out_dir = (
        export_folder
        / "SimulationResultsPlot"
        / paths_folder.name
        / "ControlDensities"
        / f"marginal_{args.comparison}"
    )
    plot_out_dir.mkdir(parents=True, exist_ok=True)
    for source in out_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, plot_out_dir / source.name)

    del evaluator
    tf.keras.backend.clear_session()
    return {
        "paths_folder": str(paths_folder),
        "seeds_used": len(seeds),
        "figures_written": figure_count,
        "output_dir": str(out_dir),
        "plot_output_dir": str(plot_out_dir),
    }


def default_paths() -> List[Path]:
    paths: List[Path] = []
    for model in DEFAULT_MODELS:
        paths.extend(sorted((ROOT / "output_001" / model / "SimulationResults").glob("paths_*")))
    return [path for path in paths if path.is_dir()]


def expand_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for raw in args.paths_folder or []:
        path = Path(raw).expanduser()
        paths.append(path if path.is_absolute() else ROOT / path)
    for raw in args.export_folder or []:
        export = Path(raw).expanduser()
        export = export if export.is_absolute() else ROOT / export
        paths.extend(sorted((export / "SimulationResults").glob("paths_*")))
    if args.all_default_models or not paths:
        paths.extend(default_paths())

    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-folder", action="append")
    parser.add_argument("--export-folder", action="append")
    parser.add_argument("--all-default-models", action="store_true")
    parser.add_argument("--comparison", choices=["pre-post", "post-next"], default="pre-post")
    parser.add_argument("--y0", type=float, default=1.2)
    parser.add_argument("--seed-min", type=int, default=None)
    parser.add_argument("--seed-max", type=int, default=None)
    parser.add_argument("--network-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--x-pctl-lo", type=float, default=1.0)
    parser.add_argument("--x-pctl-hi", type=float, default=99.0)
    parser.add_argument("--n-x", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.y0 is not None and args.y0 < 0:
        args.y0 = None
    paths_folders = expand_paths(args)
    if not paths_folders:
        raise SystemExit("No stochastic paths folders found.")

    rows = []
    for paths_folder in paths_folders:
        result = plot_paths_folder(args, paths_folder)
        rows.append(result)
        print(
            f"{result['figures_written']:2d} figures | {result['seeds_used']:3d} seeds | "
            f"{paths_folder}"
        )

    with (ROOT / "marginal_density_plot_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
