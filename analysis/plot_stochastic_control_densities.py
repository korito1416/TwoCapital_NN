#!/usr/bin/env python3
"""Plot investment-control densities around simulated jump events.

The stochastic simulator writes one folder per xi:

    <model>/SimulationResults/paths_ξ_<xi>/

This script reads those folders, collects controls immediately before/after
technology and damage jumps, and saves KDE figures plus a summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent

DEFAULT_MODELS = [
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000",
    "TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000",
    "TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000",
]

EVENTS = {
    "tech_0_to_1": {
        "group": "technology",
        "state_array": "tech_state_array",
        "from": 0,
        "to": 1,
        "title": "Pre-Tech to Interm-Tech",
    },
    "tech_1_to_2": {
        "group": "technology",
        "state_array": "tech_state_array",
        "from": 1,
        "to": 2,
        "title": "Interm-Tech to Post-Tech",
    },
    "tech_0_to_2": {
        "group": "technology",
        "state_array": "tech_state_array",
        "from": 0,
        "to": 2,
        "title": "Pre-Tech to Post-Tech",
    },
    "damage_jump": {
        "group": "damage",
        "state_array": "damage_state_array",
        "from": 0,
        "to": 1,
        "title": "Damage Jump",
    },
}

CONTROL_GROUPS = {
    "level": [
        ("I_r", "I_r_array", "R&D investment $I_r$"),
        ("I_d", "I_d_array", "Dirty investment $I_d$"),
        ("I_g", "I_g_array", "Green investment $I_g$"),
    ],
    "rate": [
        ("i_r", "i_r_array", "R&D investment rate $i_r$"),
        ("i_d", "i_d_array", "Dirty investment rate $i_d$"),
        ("i_g", "i_g_array", "Green investment rate $i_g$"),
    ],
    "value": [
        ("V", "V_array", "Value function $V$"),
    ],
}


def parse_metadata(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def metadata_y0_matches(path: Path, y0: Optional[float]) -> bool:
    if y0 is None:
        return True
    meta = parse_metadata(path)
    if "Y0" not in meta:
        return False
    try:
        return abs(float(meta["Y0"]) - float(y0)) <= 1e-10
    except ValueError:
        return False


def seed_from_metadata(path: Path) -> Optional[int]:
    stem = path.stem
    if not stem.startswith("metadata_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return None


def discover_seeds(paths_folder: Path, y0: Optional[float], seed_min: Optional[int], seed_max: Optional[int]) -> List[int]:
    seeds: List[int] = []
    for metadata_path in sorted(paths_folder.glob("metadata_*.txt")):
        seed = seed_from_metadata(metadata_path)
        if seed is None:
            continue
        if seed_min is not None and seed < seed_min:
            continue
        if seed_max is not None and seed > seed_max:
            continue
        if not metadata_y0_matches(metadata_path, y0):
            continue
        seeds.append(seed)
    return sorted(set(seeds))


_NPZ_BUNDLE_CACHE: dict = {}


def _npz_key_from_prefix(array_prefix: str) -> str:
    """Map a legacy per-array .npy prefix to its key in the .npz bundle."""
    if array_prefix == "logK_sim":
        return "logK"
    base = array_prefix[:-6] if array_prefix.endswith("_array") else array_prefix
    if base == "gamma3":
        return "lambda3"
    return base


def load_array(paths_folder: Path, array_prefix: str, seed: int) -> np.ndarray:
    # Prefer the compact per-seed .npz bundle (the loose *_array_*.npy files were
    # retired to save inodes); fall back to legacy .npy when no bundle exists.
    npz_path = paths_folder / f"paths_controls_values_{seed}.npz"
    if npz_path.exists():
        cache_key = str(npz_path)
        bundle = _NPZ_BUNDLE_CACHE.get(cache_key)
        if bundle is None:
            with np.load(npz_path, allow_pickle=True) as data:
                bundle = {name: data[name] for name in data.files}
            _NPZ_BUNDLE_CACHE[cache_key] = bundle
        key = _npz_key_from_prefix(array_prefix)
        if key in bundle:
            return bundle[key]
    return np.load(paths_folder / f"{array_prefix}_{seed}.npy", allow_pickle=True)


def clean_value(value: float, control_name: str, inactive_rd_zero: bool) -> float:
    value = float(value)
    if inactive_rd_zero and control_name in {"I_r", "i_r"} and not np.isfinite(value):
        return 0.0
    return value


def append_if_finite(bucket: List[float], value: float) -> None:
    if np.isfinite(value):
        bucket.append(float(value))


def transition_indices(state_path: np.ndarray, from_state: int, to_state: int) -> np.ndarray:
    state_path = state_path.astype(int)
    return np.where((state_path[:-1] == from_state) & (state_path[1:] == to_state))[0]


def comparison_indices(j_before: int, n_time: int, comparison: str) -> Optional[Tuple[int, int]]:
    if comparison == "pre-post":
        left, right = j_before, j_before + 1
    elif comparison == "post-next":
        left, right = j_before + 1, j_before + 2
    else:
        raise ValueError(f"Unknown comparison: {comparison}")
    if left < 0 or right >= n_time:
        return None
    return left, right


def comparison_labels(comparison: str) -> Tuple[str, str]:
    if comparison == "pre-post":
        return "right before jump", "right after jump"
    if comparison == "post-next":
        return "right after jump", "one step after jump"
    raise ValueError(comparison)


def compute_range(values: Iterable[float], lo_p: float, hi_p: float) -> Optional[Tuple[float, float]]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    lo, hi = np.percentile(arr, [lo_p, hi_p])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if np.isclose(lo, hi):
        pad = max(1e-8, abs(float(lo)) * 0.05 + 1e-8)
        return float(lo - pad), float(hi + pad)
    pad = 0.03 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def plot_one_series(
    ax: plt.Axes,
    data: Iterable[float],
    label: str,
    color: str,
    linestyle: str,
    x_range: Optional[Tuple[float, float]],
    n_x: int,
) -> bool:
    arr = np.asarray(list(data), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    if arr.size == 1 or np.isclose(np.nanstd(arr), 0.0):
        x = float(arr[0])
        ax.axvline(x, color=color, linestyle=linestyle, linewidth=2.2, label=f"{label} (mass at {x:.4g})")
        return True
    if x_range is None:
        x_range = compute_range(arr, 1.0, 99.0)
    if x_range is None:
        return False
    xs = np.linspace(x_range[0], x_range[1], n_x)
    kde = gaussian_kde(arr)
    ax.plot(xs, kde(xs), color=color, linestyle=linestyle, linewidth=2.2, label=label)
    return True


def plot_density(
    before: List[float],
    after: List[float],
    x_label: str,
    title: str,
    output_path: Path,
    x_range: Optional[Tuple[float, float]],
    n_x: int,
    labels: Tuple[str, str],
) -> bool:
    if len(before) == 0 and len(after) == 0:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ok_before = plot_one_series(ax, before, labels[0], "tab:blue", "-", x_range, n_x)
    ok_after = plot_one_series(ax, after, labels[1], "tab:red", "--", x_range, n_x)
    if not (ok_before or ok_after):
        plt.close(fig)
        return False
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_values(values: List[float], prefix: str) -> Dict[str, object]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p95": np.nan,
        }
    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p05": float(np.percentile(arr, 5)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
    }


def collect_folder(
    paths_folder: Path,
    control_group: str,
    comparison: str,
    y0: Optional[float],
    seed_min: Optional[int],
    seed_max: Optional[int],
    inactive_rd_zero: bool,
) -> Tuple[Dict[str, Dict[str, Dict[str, List[float]]]], List[int], List[Dict[str, object]]]:
    controls = CONTROL_GROUPS[control_group]
    collected: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        event_key: {control_name: {"left": [], "right": []} for control_name, _, _ in controls}
        for event_key in EVENTS
    }
    seeds = discover_seeds(paths_folder, y0=y0, seed_min=seed_min, seed_max=seed_max)
    missing_rows: List[Dict[str, object]] = []

    event_arrays: Dict[str, Dict[int, np.ndarray]] = {event_key: {} for event_key in EVENTS}
    control_arrays: Dict[str, Dict[int, np.ndarray]] = {control_name: {} for control_name, _, _ in controls}

    for seed in seeds:
        try:
            for event_key, spec in EVENTS.items():
                event_arrays[event_key][seed] = load_array(paths_folder, spec["state_array"], seed)
            for control_name, array_prefix, _ in controls:
                control_arrays[control_name][seed] = load_array(paths_folder, array_prefix, seed)
        except FileNotFoundError as exc:
            missing_rows.append({"seed": seed, "missing": str(exc)})
            continue

        sample_event = next(iter(event_arrays.values()))[seed]
        n_paths, n_time = sample_event.shape
        for path_idx in range(n_paths):
            for event_key, spec in EVENTS.items():
                state = event_arrays[event_key][seed][path_idx]
                transitions = transition_indices(state, int(spec["from"]), int(spec["to"]))
                if transitions.size == 0:
                    continue
                indices = comparison_indices(int(transitions[0]), n_time, comparison)
                if indices is None:
                    continue
                left_idx, right_idx = indices
                for control_name, _, _ in controls:
                    arr = control_arrays[control_name][seed][path_idx]
                    left_value = clean_value(arr[left_idx], control_name, inactive_rd_zero)
                    right_value = clean_value(arr[right_idx], control_name, inactive_rd_zero)
                    append_if_finite(collected[event_key][control_name]["left"], left_value)
                    append_if_finite(collected[event_key][control_name]["right"], right_value)

    return collected, seeds, missing_rows


def model_label_from_paths(paths_folder: Path) -> str:
    try:
        return paths_folder.parents[1].name
    except IndexError:
        return paths_folder.name


def xi_label_from_paths(paths_folder: Path) -> str:
    return paths_folder.name.replace("paths_", "")


def plot_folder(args: argparse.Namespace, paths_folder: Path, control_group: str) -> Dict[str, object]:
    out_dir = paths_folder / "ControlDensities" / f"{control_group}_{args.comparison}"
    out_dir.mkdir(parents=True, exist_ok=True)
    collected, seeds, missing_rows = collect_folder(
        paths_folder,
        control_group=control_group,
        comparison=args.comparison,
        y0=args.y0,
        seed_min=args.seed_min,
        seed_max=args.seed_max,
        inactive_rd_zero=not args.keep_inactive_rd_nan,
    )
    labels = comparison_labels(args.comparison)
    summary_rows: List[Dict[str, object]] = []
    range_by_control: Dict[str, Optional[Tuple[float, float]]] = {}
    for control_name, _, _ in CONTROL_GROUPS[control_group]:
        all_values: List[float] = []
        for event_key in EVENTS:
            all_values.extend(collected[event_key][control_name]["left"])
            all_values.extend(collected[event_key][control_name]["right"])
        range_by_control[control_name] = compute_range(all_values, args.x_pctl_lo, args.x_pctl_hi)

    figure_count = 0
    for event_key, event_spec in EVENTS.items():
        for control_name, _, x_label in CONTROL_GROUPS[control_group]:
            left = collected[event_key][control_name]["left"]
            right = collected[event_key][control_name]["right"]
            row = {
                "model": model_label_from_paths(paths_folder),
                "xi": xi_label_from_paths(paths_folder),
                "paths_folder": str(paths_folder),
                "control_group": control_group,
                "comparison": args.comparison,
                "event": event_key,
                "event_title": event_spec["title"],
                "control": control_name,
                "seeds_used": len(seeds),
            }
            row.update(summarize_values(left, "left"))
            row.update(summarize_values(right, "right"))
            summary_rows.append(row)
            if len(left) == 0 and len(right) == 0:
                continue
            filename = f"{control_name}_kde_{event_key}_{args.comparison}.png"
            title = f"{x_label} around {event_spec['title']}\n{xi_label_from_paths(paths_folder)}"
            ok = plot_density(
                left,
                right,
                x_label=x_label,
                title=title,
                output_path=out_dir / filename,
                x_range=range_by_control[control_name],
                n_x=args.n_x,
                labels=labels,
            )
            if ok:
                figure_count += 1

    write_csv(out_dir / "control_density_summary.csv", summary_rows)
    write_csv(out_dir / "missing_files.csv", missing_rows)
    with (out_dir / "metadata.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"paths_folder = {paths_folder}\n")
        handle.write(f"control_group = {control_group}\n")
        handle.write(f"comparison = {args.comparison}\n")
        handle.write(f"Y0_filter = {args.y0}\n")
        handle.write(f"seeds_used = {len(seeds)}\n")
        handle.write(f"seed_min = {min(seeds) if seeds else ''}\n")
        handle.write(f"seed_max = {max(seeds) if seeds else ''}\n")
        handle.write(f"inactive_rd_zero = {not args.keep_inactive_rd_nan}\n")
        handle.write(f"figures_written = {figure_count}\n")

    return {
        "paths_folder": str(paths_folder),
        "control_group": control_group,
        "seeds_used": len(seeds),
        "figures_written": figure_count,
        "output_dir": str(out_dir),
    }


def default_paths(root: Path) -> List[Path]:
    out: List[Path] = []
    for model in DEFAULT_MODELS:
        sim_root = root / model / "SimulationResults"
        out.extend(sorted(sim_root.glob("paths_*")))
    return [path for path in out if path.is_dir()]


def expand_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for raw in args.paths_folder or []:
        path = Path(raw).expanduser()
        paths.append(path if path.is_absolute() else (ROOT / path))
    for raw in args.export_folder or []:
        export = Path(raw).expanduser()
        export = export if export.is_absolute() else (ROOT / export)
        paths.extend(sorted((export / "SimulationResults").glob("paths_*")))
    if args.all_default_models or not paths:
        paths.extend(default_paths(ROOT / "output_001"))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stochastic control densities around jump events.")
    parser.add_argument("--paths-folder", action="append", help="Specific SimulationResults/paths_* folder. Can repeat.")
    parser.add_argument("--export-folder", action="append", help="Trained model folder; all paths_* folders are used. Can repeat.")
    parser.add_argument("--all-default-models", action="store_true", help="Use the three stochastic model folders from the recent simulations.")
    parser.add_argument("--control-group", choices=["level", "rate", "value", "both", "all"], default="level")
    parser.add_argument("--comparison", choices=["pre-post", "post-next"], default="pre-post")
    parser.add_argument("--y0", type=float, default=1.2, help="Only use seeds whose metadata has this Y0. Set negative to disable.")
    parser.add_argument("--seed-min", type=int, default=None)
    parser.add_argument("--seed-max", type=int, default=None)
    parser.add_argument("--x-pctl-lo", type=float, default=1.0)
    parser.add_argument("--x-pctl-hi", type=float, default=99.0)
    parser.add_argument("--n-x", type=int, default=400)
    parser.add_argument("--keep-inactive-rd-nan", action="store_true", help="Do not convert inactive post-tech R&D controls from NaN to zero.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.y0 is not None and args.y0 < 0:
        args.y0 = None
    if args.control_group == "both":
        control_groups = ["level", "rate"]
    elif args.control_group == "all":
        control_groups = ["level", "rate", "value"]
    else:
        control_groups = [args.control_group]
    paths_folders = expand_paths(args)
    if not paths_folders:
        raise SystemExit("No paths folders found.")

    rows: List[Dict[str, object]] = []
    for paths_folder in paths_folders:
        for control_group in control_groups:
            result = plot_folder(args, paths_folder, control_group)
            rows.append(result)
            print(
                f"{result['figures_written']:3d} figures | {result['seeds_used']:3d} seeds | "
                f"{control_group:5s} | {paths_folder}"
            )

    summary_path = ROOT / "control_density_plot_runs.csv"
    write_csv(summary_path, rows)
    print(f"Wrote run summary: {summary_path}")


if __name__ == "__main__":
    main()
