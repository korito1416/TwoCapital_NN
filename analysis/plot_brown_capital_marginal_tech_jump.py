#!/usr/bin/env python3
"""Plot the brown-capital FOC objects with respect to Kd and log Kd.

The trained HJB networks use transformed states

    logK = log(Kd + Kg),   Z = Kg / (Kd + Kg).

Holding green capital fixed,

    dV/dKd = (V_logK - Z * V_Z) / K,
    dV/dlogKd = Kd * dV/dKd = (1 - Z) * (V_logK - Z * V_Z).

The dirty-investment FOC implies the same log-brown-capital shadow value:

    dV/dlogKd = (1 - Z) * [delta / (C/K)] / phi_d_prime(i_d),
    phi_d_prime(i_d) = Gamma_d * theta_d / (1 + theta_d * i_d).

This script loads all six trained regime networks, varies temperature Y, fixes
the other states at baseline values, and saves figures plus the underlying CSV.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from feedforward_subnet import FeedForwardSubNet  # noqa: E402
from params import PARAMS  # noqa: E402


@dataclass(frozen=True)
class RegimeSpec:
    folder: str
    checkpoint_suffix: str
    label: str
    n_inputs: int
    has_rd: bool
    ag_key: str
    post_damage: bool
    linestyle: str = "-"


REGIMES: List[RegimeSpec] = [
    RegimeSpec(
        folder="PreDamagePreTech",
        checkpoint_suffix="PreDamagePreTech",
        label="Pre-damage, pre-tech",
        n_inputs=7,
        has_rd=True,
        ag_key="A_g",
        post_damage=False,
    ),
    RegimeSpec(
        folder="PreDamageIntermTech",
        checkpoint_suffix="PreDamageIntermTech",
        label="Pre-damage, intermediate tech",
        n_inputs=7,
        has_rd=True,
        ag_key="A_g_prime",
        post_damage=False,
    ),
    RegimeSpec(
        folder="PreDamagePostTech",
        checkpoint_suffix="PreDamagePostTech",
        label="Pre-damage, final tech",
        n_inputs=6,
        has_rd=False,
        ag_key="A_g_prime_prime",
        post_damage=False,
    ),
    RegimeSpec(
        folder="PostDamagePreTech",
        checkpoint_suffix="PostDamagePreTech",
        label="Post-damage, pre-tech",
        n_inputs=8,
        has_rd=True,
        ag_key="A_g",
        post_damage=True,
        linestyle="--",
    ),
    RegimeSpec(
        folder="PostDamageIntermTech",
        checkpoint_suffix="PostDamageIntermTech",
        label="Post-damage, intermediate tech",
        n_inputs=8,
        has_rd=True,
        ag_key="A_g_prime",
        post_damage=True,
        linestyle="--",
    ),
    RegimeSpec(
        folder="PostDamagePostTech",
        checkpoint_suffix="PostDamagePostTech",
        label="Post-damage, final tech",
        n_inputs=7,
        has_rd=False,
        ag_key="A_g_prime_prime",
        post_damage=True,
        linestyle="--",
    ),
]


def selected_regimes(regime_set: str) -> List[RegimeSpec]:
    if regime_set == "all":
        return REGIMES
    if regime_set == "pre-damage":
        return [spec for spec in REGIMES if not spec.post_damage]
    if regime_set == "post-damage":
        return [spec for spec in REGIMES if spec.post_damage]
    raise ValueError(f"Unknown regime set: {regime_set}")


def custom_investment_activation(theta: float) -> Callable[[tf.Tensor], tf.Tensor]:
    def activation(x: tf.Tensor) -> tf.Tensor:
        return 1.0 - (1.0 + 1.0 / theta) / (tf.exp(2.0 * x) + 1.0)

    return activation


def parse_scalar(value: str):
    value = value.strip()
    if value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    if value.startswith("<function"):
        return "custom"
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def read_config(path: Path, theta: Optional[float] = None) -> Dict:
    config = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = parse_scalar(value)

    if config.get("final_activation") == "custom":
        if theta is None:
            raise ValueError(f"Custom activation in {path} needs theta.")
        config["final_activation"] = custom_investment_activation(theta)

    return config


def load_network(regime_dir: Path, spec: RegimeSpec, nn_name: str, theta: Optional[float] = None):
    config = read_config(regime_dir / f"params_{nn_name}_config.txt", theta=theta)
    net = FeedForwardSubNet(config)
    dummy = tf.zeros((1, spec.n_inputs), dtype=tf.float32)
    net(dummy, training=False)
    checkpoint = regime_dir / f"{nn_name}_checkpoint_{spec.checkpoint_suffix}"
    net.load_weights(str(checkpoint))
    return net


def build_input(
    spec: RegimeSpec,
    logk: tf.Tensor,
    z: tf.Tensor,
    y: tf.Tensor,
    logr: tf.Tensor,
    lambda3: tf.Tensor,
    logxi: tf.Tensor,
    params: Dict,
) -> tf.Tensor:
    if spec.folder in ("PreDamagePreTech", "PreDamageIntermTech"):
        return tf.concat([logk, z, y, logr, logxi, logxi, logxi], axis=1)
    if spec.folder == "PreDamagePostTech":
        ag = tf.ones_like(y) * params["A_g_prime_prime"]
        return tf.concat([logk, z, y, ag, logxi, logxi], axis=1)
    if spec.folder in ("PostDamagePreTech", "PostDamageIntermTech"):
        return tf.concat([logk, z, y, logr, lambda3, logxi, logxi, logxi], axis=1)
    if spec.folder == "PostDamagePostTech":
        ag = tf.ones_like(y) * params["A_g_prime_prime"]
        return tf.concat([logk, z, y, lambda3, ag, logxi, logxi], axis=1)
    raise ValueError(f"Unsupported regime folder: {spec.folder}")


def evaluate_regime(
    export_root: Path,
    spec: RegimeSpec,
    y_grid: np.ndarray,
    logk_value: float,
    z_value: float,
    logr_value: float,
    xi_value: float,
    lambda3_value: float,
    params: Dict,
) -> Dict[str, np.ndarray]:
    regime_dir = export_root / spec.folder
    v_nn = load_network(regime_dir, spec, "v_nn")
    i_d_nn = load_network(regime_dir, spec, "i_d_nn", theta=params["theta_d"])
    i_g_nn = load_network(regime_dir, spec, "i_g_nn", theta=params["theta_g"])
    i_r_nn = None
    if spec.has_rd:
        i_r_nn = load_network(regime_dir, spec, "i_r_nn")

    n = y_grid.size
    logk = tf.ones((n, 1), dtype=tf.float32) * np.float32(logk_value)
    z = tf.ones((n, 1), dtype=tf.float32) * np.float32(z_value)
    y = tf.reshape(tf.convert_to_tensor(y_grid, dtype=tf.float32), (n, 1))
    logr = tf.ones((n, 1), dtype=tf.float32) * np.float32(logr_value)
    lambda3 = tf.ones((n, 1), dtype=tf.float32) * np.float32(lambda3_value)
    logxi = tf.ones((n, 1), dtype=tf.float32) * np.float32(np.log(xi_value))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([logk, z])
        x_value = build_input(spec, logk, z, y, logr, lambda3, logxi, params)
        value = v_nn(x_value, training=False)

    dv_dlogk = tape.gradient(value, logk)
    dv_dz = tape.gradient(value, z)
    del tape

    dvalue_dkd_times_k = dv_dlogk - z * dv_dz
    dvalue_dlogkd = (1.0 - z) * dvalue_dkd_times_k
    dvalue_dkd = dvalue_dkd_times_k / tf.exp(logk)

    x_controls = build_input(spec, logk, z, y, logr, lambda3, logxi, params)
    i_d = i_d_nn(x_controls, training=False)
    i_g = i_g_nn(x_controls, training=False)
    i_r = tf.zeros_like(i_d)
    if i_r_nn is not None:
        i_r = tf.exp(-i_r_nn(x_controls, training=False))

    ag = params[spec.ag_key]
    c_over_k = (params["A_d"] - i_d) * (1.0 - z) + (ag - i_g) * z - i_r
    marginal_utility_c = params["delta"] / c_over_k
    phi_d_prime = params["Gamma_d"] * params["theta_d"] / (1.0 + params["theta_d"] * i_d)
    dvalue_dkd_foc = marginal_utility_c / phi_d_prime / tf.exp(logk)
    dvalue_dlogkd_foc = (1.0 - z) * marginal_utility_c / phi_d_prime
    foc_kd_gap = dvalue_dkd - dvalue_dkd_foc
    foc_logkd_gap = dvalue_dlogkd - dvalue_dlogkd_foc

    return {
        "Y": y_grid,
        "value": value.numpy().reshape(-1),
        "dv_dlogK": dv_dlogk.numpy().reshape(-1),
        "dv_dZ": dv_dz.numpy().reshape(-1),
        "dV_dKd_times_K": dvalue_dkd_times_k.numpy().reshape(-1),
        "dV_dKd": dvalue_dkd.numpy().reshape(-1),
        "dV_dlogKd": dvalue_dlogkd.numpy().reshape(-1),
        "dV_dKd_foc_implied": dvalue_dkd_foc.numpy().reshape(-1),
        "dV_dlogKd_foc_implied": dvalue_dlogkd_foc.numpy().reshape(-1),
        "foc_Kd_gap": foc_kd_gap.numpy().reshape(-1),
        "foc_logKd_gap": foc_logkd_gap.numpy().reshape(-1),
        "i_d": i_d.numpy().reshape(-1),
        "i_g": i_g.numpy().reshape(-1),
        "i_r": i_r.numpy().reshape(-1),
        "c_over_k": c_over_k.numpy().reshape(-1),
        "marginal_utility_c": marginal_utility_c.numpy().reshape(-1),
        "phi_d_prime": phi_d_prime.numpy().reshape(-1),
    }


def write_csv(path: Path, results: Dict[str, Dict[str, np.ndarray]]) -> None:
    fieldnames = [
        "regime",
        "Y",
        "value",
        "dv_dlogK",
        "dv_dZ",
        "dV_dKd_times_K",
        "dV_dKd",
        "dV_dlogKd",
        "dV_dKd_foc_implied",
        "dV_dlogKd_foc_implied",
        "foc_Kd_gap",
        "foc_logKd_gap",
        "i_d",
        "i_g",
        "i_r",
        "c_over_k",
        "marginal_utility_c",
        "phi_d_prime",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for regime, data in results.items():
            for idx in range(data["Y"].size):
                row = {"regime": regime}
                for name in fieldnames[1:]:
                    row[name] = float(data[name][idx])
                writer.writerow(row)


def color_cycle() -> Iterable[str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    while True:
        for color in colors:
            yield color


def plot_shadow_value(
    path: Path,
    results: Dict[str, Dict[str, np.ndarray]],
    specs: List[RegimeSpec],
    subtitle: str,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    colors = color_cycle()
    for spec in specs:
        data = results[spec.folder]
        ax.plot(
            data["Y"],
            data[metric],
            linewidth=2.2,
            linestyle=spec.linestyle,
            color=next(colors),
            label=spec.label,
        )

    ax.set_title(title)
    ax.set_xlabel("Temperature anomaly Y")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle(subtitle, y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_foc_gap(
    path: Path,
    results: Dict[str, Dict[str, np.ndarray]],
    specs: List[RegimeSpec],
    subtitle: str,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    colors = color_cycle()
    for spec in specs:
        data = results[spec.folder]
        ax.plot(
            data["Y"],
            data[metric],
            linewidth=1.8,
            linestyle=spec.linestyle,
            color=next(colors),
            label=spec.label,
        )

    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Temperature anomaly Y")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle(subtitle, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_params() -> Dict:
    params = PARAMS.copy()
    params["delta"] = params["δ"]
    params["Gamma_d"] = params["Γ_d"]
    params["theta_d"] = params["θ_d"]
    params["Gamma_g"] = params["Γ_g"]
    params["theta_g"] = params["θ_g"]
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "export_root",
        type=Path,
        help="Model output root containing the six regime checkpoint folders.",
    )
    parser.add_argument(
        "--regime-set",
        choices=("all", "pre-damage", "post-damage"),
        default="all",
        help="Which regimes to plot. Default: all six.",
    )
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--K", type=float, default=float(PARAMS["K0"]))
    parser.add_argument("--Z", type=float, default=float(PARAMS["Z0"]))
    parser.add_argument("--R", type=float, default=float(PARAMS["R0"]))
    parser.add_argument("--lambda3", type=float, default=float(PARAMS["λ3_values"][2]))
    parser.add_argument("--Y-min", type=float, default=float(PARAMS["Y_min"]))
    parser.add_argument("--Y-max", type=float, default=float(PARAMS["Y_max"]))
    parser.add_argument("--num-points", type=int, default=201)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    params = make_params()
    export_root = args.export_root.resolve()
    specs = selected_regimes(args.regime_set)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = export_root / f"BrownCapitalFOC_Kd_logKd_{args.regime_set.replace('-', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    y_grid = np.linspace(args.Y_min, args.Y_max, args.num_points)
    logk_value = float(np.log(args.K))
    logr_value = float(np.log(args.R))

    results = {}
    for spec in specs:
        results[spec.folder] = evaluate_regime(
            export_root=export_root,
            spec=spec,
            y_grid=y_grid,
            logk_value=logk_value,
            z_value=args.Z,
            logr_value=logr_value,
            xi_value=args.xi,
            lambda3_value=args.lambda3,
            params=params,
        )

    csv_path = output_dir / "brown_capital_foc_Kd_logKd.csv"
    kd_png_path = output_dir / "brown_capital_foc_Kd.png"
    logkd_png_path = output_dir / "brown_capital_foc_logKd.png"
    kd_gap_path = output_dir / "brown_capital_foc_Kd_gap.png"
    logkd_gap_path = output_dir / "brown_capital_foc_logKd_gap.png"
    write_csv(csv_path, results)

    kd = (1.0 - args.Z) * args.K
    subtitle = (
        f"{args.regime_set}, K={args.K:g}, Kd={kd:g}, Z={args.Z:g}, "
        f"R={args.R:g}, xi={args.xi:g}, lambda3={args.lambda3:g}"
    )
    plot_shadow_value(
        kd_png_path,
        results,
        specs,
        subtitle,
        metric="dV_dKd",
        ylabel="dV/dKd",
        title="FOC shadow value with respect to brown capital",
    )
    plot_shadow_value(
        logkd_png_path,
        results,
        specs,
        subtitle,
        metric="dV_dlogKd",
        ylabel="dV/dlog Kd",
        title="FOC shadow value with respect to log brown capital",
    )
    plot_foc_gap(
        kd_gap_path,
        results,
        specs,
        subtitle,
        metric="foc_Kd_gap",
        ylabel="gradient - FOC-implied dV/dKd",
        title="Gradient minus FOC-implied dV/dKd",
    )
    plot_foc_gap(
        logkd_gap_path,
        results,
        specs,
        subtitle,
        metric="foc_logKd_gap",
        ylabel="gradient - FOC-implied dV/dlog Kd",
        title="Gradient minus FOC-implied dV/dlog Kd",
    )

    print(f"Wrote {kd_png_path}")
    print(f"Wrote {logkd_png_path}")
    print(f"Wrote {kd_gap_path}")
    print(f"Wrote {logkd_gap_path}")
    print(f"Wrote {csv_path}")
    for spec in specs:
        data = results[spec.folder]
        max_abs_gap_kd = float(np.max(np.abs(data["foc_Kd_gap"])))
        max_abs_gap_logkd = float(np.max(np.abs(data["foc_logKd_gap"])))
        kd_y0 = float(data["dV_dKd"][0])
        kd_yend = float(data["dV_dKd"][-1])
        logkd_y0 = float(data["dV_dlogKd"][0])
        logkd_yend = float(data["dV_dlogKd"][-1])
        print(
            f"{spec.label}: "
            f"dV/dKd Y_min={kd_y0:.6g}, Y_max={kd_yend:.6g}, max |Kd gap|={max_abs_gap_kd:.4e}; "
            f"dV/dlogKd Y_min={logkd_y0:.6g}, Y_max={logkd_yend:.6g}, "
            f"max |logKd gap|={max_abs_gap_logkd:.4e}"
        )


if __name__ == "__main__":
    main()
