#!/usr/bin/env python3
"""Monte Carlo SVRD marginal-value decomposition before the first jump.

This implements the decomposition described in the marginal-valuation notes:

    V_x(X0) Lambda0 = E~ int Dis_t (Lambda_t . Scf_t) dt

for the social value of R&D, using Lambda0 equal to a unit perturbation in
initial logR.  The expectation is under no-jump diffusion dynamics with the
minimizing drift distortion.  The possible first jumps enter through the
robust jump discount and through flow ii/iii cash-flow terms.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
# This script was moved into analysis/; the models package lives at repo-root/models.
MODELS_DIR = (ROOT / "models") if (ROOT / "models").exists() else (ROOT.parent / "models")
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from params import PARAMS  # noqa: E402
from SimulationStochasticJumps import RegimeModels, xi_label  # noqa: E402


TF_FLOAT = tf.float32


def pvalue(name: str) -> float:
    return float(PARAMS[name])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=1) / np.sqrt(values.size))


def split_x(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]


def make_constant_like(reference: tf.Tensor, value: float) -> tf.Tensor:
    return tf.ones_like(reference, dtype=TF_FLOAT) * np.float32(value)


class SVRDDecomposer:
    def __init__(self, export_folder: str, xi: float, batch_size: int) -> None:
        self.export_folder = os.path.abspath(export_folder)
        self.xi = float(xi)
        self.log_xi = float(np.log(self.xi))
        self.batch_size = int(batch_size)
        self.bundle = RegimeModels(self.export_folder, self.xi, batch_size=self.batch_size)
        self.params = self.bundle.params
        self.pi = float(self.bundle.pi)
        self.tech_scale = float(self.bundle.tech_jump_intensity_scale)
        self.one_tech_jump_mode = bool(self.bundle.one_tech_jump_mode)

        if "PreDamagePreTech" not in self.bundle.models:
            raise FileNotFoundError("PreDamagePreTech model is required for SVRD decomposition.")
        if "PostDamagePreTech" not in self.bundle.models:
            raise FileNotFoundError("PostDamagePreTech model is required for damage-jump continuation values.")
        if "PreDamagePostTech" not in self.bundle.models:
            raise FileNotFoundError("PreDamagePostTech model is required for final-technology continuation values.")
        if (not self.one_tech_jump_mode) and self.pi < 1.0 - 1e-12 and "PreDamageIntermTech" not in self.bundle.models:
            raise FileNotFoundError("PreDamageIntermTech model is required when the intermediary tech jump has positive intensity.")

    def stage_input(
        self,
        stage: str,
        x: tf.Tensor,
        lambda3_value: float = 0.0,
        y_override: float | None = None,
    ) -> tf.Tensor:
        logk, z, y, logr = split_x(x)
        lx = make_constant_like(logk, self.log_xi)
        if y_override is not None:
            y = make_constant_like(y, y_override)
        if stage == "PreDamagePreTech":
            return tf.concat([logk, z, y, logr, lx, lx, lx], axis=1)
        if stage == "PreDamageIntermTech":
            return tf.concat([logk, z, y, logr, lx, lx, lx], axis=1)
        if stage == "PreDamagePostTech":
            ag = make_constant_like(y, pvalue("A_g_prime_prime"))
            return tf.concat([logk, z, y, ag, lx, lx], axis=1)
        if stage == "PostDamagePreTech":
            lam = make_constant_like(y, lambda3_value)
            return tf.concat([logk, z, y, logr, lam, lx, lx, lx], axis=1)
        raise ValueError(f"Unsupported continuation stage for SVRD: {stage}")

    def value_and_gradient(
        self,
        stage: str,
        x: tf.Tensor,
        lambda3_value: float = 0.0,
        y_override: float | None = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        model = self.bundle.models[stage]
        with tf.GradientTape() as tape:
            tape.watch(x)
            inputs = self.stage_input(stage, x, lambda3_value=lambda3_value, y_override=y_override)
            value = model.v_nn(inputs, training=False)
        grad = tape.gradient(value, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return value, grad

    def current_objects(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        model = self.bundle.models["PreDamagePreTech"]
        p = self.params
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            logk, z, y, logr = split_x(x)
            inputs = self.stage_input("PreDamagePreTech", x)
            value = model.v_nn(inputs, training=False)
            i_d = model.i_d_nn(inputs, training=False)
            i_g = model.i_g_nn(inputs, training=False)
            i_r = tf.exp(-model.i_r_nn(inputs, training=False))
            c_over_k = (pvalue("A_d") - i_d) * (1.0 - z) + (pvalue("A_g") - i_g) * z - i_r
            safe_c = tf.maximum(c_over_k, 1e-10)
            log_damage = pvalue("λ1") * y + 0.5 * pvalue("λ2") * y * y
            utility = pvalue("δ") * (tf.math.log(safe_c) + logk - log_damage)
        value_grad = tape.gradient(value, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        utility_grad = tape.gradient(utility, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        del tape
        return {
            "value": value,
            "value_grad": value_grad,
            "utility_grad": utility_grad,
            "i_d": i_d,
            "i_g": i_g,
            "i_r": i_r,
            "c_over_k": c_over_k,
        }

    def damage_intensity(self, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        y_lower = pvalue("y_lower")
        r1 = pvalue("r1")
        r2 = pvalue("r2")
        dy = y - y_lower
        expo = tf.exp(0.5 * r2 * dy * dy)
        raw = r1 * (expo - 1.0)
        intensity = tf.where(y > y_lower, raw, tf.zeros_like(raw))
        denom = tf.maximum(expo - 1.0, 1e-12)
        dlog_dy = r2 * dy * expo / denom
        dlog_dy = tf.where(y > y_lower, dlog_dy, tf.zeros_like(dlog_dy))
        return intensity, dlog_dy

    def robust_g(self, continuation_value: tf.Tensor, current_value: tf.Tensor) -> tf.Tensor:
        exponent = -(continuation_value - current_value) / np.float32(self.xi)
        return tf.exp(tf.clip_by_value(exponent, -50.0, 50.0))

    def cashflow_terms(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        current = self.current_objects(x)
        logk, z, y, logr = split_x(x)
        current_value = current["value"]
        n = tf.shape(x)[0]

        zeros = tf.zeros((n, 4), dtype=TF_FLOAT)
        flow_i = current["utility_grad"]
        flow_ii_damage = tf.identity(zeros)
        flow_ii_tech = tf.identity(zeros)
        flow_iii_damage = tf.identity(zeros)
        flow_iii_tech = tf.identity(zeros)
        robust_intensity_sum = tf.zeros((n, 1), dtype=TF_FLOAT)

        j_damage_total, dlog_damage_dy = self.damage_intensity(y)
        j_damage_each = j_damage_total / float(int(PARAMS["L"]))
        dlog_damage_vec = tf.concat(
            [tf.zeros((n, 2), dtype=TF_FLOAT), dlog_damage_dy, tf.zeros((n, 1), dtype=TF_FLOAT)],
            axis=1,
        )
        for lambda3 in PARAMS["λ3_values"]:
            continuation, continuation_grad = self.value_and_gradient(
                "PostDamagePreTech",
                x,
                lambda3_value=float(lambda3),
                y_override=pvalue("y_upper"),
            )
            g = self.robust_g(continuation, current_value)
            weighted_intensity = g * j_damage_each
            robust_intensity_sum += weighted_intensity
            flow_ii_damage += self.xi * j_damage_each * (1.0 - g) * dlog_damage_vec
            flow_iii_damage += weighted_intensity * continuation_grad

        r_scaled = tf.exp(logr) / pvalue("varrho")
        if self.one_tech_jump_mode or self.pi >= 1.0 - 1e-12:
            j_intermediate = tf.zeros_like(r_scaled)
            j_final = self.tech_scale * r_scaled
        else:
            j_intermediate = self.tech_scale * (1.0 - self.pi) * r_scaled
            j_final = self.tech_scale * self.pi * r_scaled

        dlog_tech_vec = tf.concat(
            [tf.zeros((n, 3), dtype=TF_FLOAT), tf.ones((n, 1), dtype=TF_FLOAT)],
            axis=1,
        )
        if not self.one_tech_jump_mode and self.pi < 1.0 - 1e-12:
            continuation, continuation_grad = self.value_and_gradient("PreDamageIntermTech", x)
            g = self.robust_g(continuation, current_value)
            weighted_intensity = g * j_intermediate
            robust_intensity_sum += weighted_intensity
            flow_ii_tech += self.xi * j_intermediate * (1.0 - g) * dlog_tech_vec
            flow_iii_tech += weighted_intensity * continuation_grad

        continuation, continuation_grad = self.value_and_gradient("PreDamagePostTech", x)
        g = self.robust_g(continuation, current_value)
        weighted_intensity = g * j_final
        robust_intensity_sum += weighted_intensity
        flow_ii_tech += self.xi * j_final * (1.0 - g) * dlog_tech_vec
        flow_iii_tech += weighted_intensity * continuation_grad

        return {
            "flow_i": flow_i,
            "flow_ii_damage": flow_ii_damage,
            "flow_ii_tech": flow_ii_tech,
            "flow_iii_damage": flow_iii_damage,
            "flow_iii_tech": flow_iii_tech,
            "robust_intensity_sum": robust_intensity_sum,
            "current_value_grad": current["value_grad"],
            "c_over_k": current["c_over_k"],
        }

    def nojump_step(self, x: tf.Tensor, shocks: tf.Tensor, dt: float) -> tf.Tensor:
        current = self.current_objects(x)
        logk, z, y, logr = split_x(x)
        i_d = current["i_d"]
        i_g = current["i_g"]
        i_r = current["i_r"]
        value_grad = current["value_grad"]

        sigma_d = pvalue("σ_d")
        sigma_g = pvalue("σ_g")
        sigma_r = pvalue("σ_κ")
        sigma_y_bar = pvalue("ϛ")
        theta_d = pvalue("θ_d")
        theta_g = pvalue("θ_g")

        inside_d = tf.maximum(1.0 + theta_d * i_d, 1e-8)
        inside_g = tf.maximum(1.0 + theta_g * i_g, 1e-8)
        phi_d = pvalue("α_d") + pvalue("Γ_d") * tf.math.log(inside_d)
        phi_g = pvalue("α_g") + pvalue("Γ_g") * tf.math.log(inside_g)

        drift_logk = (1.0 - z) * phi_d + z * phi_g - 0.5 * (
            sigma_d**2 * (1.0 - z) ** 2 + sigma_g**2 * z**2
        )
        drift_z = z * (1.0 - z) * (phi_g - phi_d + (1.0 - z) * sigma_d**2 - z * sigma_g**2)
        emissions = pvalue("η") * pvalue("A_d") * (1.0 - z) * tf.exp(logk)
        drift_y = pvalue("θ_bar") * emissions
        diffusion_y = sigma_y_bar * emissions
        safe_ir = tf.maximum(i_r, 1e-12)
        drift_logr = (
            -pvalue("ζ")
            + pvalue("ψ0") * tf.exp(pvalue("ψ1") * (tf.math.log(safe_ir) + logk - logr))
            - 0.5 * sigma_r**2  # Itô term is -1/2 sigma^2 (PAPER_HJB_REFERENCE); prior +0.5 was a sign bug
        )

        xi = np.float32(self.xi)
        v_logk = value_grad[:, 0:1]
        v_z = value_grad[:, 1:2]
        v_y = value_grad[:, 2:3]
        v_logr = value_grad[:, 3:4]
        h_d = -((v_logk - z * v_z) * (1.0 - z) * sigma_d) / xi
        h_g = -((v_logk + (1.0 - z) * v_z) * z * sigma_g) / xi
        h_y = -(v_y * diffusion_y) / xi
        h_r = -(v_logr * sigma_r) / xi

        drift_logk += sigma_d * (1.0 - z) * h_d + sigma_g * z * h_g
        drift_z += -sigma_d * z * (1.0 - z) * h_d + sigma_g * z * (1.0 - z) * h_g
        drift_y += diffusion_y * h_y
        drift_logr += sigma_r * h_r

        d_w_g = shocks[:, 0:1]
        d_w_d = shocks[:, 1:2]
        d_w_y = shocks[:, 2:3]
        d_w_r = shocks[:, 3:4]

        new_logk = logk + drift_logk * dt + sigma_d * (1.0 - z) * d_w_d + sigma_g * z * d_w_g
        new_z = z + drift_z * dt - sigma_d * z * (1.0 - z) * d_w_d + sigma_g * z * (1.0 - z) * d_w_g
        new_y = y + drift_y * dt + diffusion_y * d_w_y
        new_logr = logr + drift_logr * dt + sigma_r * d_w_r

        new_z = tf.clip_by_value(new_z, 1e-4, 0.9999)
        new_y = tf.maximum(new_y, 0.0)
        return tf.concat([new_logk, new_z, new_y, new_logr], axis=1)

    def initial_x(self, n_paths: int, y0: float, log_r_shift: float = 0.0) -> tf.Tensor:
        base = np.array(
            [
                np.log(pvalue("K0")),
                pvalue("Z0"),
                y0,
                np.log(pvalue("R0")) + float(log_r_shift),
            ],
            dtype=np.float32,
        )
        return tf.convert_to_tensor(np.repeat(base[None, :], int(n_paths), axis=0), dtype=TF_FLOAT)

    def scaling_factor(self, y0: float) -> Dict[str, float]:
        x0 = self.initial_x(1, y0)
        current = self.current_objects(x0)
        c0 = float(tf.squeeze(current["c_over_k"]).numpy()) * pvalue("K0")
        n0 = float(np.exp(pvalue("λ1") * y0 + 0.5 * pvalue("λ2") * y0 * y0))
        mu0 = pvalue("δ") * n0 / c0
        value_grad = current["value_grad"].numpy()[0]
        scale = 1.0 / (pvalue("R0") * mu0)
        return {
            "C0": c0,
            "N0": n0,
            "MU0": mu0,
            "scale_logR_to_level_consumption": scale,
            "direct_unscaled_dV_dlogR": float(value_grad[3]),
            "direct_scaled_SVRD": float(value_grad[3] * scale),
        }

    def simulate(self, args: argparse.Namespace) -> Dict[str, np.ndarray | Dict[str, float]]:
        rng = np.random.default_rng(args.seed)
        tf.random.set_seed(args.seed)
        n_paths = int(args.n_paths)
        dt = float(args.dt)
        steps = int(round(float(args.years) / dt))
        eps = float(args.eps_logr)

        x = self.initial_x(n_paths, args.y0)
        x_perturbed = self.initial_x(n_paths, args.y0, log_r_shift=eps)
        discount = tf.ones((n_paths, 1), dtype=TF_FLOAT)
        sqrt_dt = np.sqrt(dt)

        accum = OrderedDict(
            [
                ("jump_flow_i", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("jump_flow_ii_damage", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("jump_flow_ii_tech", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("jump_flow_iii_damage", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("jump_flow_iii_tech", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("state_flow_i_capital", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("state_flow_i_temperature", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("state_flow_i_rd", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("state_flow_ii_damage_temperature", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("state_flow_ii_tech_rd", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("state_flow_iii_capital", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("state_flow_iii_temperature", tf.zeros((n_paths,), dtype=TF_FLOAT)),
                ("state_flow_iii_rd", tf.zeros((n_paths,), dtype=TF_FLOAT)),
            ]
        )
        if args.save_path_details:
            path_totals = {key: [] for key in accum}
            path_time_grid = []
        else:
            path_totals = {}
            path_time_grid = []

        for t_idx in range(steps):
            terms = self.cashflow_terms(x)
            lambda_t = (x_perturbed - x) / np.float32(eps)
            weight = tf.squeeze(discount, axis=1) * np.float32(dt)

            def dot(flow: tf.Tensor) -> tf.Tensor:
                return tf.reduce_sum(lambda_t * flow, axis=1) * weight

            flow_i = terms["flow_i"]
            flow_ii_damage = terms["flow_ii_damage"]
            flow_ii_tech = terms["flow_ii_tech"]
            flow_iii_damage = terms["flow_iii_damage"]
            flow_iii_tech = terms["flow_iii_tech"]
            flow_iii_total = flow_iii_damage + flow_iii_tech

            accum["jump_flow_i"] += dot(flow_i)
            accum["jump_flow_ii_damage"] += dot(flow_ii_damage)
            accum["jump_flow_ii_tech"] += dot(flow_ii_tech)
            accum["jump_flow_iii_damage"] += dot(flow_iii_damage)
            accum["jump_flow_iii_tech"] += dot(flow_iii_tech)

            capital_lambda = lambda_t[:, 0] * weight
            z_lambda = lambda_t[:, 1] * weight
            temp_lambda = lambda_t[:, 2] * weight
            rd_lambda = lambda_t[:, 3] * weight

            accum["state_flow_i_capital"] += capital_lambda * flow_i[:, 0] + z_lambda * flow_i[:, 1]
            accum["state_flow_i_temperature"] += temp_lambda * flow_i[:, 2]
            accum["state_flow_i_rd"] += rd_lambda * flow_i[:, 3]
            accum["state_flow_ii_damage_temperature"] += temp_lambda * flow_ii_damage[:, 2]
            accum["state_flow_ii_tech_rd"] += rd_lambda * flow_ii_tech[:, 3]
            accum["state_flow_iii_capital"] += capital_lambda * flow_iii_total[:, 0] + z_lambda * flow_iii_total[:, 1]
            accum["state_flow_iii_temperature"] += temp_lambda * flow_iii_total[:, 2]
            accum["state_flow_iii_rd"] += rd_lambda * flow_iii_total[:, 3]

            discount = discount * tf.exp(
                -(pvalue("δ") + terms["robust_intensity_sum"]) * np.float32(dt)
            )
            shocks = tf.convert_to_tensor(rng.normal(0.0, sqrt_dt, size=(n_paths, 4)), dtype=TF_FLOAT)
            x = self.nojump_step(x, shocks, dt)
            x_perturbed = self.nojump_step(x_perturbed, shocks, dt)

            if args.progress_every and (t_idx + 1) % int(args.progress_every) == 0:
                print(f"step {t_idx + 1}/{steps} complete", flush=True)

            if args.save_path_details and (t_idx + 1) % max(1, steps // 60) == 0:
                path_time_grid.append((t_idx + 1) * dt)
                for key, value in accum.items():
                    path_totals[key].append(value.numpy().copy())

        totals = {key: value.numpy().astype(np.float64) for key, value in accum.items()}
        totals["jump_total"] = (
            totals["jump_flow_i"]
            + totals["jump_flow_ii_damage"]
            + totals["jump_flow_ii_tech"]
            + totals["jump_flow_iii_damage"]
            + totals["jump_flow_iii_tech"]
        )
        totals["state_total"] = (
            totals["state_flow_i_capital"]
            + totals["state_flow_i_temperature"]
            + totals["state_flow_i_rd"]
            + totals["state_flow_ii_damage_temperature"]
            + totals["state_flow_ii_tech_rd"]
            + totals["state_flow_iii_capital"]
            + totals["state_flow_iii_temperature"]
            + totals["state_flow_iii_rd"]
        )
        metadata = self.scaling_factor(args.y0)
        metadata.update(
            {
                "n_paths": float(n_paths),
                "years": float(args.years),
                "dt": dt,
                "steps": float(steps),
                "seed": float(args.seed),
                "eps_logr": eps,
                "xi": self.xi,
                "Y0": float(args.y0),
                "tech_jump_intensity_scale": self.tech_scale,
                "tech_jump_probability_pi": self.pi,
                "one_tech_jump_mode": float(self.one_tech_jump_mode),
            }
        )
        if args.save_path_details:
            totals["_path_time_grid"] = np.asarray(path_time_grid, dtype=np.float64)
            for key, snapshots in path_totals.items():
                totals[f"_snapshot_{key}"] = np.asarray(snapshots, dtype=np.float64)
        return {"totals": totals, "metadata": metadata}


def summarize_component(name: str, values: np.ndarray, scale: float, panel: str, label: str) -> Dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "panel": panel,
        "component": name,
        "label": label,
        "unscaled_mean": float(np.nanmean(values)),
        "unscaled_se": standard_error(values),
        "scaled_mean": float(np.nanmean(values) * scale),
        "scaled_se": standard_error(values * scale),
    }


def build_output_rows(totals: Dict[str, np.ndarray], scale: float) -> Tuple[list[Dict], list[Dict], list[Dict]]:
    jump_specs = [
        ("jump_flow_i", "flow i"),
        ("jump_flow_ii_damage", "flow ii: damage"),
        ("jump_flow_ii_tech", "flow ii: tech"),
        ("jump_flow_iii_damage", "flow iii: damage"),
        ("jump_flow_iii_tech", "flow iii: tech"),
        ("jump_total", "total"),
    ]
    state_specs = [
        ("state_flow_i_capital", "flow i: capital"),
        ("state_flow_i_temperature", "flow i: temperature"),
        ("state_flow_i_rd", "flow i: R&D"),
        ("state_flow_ii_damage_temperature", "flow ii damage: temperature"),
        ("state_flow_ii_tech_rd", "flow ii tech: R&D"),
        ("state_flow_iii_capital", "flow iii: capital"),
        ("state_flow_iii_temperature", "flow iii: temperature"),
        ("state_flow_iii_rd", "flow iii: R&D"),
        ("state_total", "total"),
    ]
    jump_rows = [
        summarize_component(key, totals[key], scale, "jump_type", label)
        for key, label in jump_specs
    ]
    state_rows = [
        summarize_component(key, totals[key], scale, "state_channel", label)
        for key, label in state_specs
    ]
    all_rows = jump_rows + state_rows
    return all_rows, jump_rows, state_rows


def plot_rows(rows: list[Dict], output_path: Path, title: str) -> None:
    labels = [row["label"] for row in rows if row["label"] != "total"]
    values = [row["scaled_mean"] for row in rows if row["label"] != "total"]
    errors = [row["scaled_se"] for row in rows if row["label"] != "total"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("scaled SVRD contribution")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SVRD decomposition by potential jumps and state channels.")
    parser.add_argument("--export-folder", required=True, help="Trained output folder.")
    parser.add_argument("--xi", type=float, required=True, help="Uncertainty aversion value, e.g. 0.05, 0.1, 148.6.")
    parser.add_argument("--n-paths", type=int, default=512)
    parser.add_argument("--years", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=1.0 / 12.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eps-logr", type=float, default=1e-4)
    parser.add_argument("--y0", type=float, default=1.2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--progress-every", type=int, default=120)
    parser.add_argument("--save-path-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_folder = Path(args.export_folder).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else export_folder / "SVRDDecomposition"
    output_dir = output_root / f"xi_{xi_label(args.xi)}"
    ensure_dir(output_dir)

    start = time.time()
    decomposer = SVRDDecomposer(str(export_folder), args.xi, batch_size=max(args.batch_size, args.n_paths))
    result = decomposer.simulate(args)
    totals = result["totals"]
    metadata = result["metadata"]
    scale = metadata["scale_logR_to_level_consumption"]
    all_rows, jump_rows, state_rows = build_output_rows(totals, scale)

    write_rows(output_dir / "svrd_decomposition.csv", all_rows)
    write_rows(output_dir / "svrd_by_potential_jump.csv", jump_rows)
    write_rows(output_dir / "svrd_by_state_channel.csv", state_rows)
    plot_rows(jump_rows, output_dir / "svrd_by_potential_jump.png", f"SVRD by potential jumps, xi={args.xi:g}")
    plot_rows(state_rows, output_dir / "svrd_by_state_channel.png", f"SVRD by state channels, xi={args.xi:g}")

    save_payload = {
        key: value
        for key, value in totals.items()
        if not key.startswith("_snapshot_") and key != "_path_time_grid"
    }
    np.savez_compressed(output_dir / "svrd_path_totals.npz", **save_payload)
    if args.save_path_details:
        np.savez_compressed(
            output_dir / "svrd_path_snapshots.npz",
            **{key: value for key, value in totals.items() if key.startswith("_snapshot_")},
            time_grid=totals.get("_path_time_grid", np.array([], dtype=np.float64)),
        )

    elapsed = time.time() - start
    with (output_dir / "metadata.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"created = {time.ctime()}\n")
        handle.write(f"elapsed_seconds = {elapsed:.3f}\n")
        handle.write(f"export_folder = {export_folder}\n")
        handle.write("method = no-jump robust diffusion Monte Carlo; Lambda by common-random finite difference in initial logR\n")
        for key in sorted(metadata):
            handle.write(f"{key} = {metadata[key]}\n")
        handle.write("\nidentity_check_scaled:\n")
        handle.write(f"  direct_scaled_SVRD = {metadata['direct_scaled_SVRD']:.10g}\n")
        handle.write(f"  mc_jump_total_scaled = {jump_rows[-1]['scaled_mean']:.10g}\n")
        handle.write(f"  mc_state_total_scaled = {state_rows[-1]['scaled_mean']:.10g}\n")

    print(f"Saved SVRD decomposition to {output_dir}")
    print(f"Direct scaled SVRD: {metadata['direct_scaled_SVRD']:.6g}")
    print(f"MC total by jump type: {jump_rows[-1]['scaled_mean']:.6g} +/- {jump_rows[-1]['scaled_se']:.3g}")
    print(f"MC total by state channel: {state_rows[-1]['scaled_mean']:.6g} +/- {state_rows[-1]['scaled_se']:.3g}")


if __name__ == "__main__":
    main()
