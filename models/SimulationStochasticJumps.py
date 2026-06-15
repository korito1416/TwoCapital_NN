import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from params import PARAMS

from PostDamageIntermTech import PostDamageIntermTechModel
from PostDamagePostTech import PostDamagePostTechModel
from PostDamagePreTech import PostDamagePreTechModel
from PreDamageIntermTech import PreDamageIntermTechModel
from PreDamagePostTech import PreDamagePostTechModel
from PreDamagePreTech import PreDamagePreTechModel


STAGE_SPECS = {
    "PostDamagePostTech": (PostDamagePostTechModel, 7, False),
    "PostDamageIntermTech": (PostDamageIntermTechModel, 8, True),
    "PostDamagePreTech": (PostDamagePreTechModel, 8, True),
    "PreDamagePostTech": (PreDamagePostTechModel, 6, False),
    "PreDamageIntermTech": (PreDamageIntermTechModel, 7, True),
    "PreDamagePreTech": (PreDamagePreTechModel, 7, True),
}


def checkpoint_prefix(export_folder, stage, net):
    return os.path.join(export_folder, stage, f"{net}_nn_checkpoint_{stage}")


def checkpoint_exists(export_folder, stage, net="v"):
    return os.path.exists(checkpoint_prefix(export_folder, stage, net) + ".index")


def read_float_from_files(export_folder, labels, default):
    candidates = [
        os.path.join(export_folder, "run_manifest.txt"),
        os.path.join(export_folder, "PreDamagePreTech", "params.txt"),
        os.path.join(export_folder, "PreDamagePostTech", "params.txt"),
        os.path.join(export_folder, "PostDamagePreTech", "params.txt"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                for label in labels:
                    if label in stripped or stripped.startswith(label):
                        try:
                            return float(stripped.split(":", 1)[1].strip().split()[0])
                        except (IndexError, ValueError):
                            pass
    return float(default)


def infer_tech_jump_intensity_scale(export_folder):
    folder = os.path.basename(os.path.abspath(export_folder))
    match = re.search(r"TechIntensityScale_(\d+)p(\d+)", folder)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return read_float_from_files(
        export_folder,
        ["Tech jump intensity scale", "tech_jump_intensity_scale"],
        PARAMS.get("tech_jump_intensity_scale", 1.0),
    )


def infer_tech_jump_probability(export_folder):
    folder = os.path.basename(os.path.abspath(export_folder))
    match = re.search(r"OneTechJump_Pi_(\d+)p(\d+)", folder)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return read_float_from_files(
        export_folder,
        ["Tech jump probability pi", "π"],
        PARAMS.get("π", 0.04),
    )


def infer_one_tech_jump_mode(export_folder):
    folder = os.path.basename(os.path.abspath(export_folder))
    if folder.startswith("OneTechJump_"):
        return True
    manifest = os.path.join(export_folder, "run_manifest.txt")
    if os.path.exists(manifest):
        with open(manifest, "r") as f:
            text = f.read().lower()
        return "one technology jump" in text or "one-tech" in text
    return False


def xi_label(xi):
    return f"{xi:g}"


def make_nn_configs(num_hidden_layers=4, num_neurons=32):
    hidden = ["swish", "tanh", "tanh", "softplus"]
    output = ["softplus", "custom", "custom", "softplus"]

    v_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden[0],
        "dim": 1,
        "nn_name": "v_nn",
        "final_activation": output[0],
    }
    i_g_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden[1],
        "dim": 1,
        "nn_name": "i_g_nn",
        "final_activation": output[1],
    }
    i_d_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden[2],
        "dim": 1,
        "nn_name": "i_d_nn",
        "final_activation": output[2],
    }
    i_r_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden[3],
        "dim": 1,
        "nn_name": "i_r_nn",
        "final_activation": output[3],
    }

    phi_g = 16.7
    phi_d = 16.7
    i_g_nn_config["final_activation"] = lambda x: 1.0 - (1.0 + 1.0 / phi_g) / (tf.exp(2 * x) + 1.0)
    i_d_nn_config["final_activation"] = lambda x: 1.0 - (1.0 + 1.0 / phi_d) / (tf.exp(2 * x) + 1.0)
    return v_nn_config, i_g_nn_config, i_d_nn_config, i_r_nn_config


class RegimeModels:
    def __init__(self, export_folder, xi, batch_size=128):
        self.export_folder = os.path.abspath(export_folder)
        self.xi = float(xi)
        self.log_xi = float(np.log(self.xi))
        self.batch_size = int(batch_size)
        self.tech_jump_intensity_scale = infer_tech_jump_intensity_scale(self.export_folder)
        self.pi = infer_tech_jump_probability(self.export_folder)
        self.one_tech_jump_mode = infer_one_tech_jump_mode(self.export_folder)

        v_cfg, ig_cfg, id_cfg, ir_cfg = make_nn_configs()
        params = {
            "batch_size": self.batch_size,
            "learning_rates": [1e-4, 1e-4, 1e-4, 1e-4],
            "v_nn_config": v_cfg,
            "i_g_nn_config": ig_cfg,
            "i_d_nn_config": id_cfg,
            "i_r_nn_config": ir_cfg,
            "num_iterations": 1,
            "logging_frequency": 1000,
            "verbose": False,
            "pretrained_path": None,
            "learning_rate_schedule_type": "None",
            "tech_jump_intensity_scale": self.tech_jump_intensity_scale,
            "π": self.pi,
            "v_PostDamagePostTech_nn_path": checkpoint_prefix(self.export_folder, "PostDamagePostTech", "v"),
            "v_PostDamageIntermTech_nn_path": checkpoint_prefix(self.export_folder, "PostDamageIntermTech", "v"),
            "v_PostDamagePreTech_nn_path": checkpoint_prefix(self.export_folder, "PostDamagePreTech", "v"),
            "v_PreDamagePostTech_nn_path": checkpoint_prefix(self.export_folder, "PreDamagePostTech", "v"),
            "v_PreDamageIntermTech_nn_path": checkpoint_prefix(self.export_folder, "PreDamageIntermTech", "v"),
        }
        PARAMS.update(params)
        self.params = PARAMS.copy()
        self.models = {}
        self.load_available_models()

    def load_available_models(self):
        required_stages = [
            "PostDamagePostTech",
            "PostDamagePreTech",
            "PreDamagePostTech",
            "PreDamagePreTech",
        ]
        optional_interm_stages = ["PostDamageIntermTech", "PreDamageIntermTech"]
        stages = required_stages + ([] if self.one_tech_jump_mode else optional_interm_stages)

        for stage in stages:
            if not checkpoint_exists(self.export_folder, stage):
                raise FileNotFoundError(f"Missing checkpoint for {stage} in {self.export_folder}")
            cls, n_inputs, has_ir = STAGE_SPECS[stage]
            model = cls(self.params)
            dummy = tf.zeros((self.batch_size, n_inputs), dtype=tf.float32)
            model.v_nn(dummy, training=False)
            model.i_g_nn(dummy, training=False)
            model.i_d_nn(dummy, training=False)
            model.v_nn.load_weights(checkpoint_prefix(self.export_folder, stage, "v"))
            model.i_g_nn.load_weights(checkpoint_prefix(self.export_folder, stage, "i_g"))
            model.i_d_nn.load_weights(checkpoint_prefix(self.export_folder, stage, "i_d"))
            if has_ir:
                model.i_r_nn(dummy, training=False)
                model.i_r_nn.load_weights(checkpoint_prefix(self.export_folder, stage, "i_r"))
            self.models[stage] = model

    def stage_for_state(self, tech_state, damage_state):
        if tech_state == 0 and damage_state == 0:
            return "PreDamagePreTech"
        if tech_state == 0 and damage_state == 1:
            return "PostDamagePreTech"
        if tech_state == 1 and damage_state == 0:
            return "PreDamageIntermTech"
        if tech_state == 1 and damage_state == 1:
            return "PostDamageIntermTech"
        if tech_state == 2 and damage_state == 0:
            return "PreDamagePostTech"
        if tech_state == 2 and damage_state == 1:
            return "PostDamagePostTech"
        raise ValueError(f"Invalid regime tech={tech_state}, damage={damage_state}")

    def green_productivity(self, tech_state, a_g_level):
        if tech_state == 0:
            return float(PARAMS["A_g"])
        if tech_state == 1:
            return float(PARAMS["A_g_prime"])
        return float(a_g_level if a_g_level > 0 else PARAMS["A_g_prime_prime"])

    def state_vector(self, stage, state):
        log_k = state["logK"]
        z = state["Z"]
        y = state["Y"]
        log_r = state["logR"]
        lambda3 = state["lambda3"]
        a_g_level = state["A_g"]
        lx = self.log_xi

        if stage == "PreDamagePreTech":
            return [log_k, z, y, log_r, lx, lx, lx]
        if stage == "PostDamagePreTech":
            return [log_k, z, y, log_r, lambda3, lx, lx, lx]
        if stage == "PreDamageIntermTech":
            return [log_k, z, y, log_r, lx, lx, lx]
        if stage == "PostDamageIntermTech":
            return [log_k, z, y, log_r, lambda3, lx, lx, lx]
        if stage == "PreDamagePostTech":
            return [log_k, z, y, a_g_level, lx, lx]
        if stage == "PostDamagePostTech":
            return [log_k, z, y, lambda3, a_g_level, lx, lx]
        raise ValueError(stage)

    def evaluate(self, state):
        tech_state = int(state["tech_state"])
        damage_state = int(state["damage_state"])
        stage = self.stage_for_state(tech_state, damage_state)
        model = self.models[stage]
        x = tf.constant([self.state_vector(stage, state)], dtype=tf.float32)

        v = float(tf.squeeze(model.v_nn(x, training=False)).numpy())
        i_g = float(tf.squeeze(model.i_g_nn(x, training=False)).numpy())
        i_d = float(tf.squeeze(model.i_d_nn(x, training=False)).numpy())
        if STAGE_SPECS[stage][2]:
            i_r = float(np.exp(-float(tf.squeeze(model.i_r_nn(x, training=False)).numpy())))
        else:
            i_r = np.nan

        k = float(np.exp(state["logK"]))
        z = float(state["Z"])
        a_g_current = self.green_productivity(tech_state, state["A_g"])
        i_r_for_c = 0.0 if np.isnan(i_r) else i_r
        c_over_k = (float(PARAMS["A_d"]) - i_d) * (1.0 - z) + (a_g_current - i_g) * z - i_r_for_c
        y_over_k = float(PARAMS["A_d"]) * (1.0 - z) + a_g_current * z

        return {
            "stage": stage,
            "V": v,
            "i_g": i_g,
            "i_d": i_d,
            "i_r": i_r,
            "I_g": i_g * k * z,
            "I_d": i_d * k * (1.0 - z),
            "I_r": np.nan if np.isnan(i_r) else i_r * k,
            "C_over_K": c_over_k,
            "Y_over_K": y_over_k,
        }


def initial_state(y0):
    return {
        "logK": float(np.log(PARAMS["K0"])),
        "Z": float(PARAMS["Z0"]),
        "Y": float(y0),
        "logR": float(np.log(PARAMS["R0"])),
        "A_g": 0.0,
        "lambda3": 0.0,
        "tech_state": 0,
        "damage_state": 0,
    }


def draw_damage_lambda3(rng):
    values = np.asarray(PARAMS["λ3_values"], dtype=float)
    return float(values[rng.integers(0, len(values))])


def step_state(state, policy, evaluator, rng, dt):
    p = PARAMS
    log_k = state["logK"]
    z = state["Z"]
    y = state["Y"]
    log_r = state["logR"]
    tech_state = int(state["tech_state"])
    damage_state = int(state["damage_state"])

    k = float(np.exp(log_k))
    i_d = float(policy["i_d"])
    i_g = float(policy["i_g"])
    i_r = float(policy["i_r"]) if not np.isnan(policy["i_r"]) else 0.0

    sigma_d = float(p["σ_d"])
    sigma_g = float(p["σ_g"])
    sigma_kappa = float(p["σ_κ"])
    alpha_d = float(p["α_d"])
    alpha_g = float(p["α_g"])
    gamma_d = float(p["Γ_d"])
    gamma_g = float(p["Γ_g"])
    theta_d = float(p["θ_d"])
    theta_g = float(p["θ_g"])

    inside_d = max(1.0 + theta_d * i_d, 1e-8)
    inside_g = max(1.0 + theta_g * i_g, 1e-8)

    vkk = 0.5 * (sigma_d**2 * (1.0 - z) ** 2 + sigma_g**2 * z**2)
    drift_log_k = (
        (alpha_d + gamma_d * np.log(inside_d)) * (1.0 - z)
        + (alpha_g + gamma_g * np.log(inside_g)) * z
        - vkk
    )
    drift_z = (
        alpha_g
        + gamma_g * np.log(inside_g)
        - (alpha_d + gamma_d * np.log(inside_d))
        - z * sigma_g**2
        + (1.0 - z) * sigma_d**2
    ) * z * (1.0 - z)
    emissions = float(p["η"]) * float(p["A_d"]) * (1.0 - z) * k
    drift_y = float(p["θ_bar"]) * emissions
    diffusion_y = float(p["ϛ"]) * emissions

    d_w_g, d_w_d, d_w_y, d_w_log_r = rng.normal(0.0, np.sqrt(dt), size=4)

    new_state = dict(state)
    new_state["logK"] = log_k + drift_log_k * dt + sigma_d * (1.0 - z) * d_w_d + sigma_g * z * d_w_g
    new_state["Z"] = z + drift_z * dt - sigma_d * z * (1.0 - z) * d_w_d + sigma_g * z * (1.0 - z) * d_w_g
    new_state["Z"] = float(np.clip(new_state["Z"], 1e-4, 0.9999))
    new_state["Y"] = max(0.0, y + drift_y * dt + diffusion_y * d_w_y)

    if tech_state < 2:
        drift_log_r = -float(p["ζ"]) + float(p["ψ0"]) * np.exp(float(p["ψ1"]) * (np.log(max(i_r, 1e-12)) + log_k - log_r)) + 0.5 * sigma_kappa**2
        new_state["logR"] = log_r + drift_log_r * dt + sigma_kappa * d_w_log_r
    else:
        new_state["logR"] = 0.0

    tech_event = ""
    damage_event = False

    if damage_state == 0:
        j_damage = float(p["r1"]) * (np.exp(float(p["r2"]) / 2.0 * (new_state["Y"] - float(p["y_lower"])) ** 2) - 1.0)
        if new_state["Y"] <= float(p["y_lower"]):
            j_damage = 0.0
        if rng.random() < 1.0 - np.exp(-j_damage * dt):
            damage_event = True
            new_state["damage_state"] = 1
            new_state["lambda3"] = draw_damage_lambda3(rng)
            new_state["Y"] = float(p["y_upper"])

    if tech_state == 0:
        j_tech = evaluator.tech_jump_intensity_scale * np.exp(new_state["logR"]) / float(p["varrho"])
        if rng.random() < 1.0 - np.exp(-j_tech * dt):
            if evaluator.one_tech_jump_mode or evaluator.pi >= 1.0 - 1e-12:
                next_tech = 2
            else:
                next_tech = 1 if rng.random() < (1.0 - evaluator.pi) else 2
            new_state["tech_state"] = next_tech
            if next_tech == 1:
                new_state["A_g"] = float(p["A_g_prime"])
                tech_event = "tech_0_to_1"
            else:
                new_state["A_g"] = float(p["A_g_prime_prime"])
                new_state["logR"] = 0.0
                tech_event = "tech_0_to_2"
    elif tech_state == 1:
        j_tech_post = evaluator.tech_jump_intensity_scale * evaluator.pi * np.exp(new_state["logR"]) / float(p["varrho"])
        if rng.random() < 1.0 - np.exp(-j_tech_post * dt):
            new_state["tech_state"] = 2
            new_state["A_g"] = float(p["A_g_prime_prime"])
            new_state["logR"] = 0.0
            tech_event = "tech_1_to_2"
    else:
        new_state["tech_state"] = 2
        new_state["A_g"] = float(p["A_g_prime_prime"])
        new_state["logR"] = 0.0

    return new_state, tech_event, damage_event


def append_event(events, model_name, xi, seed, path_id, t_idx, event_group, event_type, event_order, before_state, after_state, before_policy, after_policy):
    row = {
        "model": model_name,
        "xi": xi,
        "seed": seed,
        "path_id": path_id,
        "time_index_after": t_idx,
        "time_years_after": t_idx / 12.0,
        "event_group": event_group,
        "event_type": event_type,
        "event_order": event_order,
    }
    state_keys = ["logK", "Z", "Y", "logR", "A_g", "lambda3", "tech_state", "damage_state"]
    policy_keys = ["stage", "V", "i_d", "i_g", "i_r", "I_d", "I_g", "I_r", "C_over_K", "Y_over_K"]
    for key in state_keys:
        row[f"before_{key}"] = before_state[key]
        row[f"after_{key}"] = after_state[key]
    for key in policy_keys:
        row[f"before_{key}"] = before_policy[key]
        row[f"after_{key}"] = after_policy[key]
    events.append(row)


def summarize_events(events):
    numeric_suffixes = [
        "V",
        "i_d",
        "i_g",
        "i_r",
        "I_d",
        "I_g",
        "I_r",
        "C_over_K",
        "Y_over_K",
        "logK",
        "Z",
        "Y",
        "logR",
    ]
    groups = defaultdict(list)
    for row in events:
        groups[(row["event_group"], row["event_type"], row["event_order"])].append(row)

    summaries = []
    for (event_group, event_type, event_order), rows in sorted(groups.items()):
        out = {
            "event_group": event_group,
            "event_type": event_type,
            "event_order": event_order,
            "count": len(rows),
        }
        for prefix in ["before", "after"]:
            for suffix in numeric_suffixes:
                key = f"{prefix}_{suffix}"
                vals = np.array([float(r[key]) for r in rows if key in r and r[key] != ""], dtype=float)
                out[f"mean_{key}"] = float(np.nanmean(vals)) if vals.size else np.nan
        for suffix in ["V", "i_d", "i_g", "i_r", "I_d", "I_g", "I_r", "C_over_K", "Y_over_K"]:
            before_key = f"before_{suffix}"
            after_key = f"after_{suffix}"
            vals = np.array(
                [float(r[after_key]) - float(r[before_key]) for r in rows if before_key in r and after_key in r],
                dtype=float,
            )
            out[f"mean_delta_{suffix}"] = float(np.nanmean(vals)) if vals.size else np.nan
        summaries.append(out)
    return summaries


def write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def simulate(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    evaluator = RegimeModels(args.export_folder, args.xi, args.batch_size)
    model_name = os.path.basename(os.path.abspath(args.export_folder))
    output_root = args.output_root or os.path.join(args.export_folder, "SimulationResults")
    output_folder = os.path.join(output_root, f"paths_ξ_{xi_label(args.xi)}")
    os.makedirs(output_folder, exist_ok=True)

    dt = float(args.dt)
    steps = int(round(float(args.years) / dt))
    n_paths = int(args.n_paths)

    float_arrays = {
        "logK": np.empty((n_paths, steps + 1), dtype=np.float64),
        "Z": np.empty((n_paths, steps + 1), dtype=np.float64),
        "logR": np.empty((n_paths, steps + 1), dtype=np.float64),
        "Y": np.empty((n_paths, steps + 1), dtype=np.float64),
        "A_g": np.empty((n_paths, steps + 1), dtype=np.float64),
        "lambda3": np.empty((n_paths, steps + 1), dtype=np.float64),
        "i_r": np.empty((n_paths, steps + 1), dtype=np.float64),
        "i_g": np.empty((n_paths, steps + 1), dtype=np.float64),
        "i_d": np.empty((n_paths, steps + 1), dtype=np.float64),
        "I_r": np.empty((n_paths, steps + 1), dtype=np.float64),
        "I_g": np.empty((n_paths, steps + 1), dtype=np.float64),
        "I_d": np.empty((n_paths, steps + 1), dtype=np.float64),
        "V": np.empty((n_paths, steps + 1), dtype=np.float64),
        "C_over_K": np.empty((n_paths, steps + 1), dtype=np.float64),
        "Y_over_K": np.empty((n_paths, steps + 1), dtype=np.float64),
    }
    int_arrays = {
        "tech_state": np.empty((n_paths, steps + 1), dtype=np.int16),
        "damage_state": np.empty((n_paths, steps + 1), dtype=np.int16),
        "tech_event_code": np.zeros((n_paths, steps + 1), dtype=np.int16),
        "damage_event": np.zeros((n_paths, steps + 1), dtype=np.int16),
    }
    stage_array = np.empty((n_paths, steps + 1), dtype=object)
    events = []

    def store(path_id, t_idx, state, policy):
        for key in ["logK", "Z", "logR", "Y", "A_g", "lambda3"]:
            float_arrays[key][path_id, t_idx] = state[key]
        for key in ["i_r", "i_g", "i_d", "I_r", "I_g", "I_d", "V", "C_over_K", "Y_over_K"]:
            float_arrays[key][path_id, t_idx] = policy[key]
        int_arrays["tech_state"][path_id, t_idx] = int(state["tech_state"])
        int_arrays["damage_state"][path_id, t_idx] = int(state["damage_state"])
        stage_array[path_id, t_idx] = policy["stage"]

    for path_id in range(n_paths):
        state = initial_state(args.y0)
        policy = evaluator.evaluate(state)
        store(path_id, 0, state, policy)
        tech_event_count = 0
        damage_event_count = 0

        for t_idx in range(1, steps + 1):
            before_state = dict(state)
            before_policy = dict(policy)
            state, tech_event, damage_event = step_state(state, policy, evaluator, rng, dt)
            policy = evaluator.evaluate(state)
            store(path_id, t_idx, state, policy)

            if tech_event:
                tech_event_count += 1
                int_arrays["tech_event_code"][path_id, t_idx] = {"tech_0_to_1": 1, "tech_0_to_2": 2, "tech_1_to_2": 3}[tech_event]
                append_event(
                    events,
                    model_name,
                    args.xi,
                    args.seed,
                    path_id,
                    t_idx,
                    "technology",
                    tech_event,
                    tech_event_count,
                    before_state,
                    state,
                    before_policy,
                    policy,
                )
            if damage_event:
                damage_event_count += 1
                int_arrays["damage_event"][path_id, t_idx] = 1
                append_event(
                    events,
                    model_name,
                    args.xi,
                    args.seed,
                    path_id,
                    t_idx,
                    "damage",
                    "damage_jump",
                    damage_event_count,
                    before_state,
                    state,
                    before_policy,
                    policy,
                )

    seed = args.seed
    np.save(os.path.join(output_folder, f"logK_sim_{seed}.npy"), float_arrays["logK"])
    np.save(os.path.join(output_folder, f"Z_array_{seed}.npy"), float_arrays["Z"])
    np.save(os.path.join(output_folder, f"logR_array_{seed}.npy"), float_arrays["logR"])
    np.save(os.path.join(output_folder, f"Y_array_{seed}.npy"), float_arrays["Y"])
    np.save(os.path.join(output_folder, f"A_g_array_{seed}.npy"), float_arrays["A_g"])
    np.save(os.path.join(output_folder, f"gamma3_array_{seed}.npy"), float_arrays["lambda3"])
    np.save(os.path.join(output_folder, f"tech_state_array_{seed}.npy"), int_arrays["tech_state"])
    np.save(os.path.join(output_folder, f"damage_state_array_{seed}.npy"), int_arrays["damage_state"])
    np.save(os.path.join(output_folder, f"I_r_array_{seed}.npy"), float_arrays["I_r"])
    np.save(os.path.join(output_folder, f"I_g_array_{seed}.npy"), float_arrays["I_g"])
    np.save(os.path.join(output_folder, f"I_d_array_{seed}.npy"), float_arrays["I_d"])
    np.save(os.path.join(output_folder, f"i_r_array_{seed}.npy"), float_arrays["i_r"])
    np.save(os.path.join(output_folder, f"i_g_array_{seed}.npy"), float_arrays["i_g"])
    np.save(os.path.join(output_folder, f"i_d_array_{seed}.npy"), float_arrays["i_d"])
    np.save(os.path.join(output_folder, f"V_array_{seed}.npy"), float_arrays["V"])
    np.save(os.path.join(output_folder, f"C_over_K_array_{seed}.npy"), float_arrays["C_over_K"])
    np.save(os.path.join(output_folder, f"Y_over_K_array_{seed}.npy"), float_arrays["Y_over_K"])
    np.save(os.path.join(output_folder, f"tech_event_code_array_{seed}.npy"), int_arrays["tech_event_code"])
    np.save(os.path.join(output_folder, f"damage_event_array_{seed}.npy"), int_arrays["damage_event"])
    np.save(os.path.join(output_folder, f"stage_array_{seed}.npy"), stage_array)

    np.savez_compressed(
        os.path.join(output_folder, f"paths_controls_values_{seed}.npz"),
        **float_arrays,
        **int_arrays,
        stage=stage_array,
        dt=dt,
        years=float(args.years),
        xi=float(args.xi),
        seed=int(seed),
        model=model_name,
    )

    write_csv(os.path.join(output_folder, f"event_windows_{seed}.csv"), events)
    write_csv(os.path.join(output_folder, f"event_summary_{seed}.csv"), summarize_events(events))

    with open(os.path.join(output_folder, f"metadata_{seed}.txt"), "w") as f:
        f.write(f"created = {time.ctime()}\n")
        f.write(f"export_folder = {os.path.abspath(args.export_folder)}\n")
        f.write(f"model = {model_name}\n")
        f.write(f"xi = {args.xi}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"n_paths = {n_paths}\n")
        f.write(f"years = {args.years}\n")
        f.write(f"dt = {dt}\n")
        f.write(f"steps = {steps}\n")
        f.write(f"Y0 = {args.y0}\n")
        f.write(f"tech_jump_intensity_scale = {evaluator.tech_jump_intensity_scale}\n")
        f.write(f"tech_jump_probability_pi = {evaluator.pi}\n")
        f.write(f"one_tech_jump_mode = {evaluator.one_tech_jump_mode}\n")
        f.write(f"event_rows = {len(events)}\n")

    print(f"Saved stochastic paths to {output_folder}")
    print(f"Event rows: {len(events)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stochastic simulation with jump-window policy/value reporting.")
    parser.add_argument("--export-folder", required=True)
    parser.add_argument("--xi", type=float, required=True)
    parser.add_argument("--seed", "--id", dest="seed", type=int, default=1)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--years", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=1.0 / 12.0)
    parser.add_argument("--y0", type=float, default=1.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    simulate(parse_args())
