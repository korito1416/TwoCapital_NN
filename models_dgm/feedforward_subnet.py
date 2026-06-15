"""DGM subnet and shared training utilities for the climate HJB models.

The gated layer follows the Deep Galerkin Method architecture of Sirignano and
Spiliopoulos as implemented by Al-Aradi et al. The public model interface is
kept identical to the original FeedForwardSubNet so the existing HJB and FOC
code can be reused without equation-level changes.
"""

import math
import os
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(int(os.environ.get("DGM_SEED", "0")))


def _activation(value, default="tanh"):
    value = value if value is not None else default
    return tf.keras.activations.get(value)


class DGMLayer(tf.keras.layers.Layer):
    """One gated DGM layer driven by both the original input and hidden state."""

    def __init__(self, width, activation="tanh", candidate_activation="tanh", name=None):
        super().__init__(name=name)
        self.width = int(width)
        self.activation = _activation(activation)
        self.candidate_activation = _activation(candidate_activation)

    def build(self, input_shape):
        x_shape, state_shape = input_shape
        input_dim = int(x_shape[-1])
        state_dim = int(state_shape[-1])
        for gate in ("z", "g", "r", "h"):
            setattr(
                self,
                f"u_{gate}",
                self.add_weight(
                    name=f"u_{gate}",
                    shape=(input_dim, self.width),
                    initializer="glorot_uniform",
                ),
            )
            setattr(
                self,
                f"w_{gate}",
                self.add_weight(
                    name=f"w_{gate}",
                    shape=(state_dim, self.width),
                    initializer="glorot_uniform",
                ),
            )
            setattr(
                self,
                f"b_{gate}",
                self.add_weight(
                    name=f"b_{gate}",
                    shape=(self.width,),
                    initializer="zeros",
                ),
            )
        super().build(input_shape)

    def call(self, inputs):
        x, state = inputs
        z = self.activation(x @ self.u_z + state @ self.w_z + self.b_z)
        g = self.activation(x @ self.u_g + state @ self.w_g + self.b_g)
        r = self.activation(x @ self.u_r + state @ self.w_r + self.b_r)
        candidate = self.candidate_activation(
            x @ self.u_h + (state * r) @ self.w_h + self.b_h
        )
        return (1.0 - g) * candidate + z * state


class FeedForwardSubNet(tf.keras.Model):
    """Drop-in DGM replacement for the original feedforward subnet."""

    def __init__(self, config):
        super().__init__(name=config["nn_name"] + ".dgm")
        widths = [int(width) for width in config["num_hiddens"]]
        if not widths:
            raise ValueError("DGM requires at least one hidden layer")
        if len(set(widths)) != 1:
            raise ValueError("All DGM hidden layers must have the same width")

        width = widths[0]
        gate_activation = config.get(
            "dgm_gate_activation",
            os.environ.get("DGM_GATE_ACTIVATION", "tanh"),
        )
        candidate_activation = config.get(
            "dgm_candidate_activation",
            os.environ.get("DGM_CANDIDATE_ACTIVATION", "tanh"),
        )
        initial_activation = config.get(
            "dgm_initial_activation",
            os.environ.get("DGM_INITIAL_ACTIVATION", "tanh"),
        )

        self.initial_layer = tf.keras.layers.Dense(
            width,
            activation=_activation(initial_activation),
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name=config["nn_name"] + ".initial",
        )
        self.dgm_layers = [
            DGMLayer(
                width,
                activation=gate_activation,
                candidate_activation=candidate_activation,
                name=config["nn_name"] + f".dgm_layer.{index}",
            )
            for index in range(len(widths))
        ]
        self.output_layer = tf.keras.layers.Dense(
            int(config["dim"]),
            activation=config.get("final_activation"),
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name=config["nn_name"] + ".output",
        )

    def call(self, x, training=False):
        del training
        state = self.initial_layer(x)
        for layer in self.dgm_layers:
            state = layer((x, state))
        return self.output_layer(state)


def stratified_uniform(lower, upper, batch_size):
    """Draw a shuffled one-dimensional stratified sample."""
    batch_size = tf.cast(batch_size, tf.int32)
    edges = tf.linspace(
        tf.cast(lower, tf.float32),
        tf.cast(upper, tf.float32),
        batch_size + 1,
    )
    offsets = tf.random.uniform(tf.stack([batch_size, 1]), 0.0, 1.0)
    draws = tf.reshape(edges[:-1], (-1, 1))
    widths = tf.reshape(edges[1:] - edges[:-1], (-1, 1))
    return tf.random.shuffle(draws + widths * offsets)


def sample_state_columns(params, batch_size=None):
    """Sample the common six state columns with a flexible batch dimension."""
    n = int(batch_size if batch_size is not None else params["batch_size"])
    return (
        stratified_uniform(params.get("logK_min", 4.0), params.get("logK_max", 7.0), n),
        stratified_uniform(params.get("Z_min", 0.01), params.get("Z_max", 0.99), n),
        stratified_uniform(params.get("Y_min", 0.0), params.get("Y_max", 4.0), n),
        stratified_uniform(
            params.get("logR_min", params.get("R_min", 1.0)),
            params.get("logR_max", params.get("R_max", 6.0)),
            n,
        ),
        stratified_uniform(params.get("lambda3_min", params.get("λ3_min", 0.0)),
                           params.get("lambda3_max", params.get("λ3_max", 1.0 / 3.0)), n),
        stratified_uniform(params.get("logxi_min", params.get("logξ_min", -3.0)),
                           params.get("logxi_max", params.get("logξ_max", 5.0)), n),
    )


def large_sample_validation(model):
    """Average residual diagnostics over independent validation batches."""
    n_batches = int(model.params.get("validation_batches", 4))
    batch_size = int(
        model.params.get(
            "validation_batch_size",
            max(1024, int(model.params["batch_size"])),
        )
    )
    losses = []
    for _ in range(n_batches):
        sample = model.sample(batch_size=batch_size)
        losses.append(model.objective_fn(*sample, training=False))
    return tuple(
        tf.reduce_mean(tf.stack([batch[i] for batch in losses]))
        for i in range(len(losses[0]))
    )


def validation_score(losses, control_weight=1.0):
    """Return a finite score accounting for PDE and control residuals."""
    losses = list(losses)
    if not losses:
        return float("inf")
    weighted = [losses[0]]
    if len(losses) > 2:
        weighted.extend(control_weight * loss for loss in losses[1:-1])
        weighted.append(losses[-1])
    score = float(tf.add_n(weighted).numpy())
    return score if np.isfinite(score) else float("inf")


def _stage_inputs(stage, sample, params):
    logk, z, y, logr, lambda3, logxi = sample
    ones = tf.ones_like(y)
    if stage == "PostDamagePostTech":
        columns = [
            logk,
            z,
            y,
            lambda3,
            float(params["A_g_prime_prime"]) * ones,
            logxi,
            logxi,
        ]
    elif stage in {"PostDamageIntermTech", "PostDamagePreTech"}:
        columns = [logk, z, y, logr, lambda3, logxi, logxi, logxi]
    elif stage == "PreDamagePostTech":
        columns = [
            logk,
            z,
            y,
            float(params["A_g_prime_prime"]) * ones,
            logxi,
            logxi,
        ]
    elif stage in {"PreDamageIntermTech", "PreDamagePreTech"}:
        columns = [logk, z, y, logr, logxi, logxi, logxi]
    else:
        raise ValueError(f"Unknown regime for DGM distillation: {stage}")
    return tf.concat(columns, axis=1)


def _baseline_subnet_class():
    root = Path(__file__).resolve().parents[1]
    path = root / "models" / "feedforward_subnet.py"
    spec = importlib.util.spec_from_file_location("baseline_feedforward_subnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeedForwardSubNet


def distill_from_baseline(model, stage, n_inputs):
    """Initialize DGM outputs from a baseline checkpoint using sampled states."""
    teacher_folder = os.environ.get("DGM_TEACHER_FOLDER", "").strip()
    steps = int(os.environ.get("DGM_DISTILL_STEPS", "0"))
    if not teacher_folder or steps <= 0:
        return

    teacher_width = int(os.environ.get("DGM_TEACHER_WIDTH", "32"))
    teacher_layers = int(os.environ.get("DGM_TEACHER_LAYERS", "4"))
    batch_size = int(os.environ.get("DGM_DISTILL_BATCH_SIZE", "512"))
    learning_rate = float(os.environ.get("DGM_DISTILL_LR", "1e-3"))
    log_frequency = max(1, int(os.environ.get("DGM_DISTILL_LOG_FREQUENCY", "1000")))
    baseline_class = _baseline_subnet_class()

    student_names = ["v", "i_g", "i_d"]
    if hasattr(model, "i_r_nn"):
        student_names.append("i_r")

    teachers = []
    students = []
    for name in student_names:
        config = dict(model.params[f"{name}_nn_config"])
        config["num_hiddens"] = [teacher_width] * teacher_layers
        teacher = baseline_class(config)
        teacher.build((None, n_inputs))
        checkpoint = (
            Path(teacher_folder)
            / stage
            / f"{name}_nn_checkpoint_{stage}"
        )
        teacher.load_weights(str(checkpoint))
        teacher.trainable = False
        teachers.append(teacher)
        students.append(getattr(model, f"{name}_nn"))

    variables = [
        variable
        for student in students
        for variable in student.trainable_variables
    ]
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        global_clipnorm=float(model.params.get("gradient_clip_norm", 1.0)),
    )

    @tf.function
    def distill_step(x):
        targets = [
            tf.stop_gradient(teacher(x, training=False))
            for teacher in teachers
        ]
        with tf.GradientTape() as tape:
            predictions = [student(x, training=True) for student in students]
            losses = []
            for prediction, target in zip(predictions, targets):
                scale = tf.reduce_mean(tf.square(target)) + 1e-6
                losses.append(tf.reduce_mean(tf.square(prediction - target)) / scale)
            loss = tf.add_n(losses)
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    history = []
    for step in range(steps):
        x = _stage_inputs(stage, model.sample(batch_size=batch_size), model.params)
        loss = distill_step(x)
        if step % log_frequency == 0 or step == steps - 1:
            value = float(loss.numpy())
            history.append((step, value))
            print(f"DGM distillation {stage}: step={step}, normalized_mse={value:.6e}")

    export_folder = model.params.get("export_folder")
    if export_folder:
        np.savetxt(
            str(Path(export_folder) / "distillation_history.csv"),
            np.asarray(history),
            delimiter=",",
            header="step,normalized_mse",
            comments="",
            fmt=["%d", "%.8e"],
        )


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=0.0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.min_lr + (self.base_lr - self.min_lr) * (
            step / tf.maximum(1.0, self.warmup_steps)
        )
        progress = tf.clip_by_value(
            (step - self.warmup_steps)
            / tf.maximum(1.0, self.total_steps - self.warmup_steps),
            0.0,
            1.0,
        )
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def setup_optimizers(params):
    """Create the two optimizers used by value and policy improvement."""
    learning_rates = params.get("learning_rates", [1e-5, 1e-4])
    num_iterations = int(params.get("num_iterations", 100000))
    schedule_type = params.get("learning_rate_schedule_type", "warmup_cosine")
    gradient_clip_norm = float(params.get("gradient_clip_norm", 1.0))
    optimizer_type = params.get("optimizer_type", "adam").lower()

    schedules = []
    for learning_rate in learning_rates:
        learning_rate = float(learning_rate)
        if schedule_type == "warmup_cosine":
            schedules.append(
                WarmupCosine(
                    learning_rate,
                    num_iterations,
                    warmup_steps=max(1, num_iterations // 100),
                    min_lr=learning_rate * 0.01,
                )
            )
        elif schedule_type == "cosine":
            schedules.append(
                tf.keras.optimizers.schedules.CosineDecay(
                    learning_rate,
                    num_iterations,
                    alpha=0.01,
                )
            )
        elif schedule_type == "None":
            schedules.append(learning_rate)
        else:
            raise ValueError(f"Unsupported DGM learning-rate schedule: {schedule_type}")

    optimizer_kwargs = {}
    if gradient_clip_norm > 0:
        optimizer_kwargs["global_clipnorm"] = gradient_clip_norm

    def make_optimizer(schedule):
        if optimizer_type == "adamw" and hasattr(tf.keras.optimizers, "AdamW"):
            return tf.keras.optimizers.AdamW(
                learning_rate=schedule,
                weight_decay=float(params.get("weight_decay", 0.0)),
                **optimizer_kwargs,
            )
        return tf.keras.optimizers.Adam(
            learning_rate=schedule,
            **optimizer_kwargs,
        )

    params["optimizers"] = [make_optimizer(schedule) for schedule in schedules]
