import math

import numpy as np
import tensorflow as tf


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__(name = config["nn_name"] + ".init_layer")
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5),
                name = config["nn_name"] + ".bn." + str(_)
            )
            for _ in range(len(config["num_hiddens"]) + 1)]
        
        if config['activation'] is not None and "relu" in config['activation']:
            initializer = tf.keras.initializers.HeNormal(seed=0)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers = [tf.keras.layers.Dense(config["num_hiddens"][i],
                                                   use_bias=config['use_bias'],
                                                   activation=config['activation'],
                                                   kernel_initializer = initializer,
                                                   name = config["nn_name"] + ".dense." + str(i))
                             for i in range(len(config["num_hiddens"]))]
        # final output should be gradient of size dim
        try:
            if config['final_activation'] is None:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
            elif "relu" in config['final_activation']:
                initializer = tf.keras.initializers.HeNormal(seed=0)
            else:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
        except:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers.append(tf.keras.layers.Dense(config["dim"], 
        kernel_initializer = initializer, 
        activation=config['final_activation'], use_bias = True, name = config["nn_name"] + ".output" ))

    def call(self, x, training=False):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        x_inputs = []
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x_inputs.append(x)
        x = tf.add_n(x_inputs)
        x = self.dense_layers[-1](x)
        return x


def load_weights_cast(target_net, config, ckpt_path, n_inputs):
    """Warm-start a float64 net from a *float32* TF checkpoint, casting weights.

    TF checkpoint restore does NOT auto-cast: loading a float32 checkpoint into a
    float64 variable raises "expected dtype double does not equal original dtype
    float". So we restore into a temporary FLOAT32 net, pull the float32 numpy
    weights via get_weights(), and set_weights() them into the float64 target
    (which casts on assignment).  `target_net` must already be built in float64.
    """
    prev = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx("float32")
    try:
        tmp = FeedForwardSubNet(config)
        tmp.build((None, n_inputs))
        tmp.load_weights(ckpt_path).expect_partial()
        w32 = tmp.get_weights()
    finally:
        tf.keras.backend.set_floatx(prev)
    target_net.set_weights(w32)
    return target_net


def stratified_uniform(lower, upper, batch_size):
    """Draw a shuffled one-dimensional stratified sample."""
    # float64 sandbox: state samples must match the keras backend floatx so the
    # NN inputs (and every downstream tensor) are float64.  In production this
    # resolves to float32; here set_floatx("float64") makes it float64.
    _dtype = tf.keras.backend.floatx()
    batch_size = tf.cast(batch_size, tf.int32)
    edges = tf.linspace(
        tf.cast(lower, _dtype),
        tf.cast(upper, _dtype),
        batch_size + 1,
    )
    offsets = tf.random.uniform(tf.stack([batch_size, 1]), 0.0, 1.0, dtype=_dtype)
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
        stratified_uniform(params.get("λ3_min", 0.0), params.get("λ3_max", 1.0 / 3.0), n),
        stratified_uniform(params.get("logξ_min", -3.0), params.get("logξ_max", 5.0), n),
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
    """Return a finite scalar score that accounts for PDE and control residuals."""
    losses = list(losses)
    if not losses:
        return float("inf")
    weighted = [losses[0]]
    if len(losses) > 2:
        weighted.extend(control_weight * loss for loss in losses[1:-1])
        weighted.append(losses[-1])
    score = float(tf.add_n(weighted).numpy())
    return score if np.isfinite(score) else float("inf")

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=0.0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        if self.warmup_steps > 0:
            warmup_lr = self.min_lr + (self.base_lr - self.min_lr) * (step / tf.maximum(1.0, self.warmup_steps))
        else:
            warmup_lr = self.base_lr

        # cosine phase starts at warmup_steps
        progress = tf.clip_by_value((step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps), 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {"base_lr": self.base_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "min_lr": self.min_lr}

class CyclicalLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Triangular CLR (Smith 2017)."""
    def __init__(self, min_lr, max_lr, step_size):
        super().__init__()
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.step_size = float(step_size)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        cycle = tf.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        scale = tf.maximum(0.0, 1 - x)
        return self.min_lr + (self.max_lr - self.min_lr) * scale

    def get_config(self):
        return {"min_lr": self.min_lr, "max_lr": self.max_lr, "step_size": self.step_size}

class OneCycle(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    One-Cycle LR: warmup to max_lr, then cosine down to min_lr.
    phases:
      - pct_up: fraction of total steps spent warming up.
    """
    def __init__(self, max_lr, total_steps, pct_up=0.3, min_lr_ratio=1e-2):
        super().__init__()
        self.max_lr = float(max_lr)
        self.total_steps = int(total_steps)
        self.pct_up = float(pct_up)
        self.min_lr = self.max_lr * float(min_lr_ratio)
        self.up_steps = int(round(self.total_steps * self.pct_up))
        self.down_steps = max(1, self.total_steps - self.up_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # phase 1: linear warmup 0 -> max_lr
        lr_up = (self.max_lr) * (step / tf.maximum(1.0, self.up_steps))
        # phase 2: cosine from max_lr -> min_lr
        t = tf.clip_by_value((step - self.up_steps) / tf.maximum(1.0, self.down_steps), 0.0, 1.0)
        lr_down = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + tf.cos(math.pi * t))
        return tf.where(step <= self.up_steps, lr_up, lr_down)

    def get_config(self):
        return {"max_lr": self.max_lr, "total_steps": self.total_steps,
                "pct_up": self.pct_up, "min_lr_ratio": self.min_lr/self.max_lr}










def setup_optimizers(params):
    """
    Adds many schedule options.
    params:
      - learning_rates: list of starting/base LRs (floats)
      - num_iterations: total steps
      - learning_rate_schedule_type: one of
          'None', 'piecewiseconstant', 'sgd+piecewiseconstant', 'sgd',
          'cosine', 'cosine_restarts', 'exp_decay', 'poly_decay',
          'inverse_time', 'warmup_cosine', 'cyclical', 'onecycle'
      - extra (dict): optional knobs depending on schedule:
          - for cosine: {'alpha': 0.0}  # final LR fraction of base
          - for cosine_restarts: {'first_decay_steps': 2000, 't_mul': 2.0, 'm_mul': 1.0, 'alpha': 0.0}
          - for exp_decay: {'decay_steps': 2000, 'decay_rate': 0.96, 'staircase': True}
          - for poly_decay: {'decay_steps': 20000, 'end_lr': 0.0, 'power': 1.0}
          - for inverse_time: {'decay_steps': 1000, 'decay_rate': 0.5, 'staircase': False}
          - for warmup_cosine: {'warmup_steps': 1000, 'min_lr': 0.0}
          - for cyclical: {'min_lr': 1e-5, 'max_lr': 1e-3, 'step_size': 1000}
          - for onecycle: {'max_lr': 1e-3, 'pct_up': 0.3, 'min_lr_ratio': 1e-2}
      - optimizer_type: 'adam' (default), 'adamw', 'sgd'
      - weight_decay: float (for AdamW, optional)
    """
    learning_rates = params.get("learning_rates", [1e-3])
    num_iterations = int(params.get("num_iterations", 100000))
    lr_type = params.get("learning_rate_schedule_type", "cosine")
    extra = params.get("extra", {}) or {}
    opt_type = params.get("optimizer_type", "adam").lower()
    weight_decay = float(params.get("weight_decay", 0.0))
    gradient_clip_norm = float(params.get("gradient_clip_norm", 0.0))

    schedulers = []
    for lr in learning_rates:
        if lr_type == "None":
            schedulers.append(float(lr))

        elif lr_type == "piecewiseconstant":
            import numpy as np
            boundaries = [int(round(x)) for x in np.linspace(0, num_iterations, 20)][1:-1]
            values = [float(lr) / (4 ** x) for x in range(len(boundaries) + 1)]
            schedulers.append(tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values))

        elif lr_type == "sgd+piecewiseconstant":
            import numpy as np
            boundaries = [int(round(x)) for x in np.linspace(0, num_iterations, 5)][1:-1]
            values = [float(lr) / (2 ** x) for x in range(len(boundaries) + 1)]
            schedulers.append(tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values))

        elif lr_type == "sgd":
            schedulers.append(float(lr))

        elif lr_type == "cosine":
            alpha = float(extra.get("alpha", 0.0))  # final lr fraction
            schedulers.append(tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=float(lr),
                                                                        decay_steps=num_iterations,
                                                                        alpha=alpha))

        elif lr_type == "cosine_restarts":
            schedulers.append(tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=float(lr),
                first_decay_steps=int(extra.get("first_decay_steps", max(1, num_iterations // 10))),
                t_mul=float(extra.get("t_mul", 2.0)),
                m_mul=float(extra.get("m_mul", 1.0)),
                alpha=float(extra.get("alpha", 0.0))
            ))

        elif lr_type == "exp_decay":
            schedulers.append(tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=float(lr),
                decay_steps=int(extra.get("decay_steps", max(1, num_iterations // 50))),
                decay_rate=float(extra.get("decay_rate", 0.96)),
                staircase=bool(extra.get("staircase", True))
            ))

        elif lr_type == "inverse_time":
            schedulers.append(tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=float(lr),
                decay_steps=int(extra.get("decay_steps", max(1, num_iterations // 50))),
                decay_rate=float(extra.get("decay_rate", 1.0)),
                staircase=bool(extra.get("staircase", False))
            ))

        elif lr_type == "poly_decay":
            schedulers.append(tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=float(lr),
                decay_steps=int(extra.get("decay_steps", num_iterations)),
                end_learning_rate=float(extra.get("end_lr", 0.0)),
                power=float(extra.get("power", 1.0))
            ))

        elif lr_type == "warmup_cosine":
            schedulers.append(WarmupCosine(
                base_lr=float(lr),
                total_steps=num_iterations,
                warmup_steps=int(extra.get("warmup_steps", max(1, num_iterations // 100))),
                min_lr=float(extra.get("min_lr", 0.0))
            ))

        elif lr_type == "cyclical":
            schedulers.append(CyclicalLR(
                min_lr=float(extra.get("min_lr", lr * 0.1)),
                max_lr=float(extra.get("max_lr", lr)),
                step_size=float(extra.get("step_size", max(1, num_iterations // 8)))
            ))

        elif lr_type == "onecycle":
            schedulers.append(OneCycle(
                max_lr=float(extra.get("max_lr", lr)),
                total_steps=num_iterations,
                pct_up=float(extra.get("pct_up", 0.3)),
                min_lr_ratio=float(extra.get("min_lr_ratio", 1e-2))
            ))

        else:
            # Fallback
            schedulers.append(float(lr))

    # Choose optimizer
    def make_opt(sched):
        optimizer_kwargs = {"learning_rate": sched}
        if gradient_clip_norm > 0:
            optimizer_kwargs["global_clipnorm"] = gradient_clip_norm
        if opt_type == "sgd":
            return tf.keras.optimizers.legacy.SGD(
                momentum=0.9,
                nesterov=True,
                **optimizer_kwargs,
            )
        elif opt_type == "adamw":
            try:
                return tf.keras.optimizers.AdamW(
                    weight_decay=weight_decay,
                    **optimizer_kwargs,
                )
            except AttributeError:
                # Older TF: fall back to Adam
                return tf.keras.optimizers.Adam(**optimizer_kwargs)
        else:
            return tf.keras.optimizers.Adam(**optimizer_kwargs)

    params["optimizers"] = [make_opt(s) for s in schedulers]
