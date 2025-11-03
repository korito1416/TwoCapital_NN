import tensorflow as tf
import numpy as np


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

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        x_inputs = []
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x_inputs.append(x)
        x = tf.keras.layers.Add()(x_inputs)
        x = self.dense_layers[-1](x)
        return x


import tensorflow as tf
import math

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
        if opt_type == "sgd":
            return tf.keras.optimizers.legacy.SGD(learning_rate=sched, momentum=0.9, nesterov=True)
        elif opt_type == "adamw":
            try:
                return tf.keras.optimizers.AdamW(learning_rate=sched, weight_decay=weight_decay)
            except AttributeError:
                # Older TF: fall back to Adam
                return tf.keras.optimizers.Adam(learning_rate=sched)
        else:
            return tf.keras.optimizers.Adam(learning_rate=sched)

    params["optimizers"] = [make_opt(s) for s in schedulers]