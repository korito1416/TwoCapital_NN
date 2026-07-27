import math

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Per-regime input column bounds.
# ---------------------------------------------------------------------------
# The net input X is built per-regime with a different number/order of columns
# (see each regime's `pde_rhs`).  To normalize cleanly we need the (min, max)
# of EACH column, in the SAME order the regime concatenates them.
#
# State sampling bounds (kept in sync with params.py):
#   logK    in [4, 7]      Z   in [0.01, 0.99]   Y      in [0, 4]
#   logR    in [1, 6]      lambda3 in [0, 1/3]   logxi  in [-3, 5]
#   A_g_const : a CONSTANT column (A_g_prime_prime * ones) -> maps to 0 exactly.
#
# Duplicated logxi columns (some regimes pass logxi 2-3 times) are kept so the
# input width matches each regime's existing n_inputs / checkpoint width.  Each
# copy is normalized identically; duplicates are redundant but harmless.
_LOGK    = (4.0, 7.0)
_Z       = (0.01, 0.99)
_Y       = (0.0, 4.0)
_LOGR    = (1.0, 6.0)
_LAMBDA3 = (0.0, 1.0 / 3.0)
_LOGXI   = (-3.0, 5.0)
# Sentinel for a constant column.  min==max signals the normalizer to emit 0.
_CONST   = (0.0, 0.0)


def regime_input_bounds(regime_name):
    """Return the ordered list of (min, max) column bounds for a regime.

    `regime_name` must be one of the six regime tags.  The order MUST match the
    `tf.concat([...], 1)` that builds X inside that regime's pde_rhs.
    """
    table = {
        # PostDamagePostTech: X = [logK, Z, Y, lambda3, A_g_const, logxi, logxi]  (7)
        "PostDamagePostTech": [_LOGK, _Z, _Y, _LAMBDA3, _CONST, _LOGXI, _LOGXI],
        # PreDamagePostTech:  X = [logK, Z, Y, A_g_const, logxi, logxi]           (6)
        "PreDamagePostTech": [_LOGK, _Z, _Y, _CONST, _LOGXI, _LOGXI],
        # PostDamagePreTech:  X = [logK, Z, Y, logR, lambda3, logxi, logxi, logxi] (8)
        "PostDamagePreTech": [_LOGK, _Z, _Y, _LOGR, _LAMBDA3, _LOGXI, _LOGXI, _LOGXI],
        # PreDamagePreTech:   X = [logK, Z, Y, logR, logxi, logxi, logxi]          (7)
        "PreDamagePreTech": [_LOGK, _Z, _Y, _LOGR, _LOGXI, _LOGXI, _LOGXI],
        # PostDamageIntermTech: X = [logK, Z, Y, logR, lambda3, logxi, logxi, logxi] (8)
        "PostDamageIntermTech": [_LOGK, _Z, _Y, _LOGR, _LAMBDA3, _LOGXI, _LOGXI, _LOGXI],
        # PreDamageIntermTech:  X = [logK, Z, Y, logR, logxi, logxi, logxi]          (7)
        "PreDamageIntermTech": [_LOGK, _Z, _Y, _LOGR, _LOGXI, _LOGXI, _LOGXI],
    }
    if regime_name not in table:
        raise KeyError(
            f"Unknown regime '{regime_name}'. Known: {sorted(table)}"
        )
    return table[regime_name]


# ---------------------------------------------------------------------------
# Fixed (non-trainable) input normalization to ~[-1, 1].
# ---------------------------------------------------------------------------
class InputNormalization(tf.keras.layers.Layer):
    """Affine map column j: x -> 2*(x - min_j)/(max_j - min_j) - 1, FIXED.

    - NON-trainable; constants stored as buffers.  Its Jacobian is a constant
      diagonal, so it does not distort the PINN's autodiff 1st/2nd derivatives
      (those are taken w.r.t. the RAW state tensors logK, Z, ... in pde_rhs,
      not w.r.t. the normalized features); it only improves the conditioning
      the optimizer sees.
    - Constant columns (min==max) are mapped to 0.0 exactly.
    """

    def __init__(self, input_bounds, name=None):
        super().__init__(name=name, trainable=False)
        lows = np.array([b[0] for b in input_bounds], dtype=np.float32)
        highs = np.array([b[1] for b in input_bounds], dtype=np.float32)
        spans = highs - lows
        const_mask = spans <= 0.0
        safe_span = np.where(const_mask, 1.0, spans).astype(np.float32)
        # x_norm = scale * x + shift ; scale = 2/span, shift = -(2*low/span + 1)
        scale = np.where(const_mask, 0.0, 2.0 / safe_span).astype(np.float32)
        shift = np.where(
            const_mask, 0.0, -(2.0 * lows / safe_span + 1.0)
        ).astype(np.float32)
        self._scale = tf.constant(scale.reshape(1, -1), dtype=tf.float32)
        self._shift = tf.constant(shift.reshape(1, -1), dtype=tf.float32)
        self._n_inputs = int(len(input_bounds))

    def call(self, x):
        return x * self._scale + self._shift


# ---------------------------------------------------------------------------
# The v2 net: clean residual-MLP (no BatchNorm, fixed input normalization).
# ---------------------------------------------------------------------------
class FeedForwardSubNet(tf.keras.Model):
    """Clean residual-MLP for the DGM-PIA value / control nets.

    Design (benchmarks/model_training_review.md THEME 3):
      * BatchNorm REMOVED (it was dead: always inference mode, default stats,
        zero normalization). Replaced by a FIXED input normalization to ~[-1,1]
        from the per-column state bounds -- the actual conditioning fix.
      * Capacity is ample (a 32x4 MLP fits FD truth to ~1e-4), so a plain
        pre-activation residual MLP with a smooth activation, NOT Fourier /
        PirateNet.  Residual skips only help conditioning of deeper variants and
        reduce to a plain MLP at equal widths.
      * Smooth activation (swish default; tanh selectable) for C^2 value
        derivatives.  Controls keep the bounded investment_rate_activation(theta)
        as final_activation (passed in by the regime, unchanged).
      * Seeded initializers (config['seed'], default 0).

    Public API preserved: FeedForwardSubNet(config); call(x, training=False) ->
    (batch, dim).  Regime call sites (self.v_nn(X), ...) are unchanged.

    New OPTIONAL config keys (sane defaults; old configs still build):
      "input_bounds": list[(min,max)] length n_inputs, in X-column order.
                      If omitted -> identity normalization (back-compat).
      "residual":     bool, default True.
      "seed":         int, default 0.
    Existing keys used unchanged: "num_hiddens", "activation",
      "final_activation", "dim", "nn_name", "use_bias".
    """

    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__(
            name=config["nn_name"] + ".init_layer"
        )
        self.config = config
        seed = int(config.get("seed", 0))
        activation = config.get("activation", "swish")
        if activation is None:
            activation = "swish"
        self._activation_fn = tf.keras.activations.get(activation)
        self._use_residual = bool(config.get("residual", True))

        # ---- fixed input normalization ----
        input_bounds = config.get("input_bounds", None)
        if input_bounds is not None:
            self.input_norm = InputNormalization(
                input_bounds, name=config["nn_name"] + ".input_norm"
            )
        else:
            self.input_norm = None

        # ---- initializer (seeded) ----
        if isinstance(activation, str) and "relu" in activation:
            initializer = tf.keras.initializers.HeNormal(seed=seed)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        # ---- hidden layers (no BN; activation applied explicitly in call) ----
        self.dense_layers = [
            tf.keras.layers.Dense(
                config["num_hiddens"][i],
                use_bias=config.get("use_bias", True),
                activation=None,
                kernel_initializer=initializer,
                name=config["nn_name"] + ".dense." + str(i),
            )
            for i in range(len(config["num_hiddens"]))
        ]

        # ---- final (output) layer ----
        final_activation = config.get("final_activation", None)
        try:
            if final_activation is None:
                out_init = tf.keras.initializers.GlorotUniform(seed=seed)
            elif isinstance(final_activation, str) and "relu" in final_activation:
                out_init = tf.keras.initializers.HeNormal(seed=seed)
            else:
                out_init = tf.keras.initializers.GlorotUniform(seed=seed)
        except Exception:
            out_init = tf.keras.initializers.GlorotUniform(seed=seed)

        self.output_layer = tf.keras.layers.Dense(
            config["dim"],
            kernel_initializer=out_init,
            activation=final_activation,
            use_bias=True,
            name=config["nn_name"] + ".output",
        )

    def call(self, x, training=False):
        """norm -> [dense -> act (+ residual skip)] * L -> dense(out)."""
        if self.input_norm is not None:
            x = self.input_norm(x)
        h = None
        for layer in self.dense_layers:
            pre = layer(x if h is None else h)
            act = self._activation_fn(pre)
            if (
                self._use_residual
                and h is not None
                and act.shape[-1] == h.shape[-1]
            ):
                h = h + act
            else:
                h = act
        return self.output_layer(h)


# ---------------------------------------------------------------------------
# Collocation sampling + validation helpers (carried over unchanged).
# ---------------------------------------------------------------------------
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
    """Return a finite scalar score that accounts for PDE and control residuals.

    losses[0] is the RAW value-residual rmse; losses[1:-1] are FOC rmses
    (weighted by control_weight); losses[-1] is the dv/dY monotonicity rmse.
    Length-agnostic: works for the PostTech 4-tuple and the PreTech 5-tuple
    (which carries an extra FOC_r).
    """
    losses = list(losses)
    if not losses:
        return float("inf")
    weighted = [losses[0]]
    if len(losses) > 2:
        weighted.extend(control_weight * loss for loss in losses[1:-1])
        weighted.append(losses[-1])
    score = float(tf.add_n(weighted).numpy())
    return score if np.isfinite(score) else float("inf")


# ===========================================================================
# LEGACY single-net schedule -- kept only as a FALLBACK.  The v2 regimes build
# optimizers via schedule_v2.build_optimizers_v2 (separate value/control
# schedules).  Do NOT call setup_optimizers from the v2 regime files.
# ===========================================================================
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
            warmup_lr = self.min_lr + (self.base_lr - self.min_lr) * (
                step / tf.maximum(1.0, self.warmup_steps)
            )
        else:
            warmup_lr = self.base_lr
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
    """FALLBACK only (single schedule for all nets). v2 regimes use schedule_v2."""
    learning_rates = params.get("learning_rates", [1e-3])
    num_iterations = int(params.get("num_iterations", 100000))
    lr_type = params.get("learning_rate_schedule_type", "warmup_cosine")
    extra = params.get("extra", {}) or {}
    gradient_clip_norm = float(params.get("gradient_clip_norm", 0.0))

    schedulers = []
    for lr in learning_rates:
        if lr_type in ("None", "sgd"):
            schedulers.append(float(lr))
        elif lr_type == "cosine":
            schedulers.append(
                tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=float(lr),
                    decay_steps=num_iterations,
                    alpha=float(extra.get("alpha", 0.0)),
                )
            )
        elif lr_type == "warmup_cosine":
            schedulers.append(
                WarmupCosine(
                    base_lr=float(lr),
                    total_steps=num_iterations,
                    warmup_steps=int(
                        extra.get("warmup_steps", max(1, num_iterations // 50))
                    ),
                    min_lr=float(extra.get("min_lr", float(lr) * 1e-3)),
                )
            )
        else:
            schedulers.append(float(lr))

    def make_opt(sched):
        kw = {"learning_rate": sched}
        if gradient_clip_norm > 0:
            kw["global_clipnorm"] = gradient_clip_norm
        return tf.keras.optimizers.Adam(**kw)

    params["optimizers"] = [make_opt(s) for s in schedulers]
    return params["optimizers"]
