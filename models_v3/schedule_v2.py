"""
models_v2 schedule / warm-up designer.

Self-contained module (does NOT import or modify models/ or models_precond/).
Provides:
  - WarmupCosineV2          : warmup -> cosine -> floor, with separate value/control
                              warm-up lengths and an optional cosine-restart cycle.
  - build_optimizers_v2     : drop-in replacement for setup_optimizers() that builds
                              TWO optimizers (value, control) with DISTINCT schedules.
  - EarlyStopper            : patience-based early stop on validation_score with
                              best-weights restore (cooperates with the loop's
                              existing best_*_nn tracking).
  - lbfgs_polish_v2         : optional end-of-training L-BFGS-B polish on the
                              preconditioned combined loss over a fixed collocation
                              batch, ported from nn_dgm_shock.py.

Design rationale vs the old WarmupCosine (models/feedforward_subnet.py:121-143):
  * Old schedule used a SINGLE schedule type applied to BOTH nets via one
    `learning_rates` list, and a default warmup of num_iterations//100 (~0.5-1k
    steps for a 50-100k run) which is far too short to stabilise the
    ill-conditioned (kappa(J^T J)~7e6) value residual before the cosine ramp.
  * The control net is the WEAKLY-IDENTIFIED one (controls disagree 50-360% at the
    same low loss). It should warm up LONGER and decay to a higher floor than the
    value net, so it keeps moving after the value net has frozen its geometry.
    => separate per-net schedules (different base_lr, warmup, floor).
  * total_steps follows num_iterations (matching the old auto-follow behaviour at
    feedforward_subnet.py:292) so a shorter TRANSFER budget auto-shortens the ramp.
  * Optional cosine-restart cycle lets the control net periodically re-anneal to
    escape the flat weakly-identified directions without restarting the value net.
"""

import math

import numpy as np
import tensorflow as tf


# --------------------------------------------------------------------------- #
#  LR schedule
# --------------------------------------------------------------------------- #
class WarmupCosineV2(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up -> cosine decay -> constant floor, with optional restarts.

    Args:
      base_lr:        peak LR reached at end of warm-up.
      total_steps:    total Adam steps (== num_iterations; ramp auto-follows it).
      warmup_steps:   linear ramp 0..base_lr over these steps.
      min_lr:         floor LR held after the cosine completes (>0 keeps a net
                      that is still mis-identified moving; recommended for control).
      restart_period: if >0, cosine ANNEALS within windows of this many steps and
                      jumps back up to (restart_decay * peak) at each window edge
                      (Loshchilov-Hutter SGDR). 0 => single monotone cosine.
      restart_decay:  multiplicative peak decay per restart cycle (<=1).
      warmup_floor:   LR at step 0 of the linear ramp (small but nonzero avoids a
                      dead first step under BatchNorm-free raw inputs).
    """

    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=0.0,
                 restart_period=0, restart_decay=0.7, warmup_floor=None):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.restart_period = int(restart_period)
        self.restart_decay = float(restart_decay)
        self.warmup_floor = float(warmup_floor) if warmup_floor is not None \
            else max(self.min_lr, 0.02 * self.base_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        ws = tf.maximum(1.0, float(self.warmup_steps))

        # phase 1: linear warm-up  warmup_floor -> base_lr
        warmup_lr = self.warmup_floor + (self.base_lr - self.warmup_floor) * (step / ws)

        # phase 2: cosine (optionally restarting) over [warmup_steps, total_steps]
        post = tf.maximum(0.0, step - self.warmup_steps)
        decay_span = tf.maximum(1.0, float(self.total_steps - self.warmup_steps))

        if self.restart_period > 0:
            period = float(self.restart_period)
            cycle = tf.floor(post / period)
            t = (post - cycle * period) / period               # 0..1 within cycle
            peak = self.base_lr * tf.pow(self.restart_decay, cycle)
            peak = tf.maximum(peak, self.min_lr)
            cosine_lr = self.min_lr + 0.5 * (peak - self.min_lr) * (1.0 + tf.cos(math.pi * t))
        else:
            t = tf.clip_by_value(post / decay_span, 0.0, 1.0)
            cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * t))

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {"base_lr": self.base_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "min_lr": self.min_lr,
                "restart_period": self.restart_period,
                "restart_decay": self.restart_decay,
                "warmup_floor": self.warmup_floor}


# --------------------------------------------------------------------------- #
#  Optimizer builder (drop-in replacement for setup_optimizers)
# --------------------------------------------------------------------------- #
def build_optimizers_v2(params):
    """Build optimizers[0]=value, optimizers[1]=control with DISTINCT schedules.

    Reads from params (with v2 defaults; everything overridable):
      learning_rates:        [lr_value, lr_control]  (peak LRs)
      num_iterations:        total Adam steps (== total_steps for the cosine)
      phase:                 "base" (from scratch) or "transfer" (fine-tune).
                             Sets default warm-up FRACTIONS if not given explicitly.
      gradient_clip_norm:    global clipnorm (default 1.0; keep the existing guard).
      schedule_v2 (dict), all optional:
        warmup_frac_value    default 0.05 (base) / 0.02 (transfer)
        warmup_frac_control  default 0.10 (base) / 0.04 (transfer)  (longer: weak id)
        min_lr_value         default 1e-6
        min_lr_control       default 2e-5   (higher floor keeps controls moving)
        restart_period_control default 0 (base) ; set e.g. num_iter//4 to re-anneal
        restart_decay_control  default 0.7

    Writes params["optimizers"] = [value_opt, control_opt] (same contract as the
    regime train loop, which applies [0] to v_nn and [1] to i_g_nn+i_d_nn).
    """
    lrs = list(params.get("learning_rates", [1e-4, 4e-3]))
    if len(lrs) == 1:
        lrs = [lrs[0], lrs[0]]
    lr_v, lr_c = float(lrs[0]), float(lrs[1])

    num_iter = int(params.get("num_iterations", 100000))
    phase = str(params.get("phase", "base")).lower()
    clip = float(params.get("gradient_clip_norm", 1.0))
    cfg = dict(params.get("schedule_v2", {}) or {})

    if phase == "transfer":
        wf_v_def, wf_c_def = 0.02, 0.04
    else:  # base / from-scratch
        wf_v_def, wf_c_def = 0.05, 0.10

    wf_v = float(cfg.get("warmup_frac_value", wf_v_def))
    wf_c = float(cfg.get("warmup_frac_control", wf_c_def))
    warm_v = max(1, int(round(num_iter * wf_v)))
    warm_c = max(1, int(round(num_iter * wf_c)))

    min_lr_v = float(cfg.get("min_lr_value", 1e-6))
    min_lr_c = float(cfg.get("min_lr_control", 2e-5))
    rp_c = int(cfg.get("restart_period_control", 0))
    rd_c = float(cfg.get("restart_decay_control", 0.7))

    sched_v = WarmupCosineV2(base_lr=lr_v, total_steps=num_iter,
                             warmup_steps=warm_v, min_lr=min_lr_v)
    sched_c = WarmupCosineV2(base_lr=lr_c, total_steps=num_iter,
                             warmup_steps=warm_c, min_lr=min_lr_c,
                             restart_period=rp_c, restart_decay=rd_c)

    def make_opt(sched):
        kw = {"learning_rate": sched}
        if clip > 0:
            kw["global_clipnorm"] = clip
        return tf.keras.optimizers.Adam(**kw)

    params["optimizers"] = [make_opt(sched_v), make_opt(sched_c)]
    # keep the loop's tensorboard branch happy (it checks this string)
    params.setdefault("learning_rate_schedule_type", "warmup_cosine_v2")
    return params["optimizers"]


# --------------------------------------------------------------------------- #
#  Patience-based early stopping with best-weights restore
# --------------------------------------------------------------------------- #
class EarlyStopper:
    """Patience-based early stop on validation_score, restoring best weights.

    Cooperates with the regime loop's existing best_*_nn tracking: call update()
    at every validation point. It mirrors best weights into the provided
    best_nets, decides when to stop, and at the end restore() copies them back.

    Args:
      nets:       list of live nets        [v_nn, i_g_nn, i_d_nn]
      best_nets:  list of shadow best nets [best_v_nn, best_i_g_nn, best_i_d_nn]
      patience:   #validation points without rel-improvement before stopping.
      min_delta:  relative improvement required to reset patience (default 1e-4).
      warmup_evals: ignore the first N validation points (LR still ramping; the
                    score is not yet meaningful) -- do not count toward patience.
    """

    def __init__(self, nets, best_nets, patience=20, min_delta=1e-4,
                 warmup_evals=5):
        self.nets = list(nets)
        self.best_nets = list(best_nets)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_evals = int(warmup_evals)
        self.best = float("inf")
        self.bad = 0
        self.n_eval = 0
        self.best_step = -1

    def update(self, score, step):
        """Returns True if training should stop."""
        self.n_eval += 1
        if not np.isfinite(score):
            # restore last good weights and stop
            return True
        improved = score < self.best * (1.0 - self.min_delta)
        if improved or self.best == float("inf"):
            self.best = score
            self.best_step = step
            for live, shadow in zip(self.nets, self.best_nets):
                shadow.set_weights(live.get_weights())
            self.bad = 0
        elif self.n_eval > self.warmup_evals:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self):
        for live, shadow in zip(self.nets, self.best_nets):
            live.set_weights(shadow.get_weights())


# --------------------------------------------------------------------------- #
#  Optional end-of-training L-BFGS-B polish (ported from nn_dgm_shock.py:154)
# --------------------------------------------------------------------------- #
def lbfgs_polish_v2(model, n_points=8192, maxiter=3000, precond=True):
    """L-BFGS-B polish of the COMBINED preconditioned loss over a FIXED batch.

    Optimises v_nn, i_g_nn, i_d_nn jointly with second-order curvature, which the
    two-step Adam loop cannot exploit. Runs ONCE after Adam; it does not touch the
    train_step graph, so it cannot break the two-step loop.

    Requirements on `model` (all already present in the regime classes):
      - model.v_nn, model.i_g_nn, model.i_d_nn
      - model.sample(batch_size=...) -> 6 state columns
      - model.objective_fn(*states, compute_control, training) returning, in eval
        mode (training=False), a tuple of scalar RMS residuals
        (loss_v, FOC_d, FOC_g, loss_dv_dY).

    The polish minimises loss_v + FOC_d + FOC_g + loss_dv_dY on a FIXED collocation
    batch (so the objective is deterministic, as L-BFGS requires). If the regime's
    objective_fn already applies the precond weight inside loss_v (it does in
    models_v2), set precond=True is a no-op flag kept for signature parity.
    """
    import scipy.optimize as so

    # v3: controls are closed-form, so only v_nn is trainable. Fall back to v_nn
    # alone when the regime has no control nets (semi-analytic regimes).
    vl = list(model.v_nn.trainable_variables)
    if hasattr(model, "i_g_nn"):
        vl += list(model.i_g_nn.trainable_variables)
    if hasattr(model, "i_d_nn"):
        vl += list(model.i_d_nn.trainable_variables)
    shapes = [v.shape for v in vl]
    sizes = [int(np.prod(s)) for s in shapes]

    # FIXED collocation batch (constants -> deterministic objective)
    states = [tf.constant(s) for s in model.sample(batch_size=n_points)]

    def setf(x):
        x = tf.constant(x, tf.float32)
        i = 0
        for v, s, sz in zip(vl, shapes, sizes):
            v.assign(tf.reshape(x[i:i + sz], s))
            i += sz

    @tf.function
    def loss_and_grad():
        with tf.GradientTape() as t:
            parts = model.objective_fn(*states, compute_control=False,
                                        training=False)
            # parts = (loss_v, FOC_d, FOC_g, loss_dv_dY) ; square the RMS terms so
            # the combined objective is smooth and on a comparable scale.
            loss = tf.add_n([tf.square(p) for p in parts])
        g = t.gradient(loss, vl)
        g = [tf.zeros_like(v) if gi is None else gi for gi, v in zip(g, vl)]
        return loss, tf.concat([tf.reshape(gi, [-1]) for gi in g], 0)

    def func(x):
        setf(x)
        loss, g = loss_and_grad()
        return float(loss.numpy()), g.numpy().astype(np.float64)

    x0 = tf.concat([tf.reshape(v, [-1]) for v in vl], 0).numpy().astype(np.float64)
    res = so.minimize(func, x0, jac=True, method="L-BFGS-B",
                      options={"maxiter": int(maxiter), "maxfun": int(maxiter) * 2,
                               "ftol": 1e-16, "gtol": 1e-12})
    setf(res.x)
    return res
