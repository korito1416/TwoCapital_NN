"""Value-function parameterizations for the redesign.

Three options, selected by config (all reduce to production behaviour by default):

  plain      v = phi(x)                                  [production]
  recenter   v = phi(x) - phi(x0) + v0                    [level pinned BY CONSTRUCTION]
  separable  v = A(logK) + W(Z, Y, u)  (+ re-centering)   [generalized detrend]

WHY "recenter" AND NOT COSTATE.  The level of v is a single scalar degree of freedom per regime
(proved analytically: differentiating the HJB annihilates the bare level, so the costates satisfy a
closed level-free system).  Two ways to pin it:
  * costate/EGM — parameterize p = grad v and line-integrate from one anchor.  TESTED AND REJECTED
    in this project: it parameterizes the gradient as an UNCONSTRAINED VECTOR FIELD, which admits a
    non-physical curl component, so the soft curl penalty is never exact => path-dependent level and
    non-symmetric second derivatives.  Measured: level_spread 0.0077 and true HJB residual 2.98e-3.
  * re-centering — parameterize the POTENTIAL and subtract its value at an anchor state.  Exactly
    conservative by construction, no curl penalty, no extra heads.  Measured: level_spread 0.0051 and
    residual 2.05e-3 (i.e. FREE relative to plain 2.02e-3, because the level is a flat direction).
  => re-centering wins on both metrics.  Costate was the right INSIGHT, the wrong REALIZATION.

WHY "separable" AND NOT V = a*logK + W.  A constant-a detrend forces the joint scale elasticity
a = V_logK + V_logR to be constant.  MEASURED on the reference network, a falls 0.7735 -> 0.4532 as
logK goes 4.5 -> 7.0, while varying only 0.0012 across Z and 0.0021 across Y.  A constant-a fit would
therefore inject delta*logK*Delta_a ~ 2.2e-2, TEN TIMES the HJB residual it is meant to help.  But the
fact that a moves in logK ALONE is the signature of additive separability, so we let A be a learned
1-D function instead of forcing it linear:

    V = A(logK) + W(Z, Y, u)   =>   V_logK + V_logR = A'(logK)   [function of logK only]  ✓ matches data

This absorbs the delta*logK flow term, reduces a 4-D fit to 1-D + 3-D, and still leaves EXACTLY ONE
additive constant for the anchor to pin.  A'(logK) is itself economically meaningful: it is the
marginal value of SCALE, and its decline is the climate externality (a == 1 in a climate-free CRS
economy).

ANCHOR — IS IT ECONOMICALLY DEFENSIBLE?  Within a regime, YES and it imposes nothing: the FOCs depend
only on derivatives of V, so the level is pure GAUGE and anchoring merely selects a representative of
the equivalence class.  ACROSS regimes the level DIFFERENCE is economically real (it enters the jump
distortion g = exp(-(V^l - V)/xi)), so regimes must be CO-anchored (value-matched at the jump
boundary), never anchored independently.  The anchor TARGET is externally validated: an independent
FD solve gives V(xi=0.05) = 3.669, matching the re-centered network's ~3.64.
HONEST CAVEAT: pinned != pinned-correct.  Earlier co-anchoring runs pinned the gap CONSISTENTLY but
some seeds landed on the INADMISSIBLE side, so admissibility must be reported MIN-over-seeds.
"""
import tensorflow as tf

import config
from feedforward_subnet import FeedForwardSubNet


class ValueFunction(tf.keras.Model):
    """Wraps the raw subnet(s) with the selected level/structure parameterization."""

    def __init__(self, nn_config, anchor_input, n_inputs, logk_slot=0, name_suffix=""):
        super().__init__(name=(nn_config["nn_name"] + ".wrapper" + name_suffix))
        self.mode = "plain"
        if config.DETREND:
            self.mode = "separable"
        elif config.ANCHOR == "recenter":
            self.mode = "recenter"
        self.logk_slot = logk_slot
        self.anchor_input = anchor_input           # [1, n_inputs] tensor: the state x0
        self.v0 = tf.constant(config.ANCHOR_V0, dtype=tf.float32)

        self.phi = FeedForwardSubNet(nn_config)
        self.phi.build((None, n_inputs))

        if self.mode == "separable":
            # A(logK): a small 1-D network. Deliberately narrow -- it represents a smooth,
            # monotone-ish scalar profile, not a high-dimensional surface.
            a_cfg = dict(nn_config)
            a_cfg["nn_name"] = nn_config["nn_name"] + "_A"
            a_cfg["num_hiddens"] = [16, 16]
            a_cfg["final_activation"] = None        # A is unbounded (can be negative)
            self.A = FeedForwardSubNet(a_cfg)
            self.A.build((None, 1))
        else:
            self.A = None

    def call(self, x, training=False):
        if self.mode == "plain":
            return self.phi(x, training=training)

        if self.mode == "recenter":
            # v(x) = phi(x) - phi(x0) + v0  -> level pinned, gradients untouched
            anchor = tf.stop_gradient(self.anchor_input)
            return self.phi(x, training=training) - self.phi(anchor, training=training) + self.v0

        # separable: v = A(logK) + W(rest), re-centered so the single additive constant is pinned
        logk = tf.reshape(x[:, self.logk_slot], [-1, 1])
        a_part = self.A(logk, training=training)
        w_part = self.phi(x, training=training)
        anchor = tf.stop_gradient(self.anchor_input)
        anchor_logk = tf.reshape(anchor[:, self.logk_slot], [-1, 1])
        base = self.A(anchor_logk, training=training) + self.phi(anchor, training=training)
        return a_part + w_part - base + self.v0

    @property
    def trainable_variables(self):
        v = list(self.phi.trainable_variables)
        if self.A is not None:
            v += list(self.A.trainable_variables)
        return v

    def scale_elasticity(self, logk):
        """A'(logK) = the marginal value of scale = V_logK + V_logR (separable mode only).

        Economically: how much welfare rises when the whole economy (capital AND knowledge) is
        scaled up by one log unit.  == 1 in a climate-free CRS economy; the measured decline
        (0.77 -> 0.45 over the box) is the climate externality.
        """
        if self.A is None:
            return None
        logk = tf.reshape(logk, [-1, 1])
        with tf.GradientTape() as tape:
            tape.watch(logk)
            a = self.A(logk, training=False)
        return tape.gradient(a, logk)
