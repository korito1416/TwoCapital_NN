"""Unified input layout for the four one-jump regimes.

THE PROBLEM (measured in production `models/`):

    regime                slots fed to every net                                   width
    PreDamagePreTech      [logK, Z, Y, logR,        logxi, logxi, logxi]             7
    PostDamagePreTech     [logK, Z, Y, logR, lam3,  logxi, logxi, logxi]             8
    PreDamagePostTech     [logK, Z, Y, A_g'',       logxi, logxi]                    6
    PostDamagePostTech    [logK, Z, Y, lam3, A_g'', logxi, logxi]                    7

Two of the differences are REAL economics and must be kept:
  * logR is absent post-tech      (R&D is switched off once the breakthrough has happened)
  * lam3  is absent pre-damage    (the damage curvature has not been revealed yet)
Two are historical accidents and are removed here:
  * logxi is DUPLICATED 2-3x — pure width padding to match legacy checkpoint shapes.  A network
    cannot gain expressivity from an exactly repeated input; it only wastes first-layer weights.
  * A_g'' is fed as a CONSTANT column (a dead input: it never varies within a regime).

UNIFIED LAYOUT (this module), one canonical ordering for all four regimes:

        [ logK , Z , Y , u , lam3 , xi_input ]                              width 6

  u        = logR - logK, knowledge-to-capital (0 in post-tech regimes, where R&D is off)
  lam3     = damage curvature (0 in pre-damage regimes, where it is not yet revealed)
  xi_input = the uncertainty pseudo-state (theta/theta_max, or logxi in legacy mode)

Absent slots carry an explicit sentinel 0.0 AND are identified by the regime one-hot, so the
network can tell "absent" from "genuinely zero".  Set REDESIGN_REGIME_ONEHOT=1 to append the
one-hot (width 6+4=10) when training a SHARED network across regimes.

!!! WARM-START COMPATIBILITY — LOAD-BEARING !!!
Changing the input width/order changes the first-layer kernel shape, so NONE of the existing
checkpoints can be loaded into a unified-input network.  This project's good solution is
INHERITED through warm start, so switching to unified inputs means either
  (a) training from scratch (loses the inherited basin), or
  (b) first-layer surgery: build the new net, then copy each old column's weight row into its
      new position and SUM the three duplicated logxi rows into the single xi row
      (exactly equivalent at initialization, because the duplicated inputs were identical).
`remap_first_layer_kernel` implements (b), which is the low-risk path.
"""
import numpy as np
import tensorflow as tf

import config
import uncertainty

# canonical slot order
SLOTS = ["logK", "Z", "Y", "u", "lam3", "xi"]
WIDTH = len(SLOTS)

REGIMES = ["PreDamagePreTech", "PostDamagePreTech", "PreDamagePostTech", "PostDamagePostTech"]
HAS_R = {"PreDamagePreTech": True, "PostDamagePreTech": True,
         "PreDamagePostTech": False, "PostDamagePostTech": False}
HAS_LAM3 = {"PreDamagePreTech": False, "PostDamagePreTech": True,
            "PreDamagePostTech": False, "PostDamagePostTech": True}

# legacy layouts, for the warm-start remap
LEGACY = {
    "PreDamagePreTech":   ["logK", "Z", "Y", "logR", "logxi", "logxi", "logxi"],
    "PostDamagePreTech":  ["logK", "Z", "Y", "logR", "lam3", "logxi", "logxi", "logxi"],
    "PreDamagePostTech":  ["logK", "Z", "Y", "Ag", "logxi", "logxi"],
    "PostDamagePostTech": ["logK", "Z", "Y", "lam3", "Ag", "logxi", "logxi"],
}


def build_input(regime, logK, Z, Y, logR=None, lam3=None, theta=None, logxi=None, onehot=False):
    """Assemble the unified input tensor for `regime`.

    Pass EITHER theta (when config.USE_THETA) or logxi.  Absent-by-regime slots are filled with
    0.0 sentinels.  Returns a [batch, WIDTH] (or WIDTH+4 with one-hot) tensor.
    """
    zeros = tf.zeros_like(logK)
    u = (logR - logK) if (HAS_R[regime] and logR is not None) else zeros
    l3 = lam3 if (HAS_LAM3[regime] and lam3 is not None) else zeros
    if config.USE_THETA:
        xi_col = uncertainty.network_input(theta)
    else:
        xi_col = logxi
    cols = [logK, Z, Y, u, l3, xi_col]
    if onehot:
        idx = REGIMES.index(regime)
        for k in range(len(REGIMES)):
            cols.append(tf.ones_like(logK) * (1.0 if k == idx else 0.0))
    return tf.concat(cols, 1)


def remap_first_layer_kernel(old_kernel, regime, theta_max=None):
    """Map a legacy first-layer kernel [old_width, units] onto the unified layout [WIDTH, units].

    Rules (exact at initialization):
      * logK, Z, Y, lam3 -> copied to their new slot.
      * logR -> the u slot.  NOTE this is an APPROXIMATION: the legacy net saw logR, the unified
        net sees u = logR - logK.  Since dV/dlogR is unchanged but dV/dlogK shifts by -dV/du, the
        logK row is corrected by ADDING the logR row (chain rule d/dlogK|_u = d/dlogK|_logR + d/dlogR).
      * the 2-3 duplicated logxi rows -> SUMMED into the single xi row (the duplicated inputs were
        numerically identical, so their contributions add).  If switching to theta the row is
        additionally rescaled by d(logxi)/d(xi_input), which is state-dependent; the remap therefore
        only makes sense when KEEPING logxi.  Switching parameterization needs retraining.
      * the dead A_g'' constant column -> DROPPED (it contributed a constant bias; folded into the
        layer bias by the caller if desired).
    Returns (new_kernel, dropped_constant_row) so the caller can fold the constant into the bias.
    """
    old = np.asarray(old_kernel)
    layout = LEGACY[regime]
    units = old.shape[1]
    new = np.zeros((WIDTH, units), dtype=old.dtype)
    slot = {name: i for i, name in enumerate(SLOTS)}

    logr_row = np.zeros(units, dtype=old.dtype)
    ag_row = np.zeros(units, dtype=old.dtype)
    for i, name in enumerate(layout):
        if name in ("logK", "Z", "Y", "lam3"):
            new[slot[name]] += old[i]
        elif name == "logR":
            logr_row += old[i]
        elif name == "logxi":
            new[slot["xi"]] += old[i]          # duplicated inputs -> sum
        elif name == "Ag":
            ag_row += old[i]                    # dead constant -> fold into bias
    if HAS_R[regime]:
        new[slot["u"]] += logr_row
        new[slot["logK"]] += logr_row           # chain rule: d/dlogK|_u = d/dlogK|_logR + d/dlogR
    return new, ag_row


def legacy_input(regime, logK, Z, Y, logR=None, lam3=None, logxi=None, A_g=None):
    """Reproduce the EXACT production input vector (used when REDESIGN_INPUTS=legacy)."""
    if regime == "PreDamagePreTech":
        return tf.concat([logK, Z, Y, logR, logxi, logxi, logxi], 1)
    if regime == "PostDamagePreTech":
        return tf.concat([logK, Z, Y, logR, lam3, logxi, logxi, logxi], 1)
    if regime == "PreDamagePostTech":
        return tf.concat([logK, Z, Y, A_g * tf.ones_like(Y), logxi, logxi], 1)
    if regime == "PostDamagePostTech":
        return tf.concat([logK, Z, Y, lam3, A_g * tf.ones_like(Y), logxi, logxi], 1)
    raise ValueError(regime)
