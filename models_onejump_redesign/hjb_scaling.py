"""Non-dimensionalization of the HJB residual.

WHAT THIS CAN AND CANNOT DO — state this honestly in any report.

CAN: equalize FIT QUALITY across the state box.  The natural size of the HJB varies enormously:
in the pre-damage region the equation lives at the delta-weighted log scale
(delta*logK = +0.068, -delta*V = -0.042, delta*log(C/K) = -0.031 at X0, so ~5e-2), whereas at high
(K, Y) the climate block (∝ K) and its second-order partner (∝ K^2) dominate and the natural scale is
orders larger.  An unnormalized L2 residual therefore implicitly weights high-K/high-Y states far more
than the economically-visited region.  Dividing by a natural scale s(x) makes the objective a RELATIVE
residual, uniform across the box.

CANNOT: fix the value-LEVEL identifiability.  Gauss-Newton, with any weight w(x) = 1/s(x)^2:

    H_level = delta^2 * sum_x w(x)          (because dResidual/dlevel = -delta pointwise)
    H_shape ~ O(1)   * sum_x w(x)

Both curvatures carry the SAME factor sum_x w(x), so the conditioning ratio
H_level : H_shape = delta^2 : O(1) ~ 1e-4 is INVARIANT to the choice of divisor.  No normalization,
preconditioner or reweighting can change it -- only ADDING a term that sees the level (an anchor,
cross-regime value-matching, or a structural parameterization) can.  This is why preconditioning was
already refuted in this project, and the same algebra explains why.

A THIRD, REAL EFFECT — RE-WEIGHTING AGAINST THE FOC TERMS.  The value network's loss is NOT the HJB
residual alone; it is
    RMS(rhs - pv) + RMS(FOC_g) + RMS(FOC_d) + RMS(FOC_r) + monotonicity
summed, and the value optimizer differentiates the whole sum.  Measured: RMS(HJB) ~ 2e-3 against
RMS(FOC) ~ 1e-4, so the HJB term already carries ~90% of the value loss.  Dividing the HJB residual by
s ~ 0.07 multiplies its weight by ~14x, pushing the ratio from ~20:1 to ~300:1 and effectively
switching OFF the FOC supervision of the value net.  Normalization is therefore NOT a pure
reparameterization, and HJB_WEIGHT exists to control that side effect explicitly.
"""
import tensorflow as tf

import config


def natural_scale(delta, logK, v, c_over_k, climate_term=None, eps=1e-3):
    """A smooth envelope of the dominant HJB terms at each state.

    Built from the terms that actually set the equation's size:
        delta * |logK|      the flow scale term
        delta * |v|         the discount term
        delta * |log(C/K)|  the utility term
        |climate_term|      grows ∝ K (and its 2nd-order partner ∝ K^2), dominant at high K,Y
    `eps` floors the scale so the division can never blow up.
    """
    s = delta * (tf.abs(logK) + tf.abs(v) + tf.abs(tf.math.log(tf.maximum(c_over_k, 1e-10))))
    if climate_term is not None:
        s = s + tf.abs(climate_term)
    return tf.maximum(s, eps)


def scale_residual(residual, delta, logK=None, v=None, c_over_k=None, climate_term=None):
    """Return the (possibly non-dimensionalized) residual and the divisor actually used."""
    if config.HJB_SCALE == "none":
        return residual, tf.ones_like(residual)
    if config.HJB_SCALE == "delta":
        s = tf.ones_like(residual) * delta
        return residual / s, s
    if config.HJB_SCALE == "natural":
        s = natural_scale(delta, logK, v, c_over_k, climate_term)
        return residual / s, s
    raise ValueError(f"unknown REDESIGN_HJB_SCALE={config.HJB_SCALE}")


def weighted_rms(residual, delta, logK=None, v=None, c_over_k=None, climate_term=None):
    """RMS of the scaled residual, times the explicit HJB weight."""
    scaled, _ = scale_residual(residual, delta, logK, v, c_over_k, climate_term)
    return config.HJB_WEIGHT * tf.sqrt(tf.reduce_mean(tf.square(scaled)))


# --------------------------------------------------------------------------------------
# diagnostics: the metrics that decide whether normalization did what it claims
# --------------------------------------------------------------------------------------
def evenness_report(residual, scale, region_masks):
    """Relative-residual statistics per region -- the metric normalization SHOULD improve.

    `region_masks` maps a region name (e.g. 'lowK', 'highK', 'preDamageY', 'postDamageY',
    'deepXi', 'neutralXi') to a boolean mask.  Returns per-region RMS of residual/scale.
    A well-normalized objective has SIMILAR values across regions; the production objective is
    expected to be far more accurate in some regions than others.
    """
    rel = residual / scale
    out = {}
    for name, mask in region_masks.items():
        m = tf.cast(mask, rel.dtype)
        n = tf.reduce_sum(m)
        rms = tf.sqrt(tf.reduce_sum(tf.square(rel) * m) / tf.maximum(n, 1.0))
        out[name] = rms
    return out


# --------------------------------------------------------------------------------------
# FULL-LOSS NON-DIMENSIONALIZATION  (a PRECONDITION, not an experiment)
# --------------------------------------------------------------------------------------
# The production value loss adds the RMS of FIVE DIFFERENT PHYSICAL QUANTITIES:
#     RMS(HJB residual)  [units of delta*V,      natural scale ~5e-2]
#   + RMS(FOC_g/d/r)     [units of delta/(C/K),  natural scale ~2e-1]
#   + RMS(monotonicity)  [units of v_Y,          natural scale ~1e-2]
# Their implicit relative weights are therefore set by WHICHEVER UNITS each equation happens to be
# written in -- not by any modelling choice.  Measured consequence: the HJB term (2e-3) is ~20x the
# FOC terms (1e-4), so it carries ~90% of the gradient and the FOCs are under-weighted by accident.
#
# THIS IS WHY SCALING ONE TERM ALONE BACKFIRES.  Dividing only the HJB residual by its scale (the
# "non-dimensionalisation" arm) multiplied its weight ~8x and made the imbalance WORSE: HJB improved
# to 0.72x while FOC_d/FOC_g degraded 2.6x/4.1x, for a NET 1.22x WORSE total objective.  The fix is
# not to scale one term, it is to make EVERY term a dimensionless relative error so the sum is a
# well-posed objective in the first place.
#
#     L = ||R_HJB / S_HJB|| + sum_j ||FOC_j / S_FOC|| + ||mono / S_mono||   (+ already-normalised
#                                                                            xi-sensitivity terms)
#
# with each S the scale that GENERATES that term:
#     S_HJB  = delta*(|logK| + |v| + |log(C/K)|) + |climate term|      (measured ~0.12)
#     S_FOC  = delta/(C/K) = the marginal utility of consumption        (measured ~0.2)
#     S_mono = the same marginal-utility scale (v_Y enters the value at that order)
# Only after this are cross-term comparisons, and any claim that one loss beats another, meaningful.


def foc_scale(delta, c_over_k, eps=1e-3):
    """Natural scale of a first-order condition: the marginal utility of consumption."""
    return tf.maximum(delta / tf.maximum(c_over_k, 1e-10), eps)


def nondim_terms(delta, residual, focs, mono, logK, v, c_over_k, climate_term=None):
    """Return each loss term as a DIMENSIONLESS relative error (RMS of residual / its own scale).

    `focs` is a sequence of FOC residual tensors; `mono` may be None.
    """
    s_hjb = natural_scale(delta, logK, v, c_over_k, climate_term)
    s_foc = foc_scale(delta, c_over_k)
    rms = lambda a: tf.sqrt(tf.reduce_mean(tf.square(a)))
    out = {"hjb": rms(residual / s_hjb)}
    for i, f in enumerate(focs):
        out[f"foc{i}"] = rms(f / s_foc)
    if mono is not None:
        out["mono"] = rms(mono / s_foc)
    return out


def term_envelope(terms, eps=1e-4):
    """PRINCIPLED scale of the HJB: the size of the terms the equation is balancing.

        S(x) = sqrt( sum_i term_i(x)^2 )

    Why this and not a hand-picked combination:
      * UNIT-COVARIANT.  Every entry is an ACTUAL term of the assembled equation, so the residual and
        the scale carry identical units and the ratio is invariant.  The earlier ad-hoc scale used
        delta*|logK|, whose magnitude shifts by log(c) if capital is re-denominated -- it measured the
        bookkeeping units, not the equation.
      * NOT HAND-PICKED.  No judgement about which terms "look dominant"; the data decides.
        (Measured at xi=0.05: -delta*V supplies 63% of S^2 and the flow term 28%, so S is effectively
        delta x (value scale) -- but that is an OUTPUT of the construction, not an assumption.)
      * CORRECTLY STATE-DEPENDENT.  Where the climate block grows with K it enters S automatically.

    Measured consequence: S_envelope = 4.79e-2 versus the ad-hoc 0.12, i.e. the ad-hoc scale was
    2.5x too large and UNDERSTATED the true relative HJB error (4.82%, not 1.93%).
    """
    s2 = None
    for t in terms:
        s2 = tf.square(t) if s2 is None else s2 + tf.square(t)
    return tf.maximum(tf.sqrt(s2), eps)
