"""Uncertainty-aversion parameterization: legacy logξ vs Lars's θ = 1/ξ.

WHY θ = 1/ξ IS THE BETTER PARAMETERIZATION (all three are consequences of the algebra, not
preferences):

1.  EXACT NEUTRAL LIMIT.  θ = 0 IS uncertainty-neutrality.  Under logξ the neutral limit is
    logξ → +∞ and is only approached (the code fakes it with logξ_max = 5 ⟹ ξ = 148.4).

2.  THE JUMP TERM COLLAPSES TO ONE STABLE EXPRESSION.  With g = exp(−θ Δv) the misspecification
    contribution telescopes exactly (VERIFIED numerically against the production form to ~1e-8
    over ξ ∈ {148.6, 1, 0.1, 0.05} × Δv ∈ {±0.05, ±0.5}):

        J g Δv + ξ J (1 − g + g log g)
          = J g Δv + (1/θ) J (1 − g − θ g Δv)          [since log g = −θ Δv]
          = J (1 − g)/θ
          = J · (−expm1(−θ Δv)) / θ                    → J Δv   as θ → 0.

    This replaces three separately-evaluated terms (two of which individually blow up as θ→0,
    cancelling only in exact arithmetic) with ONE expression, and expm1 keeps it accurate when
    θ·Δv is tiny.

    !! HONEST CORRECTION — θ DOES NOT REMOVE THE DEEP-UNCERTAINTY OVERFLOW !!
    The overflow lives in the PRODUCT θ·Δv, not in the parameterization.  At ξ = 0.005
    (θ = 200) with Δv = −1 the exponent is +200 and exp() overflows float32 in BOTH forms
    (measured: θ-form returns −inf without a clip).  A clip is still required at deep θ, and it
    still zeroes the gradient there.  The overflow is INHERENT to the robust jump term as
    ξ → 0 with Δv ≠ 0; only a genuine reformulation (or keeping |Δv| small via cross-regime
    value-matching, which shrinks Δv itself) removes it.  Do not claim θ fixes it.

3.  THE DRIFT-DISTORTION BLOCK BECOMES POLYNOMIAL IN θ.  h = −θ σ' ∂v, and the penalty
    ξ|h|²/2 = θ|σ'∂v|²/2, so the whole robustness block is a POLYNOMIAL in θ (degree 1 in the
    drag).  Near neutrality V is ~linear in θ, which is exactly what a network can represent
    easily — and it makes the structural ansatz V = V_∞ − θ W natural.

4.  SAMPLING — A REAL TRADE-OFF, NOT A FREE WIN (measured, n=2e5):

                            ξ < 0.1        ξ > 10 (near-neutral)
        logξ ~ U[−3, 5]       8.6%              33.8%
        θ    ~ U[0, 50]      79.9%               0.2%      ← near-neutral nearly ABANDONED

    So uniform-θ does fix the documented "deep-ξ underweight" (≈6.7% of batch) — but it
    overcorrects hard: the neutral region collapses from 33.8% to 0.2% of the batch.  Since the
    neutral solution is the anchor of every ξ-comparison we report, that is not acceptable
    as-is.  RECOMMENDATION: sample θ from a MIXTURE or stratify (e.g. half the batch uniform in
    θ, half uniform in logξ; or uniform in sqrt(θ)), and always report the residual at BOTH ends.
    Do not adopt plain uniform-θ sampling without checking the neutral-ξ residual.
"""
import tensorflow as tf

import config


# --------------------------------------------------------------------------------------
# sampling / conversion
# --------------------------------------------------------------------------------------
def theta_to_logxi(theta, floor=1e-12):
    """θ → logξ = −log θ (for interoperating with legacy code paths). θ=0 ⟹ +inf, so floor."""
    return -tf.math.log(tf.maximum(theta, floor))


def logxi_to_theta(logxi):
    """logξ → θ = 1/ξ = exp(−logξ)."""
    return tf.exp(-logxi)


def network_input(theta):
    """Transform θ into the value fed to the network.

    θ ∈ [0, 50] is badly scaled next to Z ∈ (0,1) and Y ∈ [0,4], so by default we feed
    θ/θ_max ∈ [0,1].  'sqrt' spreads the near-neutral region (where V is ~linear in θ and the
    interesting curvature lives) over more of the input range.
    """
    t = theta / config.THETA_MAX
    if config.XI_INPUT == "raw":
        return theta
    if config.XI_INPUT == "sqrt":
        return tf.sqrt(tf.maximum(t, 0.0))
    return t


# --------------------------------------------------------------------------------------
# the two θ-dependent blocks of the HJB
# --------------------------------------------------------------------------------------
def drift_distortion(theta, sigma_dot_dv):
    """h = −θ · (σ' ∂v).   (Production: h = −(1/ξ)(σ' ∂v).)"""
    return -theta * sigma_dot_dv


def drift_penalty(theta, h):
    """ξ|h|²/2 = |h|²/(2θ).  Expressed via h = −θ·s as θ|s|²/2 to stay finite at θ=0."""
    return 0.5 * h * h / tf.maximum(theta, 1e-12)


def drift_penalty_from_s(theta, s):
    """ξ|h|²/2 with h = −θ s  ⟹  = θ|s|²/2.  FINITE AND EXACT at θ = 0 (no division)."""
    return 0.5 * theta * s * s


def jump_contribution(theta, intensity, dv, clip=None):
    """Total misspecification-adjusted jump contribution  J·(1 − exp(−θ·Δv))/θ.

    Equals  J·g·Δv + ξ·J·(1 − g + g log g)  with g = exp(−θ Δv), ξ = 1/θ — see module docstring.
    Uses expm1 and a series fallback so the θ→0 limit (J·Δv) is exact rather than 0/0.

    `dv` is Δv = V_post_jump − V.  `clip` optionally bounds the exponent θ·Δv (the production
    guard); leave None to run unclipped, which the θ-form can usually afford.
    """
    x = theta * dv
    if clip is not None:
        x = tf.clip_by_value(x, -clip, clip)
    # (1 − e^{−x})/θ  computed stably; for |x| tiny use the series (1 − e^{−x})/θ ≈ Δv(1 − x/2)
    small = tf.abs(x) < 1e-6
    safe_theta = tf.maximum(theta, 1e-12)
    exact = -tf.math.expm1(-x) / safe_theta
    series = dv * (1.0 - 0.5 * x)
    return intensity * tf.where(small, series, exact)


def jump_distortion_g(theta, dv, clip=None):
    """The worst-case intensity distortion g = exp(−θ·Δv) (reported, not used in the HJB sum)."""
    x = theta * dv
    if clip is not None:
        x = tf.clip_by_value(x, -clip, clip)
    return tf.exp(-x)
