"""Experiment switches for the one-jump HJB loss/representation redesign.

EVERY switch defaults to the PRODUCTION behaviour, so with no environment variables set this
variant reproduces `models/` exactly.  Each experiment turns on ONE switch (Mike's rule:
change one variable at a time).

Switches
--------
REDESIGN_XI_PARAM   : "logxi" (default) | "theta"
    Uncertainty pseudo-state parameterization.  "theta" uses θ = 1/ξ ∈ [0, θ_max] (Lars's
    request): θ=0 is EXACTLY uncertainty-neutral, and the jump term becomes
    J·(1−exp(−θ·Δv))/θ → J·Δv as θ→0 (smooth; no exp-overflow clip needed).
REDESIGN_THETA_MAX  : float, default 50.0   (θ=50 ⟺ ξ=0.02)
REDESIGN_XI_INPUT   : "raw" | "unit" (default) | "sqrt"
    How θ enters the network.  "unit" feeds θ/θ_max ∈ [0,1] (comparable scale to Z);
    "sqrt" feeds sqrt(θ/θ_max) (resolves the near-neutral region where V is ~linear in θ).
REDESIGN_HJB_SCALE  : "none" (default) | "delta" | "natural"
    Non-dimensionalization of the HJB residual.  "delta" divides by δ; "natural" divides by a
    smooth envelope of the assembled terms.  NOTE: normalization can even out fit quality
    across the box but PROVABLY CANNOT fix level identifiability (weighting scales the level
    and shape curvatures equally, leaving the δ²:1 ratio intact) — it is a conditioning
    experiment, not a level fix.
REDESIGN_HJB_WEIGHT : float, default 1.0
    Explicit weight on the (possibly scaled) HJB residual inside the value loss.  The value
    loss is RMS(HJB) + RMS(FOC_g) + RMS(FOC_d) + RMS(FOC_r) + monotonicity, so rescaling the
    HJB term also RE-WEIGHTS it against the FOC terms; this knob makes that explicit.
REDESIGN_INPUTS     : "legacy" (default) | "unified"
    "unified" removes the logξ padding duplication and the dead constant A_g'' column, and
    uses one canonical slot ordering across the 4 regimes.
    !! Changing this BREAKS warm-start from existing checkpoints (first-layer width/order). !!
REDESIGN_DETREND    : "off" (default) | "on"
    Homogeneity-detrended value representation V = a·logK + W(Z, Y, u), u = logR − logK.
    Approximate (exp(logK) survives in the climate/tech-jump blocks), so `a` is trainable.
REDESIGN_ANCHOR     : "off" (default) | "recenter"
    "recenter" uses the re-centered value net v(s) = φ(s) − φ(x₀) + v₀, which pins the single
    level degree of freedom BY CONSTRUCTION (conservative; beat costate/EGM in torch tests).
REDESIGN_ANCHOR_V0  : float, anchor value v₀ at the anchor state (FD-verified ≈3.64-3.67).
REDESIGN_ANCHOR_XI  : "neutral" (default) | "batch"
    WHICH ξ the anchor state is evaluated at.  THIS MATTERS AND THE WRONG CHOICE IS SILENT:
    with "batch" the anchor state carries the sample's own ξ, so v(x₀,ξ) = v₀ for EVERY ξ and the
    ξ-dependence of the value AT THE ANCHOR STATE is annihilated — measured exactly 0.00000 in the
    2026-07-26 run, which destroys the welfare-cost-of-robustness V(X₀,ξ)−V(X₀,neutral) we report.
    "neutral" evaluates the anchor at a FIXED neutral ξ, so it pins ONE scalar (the level) and
    leaves the whole ξ-profile free — which is what a gauge fix is supposed to do.
"""
import os


def _s(name, default):
    return str(os.environ.get(name, default)).strip().lower()


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


XI_PARAM = _s("REDESIGN_XI_PARAM", "logxi")        # logxi | theta
THETA_MAX = _f("REDESIGN_THETA_MAX", 50.0)
XI_INPUT = _s("REDESIGN_XI_INPUT", "unit")          # raw | unit | sqrt
HJB_SCALE = _s("REDESIGN_HJB_SCALE", "none")        # none | delta | natural
HJB_WEIGHT = _f("REDESIGN_HJB_WEIGHT", 1.0)
INPUTS = _s("REDESIGN_INPUTS", "legacy")            # legacy | unified
DETREND = _s("REDESIGN_DETREND", "off") in {"on", "1", "true", "yes"}
ANCHOR = _s("REDESIGN_ANCHOR", "off")               # off | recenter
ANCHOR_V0 = _f("REDESIGN_ANCHOR_V0", 3.64)
ANCHOR_XI = _s("REDESIGN_ANCHOR_XI", "neutral")     # neutral | batch

# --- xi-SENSITIVITY losses (order-homogeneous) ------------------------------------------
# The HJB residual and each FOC must vanish for EVERY xi, so their xi-derivatives must vanish too.
# Those derivatives are NOT implied numerically: measured on the reference solution the HJB is
# satisfied to ~96% while its theta-derivative is violated by 64-97%, and the FOC elasticity
# identities are violated 4.8-8.5% versus a 0.05% pointwise FOC residual.  The xi-response is being
# ABSORBED AS RESIDUAL ERROR instead of being learned.
# ORDER-HOMOGENEITY IS ESSENTIAL: the raw dR/dln(theta) is ~2.7e-5 and would be invisible next to a
# loss dominated by ~1.4e-3 terms.  Each term is therefore divided by the scale that GENERATES it --
# the HJB one by its own theta-explicit source (theta/2)||s||^2, the FOC ones by the marginal utility
# delta/(C/K) -- making both O(1) and immune to being swamped.
# FULL-LOSS non-dimensionalisation: make EVERY loss term a dimensionless relative error.
# This is a PRECONDITION for the objective to be well-posed, not a candidate improvement: the
# production loss sums the RMS of quantities carrying different units (delta*V, delta/(C/K), v_Y),
# so their relative weights are an accident of notation and the HJB term ends up with ~90% of the
# gradient.  Scaling ONE term (the earlier "non-dimensionalisation" arm) makes that worse, which is
# exactly what was measured: total objective 1.22x WORSE.  Default off only to preserve the
# bit-identical production gate.
LOSS_NONDIM = _s("REDESIGN_LOSS_NONDIM", "off") in {"on", "1", "true", "yes"}
SENS_WEIGHT = _f("REDESIGN_SENS_WEIGHT", 0.0)      # weight on  d(HJB residual)/d ln(theta)
FOCXI_WEIGHT = _f("REDESIGN_FOCXI_WEIGHT", 0.0)    # weight on  d(FOC_j)/d ln(theta)
SENS_FLOOR = _f("REDESIGN_SENS_FLOOR", 1e-6)       # floor on the normaliser
ANCHOR_LOGXI_NEUTRAL = _f("REDESIGN_ANCHOR_LOGXI_NEUTRAL", 5.0)   # logξ=5 ⟺ ξ=148.4 (the trained neutral end)

USE_THETA = XI_PARAM == "theta"
USE_UNIFIED = INPUTS == "unified"


def summary():
    """One-line provenance string; write this into every run's MANIFEST."""
    return (
        f"xi_param={XI_PARAM}(theta_max={THETA_MAX},input={XI_INPUT}) "
        f"hjb_scale={HJB_SCALE}(w={HJB_WEIGHT}) inputs={INPUTS} "
        f"detrend={'on' if DETREND else 'off'} anchor={ANCHOR}(v0={ANCHOR_V0},xi={ANCHOR_XI})"
    )
