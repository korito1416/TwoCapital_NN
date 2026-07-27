"""
Deterministic two-capital adjustment-cost model (no random capital shocks).

Reference: adjustment_cost_no_random.tex (Barnett-Brock-Hansen two-capital note)
and PAPER_HJB_REFERENCE.md / AGENTS.md for the parent project's calibration.

State reduction (V(logK, Z) = logK + v(Z)) leaves a 1-D problem in the green
capital share Z = K^g / (K^d + K^g). The HJB is the first-order nonlinear ODE

    0 = max_{i^d,i^g} { delta*log c - delta*v(Z)
                        + (1-Z) phi_d(i^d) + Z phi_g(i^g)
                        + Z(1-Z)[phi_g(i^g) - phi_d(i^d)] v'(Z) }

with phi_j(i) = alpha_j + Gamma_j log(1 + theta_j i),
c = (1-Z)(A_d - i^d) + Z(A_g - i^g).

This module exposes the closed-form pieces (controls/consumption as functions of
the value-function slope p = v'(Z)) shared by the finite-difference and the
neural-network solvers, plus the one-capital boundary values.

Parameters are loaded from the parent project's models/params.py so the two
solvers stay in lock-step with the rest of the codebase. Per the task,
A_g is set to the POST tech-jump productivity A_g'' (params.py: A_g_prime_prime).
"""

import os
import sys

import numpy as np

# --- Load the project calibration (unicode-keyed PARAMS) ----------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.normpath(os.path.join(_THIS, "..", "..", "models"))
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)
from params import PARAMS  # noqa: E402


def load_calibration(a_g_choice="A_g_prime_prime"):
    """Return the plain-ASCII parameter dict for the two-capital model.

    a_g_choice selects which green productivity to use:
      'A_g_prime_prime' (default, post tech jump / breakthrough)  -> 0.1567
      'A_g_prime'       (intermediate / catch-up)                 -> 0.1303
      'A_g'             (pre tech jump)                           -> 0.1085
    """
    return {
        "delta": float(PARAMS["δ"]),
        "alpha_d": float(PARAMS["α_d"]),
        "alpha_g": float(PARAMS["α_g"]),
        "Gamma_d": float(PARAMS["Γ_d"]),
        "Gamma_g": float(PARAMS["Γ_g"]),
        "theta_d": float(PARAMS["θ_d"]),
        "theta_g": float(PARAMS["θ_g"]),
        "A_d": float(PARAMS["A_d"]),
        "A_g": float(PARAMS[a_g_choice]),
        "a_g_choice": a_g_choice,
    }


# --- Core algebra (vectorized over Z and the slope p = v'(Z)) ------------------

def A_bar(Z, p):
    A_d, A_g = p["A_d"], p["A_g"]
    return (1.0 - Z) * A_d + Z * A_g


def q_values(Z, slope):
    """Marginal-value elasticities q_d, q_g as functions of slope p = v'(Z)."""
    q_d = 1.0 - Z * slope
    q_g = 1.0 + (1.0 - Z) * slope
    return q_d, q_g


def consumption(Z, slope, p):
    """Scaled consumption c(Z) = C/K from the FOCs and resource constraint.

    General (possibly asymmetric Gamma/theta) closed form:
        c = delta[ Abar + (1-Z)/theta_d + Z/theta_g ]
            / [ delta + (1-Z) Gamma_d q_d + Z Gamma_g q_g ].
    Reduces to the common-adjustment-cost result c = delta(1+theta*Abar)/(theta(delta+Gamma)).
    """
    q_d, q_g = q_values(Z, slope)
    num = p["delta"] * (A_bar(Z, p) + (1.0 - Z) / p["theta_d"] + Z / p["theta_g"])
    den = p["delta"] + (1.0 - Z) * p["Gamma_d"] * q_d + Z * p["Gamma_g"] * q_g
    return num / den


def controls(Z, slope, p):
    """FOC-implied investment rates i^d, i^g given the slope p = v'(Z)."""
    q_d, q_g = q_values(Z, slope)
    c = consumption(Z, slope, p)
    i_d = p["Gamma_d"] * c * q_d / p["delta"] - 1.0 / p["theta_d"]
    i_g = p["Gamma_g"] * c * q_g / p["delta"] - 1.0 / p["theta_g"]
    return i_d, i_g, c


def phi(i, alpha, Gamma, theta):
    return alpha + Gamma * np.log(np.maximum(1.0 + theta * i, 1e-12))


def mu_Z(Z, slope, p):
    """Drift of Z: Z(1-Z)[phi_g(i^g) - phi_d(i^d)]."""
    i_d, i_g, _ = controls(Z, slope, p)
    phi_d = phi(i_d, p["alpha_d"], p["Gamma_d"], p["theta_d"])
    phi_g = phi(i_g, p["alpha_g"], p["Gamma_g"], p["theta_g"])
    return Z * (1.0 - Z) * (phi_g - phi_d)


def hjb_residual(Z, v, slope, p):
    """Pointwise HJB residual R(Z) (should be 0 at the solution)."""
    i_d, i_g, c = controls(Z, slope, p)
    phi_d = phi(i_d, p["alpha_d"], p["Gamma_d"], p["theta_d"])
    phi_g = phi(i_g, p["alpha_g"], p["Gamma_g"], p["theta_g"])
    drift = Z * (1.0 - Z) * (phi_g - phi_d)
    return (p["delta"] * np.log(np.maximum(c, 1e-12)) - p["delta"] * v
            + (1.0 - Z) * phi_d + Z * phi_g + drift * slope)


# --- One-capital boundary values (Z=0 dirty-only, Z=1 green-only) -------------

def one_capital_value(A, alpha, Gamma, theta, delta):
    c = delta * (1.0 + theta * A) / (theta * (delta + Gamma))
    v = np.log(c) + alpha / delta + (Gamma / delta) * np.log(Gamma * theta * c / delta)
    return v, c


def boundary_values(p):
    v0, _ = one_capital_value(p["A_d"], p["alpha_d"], p["Gamma_d"], p["theta_d"], p["delta"])
    vN, _ = one_capital_value(p["A_g"], p["alpha_g"], p["Gamma_g"], p["theta_g"], p["delta"])
    return v0, vN


# --- First-order perturbation slope (symmetric-benchmark expansion) -----------

def perturbation_slope(Z, p):
    """v'(Z) ~= (A_g - A_d)/c(Z) + (alpha_g - alpha_d)/delta  (small-heterogeneity)."""
    c = consumption(Z, np.zeros_like(Z), p)  # c at slope 0 (leading order)
    return (p["A_g"] - p["A_d"]) / c + (p["alpha_g"] - p["alpha_d"]) / p["delta"]
