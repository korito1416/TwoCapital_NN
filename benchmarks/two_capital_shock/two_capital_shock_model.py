"""
Two-capital model WITH Brownian capital shocks (the sigma_d, sigma_g > 0
generalization of the deterministic note). Derivation: derivation.tex (referee-
verified). Shares the FOC / consumption / phi algebra with the deterministic
model; the new pieces are the second-order (v'') HJB term, the sigma corrections,
and the boundary Ito drag -sigma_j^2/(2 delta).

Reduced second-order ODE in v(Z), V(logK,Z)=logK+v(Z):
  0 = delta*log c - delta*v
      + (1-Z)phi_d + Z phi_g - 1/2 ( sigma_d^2 (1-Z)^2 + sigma_g^2 Z^2 )
      + [ phi_g - phi_d + (1-Z) sigma_d^2 - Z sigma_g^2 ] Z(1-Z) v'
      + 1/2 Z^2 (1-Z)^2 ( sigma_d^2 + sigma_g^2 ) v''
with FOC-implied controls (identical to the deterministic case):
  i^j = Gamma_j c q_j / delta - 1/theta_j,  q_d = 1 - Z v',  q_g = 1 + (1-Z) v',
  c   = (1-Z)(A_d - i^d) + Z(A_g - i^g).
"""

import os
import sys

import numpy as np

# Reuse the deterministic model's calibration + shared algebra (phi, controls, c).
_THIS = os.path.dirname(os.path.abspath(__file__))
_DET = os.path.normpath(os.path.join(_THIS, "..", "two_capital_deterministic"))
if _DET not in sys.path:
    sys.path.insert(0, _DET)
import two_capital_model as DET           # noqa: E402
from params import PARAMS                 # noqa: E402  (models/ is on path via DET)

# re-export shared pieces
phi = DET.phi
A_bar = DET.A_bar
q_values = DET.q_values
consumption = DET.consumption
controls = DET.controls


def load_calibration(a_g_choice="A_g_prime_prime"):
    """Deterministic calibration + the capital volatilities sigma_d, sigma_g."""
    P = DET.load_calibration(a_g_choice)
    P["sigma_d"] = float(PARAMS["σ_d"])
    P["sigma_g"] = float(PARAMS["σ_g"])
    return P


def drift_coef_v1(Z, P):
    """Coefficient on v'(Z): [phi_g - phi_d + (1-Z)sigma_d^2 - Z sigma_g^2] Z(1-Z),
    where phi_g, phi_d are at the FOC-optimal controls (depend on the slope)."""
    # handled inside hjb_residual where controls are known; kept for clarity
    raise NotImplementedError


def diffusion_coef(Z, P):
    """Coefficient on v''(Z): 1/2 Z^2 (1-Z)^2 (sigma_d^2 + sigma_g^2 - 2 rho sigma_d sigma_g)."""
    sdg = P.get("rho", 0.0) * P["sigma_d"] * P["sigma_g"]
    return 0.5 * Z ** 2 * (1.0 - Z) ** 2 * (P["sigma_d"] ** 2 + P["sigma_g"] ** 2 - 2.0 * sdg)


def hjb_residual(Z, v, slope, curv, P):
    """Second-order HJB residual R(Z) (should be 0 at the solution).
    slope = v'(Z), curv = v''(Z)."""
    i_d, i_g, c = controls(Z, slope, P)
    phi_d = phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    sd2, sg2 = P["sigma_d"] ** 2, P["sigma_g"] ** 2
    sdg = P.get("rho", 0.0) * P["sigma_d"] * P["sigma_g"]   # cross-covariance of the shocks
    logK_drift = (1.0 - Z) * phi_d + Z * phi_g - 0.5 * (
        sd2 * (1.0 - Z) ** 2 + sg2 * Z ** 2 + 2.0 * sdg * Z * (1.0 - Z))
    v1_coef = (phi_g - phi_d + (1.0 - Z) * sd2 - Z * sg2
               + (2.0 * Z - 1.0) * sdg) * Z * (1.0 - Z)
    v2_coef = 0.5 * Z ** 2 * (1.0 - Z) ** 2 * (sd2 + sg2 - 2.0 * sdg)
    return (P["delta"] * np.log(np.maximum(c, 1e-12)) - P["delta"] * v
            + logK_drift + v1_coef * slope + v2_coef * curv
            + robustness_drag(Z, slope, P))


def robustness_drag(Z, slope, P):
    """Hansen-Sargent robustness drag -1/(2 xi)[(1-Z)^2 sd^2 q_d^2 + Z^2 sg^2 q_g^2].
    xi -> inf (default) gives 0 (no robustness concern)."""
    xi = P.get("xi", np.inf)
    if not np.isfinite(xi):
        return 0.0
    q_d = 1.0 - Z * slope
    q_g = 1.0 + (1.0 - Z) * slope
    sd2, sg2 = P["sigma_d"] ** 2, P["sigma_g"] ** 2
    return -(1.0 / (2.0 * xi)) * ((1.0 - Z) ** 2 * sd2 * q_d ** 2 + Z ** 2 * sg2 * q_g ** 2)


def worst_case_drifts(Z, slope, P):
    """Worst-case Brownian drift distortions h_d*, h_g* (recovered ex post)."""
    xi = P.get("xi", np.inf)
    q_d = 1.0 - Z * slope
    q_g = 1.0 + (1.0 - Z) * slope
    h_d = -(1.0 / xi) * (1.0 - Z) * P["sigma_d"] * q_d
    h_g = -(1.0 / xi) * Z * P["sigma_g"] * q_g
    return h_d, h_g


def one_capital_value(A, alpha, Gamma, theta, sigma, delta, xi=np.inf):
    """One-capital value WITH the Ito variance drag -sigma^2/(2 delta) and the
    robustness drag -sigma^2/(2 xi delta) (xi -> inf removes the latter)."""
    c = delta * (1.0 + theta * A) / (theta * (delta + Gamma))
    rob = 0.0 if not np.isfinite(xi) else sigma ** 2 / (2.0 * xi * delta)
    v = (np.log(c) + alpha / delta + (Gamma / delta) * np.log(Gamma * theta * c / delta)
         - sigma ** 2 / (2.0 * delta) - rob)
    return v, c


def boundary_values(P):
    xi = P.get("xi", np.inf)
    v0, _ = one_capital_value(P["A_d"], P["alpha_d"], P["Gamma_d"], P["theta_d"], P["sigma_d"], P["delta"], xi)
    vN, _ = one_capital_value(P["A_g"], P["alpha_g"], P["Gamma_g"], P["theta_g"], P["sigma_g"], P["delta"], xi)
    return v0, vN
