"""
Default parameters in the appendix.

This file exposes a single dictionary `PARAMS` containing all runtime
parameters used by the compact model. For readability and to match the paper's
notation, Unicode Greek-letter aliases are also added to the same dictionary
(e.g. 'δ').

Edit values in this file to change project-wide defaults.
"""

import os

# Primary defaults (ASCII keys) -------------------------------------------------

PARAMS = {
    # 1) Table: State Variable Initial Values and Ranges
    "K0": 880,
    "Z0": 0.7,
    "Y0": 1.2,
    "R0": 11.2,
    
    ## Training ranges
    "logK_min": 4.0,
    "logK_max": 7.0,
    "Z_min": 0.01,
    "Z_max": 0.99,
    "Y_min": 0.0,
    "Y_max": 4.0,
    "logR_min": 1.0,
    "logR_max": 6.0,
    "λ3_min": 0.0,
    "λ3_max": 1.0/3.0,
    "logξ_min": -3.0,
    "logξ_max": 5.0,
    

    # 2) Technology parameters 
    "A_d": 0.1303,
    "A_g": 0.1085, 
    "A_g_prime": 0.1303,
    "A_g_prime_prime": 0.1567,
    "π" : 0.04,


    # 3) Table: Economic Parameters
    "δ": 0.01,  
    "α_d": -0.035,    "Γ_d": 0.060,    "θ_d": 16.7,    "σ_d": 0.01,     
    "α_g": -0.035,    "Γ_g": 0.060,    "θ_g": 16.7,    "σ_g": 0.01,     
    "ζ": 0.0,         "ψ0": 0.10583,   "ψ1": 0.5,      "σ_κ": 0.0078,   
    "varrho": 746.67,
    "tech_jump_intensity_scale": 1.0,


    # 4) Table: Climate Dynamics and Damages Parameters
    "θ_bar": 1.86 / 1000,     
    "η": 0.291,                  
    "ϛ": 1.2 * 1.86 / 1000,      
    "λ1": 0.00017675,    "λ2": 2 * 0.0022,   
    # Damage nonlinearity grid (functional form from table)
    "L": 5, # Number of damage realizations
    "λ3_values": [0.,1/12,1/6,1/4,1/3],
    "r1": 1.5,    "r2": 0.36,
    "y_lower": 1.5,      "y_upper": 2.5      
}


# Run-specific sensitivity overrides ------------------------------------------

# Slurm sensitivity jobs use these environment variables so the baseline
# calibration above remains unchanged.  Because model classes copy PARAMS at
# construction time, applying the overrides here also makes export_parameters()
# record the actual calibration used by each stage.
_ENVIRONMENT_OVERRIDES = {
    "MODEL_SIGMA_D": "σ_d",
    "MODEL_SIGMA_G": "σ_g",
    "MODEL_GAMMA_D": "Γ_d",
    "MODEL_GAMMA_G": "Γ_g",
    "MODEL_THETA_D": "θ_d",
    "MODEL_THETA_G": "θ_g",
    "MODEL_PSI0": "ψ0",
}

for environment_name, parameter_name in _ENVIRONMENT_OVERRIDES.items():
    raw_value = os.environ.get(environment_name)
    if raw_value not in (None, ""):
        PARAMS[parameter_name] = float(raw_value)

if PARAMS["θ_d"] <= 0.0 or PARAMS["θ_g"] <= 0.0:
    raise ValueError("theta_d and theta_g must be positive")


def investment_rate_activation(theta):
    """Return the bounded control activation compatible with log(1 + theta*i)."""
    import tensorflow as tf

    theta = float(theta)
    if theta <= 0.0:
        raise ValueError("theta must be positive")

    return lambda x: 1.0 - (1.0 + 1.0 / theta) / (tf.exp(2.0 * x) + 1.0)


# --- Optional deterministic seeding for reproducible / cross-arm-comparable runs ---
# Set MODEL_SEED in the environment to fix the stochastic collocation sampling and the
# weight init. No-op when unset, so simulation / plotting importers are unaffected.
_model_seed = os.environ.get("MODEL_SEED")
if _model_seed not in (None, ""):
    import tensorflow as _tf
    import numpy as _np
    _tf.random.set_seed(int(_model_seed))
    _np.random.seed(int(_model_seed))
