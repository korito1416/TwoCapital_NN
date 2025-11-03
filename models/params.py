"""
Default parameters in the appendix.

This file exposes a single dictionary `PARAMS` containing all runtime
parameters used by the compact model. For readability and to match the paper's
notation, Unicode Greek-letter aliases are also added to the same dictionary
(e.g. 'δ').

Edit values in this file to change project-wide defaults.
"""

# Primary defaults (ASCII keys) -------------------------------------------------

PARAMS = {
    # 1) Table: State Variable Initial Values and Ranges
    "K0": 880,
    "Z0": 0.7,
    "Y0": 1.1,
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
    "δ": 0.025,  
    "α_d": -0.035,    "Γ_d": 0.060,    "θ_d": 16.7,    "σ_d": 0.01,     
    "α_g": -0.035,    "Γ_g": 0.060,    "θ_g": 16.7,    "σ_g": 0.01,     
    "ζ": 0.0,         "ψ0": 0.10583,   "ψ1": 0.5,      "σ_κ": 0.0078,   
    "varrho": 746.67,


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

