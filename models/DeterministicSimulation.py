"""
Deterministic simulation utilities for trained networks.
Provides simulate_path(...) and simulate_path_post_tech_post_jump(...)
Both functions expect a `model` object exposing the same attributes as your trained classes
(e.g. model.v_nn, model.i_g_nn, model.i_d_nn, model.i_I_nn, model.v_pre_tech_post_damage_nn,
model.v_post_tech_pre_damage_nn and model.params)

This module adapts the code you provided into reusable functions.
"""

import os
import tensorflow as tf
from params import PARAMS
from PreDamagePreTech import PreDamagePreTechModel
import matplotlib.pyplot as plt

def load_PreDamagePreTech_model(export_root, params_override=None):
    """
    export_root: parent folder that contains:
      - PreDamagePreTech/
      - PreDamagePostTech/
      - PostDamagePreTech/
      - PreDamageIntermTech/
    Returns a PreDamagePreTechModel with value, control nets,
    and helper value nets loaded.
    """
    params = PARAMS.copy()
    if params_override:
        params.update(params_override)

    # --- helper nets (these get loaded by __init__) ---
    params["v_PreDamagePostTech_nn_path"]   = os.path.join(export_root, "PreDamagePostTech",   "v_nn_checkpoint_PreDamagePostTech")
    params["v_PreDamageIntermTech_nn_path"] = os.path.join(export_root, "PreDamageIntermTech", "v_nn_checkpoint_PreDamageIntermTech")
    params["v_PostDamagePreTech_nn_path"]   = os.path.join(export_root, "PostDamagePreTech",   "v_nn_checkpoint_PostDamagePreTech")

    # --- main nets directory ---
    params["export_folder"] = os.path.join(export_root, "PreDamagePreTech")

    # Instantiate model (loads helper nets too)
    model = PreDamagePreTechModel(params)

    # Build networks before loading weights
    n_inputs = 6  
    bs = params.get("batch_size", 128)
    model.v_nn.build((bs, n_inputs))
    model.i_g_nn.build((bs, n_inputs))
    model.i_d_nn.build((bs, n_inputs))
    model.i_r_nn.build((bs, n_inputs))

    # --- main nets checkpoints ---
    pre_dir = params["export_folder"]
    model.v_nn.load_weights(os.path.join(pre_dir, "v_nn_checkpoint_PreDamagePreTech"))
    model.i_g_nn.load_weights(os.path.join(pre_dir, "i_g_nn_checkpoint_PreDamagePreTech"))
    model.i_d_nn.load_weights(os.path.join(pre_dir, "i_d_nn_checkpoint_PreDamagePreTech"))
    model.i_r_nn.load_weights(os.path.join(pre_dir, "i_r_nn_checkpoint_PreDamagePreTech"))

    return model


from typing import Dict, Any, Tuple, List
import numpy as np
def simulate_path_pre_damage_pretech(
    model,
    ξ ,
    T ,
    dt , 
    make_plots: bool = True):
    """
    Deterministic path simulation for the Pre-Damage Pre-Technology model. 
    Uses your nets and params on an already-loaded `model`.

    State:    X = [logK, Z, Y, logR, λ3, logξ]
    Controls: i_g(X), i_d(X), i_r(X) with i_r = exp(-i_r_nn(X))

    Dynamics (reference measure, no h-* in drifts):
        d logK = [(α_d + Γ_d log(1+θ_d i_d))(1-Z) + (α_g + Γ_g log(1+θ_g i_g))Z
                  - 0.5(σ_d^2(1-Z)^2 + σ_g^2 Z^2)] dt
        dZ     = [ (α_g + Γ_g log(1+θ_g i_g) - (α_d + Γ_d log(1+θ_d i_d)) - Z σ_g^2 + (1-Z) σ_d^2 )
                   Z(1-Z) ] dt
        dY     = [ θ_bar * η * A_d * (1-Z) * K ] dt
        d logR = [ -ζ + ψ0 * exp( ψ1 * (log i_r + logK - logR) ) - 0.5 σ_κ^2 ] dt
        (K = exp(logK), i_r = exp(-i_r_nn(X)))

    We also compute h_d, h_g, h_y, h_r and distorted jump probabilities
    using gradients of v(X), following your pde_rhs structure.

    Args:
        model: Your loaded PreDamagePreTechModel.
        T: horizon in years.
        dt: time step. 
        export_folder: where to save .txt and .png outputs.
        make_plots: whether to write plots.

    Returns:
        data: dict[str, np.ndarray] with time series and diagnostics.
    """
    

    params  = model.params 
 
    # --- Shorthands for parameters used in drifts / diagnostics ---
    A_d = params['A_d']; A_g = params['A_g']
    α_d = params['α_d']; Γ_d = params['Γ_d']; θ_d = params['θ_d']; σ_d = params['σ_d']
    α_g = params['α_g']; Γ_g = params['Γ_g']; θ_g = params['θ_g']; σ_g = params['σ_g']
    ζ   = params['ζ'];   ψ0  = params['ψ0'];  ψ1  = params['ψ1'];  σ_κ = params['σ_κ']
    θ_bar = params['θ_bar']; η = params['η'];  ϛ = params['ϛ']
    varrho = params['varrho']
    π = params['π']
    # Damage jump
    r1 = params['r1']; r2 = params['r2']; y_lower = params['y_lower']; y_upper = params['y_upper']
    # Damage λ3 mixture
    L = params['L']; λ3_values = params['λ3_values']

    # --- Time grid ---
    nT = int(np.round(T / dt))
    tgrid = np.linspace(0.0, T, nT)


    # --- Initialize state (1,6) tensor ---
    logK = tf.math.log(tf.constant(params["K0"], dtype=tf.float32))
    Z    = tf.constant(params["Z0"],    dtype=tf.float32)
    Y    = tf.constant(params["Y0"],    dtype=tf.float32)
    logR = tf.math.log(tf.constant(params["R0"], dtype=tf.float32))
    λ3   = tf.constant(0.0,   dtype=tf.float32)   # start at no damage
    logξ = tf.math.log(tf.constant(ξ, dtype=tf.float32))


    x = tf.reshape(tf.stack([logK, Z, Y, logR, λ3, logξ], axis=0), (1, 6))

    # --- Storage ---
    # Use TensorArray for efficient accumulation inside Python loops
    ta_xs = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_ig = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_id = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_ir = tf.TensorArray(dtype=tf.float32, size=nT)

    # For distorted tech/damage weights per step: store scalars for the two tech branches
    ta_g_interm = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_g_post = tf.TensorArray(dtype=tf.float32, size=nT)

    # damage weights are vectors of length L at each time 
    if L > 0:
        ta_g_dmg = tf.TensorArray(dtype=tf.float32, size=nT, element_shape=(L,))
    else:
        ta_g_dmg = None

    # --- Helper functions (defined before first call) ---
    def _controls(xx: tf.Tensor):
        ig = model.i_g_nn(xx)
        id = model.i_d_nn(xx)
        ir = tf.exp(-model.i_r_nn(xx))
        return ig, id, ir

    def _v(xx: tf.Tensor) -> tf.Tensor:
        return model.v_nn(xx)

    def _tech_weights(xx: tf.Tensor, logξ_scalar: tf.Tensor) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """
        Returns [g_l_prime for Interm], [g_l_prime_prime for PostTech]
        """
        v_now = _v(xx)
        v_post = model.v_PreDamagePostTech_nn(xx)        # same X
        v_interm = model.v_PreDamageIntermTech_nn(xx)    # same X
        ξ = tf.exp(logξ_scalar)
        g_primeprime = [tf.exp(-1.0/ξ * (v_post - v_now))]    # π branch
        g_prime      = [tf.exp(-1.0/ξ * (v_interm - v_now))]  # (1-π) branch
        return g_prime, g_primeprime

    def _damage_weights(logK, Z, Y, logR, λ3, logξ_scalar) -> List[tf.Tensor]:
        """
        For each λ3_l, evaluate v_PostDamagePreTech at Y=y_upper and λ3=λ3_l,
        then return the list of g_l.
        """
        v_now = _v(tf.concat([logK, Z, Y, logR, λ3, logξ_scalar], axis=1))
        ξ = tf.exp(logξ_scalar)
        out = []
        for l in range(L):
            λ3_l = tf.ones_like(Y) * λ3_values[l]
            v_post = model.v_PostDamagePreTech_nn(
                tf.concat([logK, Z, tf.ones_like(Y) * y_upper, logR, λ3_l, logξ_scalar], axis=1)
            )
            g_l = tf.exp(-1.0/ξ * (v_post - v_now))
            out.append(g_l)
        return out

    # write initial entries (index 0)
    ta_xs = ta_xs.write(0, tf.reshape(x, (6,)))
    ig0, id0, ir0 = _controls(x)
    ta_ig = ta_ig.write(0, tf.reshape(ig0, ()))
    ta_id = ta_id.write(0, tf.reshape(id0, ()))
    ta_ir = ta_ir.write(0, tf.reshape(ir0, ()))

    gp0, gpp0 = _tech_weights(x, tf.reshape(logξ, (1,1)))
    # gp0 and gpp0 are lists of tensors (one element per branch); store the first scalar
    ta_g_interm = ta_g_interm.write(0, tf.reshape(gp0[0], ()))
    ta_g_post = ta_g_post.write(0, tf.reshape(gpp0[0], ()))

    if L > 0:
        gd0 = _damage_weights(x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4], x[:,4:5], x[:,5:6])
        # gd0 is list of length L with (1,1) tensors; convert to (L,) vector
        gd0_vec = tf.stack([tf.reshape(g, ()) for g in gd0])
        ta_g_dmg = ta_g_dmg.write(0, gd0_vec)

    # --- Step forward deterministically ---
    # loop uses explicit index to write into TensorArrays
    x_curr = x
    for i in range(1, nT):
        logK_t, Z_t, Y_t, logR_t, λ3_t, logξ_t = tf.split(x_curr, 6, axis=1)
        K_t = tf.exp(logK_t)

        # controls (1,1) each
        ig, id_, ir = _controls(x_curr)

        # inside_log terms
        inside_log_i_d = tf.math.maximum(1.0 + θ_d * id_, 1e-8)
        inside_log_i_g = tf.math.maximum(1.0 + θ_g * ig, 1e-8)

        # --- drifts under reference measure (deterministic path, no noise) ---
        v_logKlogK_term = 0.5 * (σ_d**2 * (1.0 - Z_t)**2 + σ_g**2 * Z_t**2)

        drift_logK = (α_d + Γ_d * tf.math.log(inside_log_i_d)) * (1.0 - Z_t) \
                   + (α_g + Γ_g * tf.math.log(inside_log_i_g)) * Z_t \
                   - v_logKlogK_term

        drift_Z = (
            (α_g + Γ_g * tf.math.log(inside_log_i_g))
            - (α_d + Γ_d * tf.math.log(inside_log_i_d))
            - Z_t * σ_g**2
            + (1.0 - Z_t) * σ_d**2
        ) * Z_t * (1.0 - Z_t)

        drift_Y = θ_bar * η * A_d * (1.0 - Z_t) * K_t

        # i_r appears via exp(ψ1 (log i_r + logK - logR))
        drift_logR = -ζ + ψ0 * tf.exp(ψ1 * (tf.math.log(ir) + logK_t - logR_t)) - 0.5 * σ_κ**2

        # step
        x_next = tf.concat(
            [
                logK_t + drift_logK * dt,
                tf.clip_by_value(Z_t + drift_Z * dt, 1e-6, 1.0 - 1e-6),  # keep in (0,1)
                Y_t + drift_Y * dt,
                logR_t + drift_logR * dt,
                λ3_t,          # λ3 is frozen in this pre-tech regime
                logξ_t,        # scenario preference kept fixed
            ],
            axis=1,
        )

        # store & advance
        ta_xs = ta_xs.write(i, tf.reshape(x_next, (6,)))
        ta_ig = ta_ig.write(i, tf.reshape(ig, ()))
        ta_id = ta_id.write(i, tf.reshape(id_, ()))
        ta_ir = ta_ir.write(i, tf.reshape(ir, ()))

        # weights for distorted jumps at x_next
        gp, gpp = _tech_weights(x_next, logξ_t)
        ta_g_interm = ta_g_interm.write(i, tf.reshape(gp[0], ()))
        ta_g_post = ta_g_post.write(i, tf.reshape(gpp[0], ()))
        if L > 0:
            gd = _damage_weights(x_next[:,0:1], x_next[:,1:2], x_next[:,2:3], x_next[:,3:4], x_next[:,4:5], x_next[:,5:6])
            gd_vec = tf.stack([tf.reshape(g, ()) for g in gd])
            ta_g_dmg = ta_g_dmg.write(i, gd_vec)

        x_curr = x_next
 
    # --- Stack results (tensor) ---
    X_tf = ta_xs.stack()  # shape (nT, 6)
    logK_path_tf = X_tf[:, 0]; Z_path_tf = X_tf[:, 1]; Y_path_tf = X_tf[:, 2]; logR_path_tf = X_tf[:, 3]
    λ3_path_tf = X_tf[:, 4]; logξ_path_tf = X_tf[:, 5]
    K_path_tf = tf.exp(logK_path_tf); R_path_tf = tf.exp(logR_path_tf); ξ_path_tf = tf.exp(logξ_path_tf)

    ig_path_tf = ta_ig.stack()
    id_path_tf = ta_id.stack()
    ir_path_tf = ta_ir.stack()  # this is I_r / K

    # stacked tech/damage weights as tensors
    g_interm_tf = ta_g_interm.stack()
    g_post_tf = ta_g_post.stack()
    if L > 0:
        g_dmg_arr_tf = ta_g_dmg.stack()  # shape (nT, L)
    else:
        g_dmg_arr_tf = tf.zeros((nT, 0), dtype=tf.float32)

    # --- Compute gradients once along the path to get h_* (distortion diagnostics) ---
    with tf.GradientTape() as tape:
        tape.watch(X_tf)
        v_vals = model.v_nn(X_tf)  # shape (nT, 1)
    grads = tape.gradient(v_vals, X_tf)

    dv_dlogK_tf = grads[:, 0]
    dv_dZ_tf    = grads[:, 1]
    dv_dY_tf    = grads[:, 2]
    dv_dlogR_tf = grads[:, 3]
    # grads wrt λ3, logξ not needed for h_*

    # h_* (under your pde_rhs definitions)
    h_d_tf = -1.0 / ξ  * ((dv_dlogK_tf - Z_path_tf * dv_dZ_tf) * (1.0 - Z_path_tf) * (σ_d))
    h_g_tf = -1.0 / ξ  * ((dv_dlogK_tf + (1.0 - Z_path_tf) * dv_dZ_tf) * (Z_path_tf) * (σ_g))
    # We are solving v = V - log N, so d(log N)/dY = λ1 + λ2 Y shows up below
    λ1 = params['λ1']; λ2 = params['λ2']
    h_y_tf = -1.0 / ξ  * ((dv_dY_tf - (λ1 + λ2 * Y_path_tf)) * (η * A_d * (1.0 - Z_path_tf) * K_path_tf) * ϛ)
    h_r_tf = -1.0 / ξ  * (σ_κ * dv_dlogR_tf)

    # --- Distorted jump intensities & probabilities (tech and damage) ---
    # Tech jump intensities (two branches): J_g' = (1-π) * exp(logR)/varrho, J_g'' = π * exp(logR)/varrho
    J_g_prime = (1.0 - π) * R_path_tf / varrho
    J_g_primeprime = π * R_path_tf / varrho

    # Damage intensity (tensor): use indicator (Y > y_lower) 
    J_d_tf = r1  * (tf.exp(r2  / 2.0 * tf.pow(Y_path_tf - y_lower , 2.0)) - 1.0) * tf.cast(Y_path_tf > y_lower , tf.float32)

    # Damage weights averaged over λ3 grid at Y=y_upper:
    if L > 0:
        g_dmg_avg_tf = tf.reduce_mean(g_dmg_arr_tf, axis=1)
    else:
        g_dmg_avg_tf = tf.zeros_like(J_d_tf)

    # "Distorted" intensities (Radon-Nikodym weighted), then cumulative probs
    λ_tech_interm_tf = g_interm_tf * J_g_prime 
    λ_tech_post_tf   = g_post_tf   * J_g_primeprime 
    λ_tech_total_tf  = λ_tech_interm_tf + λ_tech_post_tf

    λ_dmg_dist_tf = g_dmg_avg_tf * J_d_tf

    tech_jump_prob_tf = 1.0 - tf.exp(-tf.cumsum(λ_tech_total_tf * dt, axis=0))
    dmg_jump_prob_tf  = 1.0 - tf.exp(-tf.cumsum(λ_dmg_dist_tf * dt, axis=0))

    # --- Other diagnostics like output/consumption pieces ---
    # i_r is I_r / K, emissions E = η * A_d * (1-Z) * K
    E_path_tf = η * A_d * (1.0 - Z_path_tf) * K_path_tf
    I_g_tf = K_path_tf * Z_path_tf * ig_path_tf
    I_d_tf = K_path_tf * (1.0 - Z_path_tf) * id_path_tf
    I_r_tf = K_path_tf * ir_path_tf
    # Reference-output proxy
    y_path_tf = θ_bar * A_d * (1.0 - Z_path_tf) * K_path_tf 
    c_path_tf = (A_d - id_path_tf) * (1.0 - Z_path_tf) + (A_g - ig_path_tf) * Z_path_tf - ir_path_tf
 
    # --- Convert tensors to numpy once for return/saving ---
    X_mat = X_tf.numpy()
    logK_path = logK_path_tf.numpy(); Z_path = Z_path_tf.numpy(); Y_path = Y_path_tf.numpy(); logR_path = logR_path_tf.numpy()
    λ3_path = λ3_path_tf.numpy(); logξ_path = logξ_path_tf.numpy()
    K_path = K_path_tf.numpy(); R_path = R_path_tf.numpy(); ξ_path = ξ_path_tf.numpy()

    ig_path = ig_path_tf.numpy(); id_path = id_path_tf.numpy(); ir_path = ir_path_tf.numpy()

    g_interm = g_interm_tf.numpy(); g_post = g_post_tf.numpy()
    g_dmg_arr = g_dmg_arr_tf.numpy() if L > 0 else np.zeros((nT, 0))

    h_d = h_d_tf.numpy(); h_g = h_g_tf.numpy(); h_y = h_y_tf.numpy(); h_r = h_r_tf.numpy()
    tech_jump_prob = tech_jump_prob_tf.numpy(); dmg_jump_prob = dmg_jump_prob_tf.numpy()
    E_path = E_path_tf.numpy(); I_g = I_g_tf.numpy(); I_d = I_d_tf.numpy(); c_path = c_path_tf.numpy()

    # convert the new diagnostics
    I_r = I_r_tf.numpy()
    y_path = y_path_tf.numpy()

    # ensure export folder is available
    export_folder = params.get("export_folder", ".") + "/SimulationOutputs"
    # create directory if it doesn't exist
    os.makedirs(export_folder, exist_ok=True)

    # --- Package results ---
    data = {
        "t": tgrid,
        "logK": logK_path, "Z": Z_path, "Y": Y_path, "logR": logR_path, "λ3": λ3_path, "logξ": logξ_path,
        "K": K_path, "R": R_path, "ξ": ξ_path,
        "i_g": ig_path, "i_d": id_path, "i_r": ir_path,
        "I_g": I_g, "I_d": I_d, "I_r": I_r, "E": E_path, "c": c_path,
        "y_consumption": y_path,
        "h_d": h_d, "h_g": h_g, "h_y": h_y, "h_r": h_r,
        "tech_jump_prob": tech_jump_prob,
        "dmg_jump_prob": dmg_jump_prob,
        "g_interm": g_interm, "g_post": g_post,
    }

    # --- Save numerics ---
    for k, v in data.items():
        np.savetxt(os.path.join(export_folder, f"{k}.txt"), np.asarray(v))

    # --- Plots (optional, simple set) ---
    if make_plots:
        def _plt(y, title, fname, ylim=None):
            plt.figure()
            plt.plot(tgrid, y)
            plt.xlabel("Years")
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.savefig(os.path.join(export_folder, fname))
            plt.close()

        _plt(data["logK"], r"$\log K$", "logK.png")
        _plt(data["Z"],    r"$Z$",       "Z.png", ylim=(0,1))
        _plt(data["Y"],    r"$Y$",       "Y.png")
        _plt(data["logR"], r"$\log R$",  "logR.png")
        _plt(data["i_g"],  r"$i_g$",     "i_g.png")
        _plt(data["i_d"],  r"$i_d$",     "i_d.png")
        _plt(data["i_r"],  r"$i_r$",     "i_r.png")
        _plt(data["E"],    "Emissions",  "E.png")
        _plt(data["tech_jump_prob"], "Distorted Tech Jump Prob.", "tech_jump_prob.png", ylim=(0,1))
        _plt(data["dmg_jump_prob"],  "Distorted Damage Jump Prob.", "dmg_jump_prob.png", ylim=(0,1))
        _plt(data["h_y"],  r"$h_Y$", "h_y.png")
        _plt(data["h_d"],  r"$h_d$", "h_d.png")
        _plt(data["h_g"],  r"$h_g$", "h_g.png")
        _plt(data["h_r"],  r"$h_r$", "h_r.png")
        
    # ============================
    # Extra plots: Distortion over λ3 (damage models) and Climate sensitivity
    # ============================
    # ---- Distorted Probability of Damage Models (over λ3) ----
    try:
        # final-time per-λ3 weights g_l at Y=y_upper:
        # g_dmg_arr has shape (nT, L); grab last row and normalize.
        final_g = g_dmg_arr[-1, :] if g_dmg_arr.size else np.array([])
        if final_g.size:
            distorted = final_g / final_g.sum()
            L_eff = distorted.size
            baseline = np.ones(L_eff) / L_eff

            # λ3 grid on x-axis
            x1 = np.array(λ3_values, dtype=float).reshape(-1)
            # edges for a smooth-ish histogram regardless of grid shape
            lo, hi = float(x1.min()), float(x1.max())
            bin_edges = np.linspace(lo, hi, L+1)

            print(f"Climate (damage) models distorted weights over λ3: {distorted}")

            plt.figure()
            plt.hist(x1, weights=baseline, bins=bin_edges, label='Baseline',
                     color='C3', alpha=0.5, ec='darkgrey')
            plt.hist(x1, weights=distorted, bins=bin_edges, label='Distorted',
                     color='C0', alpha=0.5, ec='darkgrey')
            plt.title("Distorted Probability of Damage Models")
            plt.xlabel(r"$\lambda_3$")
            plt.legend()
            plt.xlim([lo, hi])
            # pick a reasonable y-limit — similar to your old 0.3 default
            plt.ylim([0,  1.1 * distorted.max()])
            plt.savefig(os.path.join(export_folder, 'Dmg_Dist.png'))
            plt.close()

            # optionally store the vectors
            np.savetxt(os.path.join(export_folder, "lambda3_grid.txt"), x1)
            np.savetxt(os.path.join(export_folder, "lambda3_weights_distorted.txt"), distorted)

            data["λ3_grid"] = x1 
            data["λ3_weights_distorted"] = distorted
    except Exception as e:
        print("Warning: could not plot damage-model distortion over λ3:", e)

    # ---- Distorted Probability of Climate Models (θℓ shifted by varsigma * h_y(T)) ----
    theta_ell_csv =  "./model144.csv" 
    try:
        if theta_ell_csv is not None and os.path.exists(theta_ell_csv):
            import pandas as pd
            theta_ell = (pd.read_csv(theta_ell_csv, header=None).to_numpy()[:,0]/1000).astype(np.float32)
            pi_c_o = np.ones_like(theta_ell, dtype=np.float32) / len(theta_ell)
            ϛ   = params['ϛ'] 
            θ_bar = params['θ_bar']
            shift = ϛ  * data["h_y"][-1]           # your formula
            print("Climate sensitivity distortion shift (varsigma * h_y_T):", shift)

            # match your old scaling to 'thousandths'
            theta_base = 1000.0 * theta_ell
            theta_dist = 1000.0 * (theta_ell + shift)

            bins = np.linspace(0.8, 3.0, 16)

            plt.figure()
            plt.hist(theta_base, weights=pi_c_o, bins=bins, label="Baseline",
                     color='C3', alpha=0.5, density=True, ec='darkgrey')
            plt.hist(theta_dist, weights=pi_c_o, bins=bins, label="Distorted",
                     color='C0', alpha=0.5, density=True, ec='darkgrey')
            plt.title("Distorted Probability of Climate Models")
            plt.xlabel("Climate Sensitivity")
            plt.xlim(0.8, 3.0)
            plt.ylim(0, None)  # autoscale height
            plt.legend()
            plt.savefig(os.path.join(export_folder, 'Climate_Dist.png'))
            plt.close()

            # store arrays too
            np.savetxt(os.path.join(export_folder, "theta_ell_baseline_x1000.txt"), theta_base) 
            np.savetxt(os.path.join(export_folder, "theta_ell_weights_uniform.txt"), pi_c_o)

            data["theta_ell_baseline_x1000"] = theta_base
            data["theta_ell_distorted_x1000"] = theta_dist
        else:
            if theta_ell_csv is None:
                print("Info: theta_ell_csv not provided; skipping Climate_Dist plot.")
            else:
                print(f"Warning: theta_ell_csv not found at {theta_ell_csv}; skipping Climate_Dist plot.")
    except Exception as e:
        print("Warning: could not plot climate sensitivity distortion:", e)

    return data

 


# Example usage
if __name__ == "__main__":

    #####################
    #### change your folder for different results 
    ####################
    export_folder                    = "/project/lhansen/Cap_damage/TwoStageTechJump/output_reformed/TechSearch_LR_piecewiseconstant_40e-6,40e-4,40e-4,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations100000"
    batch_size                       = int("128")
    num_iterations                   = int("2000000")
    pretrained_path = None
    logging_frequency                = 1000
    learning_rates                   = [float(x) for x in "10e-4,10e-4,10e-4,10e-4".split(",")]
    hidden_layer_activations         = "swish,tanh,tanh,softplus".split(",")
    output_layer_activations         = "softplus,custom,custom,softplus".split(",")
    num_hidden_layers                = int("4")
    num_neurons                      = int("32")
    learning_rate_schedule_type      = "piecewiseconstant"
    export_folder_output             = "DeterministicSimulation"
        
        
    hidden_layer_activations   = [None if x == "None" else x for x in hidden_layer_activations]
    output_layer_activations   = [None if x == "None" else x for x in output_layer_activations]


    v_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_nn"}
    v_nn_config["final_activation"] = output_layer_activations[0]

    i_g_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[1], "dim" : 1, "nn_name" : "i_g_nn"}
    i_g_nn_config["final_activation"] = output_layer_activations[1]

    i_d_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[2], "dim" : 1, "nn_name" : "i_d_nn"}
    i_d_nn_config["final_activation"] = output_layer_activations[2]


    i_r_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[3], "dim" : 1, "nn_name" : "i_r_nn"}
    i_r_nn_config["final_activation"] = output_layer_activations[3]

    params = {"batch_size" : batch_size, "learning_rates":learning_rates,
    "v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config, "i_r_nn_config" : i_r_nn_config,
    "num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, 
    "pretrained_path" : pretrained_path, "learning_rate_schedule_type" : learning_rate_schedule_type}

    params["export_folder"]  = export_folder +  "/PreDamagePreTech"
    params["job_name"] =export_folder
    params["v_PostDamagePostTech_nn_path"]  = export_folder +  "/PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech"
    params["v_PreDamagePostTech_nn_path"]  = export_folder +  "/PreDamagePostTech/v_nn_checkpoint_PreDamagePostTech"
    params["v_PostDamagePreTech_nn_path"]  = export_folder +  "/PostDamagePreTech/v_nn_checkpoint_PostDamagePreTech"
    params["v_PostDamageIntermTech_nn_path"]  = export_folder +  "/PostDamageIntermTech/v_nn_checkpoint_PostDamageIntermTech"
    params["v_PreDamageIntermTech_nn_path"]  = export_folder +  "/PreDamageIntermTech/v_nn_checkpoint_PreDamageIntermTech"
    ## i_g and i_d activations come after params because we amy want to use phi_g and phi_d
    phi_g = 16.7
    phi_d = 16.7
    if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
        params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_g) / (tf.exp(2 * x) + 1.0)
        params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_d) / (tf.exp(2 * x) + 1.0)

    PARAMS.update(params)

    model = load_PreDamagePreTech_model(export_folder)
    simulate_path_pre_damage_pretech(
                                    model,
                                    ξ= 0.1,
                                    T=5.0,
                                    dt=1/12,  )