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
from PreDamageIntermTech import PreDamageIntermTechModel
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
    n_inputs = 7 
    model.v_nn.build((None, n_inputs))
    model.i_g_nn.build((None, n_inputs))
    model.i_d_nn.build((None, n_inputs))
    model.i_r_nn.build((None, n_inputs))

    # --- main nets checkpoints ---
    pre_dir = params["export_folder"]
    model.v_nn.load_weights(os.path.join(pre_dir, "v_nn_checkpoint_PreDamagePreTech"))
    model.i_g_nn.load_weights(os.path.join(pre_dir, "i_g_nn_checkpoint_PreDamagePreTech"))
    model.i_d_nn.load_weights(os.path.join(pre_dir, "i_d_nn_checkpoint_PreDamagePreTech"))
    model.i_r_nn.load_weights(os.path.join(pre_dir, "i_r_nn_checkpoint_PreDamagePreTech"))

    return model


def load_PreDamageIntermTech_model(export_root, params_override=None):
    """Load the four-regime experiment whose initial technology state is intermediate."""
    params = PARAMS.copy()
    if params_override:
        params.update(params_override)

    params["v_PreDamagePostTech_nn_path"] = os.path.join(
        export_root, "PreDamagePostTech", "v_nn_checkpoint_PreDamagePostTech"
    )
    params["v_PostDamageIntermTech_nn_path"] = os.path.join(
        export_root, "PostDamageIntermTech", "v_nn_checkpoint_PostDamageIntermTech"
    )
    params["export_folder"] = os.path.join(export_root, "PreDamageIntermTech")

    model = PreDamageIntermTechModel(params)

    n_inputs = 7
    model.v_nn.build((None, n_inputs))
    model.i_g_nn.build((None, n_inputs))
    model.i_d_nn.build((None, n_inputs))
    model.i_r_nn.build((None, n_inputs))

    pre_dir = params["export_folder"]
    model.v_nn.load_weights(os.path.join(pre_dir, "v_nn_checkpoint_PreDamageIntermTech"))
    model.i_g_nn.load_weights(os.path.join(pre_dir, "i_g_nn_checkpoint_PreDamageIntermTech"))
    model.i_d_nn.load_weights(os.path.join(pre_dir, "i_d_nn_checkpoint_PreDamageIntermTech"))
    model.i_r_nn.load_weights(os.path.join(pre_dir, "i_r_nn_checkpoint_PreDamageIntermTech"))

    return model


def checkpoint_exists(export_root, stage):
    checkpoint = os.path.join(export_root, stage, f"v_nn_checkpoint_{stage}.index")
    return os.path.exists(checkpoint)


def infer_initial_tech_regime(export_root):
    """Infer which pre-damage technology regime supplies the deterministic policy."""
    if checkpoint_exists(export_root, "PreDamagePreTech"):
        return "pre"
    if checkpoint_exists(export_root, "PreDamageIntermTech"):
        return "intermediate"
    raise FileNotFoundError(
        f"No PreDamagePreTech or PreDamageIntermTech checkpoint found in {export_root}"
    )


def infer_tech_jump_intensity_scale(export_root):
    """Read the run-specific tech intensity scale; old baseline folders default to 1."""
    candidates = [
        os.path.join(export_root, "run_manifest.txt"),
        os.path.join(export_root, "PreDamagePreTech", "params.txt"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                if "Tech jump intensity scale:" in line or "tech_jump_intensity_scale:" in line:
                    try:
                        return float(line.split(":", 1)[1].strip().split()[0])
                    except (IndexError, ValueError):
                        pass
    return float(PARAMS.get("tech_jump_intensity_scale", 1.0))


def infer_tech_jump_probability(export_root):
    """Read the run-specific technology branch probability pi."""
    candidates = [
        os.path.join(export_root, "run_manifest.txt"),
        os.path.join(export_root, "PreDamagePreTech", "params.txt"),
        os.path.join(export_root, "PreDamageIntermTech", "params.txt"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                if "Tech jump probability pi:" in line or line.strip().startswith("π:"):
                    try:
                        return float(line.split(":", 1)[1].strip().split()[0])
                    except (IndexError, ValueError):
                        pass
    return float(PARAMS.get("π", 0.04))


def infer_one_tech_jump_mode(export_root):
    """True for four-regime one-tech-jump folders simulated from PreDamageIntermTech."""
    folder_name = os.path.basename(os.path.abspath(export_root))
    if folder_name.startswith("OneTechJump_"):
        return True

    manifest = os.path.join(export_root, "run_manifest.txt")
    if not os.path.exists(manifest):
        return False
    with open(manifest, "r") as f:
        text = f.read().lower()
    return "one technology jump" in text or "one-tech" in text


def grouped_first_jump_statistics(damage_intensity, tech_intensity, dt):
    """
    Compute path-conditional competing-risk first-jump statistics.

    Let lambda_d and lambda_g be the grouped distorted damage and technology
    intensities. Then S(t) = exp(-integral_0^t(lambda_d + lambda_g) ds),
    q_d(t) = lambda_d(t) S(t), and q_g(t) = lambda_g(t) S(t) are the
    first-jump subdensities. Conditional on a first jump occurring by the
    simulation horizon H, the plotted densities are
    f_j(t | T_first <= H) = q_j(t) / (1 - S(H)).
    """
    total_intensity = damage_intensity + tech_intensity
    cumulative_hazard = tf.cumsum(total_intensity * dt, exclusive=True)
    survival = tf.exp(-cumulative_hazard)

    interval_jump_probability = survival * (1.0 - tf.exp(-total_intensity * dt))
    # The last grid point is the horizon H. There is no [H, H + dt] interval
    # in the plotted horizon, so exclude it from probability accounting.
    horizon_mask = tf.concat(
        [tf.ones_like(total_intensity[:-1]), tf.zeros_like(total_intensity[-1:])],
        axis=0,
    )
    interval_jump_probability = interval_jump_probability * horizon_mask
    positive_total = total_intensity > 0.0
    damage_share = tf.where(
        positive_total, damage_intensity / total_intensity, tf.zeros_like(total_intensity)
    )
    tech_share = tf.where(
        positive_total, tech_intensity / total_intensity, tf.zeros_like(total_intensity)
    )
    damage_interval_probability = damage_share * interval_jump_probability
    tech_interval_probability = tech_share * interval_jump_probability
    horizon_jump_probability = tf.reduce_sum(interval_jump_probability)
    positive_horizon_probability = horizon_jump_probability > 0.0

    # These interval-density versions integrate exactly under the simulation
    # grid: sum_t (damage_density_t + tech_density_t) * dt = 1.
    conditional_damage_density = tf.where(
        positive_horizon_probability,
        damage_interval_probability / (horizon_jump_probability * dt),
        tf.zeros_like(damage_interval_probability),
    )
    conditional_tech_density = tf.where(
        positive_horizon_probability,
        tech_interval_probability / (horizon_jump_probability * dt),
        tf.zeros_like(tech_interval_probability),
    )

    damage_subdensity = damage_interval_probability / dt
    tech_subdensity = tech_interval_probability / dt
    damage_cumulative = tf.cumsum(
        damage_interval_probability, exclusive=True
    )
    tech_cumulative = tf.cumsum(
        tech_interval_probability, exclusive=True
    )
    conditional_damage_cumulative = tf.where(
        positive_horizon_probability,
        damage_cumulative / horizon_jump_probability,
        tf.zeros_like(damage_cumulative),
    )
    conditional_tech_cumulative = tf.where(
        positive_horizon_probability,
        tech_cumulative / horizon_jump_probability,
        tf.zeros_like(tech_cumulative),
    )

    return {
        "damage_intensity": damage_intensity,
        "tech_intensity": tech_intensity,
        "total_intensity": total_intensity,
        "survival": survival,
        "any_jump_probability": 1.0 - survival,
        "horizon_jump_probability": horizon_jump_probability,
        "damage_subdensity": damage_subdensity,
        "tech_subdensity": tech_subdensity,
        "damage_density": conditional_damage_density,
        "tech_density": conditional_tech_density,
        "conditional_damage_density": conditional_damage_density,
        "conditional_tech_density": conditional_tech_density,
        "damage_type_probability": damage_share,
        "tech_type_probability": tech_share,
        "damage_cumulative": damage_cumulative,
        "tech_cumulative": tech_cumulative,
        "conditional_damage_cumulative": conditional_damage_cumulative,
        "conditional_tech_cumulative": conditional_tech_cumulative,
    }


def first_jump_accounting_report(data, dt):
    damage_mass = float(np.sum(np.asarray(data["conditional_dmg_jump_density"]) * dt))
    tech_mass = float(np.sum(np.asarray(data["conditional_tech_jump_density"]) * dt))
    total_mass = damage_mass + tech_mass
    horizon_probability = float(np.asarray(data["first_jump_horizon_prob"]).reshape(-1)[0])
    return {
        "horizon_first_jump_probability": horizon_probability,
        "damage_conditional_mass": damage_mass,
        "technology_conditional_mass": tech_mass,
        "total_conditional_mass": total_mass,
        "damage_percent": 100.0 * damage_mass,
        "technology_percent": 100.0 * tech_mass,
        "integration_error": total_mass - 1.0,
    }


from typing import Dict, Any, Tuple, List
import numpy as np
def simulate_path_PreDamagePreTech(
    model,
    ξ ,
    T ,
    dt , 
    make_plots: bool = True):
    """
    Deterministic path simulation before the first damage or technology jump.
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
    A_g_prime = params['A_g_prime']
    A_g_prime_prime = params['A_g_prime_prime']
    initial_tech_regime = params.get("initial_tech_regime", "pre")
    A_g_current = A_g if initial_tech_regime == "pre" else A_g_prime
    α_d = params['α_d']; Γ_d = params['Γ_d']; θ_d = params['θ_d']; σ_d = params['σ_d']
    α_g = params['α_g']; Γ_g = params['Γ_g']; θ_g = params['θ_g']; σ_g = params['σ_g']
    ζ   = params['ζ'];   ψ0  = params['ψ0'];  ψ1  = params['ψ1'];  σ_κ = params['σ_κ']
    θ_bar = params['θ_bar']; η = params['η'];  ϛ = params['ϛ']
    varrho = params['varrho']
    tech_jump_intensity_scale = params.get('tech_jump_intensity_scale', 1.0)
    π = params['π']
    one_tech_jump_mode = bool(params.get("one_tech_jump_mode", False))
    # Damage jump
    r1 = params['r1']; r2 = params['r2']; y_lower = params['y_lower']; y_upper = params['y_upper']
    # Damage λ3 mixture
    L = params['L']; λ3_values = params['λ3_values']

    # --- Time grid ---
    nT = int(np.round(T / dt)) + 1
    tgrid = np.arange(nT, dtype=float) * dt


    # --- Initialize state (1,6) tensor ---
    logK = tf.math.log(tf.constant(params["K0"], dtype=tf.float32))
    Z    = tf.constant(params["Z0"],    dtype=tf.float32)
    Y    = tf.constant(params["Y0"],    dtype=tf.float32)
    logR = tf.math.log(tf.constant(params["R0"], dtype=tf.float32))
    λ3   = tf.constant(0.0,   dtype=tf.float32)   # start at no damage
    logξ = tf.math.log(tf.constant(ξ, dtype=tf.float32))


    # state layout: [logK, Z, Y, logR, λ3, logξ]
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
        """
        Build a 7-column input for the control networks from state `xx` (works for 6- or 7-col states).
        """
        # use the last column as logξ (works whether xx is (batch,6) or (batch,7))
        logK = xx[:, 0:1]
        Z = xx[:, 1:2]
        Y = xx[:, 2:3]
        logR = xx[:, 3:4]
        logξ_scalar = xx[:, -1:]

        nn_in = tf.concat([logK, Z, Y, logR, logξ_scalar, logξ_scalar, logξ_scalar], axis=1)

        ig = model.i_g_nn(nn_in)
        id = model.i_d_nn(nn_in)
        ir = tf.exp(-model.i_r_nn(nn_in))
        return ig, id, ir

    def _v(xx: tf.Tensor) -> tf.Tensor:
        return model.v_nn(xx)

    def _tech_weights(x, logξ_scalar):
        """
        Compute distorted tech weights given a state tensor `x` and `logξ_scalar`.

        Supports `x` with either 6 or 7 columns. Returns two lists:
        (g_prime for the intermediate branch, g_primeprime for the post-tech branch).
        """
        # Determine number of columns if available (eager); otherwise use tf.shape for graph mode
        try:
            ncols = x.shape[1]
        except Exception:
            ncols = None
        if ncols is None:
            ncols = tf.shape(x)[1]

        # Extract common fields (works for both 6- and 7-column layouts)
        # layout expected: [logK, Z, Y, logR, ...]
        logK = x[:, 0:1]
        Z = x[:, 1:2]
        Y = x[:, 2:3]
        logR = x[:, 3:4]

        # Build v inputs replicating logξ into the three trailing slots as older nets expect
        v_input = tf.concat([logK, Z, Y, logR, logξ_scalar, logξ_scalar, logξ_scalar], axis=1)
        v_now = _v(v_input)

        # v_post uses A_g_prime_prime as the 4th component for the post-tech helper net
        v_post_input = tf.concat([
            logK,
            Z,
            Y,
            A_g_prime_prime * tf.ones_like(Y),
            logξ_scalar,
            logξ_scalar,
        ], axis=1)
        v_post = model.v_PreDamagePostTech_nn(v_post_input)

        ξ = tf.exp(logξ_scalar)
        g_primeprime = [tf.exp(-1.0 / ξ * (v_post - v_now))]
        if (
            initial_tech_regime == "pre"
            and float(tech_jump_probability) < 1.0 - 1e-12
            and getattr(model, "v_PreDamageIntermTech_nn", None) is not None
        ):
            # v_interm uses the same full-state layout as v_now
            v_interm = model.v_PreDamageIntermTech_nn(v_input)
            g_prime = [tf.exp(-1.0 / ξ * (v_interm - v_now))]
        else:
            # The intermediate branch is absent for direct pi=1 one-jump runs,
            # and inactive outside the pre-technology state.
            g_prime = [tf.zeros_like(v_now)]
        return g_prime, g_primeprime

    def _damage_weights(logK, Z, Y, logR, λ3, logξ_scalar) -> List[tf.Tensor]:
        """
        For each λ3_l, evaluate the matching post-damage continuation value,
        then return the list of g_l.
        """
        v_now = _v(tf.concat([logK, Z, Y, logR,   logξ_scalar, logξ_scalar, logξ_scalar], axis=1))
        ξ = tf.exp(logξ_scalar)
        out = []
        for l in range(L):
            λ3_l = tf.ones_like(Y) * λ3_values[l]
            damage_continuation = (
                model.v_PostDamagePreTech_nn
                if initial_tech_regime == "pre"
                else model.v_PostDamageIntermTech_nn
            )
            v_post = damage_continuation(
              tf.concat([logK, Z, tf.ones(tf.shape(Y)) * y_upper, logR,  λ3_l , logξ_scalar, logξ_scalar, logξ_scalar], 1)
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
    # v_nn expects a 7-column input (logK, Z, Y, logR, logξ, logξ, logξ).
    # Build that input from X_tf by replicating the logξ column.
    logξ_col = tf.expand_dims(X_tf[:, 5], axis=1)  # shape (nT, 1)
    v_inp = tf.concat([X_tf[:, 0:4], logξ_col, logξ_col, logξ_col], axis=1)

    with tf.GradientTape() as tape:
        tape.watch(v_inp)
        v_vals = model.v_nn(v_inp)  # shape (nT, 1)
    grads_vinp = tape.gradient(v_vals, v_inp)

    # Map gradients back to original state variables (first four positions)
    dv_dlogK_tf = grads_vinp[:, 0]
    dv_dZ_tf    = grads_vinp[:, 1]
    dv_dY_tf    = grads_vinp[:, 2]
    dv_dlogR_tf = grads_vinp[:, 3]

    # h_* (under your pde_rhs definitions)
    h_d_tf = -1.0 / ξ  * ((dv_dlogK_tf - Z_path_tf * dv_dZ_tf) * (1.0 - Z_path_tf) * (σ_d))
    h_g_tf = -1.0 / ξ  * ((dv_dlogK_tf + (1.0 - Z_path_tf) * dv_dZ_tf) * (Z_path_tf) * (σ_g))
    # We are solving v = V - log N, so d(log N)/dY = λ1 + λ2 Y shows up below
    λ1 = params['λ1']; λ2 = params['λ2']
    h_y_tf = -1.0 / ξ  * ((dv_dY_tf - (λ1 + λ2 * Y_path_tf)) * (η * A_d * (1.0 - Z_path_tf) * K_path_tf) * ϛ)
    h_r_tf = -1.0 / ξ  * (σ_κ * dv_dlogR_tf)

    # --- Distorted grouped intensities and competing-risk first-jump densities ---
    if initial_tech_regime == "pre":
        J_g_prime = tech_jump_intensity_scale * (1.0 - π) * R_path_tf / varrho
        J_g_primeprime = tech_jump_intensity_scale * π * R_path_tf / varrho
    elif one_tech_jump_mode:
        # Four-regime one-jump experiments use PreDamageIntermTech as the
        # initial policy/value model.  For first-jump diagnostics, treat the
        # remaining technology arrival as one grouped jump with intensity
        # scale * R / varrho, matching the two-tech-jump grouped intensity.
        J_g_prime = tf.zeros_like(R_path_tf)
        J_g_primeprime = tech_jump_intensity_scale * R_path_tf / varrho
    else:
        # Mirror the intermediate-tech HJB exactly: only its pi-weighted
        # intermediate-to-post branch is active.
        J_g_prime = tf.zeros_like(R_path_tf)
        J_g_primeprime = tech_jump_intensity_scale * π * R_path_tf / varrho

    # Damage intensity (tensor): use indicator (Y > y_lower) 
    J_d_tf = r1  * (tf.exp(r2  / 2.0 * tf.pow(Y_path_tf - y_lower , 2.0)) - 1.0) * tf.cast(Y_path_tf > y_lower , tf.float32)

    # Damage weights averaged over λ3 grid at Y=y_upper:
    if L > 0:
        g_dmg_avg_tf = tf.reduce_mean(g_dmg_arr_tf, axis=1)
    else:
        g_dmg_avg_tf = tf.zeros_like(J_d_tf)

    # Group all technology branches and all damage realizations.
    λ_tech_interm_tf = g_interm_tf * J_g_prime 
    λ_tech_post_tf   = g_post_tf   * J_g_primeprime 
    λ_tech_total_tf  = λ_tech_interm_tf + λ_tech_post_tf

    λ_dmg_dist_tf = g_dmg_avg_tf * J_d_tf
    first_jump = grouped_first_jump_statistics(λ_dmg_dist_tf, λ_tech_total_tf, dt)
    tech_jump_prob_tf = first_jump["tech_cumulative"]
    dmg_jump_prob_tf = first_jump["damage_cumulative"]
    tech_jump_density_tf = first_jump["tech_density"]
    dmg_jump_density_tf = first_jump["damage_density"]

    # --- Other diagnostics like output/consumption pieces ---
    # i_r is I_r / K, emissions E = η * A_d * (1-Z) * K
    E_path_tf = η * A_d * (1.0 - Z_path_tf) * K_path_tf
    I_g_tf = K_path_tf * Z_path_tf * ig_path_tf
    I_d_tf = K_path_tf * (1.0 - Z_path_tf) * id_path_tf
    I_r_tf = K_path_tf * ir_path_tf
    output_per_capital_tf = A_d * (1.0 - Z_path_tf) + A_g_current * Z_path_tf
    y_path_tf = output_per_capital_tf * K_path_tf
    c_path_tf = (A_d - id_path_tf) * (1.0 - Z_path_tf) + (A_g_current - ig_path_tf) * Z_path_tf - ir_path_tf
    C_path_tf = c_path_tf * K_path_tf
    ConsumptionOutputRatio_path_tf = c_path_tf / output_per_capital_tf
    RD_path_tf = ir_path_tf / output_per_capital_tf
    DirtyInvestment_path_tf = (1.0 - Z_path_tf) * id_path_tf / output_per_capital_tf
    GreenInvestment_path_tf = Z_path_tf * ig_path_tf / output_per_capital_tf
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
    tech_jump_density = tech_jump_density_tf.numpy(); dmg_jump_density = dmg_jump_density_tf.numpy()
    tech_jump_subdensity = first_jump["tech_subdensity"].numpy()
    dmg_jump_subdensity = first_jump["damage_subdensity"].numpy()
    conditional_tech_jump_density = first_jump["conditional_tech_density"].numpy()
    conditional_dmg_jump_density = first_jump["conditional_damage_density"].numpy()
    conditional_tech_jump_prob = first_jump["conditional_tech_cumulative"].numpy()
    conditional_dmg_jump_prob = first_jump["conditional_damage_cumulative"].numpy()
    first_jump_type_tech_prob = first_jump["tech_type_probability"].numpy()
    first_jump_type_dmg_prob = first_jump["damage_type_probability"].numpy()
    tech_jump_intensity = first_jump["tech_intensity"].numpy()
    dmg_jump_intensity = first_jump["damage_intensity"].numpy()
    total_jump_intensity = first_jump["total_intensity"].numpy()
    first_jump_survival = first_jump["survival"].numpy()
    any_jump_prob = first_jump["any_jump_probability"].numpy()
    first_jump_horizon_prob = first_jump["horizon_jump_probability"].numpy()
    E_path = E_path_tf.numpy(); I_g = I_g_tf.numpy(); I_d = I_d_tf.numpy(); c_path = c_path_tf.numpy()
    C_path = C_path_tf.numpy()
    ConsumptionOutputRatio_path = ConsumptionOutputRatio_path_tf.numpy()

    # convert the new diagnostics
    I_r = I_r_tf.numpy()
    y_path = y_path_tf.numpy()
    RD_path = RD_path_tf.numpy()
    # ensure export folder is available
    export_folder = params.get("save_folder", ".") + "/SimulationOutputs" +f"_ξ_{ξ:.3f}"
    # create directory if it doesn't exist
    os.makedirs(export_folder, exist_ok=True)

    # --- Package results ---
    data = {
        "t": tgrid,
        "logK": logK_path, "Z": Z_path, "Y": Y_path, "logR": logR_path, "λ3": λ3_path, "logξ": logξ_path,
        "K": K_path, "R": R_path, "ξ": ξ_path,
        "i_g": ig_path, "i_d": id_path, "i_r": ir_path,
        "I_g": I_g, "I_d": I_d, "I_r": I_r, "E": E_path, "c": c_path, "C": C_path,
        "Output": y_path, "ConsumptionOutputRatio": ConsumptionOutputRatio_path,
        "RD": RD_path, "DirtyInvestment": DirtyInvestment_path_tf, "GreenInvestment": GreenInvestment_path_tf,
        "y_consumption": y_path,
        "h_d": h_d, "h_g": h_g, "h_y": h_y, "h_r": h_r,
        "tech_jump_prob": tech_jump_prob,
        "dmg_jump_prob": dmg_jump_prob,
        "tech_jump_density": tech_jump_density,
        "dmg_jump_density": dmg_jump_density,
        "tech_jump_subdensity": tech_jump_subdensity,
        "dmg_jump_subdensity": dmg_jump_subdensity,
        "conditional_tech_jump_density": conditional_tech_jump_density,
        "conditional_dmg_jump_density": conditional_dmg_jump_density,
        "conditional_tech_jump_prob": conditional_tech_jump_prob,
        "conditional_dmg_jump_prob": conditional_dmg_jump_prob,
        "first_jump_type_tech_prob": first_jump_type_tech_prob,
        "first_jump_type_dmg_prob": first_jump_type_dmg_prob,
        "tech_jump_intensity": tech_jump_intensity,
        "dmg_jump_intensity": dmg_jump_intensity,
        "total_jump_intensity": total_jump_intensity,
        "first_jump_survival": first_jump_survival,
        "any_jump_prob": any_jump_prob,
        "first_jump_horizon_prob": np.atleast_1d(first_jump_horizon_prob),
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
            if "Density Conditional on a First Jump" in title:
                area = float(np.sum(np.asarray(y) * dt))
                title = f"{title}\narea = {area:.4f}"
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
        _plt(data["ConsumptionOutputRatio"] * 100.0, "Consumption as % of Output (C/Y)", "ConsumptionOutputRatio.png")
        _plt(data["tech_jump_prob"], "Technology First-Jump Cumulative Incidence", "tech_jump_prob.png", ylim=(0,1))
        _plt(data["dmg_jump_prob"],  "Damage First-Jump Cumulative Incidence", "dmg_jump_prob.png", ylim=(0,1))
        _plt(data["tech_jump_density"], "Technology First-Jump Density", "tech_jump_density.png")
        _plt(data["dmg_jump_density"],  "Damage First-Jump Density", "dmg_jump_density.png")
        _plt(data["conditional_tech_jump_density"], "Technology Density Conditional on a First Jump by Horizon", "conditional_tech_jump_density.png")
        _plt(data["conditional_dmg_jump_density"], "Damage Density Conditional on a First Jump by Horizon", "conditional_dmg_jump_density.png")
        _plt(data["conditional_tech_jump_prob"], "Conditional Cumulative Probability: Technology First Jump", "conditional_tech_jump_prob.png", ylim=(0,1))
        _plt(data["conditional_dmg_jump_prob"], "Conditional Cumulative Probability: Damage First Jump", "conditional_dmg_jump_prob.png", ylim=(0,1))
        _plt(data["first_jump_type_tech_prob"], "Probability First Jump Is Technology Given Its Time", "first_jump_type_tech_prob.png", ylim=(0,1))
        _plt(data["first_jump_type_dmg_prob"], "Probability First Jump Is Damage Given Its Time", "first_jump_type_dmg_prob.png", ylim=(0,1))
        _plt(data["h_y"],  r"$h_Y$", "h_y.png")
        _plt(data["h_d"],  r"$h_d$", "h_d.png")
        _plt(data["h_g"],  r"$h_g$", "h_g.png")
        _plt(data["h_r"],  r"$h_r$", "h_r.png")

    accounting = first_jump_accounting_report(data, dt)
    with open(os.path.join(export_folder, "first_jump_density_accounting.txt"), "w") as f:
        f.write(f"xi = {ξ}\n")
        f.write(f"dt = {dt}\n")
        for key, value in accounting.items():
            f.write(f"{key}: {value:.12g}\n")
    print(
        "First-jump density accounting "
        f"(xi={ξ}): damage={accounting['damage_percent']:.6f}%, "
        f"technology={accounting['technology_percent']:.6f}%, "
        f"total_mass={accounting['total_conditional_mass']:.12f}, "
        f"integration_error={accounting['integration_error']:.3e}"
    )
        
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
            plt.ylim([0,  0.6])
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    theta_ell_csv = os.path.join(script_dir, "model144.csv")
    
    # theta_ell_csv =  "./model144.csv" 
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
            plt.ylim(0, 1.3)  # autoscale height
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


    # ---- Distorted Probability of Technology Models (two branches) ----
    try:
        # positions on x-axis: A_g_prime (interm) and A_g_prime_prime (post)
        x_pos = np.array([float(A_g_prime), float(A_g_prime_prime)], dtype=float)

        # Conditional destination probabilities for a technology jump.
        if initial_tech_regime == "pre":
            baseline_tech = np.array([float(1.0 - π), float(π)], dtype=float)
        else:
            baseline_tech = np.array([0.0, 1.0], dtype=float)

        # final-time distortion multipliers from g_interm / g_post (numpy arrays)
        final_g_interm = float(g_interm[-1]) if np.asarray(g_interm).size else 1.0
        final_g_post   = float(g_post[-1])   if np.asarray(g_post).size else 1.0

        # apply baseline weights and distort by g-factors (interm -> A_g_prime, post -> A_g_prime_prime)
        weighted = baseline_tech * np.array([final_g_interm, final_g_post], dtype=float)

        # normalize to get distorted probability vector (fallback to baseline if zero)
        if weighted.sum() > 0.0:
            distorted_tech = weighted / weighted.sum()
        else:
            distorted_tech = baseline_tech.copy()

        # bar width: small relative to spacing (handle equal or very-close positions)
        span = float(np.abs(x_pos[1] - x_pos[0]))
        width = 0.02 if span == 0.0 else max(0.02, 0.08 * span)
        print(distorted_tech)
        # plot baseline vs distorted at the same x positions (side-by-side)
        plt.figure()
        plt.bar(x_pos  , baseline_tech, width=width, label="Baseline", color="C3", alpha=0.6, ec="darkgrey")
        plt.bar(x_pos , distorted_tech, width=width, label="Distorted", color="C0", alpha=0.6, ec="darkgrey")
        # label ticks with the numeric A_g values
        plt.xticks(x_pos, [f"A_g'={x_pos[0]:.3f}", f"A_g''={x_pos[1]:.3f}"])
        plt.ylim(0, 1.0)
        plt.title("Distorted Probability of Technology Models")
        plt.xlabel(r"$A'_g$ positions (interm at left, post at right)")
        plt.legend()
        plt.savefig(os.path.join(export_folder, "Tech_Dist_bar_at_Agprime.png"))
        plt.close()

        # save numeric outputs and add to data dict
        np.savetxt(os.path.join(export_folder, "tech_baseline_at_Agprime.txt"), baseline_tech)
        np.savetxt(os.path.join(export_folder, "tech_distorted_at_Agprime.txt"), distorted_tech)
        data["tech_baseline_at_Agprime"] = baseline_tech
        data["tech_distorted_at_Agprime"] = distorted_tech

    except Exception as e:
        print("Warning: could not compute/plot tech distortion:", e)

    return data

 
if __name__ == "__main__":
    import sys 
    
    # Promising one
    # export_folder = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
    export_folder = sys.argv[1]
    # ============================================================
    # Network/training metadata (used to reconstruct the nets)
    # ============================================================
    batch_size = 128
    num_iterations = 2_000_000
    pretrained_path = None
    logging_frequency = 1000
    learning_rates = [float(x) for x in "10e-4,10e-4,10e-4,10e-4".split(",")]

    hidden_layer_activations = "swish,tanh,tanh,softplus".split(",")
    output_layer_activations = "softplus,custom,custom,softplus".split(",")

    num_hidden_layers = 4
    num_neurons = 32
    learning_rate_schedule_type = "piecewiseconstant"

    # Normalize "None" strings if you ever pass them
    hidden_layer_activations = [None if x == "None" else x for x in hidden_layer_activations]
    output_layer_activations = [None if x == "None" else x for x in output_layer_activations]

    v_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden_layer_activations[0],
        "dim": 1,
        "nn_name": "v_nn",
        "final_activation": output_layer_activations[0],
    }

    i_g_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden_layer_activations[1],
        "dim": 1,
        "nn_name": "i_g_nn",
        "final_activation": output_layer_activations[1],
    }

    i_d_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden_layer_activations[2],
        "dim": 1,
        "nn_name": "i_d_nn",
        "final_activation": output_layer_activations[2],
    }

    i_r_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden_layer_activations[3],
        "dim": 1,
        "nn_name": "i_r_nn",
        "final_activation": output_layer_activations[3],
    }

    tech_jump_intensity_scale = infer_tech_jump_intensity_scale(export_folder)
    tech_jump_probability = infer_tech_jump_probability(export_folder)
    initial_tech_regime = infer_initial_tech_regime(export_folder)
    one_tech_jump_mode = infer_one_tech_jump_mode(export_folder)
    simulation_y0 = float(os.environ.get("SIMULATION_Y0", PARAMS.get("Y0", 1.1)))

    params = {
        "batch_size": batch_size,
        "learning_rates": learning_rates,
        "v_nn_config": v_nn_config,
        "i_g_nn_config": i_g_nn_config,
        "i_d_nn_config": i_d_nn_config,
        "i_r_nn_config": i_r_nn_config,
        "num_iterations": num_iterations,
        "logging_frequency": logging_frequency,
        "verbose": True,
        "pretrained_path": pretrained_path,
        "learning_rate_schedule_type": learning_rate_schedule_type,
        "tech_jump_intensity_scale": tech_jump_intensity_scale,
        "π": tech_jump_probability,
        "initial_tech_regime": initial_tech_regime,
        "one_tech_jump_mode": one_tech_jump_mode,
        "Y0": simulation_y0,
    }
    

    # IMPORTANT: these paths are used by your model class to load helper nets
    # params["export_folder"] = os.path.join(export_folder, "PreDamagePreTech")
    params["save_folder"] = os.path.join(export_folder, "SimulationDeterministic") 
    params["v_PostDamagePostTech_nn_path"] = os.path.join(export_folder, "PostDamagePostTech", "v_nn_checkpoint_PostDamagePostTech")
    params["v_PreDamagePostTech_nn_path"] = os.path.join(export_folder, "PreDamagePostTech", "v_nn_checkpoint_PreDamagePostTech")
    params["v_PostDamagePreTech_nn_path"] = os.path.join(export_folder, "PostDamagePreTech", "v_nn_checkpoint_PostDamagePreTech")
    params["v_PostDamageIntermTech_nn_path"] = os.path.join(export_folder, "PostDamageIntermTech", "v_nn_checkpoint_PostDamageIntermTech")
    params["v_PreDamageIntermTech_nn_path"] = os.path.join(export_folder, "PreDamageIntermTech", "v_nn_checkpoint_PreDamageIntermTech")

    # Custom final activations (must be set BEFORE loading weights)
    phi_g = 16.7
    phi_d = 16.7
    if (output_layer_activations[1] == "custom") or (output_layer_activations[2] == "custom"):
        params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0 / phi_g) / (tf.exp(2.0 * x) + 1.0)
        params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0 / phi_d) / (tf.exp(2.0 * x) + 1.0)

    # Push into global PARAMS used by your model modules
    PARAMS.update(params)
    
    

    # ============================================================
    # Load + simulate
    # ============================================================
    if initial_tech_regime == "pre":
        model = load_PreDamagePreTech_model(export_folder)
        print("Loaded PreDamagePreTech model with value, controls, and helper nets.")
    else:
        model = load_PreDamageIntermTech_model(export_folder)
        print("Loaded PreDamageIntermTech model with value, controls, and helper nets.")
    print(f"Tech jump intensity scale: {tech_jump_intensity_scale}")
    print(f"Tech jump probability pi: {tech_jump_probability}")
    print(f"Initial technology regime: {initial_tech_regime}")
    print(f"One-tech-jump simulation mode: {one_tech_jump_mode}")
    print(f"Initial deterministic temperature Y0: {simulation_y0}")
    if initial_tech_regime == "intermediate" and one_tech_jump_mode:
        print(
            "Specification note: one-tech-jump simulation uses the "
            "PreDamageIntermTech model for controls/value and a single grouped "
            "technology intensity scale * R / varrho."
        )
    elif initial_tech_regime == "intermediate" and tech_jump_probability == 0.0:
        print(
            "Specification note: the intermediate-tech HJB's only technology "
            "jump intensity is scale * pi * R / varrho, so technology first-jump "
            "density is identically zero when pi = 0."
        )

    os.makedirs(params["save_folder"], exist_ok=True)
    with open(os.path.join(params["save_folder"], "first_jump_density_specification.txt"), "w") as f:
        f.write("Grouped path-conditional competing-risk density\n")
        f.write("lambda_damage = sum_l g_l J_d_l\n")
        f.write("lambda_technology = sum_k g_k J_g_k\n")
        f.write("survival = exp(-integral(lambda_damage + lambda_technology) dt)\n")
        f.write("damage_subdensity = lambda_damage * survival\n")
        f.write("technology_subdensity = lambda_technology * survival\n")
        f.write("horizon_jump_probability = 1 - survival(H)\n")
        f.write("conditional_damage_density = damage_subdensity / horizon_jump_probability\n")
        f.write("conditional_technology_density = technology_subdensity / horizon_jump_probability\n")
        f.write("conditional_damage_cumulative = damage_cumulative / horizon_jump_probability\n")
        f.write("conditional_technology_cumulative = technology_cumulative / horizon_jump_probability\n")
        f.write("damage_type_probability = damage_subdensity / (damage_subdensity + technology_subdensity)\n")
        f.write("technology_type_probability = technology_subdensity / (damage_subdensity + technology_subdensity)\n")
        f.write(f"initial_tech_regime = {initial_tech_regime}\n")
        f.write(f"tech_jump_probability_pi = {tech_jump_probability}\n")
        f.write(f"tech_jump_intensity_scale = {tech_jump_intensity_scale}\n")
        f.write(f"one_tech_jump_mode = {one_tech_jump_mode}\n")
        f.write(f"initial_temperature_Y0 = {simulation_y0}\n")
        if initial_tech_regime == "intermediate" and one_tech_jump_mode:
            f.write("one_jump_lambda_technology = tech_jump_intensity_scale * R / varrho\n")

    simulation_T = float(os.environ.get("SIMULATION_T", "60.0"))
    simulation_dt = float(os.environ.get("SIMULATION_DT", str(1 / 12)))
    xis = [
        float(value)
        for value in os.environ.get("SIMULATION_XIS", "0.01,0.05,0.1,148.6").split(",")
    ]
    for xi in xis:
        print(f"\n=== Simulating ξ = {xi} ===")
        _ = simulate_path_PreDamagePreTech(
            model,
            ξ=xi,
            T=simulation_T,
            dt=simulation_dt,
            make_plots=True,   # set False if you want faster runs
        )
        
