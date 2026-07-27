"""
v3-AWARE deterministic forward simulation (closed-form-control models_v3).

Adapted from models/SimulationDeterministic.py. The forward dynamics, regime/jump
structure, diagnostics, outputs, and plots are IDENTICAL to the original so the
existing plotting can consume the same time series. The ONLY change is the control
map: models_v3 has NO i_g_nn/i_d_nn/i_r_nn -- controls are CLOSED FORM from v's
derivatives. `_closed_form_controls(...)` below reproduces EXACTLY the v3 pde_rhs
control algebra (q_d = dv_dlogK - Z*dv_dZ, q_g = dv_dlogK + (1-Z)*dv_dZ, the c
closed form, and the i_r bisection of FOC_r) so the simulated controls match
training. Only v_nn (and the neighbor value nets for jumps) are loaded.

INVOKE (once all 4 regimes are trained into <RUN>):
    cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
    module load python/anaconda-2021.05
    MODEL_GAMMA_D=0.12 MODEL_GAMMA_G=0.12 MODEL_THETA_D=8.35 MODEL_THETA_G=8.35 \
    python models_v3/SimulationDeterministic.py <RUN_FOLDER>
where RUN_FOLDER contains PreDamagePreTech/, PreDamagePostTech/, PostDamagePreTech/
(and PreDamageIntermTech/ when pi<1) each with v_nn_checkpoint_<stage>.
"""

import os
import ast
import tensorflow as tf
from params import PARAMS
from PreDamagePreTech import PreDamagePreTechModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List
import numpy as np


# ===========================================================================
#  CLOSED-FORM control map (EXACT copy of the v3 pde_rhs control algebra).
#  Pre-tech (has_ir=True): solves i_d, i_g AND i_r (R&D) consistently.
#  Post-tech (has_ir=False): i_r := 0, c = closed form without R&D.
# ===========================================================================
def _closed_form_controls(model, logK, Z, Y, logR, logxi_scalar, has_ir, A_g_use):
    """Return (i_g, i_d, i_r) for a (batch,1) state, via the v3 closed form.

    `A_g_use` is the green productivity entering the resource constraint
    (pre-tech: A_g; post-tech: A_g'').  v-derivatives are taken on the regime's
    own v_nn input layout (built here to match each regime's pde_rhs X).
    """
    P = model.params
    delta = P["δ"]; Gd = P["Γ_d"]; Gg = P["Γ_g"]
    td = P["θ_d"]; tg = P["θ_g"]; A_d = P["A_d"]
    psi0 = P["ψ0"]; psi1 = P["ψ1"]
    eps_q = tf.constant(1e-3, tf.float32)

    # Build the v_nn input EXACTLY as each regime's pde_rhs does, then autodiff.
    regime = model.__class__.__name__
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([logK, Z, logR])
        if regime == "PreDamagePreTechModel":
            X = tf.concat([logK, Z, Y, logR, logxi_scalar, logxi_scalar, logxi_scalar], 1)
        elif regime == "PostDamagePreTechModel":
            lam3 = tf.zeros_like(Y)  # deterministic pre-damage path: lambda3 = 0
            X = tf.concat([logK, Z, Y, logR, lam3, logxi_scalar, logxi_scalar, logxi_scalar], 1)
        elif regime == "PreDamagePostTechModel":
            X = tf.concat([logK, Z, Y, P["A_g_prime_prime"] * tf.ones_like(Y),
                           logxi_scalar, logxi_scalar], 1)
        else:  # PostDamagePostTechModel
            lam3 = tf.zeros_like(Y)
            X = tf.concat([logK, Z, Y, lam3, P["A_g_prime_prime"] * tf.ones_like(Y),
                           logxi_scalar, logxi_scalar], 1)
        v = model.v_nn(X)
    dv_dlogK = tape.gradient(v, logK)
    dv_dZ = tape.gradient(v, Z)
    dv_dlogR = tape.gradient(v, logR) if has_ir else None
    del tape

    # admissibility clamp on dv_dZ so q_d>0, q_g>0 (mirror FD slope clamp)
    dvZ_hi = (dv_dlogK - eps_q) / tf.maximum(Z, 1e-6)
    dvZ_lo = (eps_q - dv_dlogK) / tf.maximum(1.0 - Z, 1e-6)
    dv_dZ = tf.clip_by_value(dv_dZ, dvZ_lo, tf.maximum(dvZ_hi, dvZ_lo))
    q_d = tf.maximum(dv_dlogK - Z * dv_dZ, eps_q)
    q_g = tf.maximum(dv_dlogK + (1.0 - Z) * dv_dZ, eps_q)

    Abar = (1.0 - Z) * A_d + Z * A_g_use
    N_r = Abar + (1.0 - Z) / td + Z / tg
    S_r = (1.0 - Z) * Gd * q_d + Z * Gg * q_g

    if has_ir:
        B_r = psi0 * tf.exp(psi1 * (logK - logR)) * dv_dlogR
        has_root = B_r > 0.0
        i_r_min = tf.constant(1e-8, tf.float32)
        i_r_hi = tf.maximum(N_r * (1.0 - 1e-3), i_r_min * 10.0)

        def _focr(ir):
            c_of = delta * (N_r - ir) / (delta + S_r)
            return -delta / tf.maximum(c_of, 1e-8) + psi1 * B_r * tf.pow(ir, psi1 - 1.0)

        lo = tf.fill(tf.shape(N_r), i_r_min)
        hi = tf.ones_like(N_r) * i_r_hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            go_right = _focr(mid) > 0.0
            lo = tf.where(go_right, mid, lo)
            hi = tf.where(go_right, hi, mid)
        i_r = tf.where(has_root, 0.5 * (lo + hi), i_r_min)
        i_r = tf.maximum(i_r, i_r_min)
        c = delta * (N_r - i_r) / (delta + S_r)
    else:
        i_r = tf.zeros_like(N_r)
        c = delta * N_r / (delta + S_r)

    i_d = Gd * c * q_d / delta - 1.0 / td
    i_g = Gg * c * q_g / delta - 1.0 / tg
    return i_g, i_d, i_r


def load_PreDamagePreTech_model(export_root, params_override=None):
    """Load the PreDamagePreTech v3 model: v_nn + neighbor VALUE nets only
    (no control nets -- controls are closed-form)."""
    params = PARAMS.copy()
    if params_override:
        params.update(params_override)

    params["v_PreDamagePostTech_nn_path"]   = os.path.join(export_root, "PreDamagePostTech",   "v_nn_checkpoint_PreDamagePostTech")
    params["v_PreDamageIntermTech_nn_path"] = os.path.join(export_root, "PreDamageIntermTech", "v_nn_checkpoint_PreDamageIntermTech")
    params["v_PostDamagePreTech_nn_path"]   = os.path.join(export_root, "PostDamagePreTech",   "v_nn_checkpoint_PostDamagePreTech")
    params["export_folder"] = os.path.join(export_root, "PreDamagePreTech")

    model = PreDamagePreTechModel(params)

    n_inputs = 7
    model.v_nn.build((None, n_inputs))
    model.v_nn.load_weights(os.path.join(params["export_folder"], "v_nn_checkpoint_PreDamagePreTech"))
    return model


def checkpoint_exists(export_root, stage):
    checkpoint = os.path.join(export_root, stage, f"v_nn_checkpoint_{stage}.index")
    return os.path.exists(checkpoint)


def infer_initial_tech_regime(export_root):
    if checkpoint_exists(export_root, "PreDamagePreTech"):
        return "pre"
    raise FileNotFoundError(
        f"No PreDamagePreTech checkpoint found in {export_root} "
        f"(v3 one-jump simulation expects the pre-tech initial regime)."
    )


def infer_tech_jump_intensity_scale(export_root):
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


def infer_saved_economic_parameters(export_root):
    parameter_names = {"σ_d", "σ_g", "Γ_d", "Γ_g", "θ_d", "θ_g", "ψ0"}
    candidates = [
        os.path.join(export_root, "PreDamagePreTech", "params.txt"),
        os.path.join(export_root, "PostDamagePreTech", "params.txt"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        overrides = {}
        with open(path, "r") as parameter_file:
            for line in parameter_file:
                name, separator, value = line.partition(":")
                if separator and name in parameter_names:
                    try:
                        overrides[name] = float(value.strip().split()[0])
                    except (IndexError, ValueError):
                        pass
        if overrides:
            return overrides
    return {}


def infer_tech_jump_probability(export_root):
    candidates = [
        os.path.join(export_root, "run_manifest.txt"),
        os.path.join(export_root, "PreDamagePreTech", "params.txt"),
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


def grouped_first_jump_statistics(damage_intensity, tech_intensity, dt):
    total_intensity = damage_intensity + tech_intensity
    cumulative_hazard = tf.cumsum(total_intensity * dt, exclusive=True)
    survival = tf.exp(-cumulative_hazard)

    interval_jump_probability = survival * (1.0 - tf.exp(-total_intensity * dt))
    horizon_mask = tf.concat(
        [tf.ones_like(total_intensity[:-1]), tf.zeros_like(total_intensity[-1:])],
        axis=0,
    )
    interval_jump_probability = interval_jump_probability * horizon_mask
    positive_total = total_intensity > 0.0
    damage_share = tf.where(positive_total, damage_intensity / total_intensity, tf.zeros_like(total_intensity))
    tech_share = tf.where(positive_total, tech_intensity / total_intensity, tf.zeros_like(total_intensity))
    damage_interval_probability = damage_share * interval_jump_probability
    tech_interval_probability = tech_share * interval_jump_probability
    horizon_jump_probability = tf.reduce_sum(interval_jump_probability)
    positive_horizon_probability = horizon_jump_probability > 0.0

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
    damage_cumulative = tf.cumsum(damage_interval_probability, exclusive=True)
    tech_cumulative = tf.cumsum(tech_interval_probability, exclusive=True)
    conditional_damage_cumulative = tf.where(
        positive_horizon_probability, damage_cumulative / horizon_jump_probability,
        tf.zeros_like(damage_cumulative))
    conditional_tech_cumulative = tf.where(
        positive_horizon_probability, tech_cumulative / horizon_jump_probability,
        tf.zeros_like(tech_cumulative))

    return {
        "damage_intensity": damage_intensity, "tech_intensity": tech_intensity,
        "total_intensity": total_intensity, "survival": survival,
        "any_jump_probability": 1.0 - survival,
        "horizon_jump_probability": horizon_jump_probability,
        "damage_subdensity": damage_subdensity, "tech_subdensity": tech_subdensity,
        "damage_density": conditional_damage_density, "tech_density": conditional_tech_density,
        "conditional_damage_density": conditional_damage_density,
        "conditional_tech_density": conditional_tech_density,
        "damage_type_probability": damage_share, "tech_type_probability": tech_share,
        "damage_cumulative": damage_cumulative, "tech_cumulative": tech_cumulative,
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


def simulate_path_PreDamagePreTech(model, ξ, T, dt, make_plots: bool = True):
    """Deterministic path simulation before the first damage or technology jump.
    v3: controls are CLOSED FORM (no control nets).

    State:    X = [logK, Z, Y, logR, λ3, logξ]
    Controls: i_g, i_d, i_r via _closed_form_controls (matches v3 pde_rhs).
    Dynamics identical to models/SimulationDeterministic.py.
    """
    params = model.params

    A_d = params['A_d']; A_g = params['A_g']
    A_g_prime = params['A_g_prime']
    A_g_prime_prime = params['A_g_prime_prime']
    initial_tech_regime = params.get("initial_tech_regime", "pre")
    A_g_current = A_g if initial_tech_regime == "pre" else A_g_prime
    α_d = params['α_d']; Γ_d = params['Γ_d']; θ_d = params['θ_d']; σ_d = params['σ_d']
    α_g = params['α_g']; Γ_g = params['Γ_g']; θ_g = params['θ_g']; σ_g = params['σ_g']
    ζ = params['ζ']; ψ0 = params['ψ0']; ψ1 = params['ψ1']; σ_κ = params['σ_κ']
    θ_bar = params['θ_bar']; η = params['η']; ϛ = params['ϛ']
    varrho = params['varrho']
    tech_jump_intensity_scale = params.get('tech_jump_intensity_scale', 1.0)
    π = params['π']
    one_tech_jump_mode = bool(params.get("one_tech_jump_mode", False))
    r1 = params['r1']; r2 = params['r2']; y_lower = params['y_lower']; y_upper = params['y_upper']
    L = params['L']; λ3_values = params['λ3_values']

    nT = int(np.round(T / dt)) + 1
    tgrid = np.arange(nT, dtype=float) * dt

    logK = tf.math.log(tf.constant(params["K0"], dtype=tf.float32))
    Z = tf.constant(params["Z0"], dtype=tf.float32)
    Y = tf.constant(params["Y0"], dtype=tf.float32)
    logR = tf.math.log(tf.constant(params["R0"], dtype=tf.float32))
    λ3 = tf.constant(0.0, dtype=tf.float32)
    logξ = tf.math.log(tf.constant(ξ, dtype=tf.float32))

    x = tf.reshape(tf.stack([logK, Z, Y, logR, λ3, logξ], axis=0), (1, 6))

    ta_xs = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_ig = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_id = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_ir = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_g_interm = tf.TensorArray(dtype=tf.float32, size=nT)
    ta_g_post = tf.TensorArray(dtype=tf.float32, size=nT)
    if L > 0:
        ta_g_dmg = tf.TensorArray(dtype=tf.float32, size=nT, element_shape=(L,))
    else:
        ta_g_dmg = None

    def _controls(xx: tf.Tensor):
        """v3 CLOSED-FORM controls from the state `xx` (batch,6/7)."""
        logK = xx[:, 0:1]; Z = xx[:, 1:2]; Y = xx[:, 2:3]; logR = xx[:, 3:4]
        logξ_scalar = xx[:, -1:]
        # PreDamagePreTech is a pre-tech regime (has R&D); A_g enters the constraint.
        ig, id, ir = _closed_form_controls(model, logK, Z, Y, logR, logξ_scalar,
                                           has_ir=True, A_g_use=A_g_current)
        return ig, id, ir

    def _v(xx: tf.Tensor) -> tf.Tensor:
        return model.v_nn(xx)

    def _tech_weights(x, logξ_scalar):
        logK = x[:, 0:1]; Z = x[:, 1:2]; Y = x[:, 2:3]; logR = x[:, 3:4]
        v_input = tf.concat([logK, Z, Y, logR, logξ_scalar, logξ_scalar, logξ_scalar], axis=1)
        v_now = _v(v_input)
        v_post_input = tf.concat([logK, Z, Y, A_g_prime_prime * tf.ones_like(Y),
                                  logξ_scalar, logξ_scalar], axis=1)
        v_post = model.v_PreDamagePostTech_nn(v_post_input)
        ξ = tf.exp(logξ_scalar)
        g_primeprime = [tf.exp(-1.0 / ξ * (v_post - v_now))]
        if (initial_tech_regime == "pre" and float(π) < 1.0 - 1e-12
                and getattr(model, "v_PreDamageIntermTech_nn", None) is not None):
            v_interm = model.v_PreDamageIntermTech_nn(v_input)
            g_prime = [tf.exp(-1.0 / ξ * (v_interm - v_now))]
        else:
            g_prime = [tf.zeros_like(v_now)]
        return g_prime, g_primeprime

    def _damage_weights(logK, Z, Y, logR, λ3, logξ_scalar) -> List[tf.Tensor]:
        v_now = _v(tf.concat([logK, Z, Y, logR, logξ_scalar, logξ_scalar, logξ_scalar], axis=1))
        ξ = tf.exp(logξ_scalar)
        out = []
        for l in range(L):
            λ3_l = tf.ones_like(Y) * λ3_values[l]
            v_post = model.v_PostDamagePreTech_nn(
                tf.concat([logK, Z, tf.ones(tf.shape(Y)) * y_upper, logR, λ3_l,
                           logξ_scalar, logξ_scalar, logξ_scalar], 1))
            g_l = tf.exp(-1.0 / ξ * (v_post - v_now))
            out.append(g_l)
        return out

    ta_xs = ta_xs.write(0, tf.reshape(x, (6,)))
    ig0, id0, ir0 = _controls(x)
    ta_ig = ta_ig.write(0, tf.reshape(ig0, ()))
    ta_id = ta_id.write(0, tf.reshape(id0, ()))
    ta_ir = ta_ir.write(0, tf.reshape(ir0, ()))

    gp0, gpp0 = _tech_weights(x, tf.reshape(logξ, (1, 1)))
    ta_g_interm = ta_g_interm.write(0, tf.reshape(gp0[0], ()))
    ta_g_post = ta_g_post.write(0, tf.reshape(gpp0[0], ()))

    if L > 0:
        gd0 = _damage_weights(x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], x[:, 4:5], x[:, 5:6])
        gd0_vec = tf.stack([tf.reshape(g, ()) for g in gd0])
        ta_g_dmg = ta_g_dmg.write(0, gd0_vec)

    x_curr = x
    for i in range(1, nT):
        logK_t, Z_t, Y_t, logR_t, λ3_t, logξ_t = tf.split(x_curr, 6, axis=1)
        K_t = tf.exp(logK_t)

        ig, id_, ir = _controls(x_curr)
        inside_log_i_d = tf.math.maximum(1.0 + θ_d * id_, 1e-8)
        inside_log_i_g = tf.math.maximum(1.0 + θ_g * ig, 1e-8)

        v_logKlogK_term = 0.5 * (σ_d**2 * (1.0 - Z_t)**2 + σ_g**2 * Z_t**2)
        drift_logK = (α_d + Γ_d * tf.math.log(inside_log_i_d)) * (1.0 - Z_t) \
                   + (α_g + Γ_g * tf.math.log(inside_log_i_g)) * Z_t \
                   - v_logKlogK_term
        drift_Z = (
            (α_g + Γ_g * tf.math.log(inside_log_i_g))
            - (α_d + Γ_d * tf.math.log(inside_log_i_d))
            - Z_t * σ_g**2 + (1.0 - Z_t) * σ_d**2
        ) * Z_t * (1.0 - Z_t)
        drift_Y = θ_bar * η * A_d * (1.0 - Z_t) * K_t
        drift_logR = -ζ + ψ0 * tf.exp(ψ1 * (tf.math.log(tf.maximum(ir, 1e-12)) + logK_t - logR_t)) - 0.5 * σ_κ**2

        x_next = tf.concat(
            [
                logK_t + drift_logK * dt,
                tf.clip_by_value(Z_t + drift_Z * dt, 1e-6, 1.0 - 1e-6),
                Y_t + drift_Y * dt,
                logR_t + drift_logR * dt,
                λ3_t,
                logξ_t,
            ],
            axis=1,
        )

        ta_xs = ta_xs.write(i, tf.reshape(x_next, (6,)))
        ta_ig = ta_ig.write(i, tf.reshape(ig, ()))
        ta_id = ta_id.write(i, tf.reshape(id_, ()))
        ta_ir = ta_ir.write(i, tf.reshape(ir, ()))

        gp, gpp = _tech_weights(x_next, logξ_t)
        ta_g_interm = ta_g_interm.write(i, tf.reshape(gp[0], ()))
        ta_g_post = ta_g_post.write(i, tf.reshape(gpp[0], ()))
        if L > 0:
            gd = _damage_weights(x_next[:, 0:1], x_next[:, 1:2], x_next[:, 2:3], x_next[:, 3:4], x_next[:, 4:5], x_next[:, 5:6])
            gd_vec = tf.stack([tf.reshape(g, ()) for g in gd])
            ta_g_dmg = ta_g_dmg.write(i, gd_vec)

        x_curr = x_next

    X_tf = ta_xs.stack()
    logK_path_tf = X_tf[:, 0]; Z_path_tf = X_tf[:, 1]; Y_path_tf = X_tf[:, 2]; logR_path_tf = X_tf[:, 3]
    λ3_path_tf = X_tf[:, 4]; logξ_path_tf = X_tf[:, 5]
    K_path_tf = tf.exp(logK_path_tf); R_path_tf = tf.exp(logR_path_tf); ξ_path_tf = tf.exp(logξ_path_tf)

    ig_path_tf = ta_ig.stack(); id_path_tf = ta_id.stack(); ir_path_tf = ta_ir.stack()

    g_interm_tf = ta_g_interm.stack(); g_post_tf = ta_g_post.stack()
    if L > 0:
        g_dmg_arr_tf = ta_g_dmg.stack()
    else:
        g_dmg_arr_tf = tf.zeros((nT, 0), dtype=tf.float32)

    logξ_col = tf.expand_dims(X_tf[:, 5], axis=1)
    v_inp = tf.concat([X_tf[:, 0:4], logξ_col, logξ_col, logξ_col], axis=1)
    with tf.GradientTape() as tape:
        tape.watch(v_inp)
        v_vals = model.v_nn(v_inp)
    grads_vinp = tape.gradient(v_vals, v_inp)
    dv_dlogK_tf = grads_vinp[:, 0]; dv_dZ_tf = grads_vinp[:, 1]
    dv_dY_tf = grads_vinp[:, 2]; dv_dlogR_tf = grads_vinp[:, 3]

    h_d_tf = -1.0 / ξ * ((dv_dlogK_tf - Z_path_tf * dv_dZ_tf) * (1.0 - Z_path_tf) * (σ_d))
    h_g_tf = -1.0 / ξ * ((dv_dlogK_tf + (1.0 - Z_path_tf) * dv_dZ_tf) * (Z_path_tf) * (σ_g))
    λ1 = params['λ1']; λ2 = params['λ2']
    h_y_tf = -1.0 / ξ * ((dv_dY_tf - (λ1 + λ2 * Y_path_tf)) * (η * A_d * (1.0 - Z_path_tf) * K_path_tf) * ϛ)
    h_r_tf = -1.0 / ξ * (σ_κ * dv_dlogR_tf)

    if initial_tech_regime == "pre":
        J_g_prime = tech_jump_intensity_scale * (1.0 - π) * R_path_tf / varrho
        J_g_primeprime = tech_jump_intensity_scale * π * R_path_tf / varrho
    elif one_tech_jump_mode:
        J_g_prime = tf.zeros_like(R_path_tf)
        J_g_primeprime = tech_jump_intensity_scale * R_path_tf / varrho
    else:
        J_g_prime = tf.zeros_like(R_path_tf)
        J_g_primeprime = tech_jump_intensity_scale * π * R_path_tf / varrho

    J_d_tf = r1 * (tf.exp(r2 / 2.0 * tf.pow(Y_path_tf - y_lower, 2.0)) - 1.0) * tf.cast(Y_path_tf > y_lower, tf.float32)

    if L > 0:
        g_dmg_avg_tf = tf.reduce_mean(g_dmg_arr_tf, axis=1)
    else:
        g_dmg_avg_tf = tf.zeros_like(J_d_tf)

    λ_tech_interm_tf = g_interm_tf * J_g_prime
    λ_tech_post_tf = g_post_tf * J_g_primeprime
    λ_tech_total_tf = λ_tech_interm_tf + λ_tech_post_tf
    λ_dmg_dist_tf = g_dmg_avg_tf * J_d_tf
    first_jump = grouped_first_jump_statistics(λ_dmg_dist_tf, λ_tech_total_tf, dt)
    tech_jump_prob_tf = first_jump["tech_cumulative"]
    dmg_jump_prob_tf = first_jump["damage_cumulative"]
    tech_jump_density_tf = first_jump["tech_density"]
    dmg_jump_density_tf = first_jump["damage_density"]

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
    I_r = I_r_tf.numpy(); y_path = y_path_tf.numpy(); RD_path = RD_path_tf.numpy()

    export_folder = params.get("save_folder", ".") + "/SimulationOutputs" + f"_ξ_{ξ:.3f}"
    os.makedirs(export_folder, exist_ok=True)

    data = {
        "t": tgrid,
        "logK": logK_path, "Z": Z_path, "Y": Y_path, "logR": logR_path, "λ3": λ3_path, "logξ": logξ_path,
        "K": K_path, "R": R_path, "ξ": ξ_path,
        "i_g": ig_path, "i_d": id_path, "i_r": ir_path,
        "I_g": I_g, "I_d": I_d, "I_r": I_r, "E": E_path, "c": c_path, "C": C_path,
        "Output": y_path, "ConsumptionOutputRatio": ConsumptionOutputRatio_path,
        "RD": RD_path, "DirtyInvestment": DirtyInvestment_path_tf.numpy(),
        "GreenInvestment": GreenInvestment_path_tf.numpy(),
        "y_consumption": y_path,
        "h_d": h_d, "h_g": h_g, "h_y": h_y, "h_r": h_r,
        "tech_jump_prob": tech_jump_prob, "dmg_jump_prob": dmg_jump_prob,
        "tech_jump_density": tech_jump_density, "dmg_jump_density": dmg_jump_density,
        "tech_jump_subdensity": tech_jump_subdensity, "dmg_jump_subdensity": dmg_jump_subdensity,
        "conditional_tech_jump_density": conditional_tech_jump_density,
        "conditional_dmg_jump_density": conditional_dmg_jump_density,
        "conditional_tech_jump_prob": conditional_tech_jump_prob,
        "conditional_dmg_jump_prob": conditional_dmg_jump_prob,
        "first_jump_type_tech_prob": first_jump_type_tech_prob,
        "first_jump_type_dmg_prob": first_jump_type_dmg_prob,
        "tech_jump_intensity": tech_jump_intensity, "dmg_jump_intensity": dmg_jump_intensity,
        "total_jump_intensity": total_jump_intensity, "first_jump_survival": first_jump_survival,
        "any_jump_prob": any_jump_prob,
        "first_jump_horizon_prob": np.atleast_1d(first_jump_horizon_prob),
        "g_interm": g_interm, "g_post": g_post,
    }

    for k, v in data.items():
        np.savetxt(os.path.join(export_folder, f"{k}.txt"), np.asarray(v))

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
        _plt(data["Z"], r"$Z$", "Z.png", ylim=(0, 1))
        _plt(data["Y"], r"$Y$", "Y.png")
        _plt(data["logR"], r"$\log R$", "logR.png")
        _plt(data["i_g"], r"$i_g$", "i_g.png")
        _plt(data["i_d"], r"$i_d$", "i_d.png")
        _plt(data["i_r"], r"$i_r$", "i_r.png")
        _plt(data["DirtyInvestment"], "Dirty Investment / Output", "DirtyInvestment.png")
        _plt(data["GreenInvestment"], "Green Investment / Output", "GreenInvestment.png")
        _plt(data["E"], "Emissions", "E.png")
        _plt(data["ConsumptionOutputRatio"] * 100.0, "Consumption as % of Output (C/Y)", "ConsumptionOutputRatio.png")
        _plt(data["tech_jump_prob"], "Technology First-Jump Cumulative Incidence", "tech_jump_prob.png", ylim=(0, 1))
        _plt(data["dmg_jump_prob"], "Damage First-Jump Cumulative Incidence", "dmg_jump_prob.png", ylim=(0, 1))
        _plt(data["tech_jump_density"], "Technology First-Jump Density", "tech_jump_density.png")
        _plt(data["dmg_jump_density"], "Damage First-Jump Density", "dmg_jump_density.png")
        _plt(data["conditional_tech_jump_density"], "Technology Density Conditional on a First Jump by Horizon", "conditional_tech_jump_density.png")
        _plt(data["conditional_dmg_jump_density"], "Damage Density Conditional on a First Jump by Horizon", "conditional_dmg_jump_density.png")
        _plt(data["h_y"], r"$h_Y$", "h_y.png")
        _plt(data["h_d"], r"$h_d$", "h_d.png")
        _plt(data["h_g"], r"$h_g$", "h_g.png")
        _plt(data["h_r"], r"$h_r$", "h_r.png")

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
    return data


if __name__ == "__main__":
    import sys

    export_folder = sys.argv[1]
    saved_economic_parameters = infer_saved_economic_parameters(export_folder)
    PARAMS.update(saved_economic_parameters)

    def infer_network_widths(export_root, initial_regime):
        stage = "PreDamagePreTech"
        config_path = os.path.join(export_root, stage, "params_v_nn_config.txt")
        if not os.path.exists(config_path):
            return [32, 32, 32, 32]
        with open(config_path, "r") as f:
            for line in f:
                if line.startswith("num_hiddens:"):
                    try:
                        return list(ast.literal_eval(line.split(":", 1)[1].strip()))
                    except (SyntaxError, ValueError):
                        pass
        return [32, 32, 32, 32]

    initial_tech_regime = infer_initial_tech_regime(export_folder)
    network_widths = infer_network_widths(export_folder, initial_tech_regime)

    batch_size = 128
    num_iterations = 2_000_000
    pretrained_path = None
    logging_frequency = 1000
    learning_rates = [float(x) for x in "10e-4,10e-4,10e-4,10e-4".split(",")]

    hidden_layer_activations = "swish,tanh,tanh,softplus".split(",")
    output_layer_activations = "softplus,custom,custom,softplus".split(",")
    num_hidden_layers = len(network_widths)
    num_neurons = int(network_widths[0])
    learning_rate_schedule_type = "warmup_cosine"

    # v3: ONLY the value-net config is needed (controls are closed-form). The
    # input_bounds make InputNormalization match the trained net exactly.
    from feedforward_subnet import regime_input_bounds
    v_nn_config = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": hidden_layer_activations[0],
        "dim": 1,
        "nn_name": "v_nn",
        "final_activation": output_layer_activations[0],
        "input_bounds": regime_input_bounds("PreDamagePreTech"),
        "seed": int(os.environ.get("MODEL_SEED", 0)),
    }

    tech_jump_intensity_scale = infer_tech_jump_intensity_scale(export_folder)
    tech_jump_probability = infer_tech_jump_probability(export_folder)
    simulation_y0 = float(os.environ.get("SIMULATION_Y0", PARAMS.get("Y0", 1.1)))

    params = {
        "batch_size": batch_size,
        "learning_rates": learning_rates,
        "v_nn_config": v_nn_config,
        "num_iterations": num_iterations,
        "logging_frequency": logging_frequency,
        "verbose": True,
        "pretrained_path": pretrained_path,
        "learning_rate_schedule_type": learning_rate_schedule_type,
        "tech_jump_intensity_scale": tech_jump_intensity_scale,
        "π": tech_jump_probability,
        "initial_tech_regime": initial_tech_regime,
        "one_tech_jump_mode": False,
        "Y0": simulation_y0,
        "train_from_scratch": True,
        "phase": "base",
    }

    params["save_folder"] = os.path.join(export_folder, "SimulationDeterministic")
    params["v_PreDamagePostTech_nn_path"] = os.path.join(export_folder, "PreDamagePostTech", "v_nn_checkpoint_PreDamagePostTech")
    params["v_PostDamagePreTech_nn_path"] = os.path.join(export_folder, "PostDamagePreTech", "v_nn_checkpoint_PostDamagePreTech")
    params["v_PreDamageIntermTech_nn_path"] = os.path.join(export_folder, "PreDamageIntermTech", "v_nn_checkpoint_PreDamageIntermTech")
    params["v_PostDamagePostTech_nn_path"] = os.path.join(export_folder, "PostDamagePostTech", "v_nn_checkpoint_PostDamagePostTech")
    params["v_PostDamageIntermTech_nn_path"] = os.path.join(export_folder, "PostDamageIntermTech", "v_nn_checkpoint_PostDamageIntermTech")

    PARAMS.update(params)

    model = load_PreDamagePreTech_model(export_folder)
    print("Loaded PreDamagePreTech v3 model (value + neighbor value nets; controls closed-form).")
    print(f"Tech jump intensity scale: {tech_jump_intensity_scale}")
    print(f"Tech jump probability pi: {tech_jump_probability}")
    print(f"Initial technology regime: {initial_tech_regime}")
    print(f"Initial deterministic temperature Y0: {simulation_y0}")
    print(f"Saved economic parameter overrides: {saved_economic_parameters}")

    os.makedirs(params["save_folder"], exist_ok=True)

    simulation_T = float(os.environ.get("SIMULATION_T", "60.0"))
    simulation_dt = float(os.environ.get("SIMULATION_DT", str(1 / 12)))
    xis = [float(value) for value in os.environ.get("SIMULATION_XIS", "0.01,0.05,0.1,148.6").split(",")]
    for xi in xis:
        print(f"\n=== Simulating ξ = {xi} ===")
        _ = simulate_path_PreDamagePreTech(model, ξ=xi, T=simulation_T, dt=simulation_dt, make_plots=True)
