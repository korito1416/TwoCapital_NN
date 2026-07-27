"""
Solve  post-damage-pre-technology  model.

The value function we are solving is v = V - log N.  
"""

import numpy as np
import tensorflow as tf
import pathlib
import time
from feedforward_subnet import (
    FeedForwardSubNet,
    large_sample_validation,
    sample_state_columns,
    setup_optimizers,
    validation_score,
)
from params import PARAMS, investment_rate_activation
from pretrained_paths import legacy_nber_folder

# ---- redesign switches (all default to production behaviour; see config.py) --------------
import config
import uncertainty
import state_layout
import hjb_scaling
from value_net import ValueFunction


class PostDamagePreTechModel:
    """Post-damage & pre-technology HJB.
    """

    def __init__(self, params):
        # Econcomic Parameters described in the appendix
        self.params = PARAMS.copy()
        # Nerual network parameters
        self.params.update(params or {})

        # ensure optimizers are prepared
        setup_optimizers(self.params)
        
        self.params['tensorboard'] = bool(self.params.get('tensorboard', True))
        
        self.v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        self.i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        self.i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
        self.i_r_nn  = FeedForwardSubNet(self.params['i_r_nn_config'])

        # --- redesign: level anchoring / separable structure -------------------------------
        # v_nn stays a plain FeedForwardSubNet so ALL the existing build/checkpoint machinery is
        # untouched; the re-centering is applied at the CALL SITE in pde_rhs.
        # anchor state x0 (the calibrated initial condition, evaluated at the neutral end).
        self._anchor_state = None          # built lazily in pde_rhs (needs the input layout)
        self.A_nn = None
        if config.DETREND:
            # A(logK): small 1-D net for the separable structure V = A(logK) + W(Z,Y,u).
            # Justification: the measured scale elasticity V_logK+V_logR varies 0.77->0.45 in logK
            # but only ~0.002 across Z and Y, so a CONSTANT-a detrend is mis-specified (it would
            # inject ~10x the residual) while an additively-separable A(logK) is not.
            a_cfg = dict(self.params['v_nn_config'])
            a_cfg['nn_name'] = 'A_nn'
            a_cfg['num_hiddens'] = [16, 16]
            a_cfg['final_activation'] = None
            self.A_nn = FeedForwardSubNet(a_cfg)
            self.A_nn.build((None, 1))

        ### Load weights from PostDamagePostTech and PostDamageIntermTech model
        self.v_PostDamagePostTech_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        self.v_PostDamagePostTech_nn.build((None, 7))
        self.v_PostDamagePostTech_nn.load_weights(self.params["v_PostDamagePostTech_nn_path"] )

        self.use_intermediate_tech_jump = float(self.params.get("π", PARAMS.get("π", 0.04))) < 1.0 - 1e-12
        if self.use_intermediate_tech_jump:
            self.v_PostDamageIntermTech_nn    = FeedForwardSubNet(self.params['v_nn_config'])
            self.v_PostDamageIntermTech_nn.build((None, 8))
            self.v_PostDamageIntermTech_nn.load_weights(self.params["v_PostDamageIntermTech_nn_path"] )
        else:
            self.v_PostDamageIntermTech_nn = None
        
        ## Create ranges for sampling later 

        self.params["state_intervals"] = {}

        # logK intervals
        self.params["state_intervals"]["logK"] = tf.reshape(
            tf.linspace(self.params.get('logK_min', 4.0), self.params.get('logK_max', 7.0), self.params["batch_size"] + 1),
            (self.params["batch_size"] + 1, 1)
        )
        self.params["state_intervals"]["logK_interval_size"] = self.params["state_intervals"]["logK"][1] - self.params["state_intervals"]["logK"][0]

        # Z intervals
        self.params["state_intervals"]["Z"] = tf.reshape(
            tf.linspace(self.params.get('Z_min', 0.01), self.params.get('Z_max', 0.99), self.params["batch_size"] + 1),
            (self.params["batch_size"] + 1, 1)
        )
        self.params["state_intervals"]["Z_interval_size"] = self.params["state_intervals"]["Z"][1] - self.params["state_intervals"]["Z"][0]

        # Y intervals
        self.params["state_intervals"]["Y"] = tf.reshape(
            tf.linspace(self.params.get('Y_min', 0.0), self.params.get('Y_max', 4.0), self.params["batch_size"] + 1),
            (self.params["batch_size"] + 1, 1)
        )
        self.params["state_intervals"]["Y_interval_size"] = self.params["state_intervals"]["Y"][1] - self.params["state_intervals"]["Y"][0]

        # logR intervals (fallback to R_min/R_max if logR_min/logR_max not present)
        self.params["state_intervals"]["logR"] = tf.reshape(
            tf.linspace(self.params.get('logR_min', self.params.get('R_min', 1.0)), self.params.get('logR_max', self.params.get('R_max', 6.0)), self.params["batch_size"] + 1),
            (self.params["batch_size"] + 1, 1)
        )
        self.params["state_intervals"]["logR_interval_size"] = self.params["state_intervals"]["logR"][1] - self.params["state_intervals"]["logR"][0]

        # λ3 (gamma_3) intervals: respect explicit length setting if provided
        self.params["state_intervals"]["λ3"] = tf.reshape(
            tf.linspace(self.params.get('λ3_min', 0.0), self.params.get('λ3_max', 1.0/3.0), int(self.params.get('gamma_3_length', self.params.get('λ3_length', self.params["batch_size"] + 1)))),
            (int(self.params.get('gamma_3_length', self.params.get('λ3_length', self.params["batch_size"] + 1))), 1)
        )
        self.params["state_intervals"]["λ3_interval_size"] = self.params["state_intervals"]["λ3"][1] - self.params["state_intervals"]["λ3"][0]
         

        # logξ intervals
        self.params["state_intervals"]["logξ"] = tf.reshape(
            tf.linspace(self.params.get('logξ_min', -3.0), self.params.get('logξ_max', 5.0), self.params["batch_size"] + 1),
            (self.params["batch_size"] + 1, 1)
        )
        self.params["state_intervals"]["logξ_interval_size"] = self.params["state_intervals"]["logξ"][1] - self.params["state_intervals"]["logξ"][0]

 

        # ensure export folder exists if provided
        if self.params.get('export_folder'):
            pathlib.Path(self.params['export_folder']).mkdir(parents=True, exist_ok=True)
        if self.params.get('export_folder') and self.params['tensorboard']:
            ## Create objects to generate checkpoints for tensorboard
            pathlib.Path(self.params["export_folder"] + '/logs/train/').mkdir(parents=True, exist_ok=True) 
            pathlib.Path(self.params["export_folder"] + '/logs/test/').mkdir(parents=True, exist_ok=True) 
            self.train_writer = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/train/')
            self.test_writer  = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/test/')
 
 
 
    def sample(self, batch_size=None):
        return sample_state_columns(self.params, batch_size=batch_size)

 
    @tf.function
    def pde_rhs(self, logK, Z, Y, logR, λ3, logξ):
        """
        Y_hat is the temperature value when jump occurs. 
        (rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d,   FOC_g, FOC_d)
        """
        ###############
        #### load parameters into local variables 
        ###############
        A_d = self.params['A_d']
        A_g = self.params['A_g']
        A_g_prime = self.params['A_g_prime']
        A_g_prime_prime = self.params['A_g_prime_prime']
        π = self.params['π']

        δ = self.params['δ']

        α_d = self.params['α_d']
        Γ_d = self.params['Γ_d']
        θ_d = self.params['θ_d']
        σ_d = self.params['σ_d']

        α_g = self.params['α_g']
        Γ_g = self.params['Γ_g']
        θ_g = self.params['θ_g']
        σ_g = self.params['σ_g']

        ζ = self.params['ζ']
        ψ0 = self.params['ψ0']
        ψ1 = self.params['ψ1']
        σ_κ = self.params['σ_κ']
        varrho = self.params['varrho']
        tech_jump_intensity_scale = self.params.get('tech_jump_intensity_scale', 1.0)

        θ_bar = self.params['θ_bar']
        η = self.params['η']
        ϛ = self.params['ϛ']

        λ1 = self.params['λ1']
        λ2 = self.params['λ2']
        L = self.params['L']
        λ3_values = self.params['λ3_values']
        r1 = self.params['r1']
        r2 = self.params['r2']
        y_lower = self.params['y_lower']
        y_upper = self.params['y_upper']


        ###############
        #### Compute value functions and derivatives
        ###############
        
        # X = tf.concat([logK, Z, Y, logR, λ3, logξ], 1)

        # --- redesign: input layout (legacy == production; unified drops the logξ padding) ---
        if config.USE_UNIFIED:
            θ_unc = uncertainty.logxi_to_theta(logξ) if not config.USE_THETA else logξ
            X = state_layout.build_input("PostDamagePreTech", logK, Z, Y,
                                         logR=logR, lam3=λ3,
                                         theta=θ_unc if config.USE_THETA else None,
                                         logxi=None if config.USE_THETA else logξ)
        else:
            X = tf.concat([logK, Z, Y, logR, λ3, logξ, logξ, logξ], 1)

        # Controls defined in section 3.4
        v = self.v_nn(X)
        i_g = self.i_g_nn(X)
        i_d = self.i_d_nn(X)
        i_r = tf.exp(-self.i_r_nn(X)) # I_r / K
        # i_r =  self.i_r_nn(X)

        # --- redesign: level parameterization ---------------------------------------------
        # recenter : v(x) = phi(x) - phi(x0) + v0   -> the single level DOF is pinned BY
        #            CONSTRUCTION, exactly conservative (unlike costate, which parameterizes an
        #            unconstrained gradient field and admits a non-physical curl).
        # separable: v = A(logK) + W(...) , re-centered the same way.
        # Both leave every DERIVATIVE of v unchanged, so the FOCs and the economics are untouched:
        # within a regime the level is pure GAUGE (the FOCs use only derivatives).
        if config.ANCHOR == "recenter" or config.DETREND:
            logK0 = tf.ones_like(logK) * 6.7799       # log(880)
            Z0    = tf.ones_like(Z)    * 0.70
            # post-damage regimes are only economically visited at Y >= yhat, so anchor at the
            # entry slice Y = yhat rather than at the pre-damage Y0.
            Y0    = tf.ones_like(Y)    * self.params['y_upper']
            # logR is a TRUE state in this regime (dv_dlogR enters the HJB), so it must be PINNED
            # in the anchor state: reusing the batch's logR would make v_anchor a function of logR
            # and would therefore change dv_dlogR (the level must stay pure gauge).
            logR0 = tf.ones_like(logR) * 2.4159       # log(11.2) = log(R0)
            l30   = tf.ones_like(λ3)   * (1.0/6.0)
            # xi AT THE ANCHOR STATE: pinned at a FIXED neutral xi.  Inheriting the batch's own xi
            # forces v(x0,xi)=v0 for EVERY xi and annihilates the xi-response at x0, destroying the
            # welfare-cost-of-robustness object.  Defined BEFORE the branch so BOTH layouts see it.
            _xa = tf.ones_like(logK) * (0.0 if config.USE_THETA else config.ANCHOR_LOGXI_NEUTRAL)
            _xa = uncertainty.network_input(_xa) if config.USE_THETA else _xa
            if config.USE_UNIFIED:
                # unified slots: [logK, Z, Y, u, lam3, xi]; u = logR - logK at the anchor state
                # ξ AT THE ANCHOR STATE: pin at a FIXED neutral ξ.  Inheriting the batch's own ξ
                # forces v(x0,ξ)=v0 for EVERY ξ and annihilates the ξ-response at x0 (measured
                # exactly 0.00000 on 2026-07-26), destroying the welfare-cost-of-robustness object.
                x0 = tf.concat([logK0, Z0, Y0, logR0 - logK0, l30, _xa], 1)
            else:
                # legacy slots: [logK, Z, Y, logR, lam3, logxi, logxi, logxi] -> reuse cols 5: verbatim
                x0 = tf.concat([logK0, Z0, Y0, logR0, l30, _xa, _xa, _xa], 1)
            v_anchor = self.v_nn(x0)
            if config.DETREND:
                v = v + self.A_nn(logK) - self.A_nn(tf.ones_like(logK) * 6.7799)
            v = v - v_anchor + tf.constant(config.ANCHOR_V0, dtype=v.dtype)

        # State Variables Transformations
        # --- redesign: uncertainty parameterization.  In θ-mode the SAMPLED column `logξ` already
        # carries θ = 1/ξ (see sample()), so ξ is recovered as 1/θ and θ is used directly in the
        # robustness blocks (θ = 0 is exactly uncertainty-neutral).
        if config.USE_THETA:
            θ_unc = logξ
            ξ = 1.0 / tf.maximum(θ_unc, 1e-12)
        else:
            ξ = tf.exp(logξ)
            θ_unc = 1.0 / ξ
        K = tf.exp(logK)


        ###########
        #### Calculate derivatives
        ###########
        
        dv_dlogK                 = tf.reshape(tf.gradients(v, logK, unconnected_gradients='zero')[0], [-1, 1])
        d2v_dlogK2                = tf.reshape(tf.gradients(dv_dlogK, logK, unconnected_gradients='zero')[0], [-1, 1])
        d2v_dlogKdZ               = tf.reshape(tf.gradients(dv_dlogK, Z, unconnected_gradients='zero')[0], [-1, 1])

        dv_dZ                    = tf.reshape(tf.gradients(v, Z, unconnected_gradients='zero')[0], [-1, 1])
        d2v_dZ2                   = tf.reshape(tf.gradients(dv_dZ, Z, unconnected_gradients='zero')[0], [-1, 1])

        dv_dY                    = tf.reshape(tf.gradients(v, Y, unconnected_gradients='zero')[0], [-1, 1])
        d2v_dY2                   = tf.reshape(tf.gradients(dv_dY, Y, unconnected_gradients='zero')[0], [-1, 1])

        dv_dlogR                 = tf.reshape(tf.gradients(v, logR, unconnected_gradients='zero')[0], [-1, 1])
        d2v_dlogR2                = tf.reshape(tf.gradients(dv_dlogR, logR, unconnected_gradients='zero')[0], [-1, 1])
         
        ###################
        ###### drift distortions
        ###################
        
        # `s_j` are the σ'∂v loadings; the distortion is h_j = -θ·s_j with θ = 1/ξ, so the penalty
        # ξ|h|²/2 = θ|s|²/2 stays EXACTLY finite at θ = 0 (production computes ξ·h² = ∞·0 there).
        s_d = (dv_dlogK - Z * dv_dZ ) * (1-Z) * σ_d
        s_g = (dv_dlogK + (1-Z) * dv_dZ ) * Z * σ_g
        # We are solving for v = V - log N, so the damage term in the HJB is modified accordingly
        # dV/dY = dv_dY - d(log N)/dY and d(log N)/dY = λ1 + λ2 * Y + λ3 * (Y - y_upper)  (post-damage)
        s_y = ( dv_dY  - (λ1  + λ2 * Y + λ3 * (Y - y_upper)   )   ) *    η *  A_d * (1-Z) * K     *  ϛ
        s_r = dv_dlogR   * σ_κ

        h_d = - θ_unc * s_d
        h_g = - θ_unc * s_g
        h_y = - θ_unc * s_y
        h_r = - θ_unc * s_r

        # ξ(h_d²+h_g²+h_y²+h_r²)/2 in the finite θ-form
        drift_penalty = 0.5 * θ_unc * (tf.pow(s_d, 2) + tf.pow(s_g, 2) + tf.pow(s_y, 2) + tf.pow(s_r, 2))


        ######################
        #### consumption and flow
        #######################
        pv   =   δ * v

        c = ( A_d  - i_d) * (1 - Z) + (A_g - i_g) * Z -i_r
        inside_log = tf.reshape(tf.math.maximum(c, 1e-8), (-1, 1))
        flow = δ * (tf.math.log(inside_log) + logK)


        # drift and drift-corrections (simplified, retains original structure)
        v_logKlogK_term = (σ_d**2 * (1 - Z)**2 + σ_g**2 * Z**2) / 2.0
        
        
        inside_log_i_d   = tf.reshape(tf.math.maximum(1.0 + θ_d * i_d, 1e-8), [-1, 1])
        inside_log_i_g   = tf.reshape(tf.math.maximum(1.0 + θ_g * i_g, 1e-8), [-1, 1])
 
        v_logK_term = (α_d + Γ_d * tf.math.log(inside_log_i_d)) * (1 - Z) \
                      + (α_g + Γ_g * tf.math.log(inside_log_i_g)) * Z  \
                      - v_logKlogK_term

        v_Z_term = (α_g + Γ_g * tf.math.log(inside_log_i_g)   \
                    - (α_d + Γ_d * tf.math.log(inside_log_i_d)) \
                    - Z * σ_g**2\
                    + (1-Z) * σ_d**2) * Z * (1 - Z)
        
        v_ZZ_term = 0.5 * Z**2 * (1 - Z)**2 * (σ_g**2 + σ_d**2)

        v_logK_Z_term = - Z * (1 - Z)**2 * σ_d**2 + Z**2 * (1.0 - Z) * σ_g**2

        v_y_term = (θ_bar + h_y * ϛ) * η * A_d * (1 - Z) * K    
        
        v_yy_term = 0.5 * ϛ**2 * (η * A_d * (1 - Z) * K)**2
        
        # Damage function is from the 2024 SITE Paper.
        v_logN_term = (λ1 + λ2 * Y + λ3 * (Y - y_upper)) * v_y_term + (λ2 + λ3) * v_yy_term

        v_logR_term = - ζ + ψ0 * tf.exp( ψ1  *   ( tf.math.log(i_r) +logK -  logR) )  - 0.5 * σ_κ**2    +   σ_κ * h_r
        v_logRlogR_term = 0.5 * σ_κ**2 
         
        ######################
        ### Jump terms
        ######################
        J_g_prime  = tech_jump_intensity_scale * (1- π) * tf.exp(logR) / varrho ; J_g_prime_prime  =  tech_jump_intensity_scale * π * tf.exp(logR) / varrho 
        
        v_PostDamagePostTech  = self.v_PostDamagePostTech_nn( tf.concat([logK, Z, Y,  λ3, A_g_prime_prime *tf.ones(tf.shape(Y)) ,logξ, logξ], 1) )

        g_l_prime_prime = tf.exp(-1/ξ * (v_PostDamagePostTech - v) )
        Jump_term = J_g_prime_prime * g_l_prime_prime * (v_PostDamagePostTech - v) + ξ * J_g_prime_prime * (1- g_l_prime_prime+ g_l_prime_prime * tf.math.log(g_l_prime_prime) )

        if self.use_intermediate_tech_jump:
            v_PostDamageIntermTech  = self.v_PostDamageIntermTech_nn( tf.concat([logK, Z, Y, logR, λ3, logξ, logξ, logξ], 1))
            g_l_prime = tf.exp(-1/ξ * (v_PostDamageIntermTech - v) )
            Jump_term += J_g_prime * g_l_prime * (v_PostDamageIntermTech - v) + ξ * J_g_prime * (1- g_l_prime+ g_l_prime * tf.math.log(g_l_prime) )
                    
                    
        #####################
        ###### HJB Equation
        #####################
        rhs = flow \
            + v_logK_term * dv_dlogK  + v_logKlogK_term * d2v_dlogK2 \
            + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
            +   h_d *  (dv_dlogK - Z * dv_dZ)*(1-Z)*σ_d +    h_g * (dv_dlogK + (1-Z) * dv_dZ)*Z*σ_g    \
            + v_logK_Z_term * d2v_dlogKdZ \
            +  dv_dY  * v_y_term  + v_yy_term * d2v_dY2  + drift_penalty  \
            + (-1.0) * v_logN_term \
            + v_logR_term * dv_dlogR + v_logRlogR_term * d2v_dlogR2 \
            + Jump_term

        ####################
        #### FOCs
        ####################
        marginal_util_c = δ / inside_log
 
        
        FOC_d = -marginal_util_c + Γ_d * θ_d / ( inside_log_i_d ) * (dv_dlogK - Z * dv_dZ)
        FOC_g = -marginal_util_c + Γ_g * θ_g / ( inside_log_i_g )   * (dv_dlogK +  (1.0 - Z) * dv_dZ)
        FOC_r = -marginal_util_c + ψ0 * ψ1 * tf.exp( ψ1  *   ( tf.math.log(i_r) +logK -  logR) )* dv_dlogR  /i_r
        
        # `scale_parts` carries what hjb_scaling needs to build the per-state natural scale.
        scale_parts = (logK, v, c, dv_dY * v_y_term)
        return rhs, pv, dv_dY, c, 1.0 + θ_g * i_g , 1.0 + θ_d * i_d,  FOC_d, FOC_g , FOC_r , dv_dlogR, scale_parts


    @tf.function
    def objective_fn(self, logK, Z, Y, logR, λ3, logξ,  compute_control = False, training = True):

        ## This is the objective function that stochastic gradient descend will try to minimize
        ## It depends on which NN it is training. Controls and value functions have different
        ## objectives.
        
        rhs, pv, dv_dY, c, inside_log_i_g , inside_log_i_d ,  FOC_d, FOC_g,  FOC_r ,dv_dlogR, scale_parts  = self.pde_rhs(logK, Z, Y, logR, λ3, logξ)
        δ_scale = self.params['δ']

        epsilon = 10e-8
        
        negative_consumption_boolean = tf.reshape( tf.cast( c < 1e-8, tf.float32 ),  [-1, 1])
        loss_c  = - c  * negative_consumption_boolean + epsilon
        
        negative_inside_log_i_g_boolean = tf.reshape( tf.cast( inside_log_i_g < 1e-8, tf.float32 ),  [-1, 1])
        loss_inside_log_i_g             = - inside_log_i_g  * negative_inside_log_i_g_boolean + epsilon
        
        negative_inside_log_i_d_boolean = tf.reshape( tf.cast( inside_log_i_d < 1e-8, tf.float32 ),  [-1, 1])
        loss_inside_log_i_d             = - inside_log_i_d  * negative_inside_log_i_d_boolean + epsilon

 
        if training:    
            ## Take care of nonsensical controls first
 
            control_constraints = tf.reduce_sum(negative_consumption_boolean) + tf.reduce_sum(negative_inside_log_i_g_boolean) + tf.reduce_sum(negative_inside_log_i_d_boolean)
 
            if control_constraints > 0:
                loss_c_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_c  )))      
                loss_inside_log_i_g_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_g  )))
                loss_inside_log_i_d_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_d  )))
                
                loss_constraints    = loss_c_mse + loss_inside_log_i_g_mse + loss_inside_log_i_d_mse
                return loss_constraints

            if compute_control:
                ## Optimizing all three together
                return -tf.reduce_mean( (rhs - pv ) ) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_g  )))  + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_d  ))) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_r  )))
                        
            else:

                ## loss associated with dv/dY > 0
                loss_dv_dY = dv_dY  * tf.reshape( tf.cast(Y > self.params['y_upper'], tf.float32 ),  [-1, 1]) \
                    * tf.reshape( tf.cast( dv_dY > 0, tf.float32 ),  [-1, 1]) + 10e-8
                loss_dv_dlogR =  dv_dlogR  * tf.reshape( tf.cast( dv_dlogR < 0.0, tf.float32 ),  [-1, 1]) + 10e-8

                    
                # --- redesign: optional non-dimensionalization of the HJB term.  With
                # REDESIGN_HJB_SCALE=none this is EXACTLY the production RMS(rhs-pv).
                # NOTE this term is summed with the FOC terms, so scaling it also re-weights it
                # against them; REDESIGN_HJB_WEIGHT controls that explicitly.
                hjb_term = hjb_scaling.weighted_rms(rhs - pv, δ_scale, *scale_parts)

                loss = hjb_term  \
                       + tf.sqrt(tf.reduce_mean(tf.square(FOC_g ))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(FOC_d  ))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(FOC_r  ))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY  )))  + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dlogR  )))

                return loss

        else:

            ## loss associated with dv/dY > 0
            loss_dv_dY = dv_dY * tf.reshape( tf.cast(Y > self.params['y_upper'], tf.float32 ),  [-1, 1]) \
                * tf.reshape( tf.cast( dv_dY > 0.0, tf.float32 ),  [-1, 1])  + 10e-8

            # eval mode ALWAYS reports the TRUE unscaled residual, so runs stay comparable
            # across REDESIGN_HJB_SCALE settings (never report a scaled loss as if it were the residual).
            return tf.sqrt(tf.reduce_mean(tf.square((rhs - pv)  ))), tf.sqrt(tf.reduce_mean(tf.square(FOC_d))), tf.sqrt(tf.reduce_mean(tf.square(FOC_g))) , tf.sqrt(tf.reduce_mean(tf.square(FOC_r)))   ,  tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY )))

    def grad(self, logK, Z, Y, logR, λ3, logξ, compute_control = False, training = True):

        if compute_control:
            with tf.GradientTape(persistent=True) as tape:
                objective = self.objective_fn(logK, Z, Y, logR, λ3, logξ, compute_control, training)

            trainable_variables = self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_r_nn.trainable_variables

            grad = tape.gradient(objective, trainable_variables)

            del tape

            return grad, objective
        else:
            with tf.GradientTape(persistent=True) as tape:
                objective = self.objective_fn(logK, Z, Y, logR, λ3, logξ, compute_control, training)

            grad = tape.gradient(objective, self.value_variables())
            del tape

            return grad , objective

    def value_variables(self):
        """Trainable variables of the value parameterization (adds A_nn in separable mode)."""
        v = list(self.v_nn.trainable_variables)
        if self.A_nn is not None:
            v = v + list(self.A_nn.trainable_variables)
        return v

    @tf.function
    def train_step(self):
        logK, Z, Y, logR, λ3, logξ = self.sample()


        ## First, train value function

        grad, loss_v_train = self.grad(logK, Z, Y, logR, λ3, logξ, compute_control= False, training=True)
        self.params["optimizers"][0].apply_gradients(zip(grad, self.value_variables()))

        ## Second, train controls
        grad, loss_c_train = self.grad(logK, Z, Y, logR, λ3, logξ, compute_control= True, training=True)
        self.params["optimizers"][1].apply_gradients(zip(grad, self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables  + self.i_r_nn.trainable_variables))

        return loss_v_train, loss_c_train
    
    
    def train(self):

        start_time = time.time()
        training_history = []

        # Prepare to store best neural networks and initialize networks
        min_loss = float("inf")
        
        # redesign: the unified layout is a single canonical width for all regimes;
        # legacy keeps this regime's production width so warm-start checkpoints still load.
        n_inputs = state_layout.WIDTH if config.USE_UNIFIED else 8

        best_v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        best_v_nn.build((None, n_inputs)) 
        self.v_nn.build((None, n_inputs))

        best_i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        best_i_g_nn.build((None, n_inputs)) 
        self.i_g_nn.build((None, n_inputs))

        best_i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
        best_i_d_nn.build((None, n_inputs)) 
        self.i_d_nn.build((None, n_inputs))

        best_i_r_nn  = FeedForwardSubNet(self.params['i_r_nn_config'])
        best_i_r_nn.build((None, n_inputs)) 
        self.i_r_nn.build((None, n_inputs))

        best_v_nn.set_weights(self.v_nn.get_weights())
        best_i_g_nn.set_weights(self.i_g_nn.get_weights())
        best_i_d_nn.set_weights(self.i_d_nn.get_weights())
        best_i_r_nn.set_weights(self.i_r_nn.get_weights())
 
 
        # self.v_nn.load_weights( self.params["job_name"]  + '/PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech')
        # self.i_g_nn.load_weights( self.params["job_name"]  + '/PostDamagePostTech/i_g_nn_checkpoint_PostDamagePostTech')
        # self.i_d_nn.load_weights( self.params["job_name"]  + '/PostDamagePostTech/i_d_nn_checkpoint_PostDamagePostTech')
        # self.i_r_nn.load_weights( self.params["job_name"]  + '/PostDamageIntermTech/i_r_nn_checkpoint_PostDamageIntermTech')
 
        NBER_folder = legacy_nber_folder(required=self.params.get("pretrained_path") is None)
        if NBER_folder is not None:
            self.v_nn.load_weights( NBER_folder + "/pre_tech_post_damage/v_nn_checkpoint_pre_tech_post_damage" )
            self.i_g_nn.load_weights( NBER_folder  + "/pre_tech_post_damage/i_g_nn_checkpoint_pre_tech_post_damage")
            self.i_d_nn.load_weights( NBER_folder + "/pre_tech_post_damage/i_d_nn_checkpoint_pre_tech_post_damage" )
            self.i_r_nn.load_weights(NBER_folder+  "/pre_tech_post_damage/i_I_nn_checkpoint_pre_tech_post_damage" )

        ## Load pretrained weights
        if self.params['pretrained_path'] is not None:
            self.v_nn.load_weights( self.params["pretrained_path"]  + '/PostDamagePreTech/v_nn_checkpoint_PostDamagePreTech')
            self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/PostDamagePreTech/i_g_nn_checkpoint_PostDamagePreTech')
            self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/PostDamagePreTech/i_d_nn_checkpoint_PostDamagePreTech')
            self.i_r_nn.load_weights( self.params["pretrained_path"]  + '/PostDamagePreTech/i_r_nn_checkpoint_PostDamagePreTech')

        # Preserve the loaded checkpoint if fine-tuning becomes nonfinite.
        best_v_nn.set_weights(self.v_nn.get_weights())
        best_i_g_nn.set_weights(self.i_g_nn.get_weights())
        best_i_d_nn.set_weights(self.i_d_nn.get_weights())
        best_i_r_nn.set_weights(self.i_r_nn.get_weights())
        min_loss = validation_score(
            large_sample_validation(self),
            self.params.get("validation_control_weight", 1.0),
        )
 
      

        # begin sgd iteration
        # begin sgd iteration
        for step in range(self.params["num_iterations"]):
            loss_v_train, loss_c_train = self.train_step()
            if step % self.params["logging_frequency"] == 0:
                test_losses = large_sample_validation(self)
                logK, Z, Y, logR, λ3, logξ = self.sample()
                ## Update normalization constants
                # rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, marginal_utility_of_consumption_norm, FOC_g, FOC_d, y_test, h_y = self.pde_rhs(logK, Z, Y, logR, λ3, logξ)
                # self.flow_pv_norm = (1.0 - self.params['norm_weight']) * self.flow_pv_norm + self.params['norm_weight'] * pv
                # self.marginal_utility_of_consumption_norm = (1.0 - self.params['norm_weight']) * self.marginal_utility_of_consumption_norm + self.params['norm_weight'] * marginal_utility_of_consumption_norm

                ## Store best neural networks
                score = validation_score(
                    test_losses,
                    self.params.get("validation_control_weight", 1.0),
                )
                if not np.isfinite(score):
                    print(f"Stopping at step {step}: validation residual is nonfinite.")
                    break
                if score < min_loss:
                    min_loss = score

                    best_v_nn.set_weights(self.v_nn.get_weights())
                    best_i_g_nn.set_weights(self.i_g_nn.get_weights())
                    best_i_d_nn.set_weights(self.i_d_nn.get_weights())
                    best_i_r_nn.set_weights(self.i_r_nn.get_weights())
 
                ## Generate checkpoints for tensorboard
                if self.params['tensorboard']:
                    grad_v_nn,loss_v_train = self.grad(logK, Z, Y, logR, λ3, logξ, compute_control=False, training=True)
                    grad_controls,loss_c_train = self.grad(logK, Z, Y, logR, λ3, logξ, compute_control=True, training=True)

                    with self.test_writer.as_default():
                        ## Export learning rates
                        for optimizer_idx in range(len(self.params['optimizers'])):
                            if "sgd" in self.params['learning_rate_schedule_type']:
                                tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx]._decayed_lr(tf.float32), step=step)
                            elif "piecewiseconstant" in self.params['learning_rate_schedule_type']:
                                optimizer = self.params["optimizers"][optimizer_idx]
                                current_lr = optimizer.learning_rate(step) if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule) else optimizer.lr
                                tf.summary.scalar(f'learning_rate_{optimizer_idx}', current_lr, step=step)
                            else:
                                tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx].lr, step=step)

                        ## Export losses
                        # tf.summary.scalar('loss_v_training', train_loss, step=step)
                        tf.summary.scalar('loss_value_function', test_losses[0], step=step)
                        tf.summary.scalar('loss_FOC_d', test_losses[1], step=step)
                        tf.summary.scalar('loss_FOC_g', test_losses[2], step=step)
                        tf.summary.scalar('loss_FOC_r', test_losses[3], step=step)
                        tf.summary.scalar('loss_dv_dY', test_losses[4], step=step)
                        
                        tf.summary.scalar('loss_value_train', loss_v_train, step=step)
                        tf.summary.scalar('loss_control_train', loss_c_train, step=step)
                         

                        ## Export weights and gradients
                        for layer in self.v_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step)

                        for g in range(len(self.v_nn.trainable_variables)):
                            tf.summary.histogram(self.v_nn.trainable_variables[g].name + '_grads', grad_v_nn[g], step=step)

                        for layer in self.i_g_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step)

                        for g in range(len(self.i_g_nn.trainable_variables)):
                            tf.summary.histogram(self.i_g_nn.trainable_variables[g].name + '_grads', grad_controls[g], step=step)

                        for layer in self.i_d_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step)

                        for g in range(len(self.i_d_nn.trainable_variables)):
                            tf.summary.histogram(self.i_d_nn.trainable_variables[g].name + '_grads', grad_controls[len(self.i_g_nn.trainable_variables) + g], step=step)

                        for layer in self.i_r_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step) 
                        
                        for g in range(len(self.i_r_nn.trainable_variables)):
                            tf.summary.histogram(self.i_r_nn.trainable_variables[g].name + '_grads', grad_controls[len(self.i_g_nn.trainable_variables) + len(self.i_d_nn.trainable_variables) + g], step=step)
                        
                elapsed_time = time.time() - start_time

                ## Appending to training history
                entry = [step] + list(test_losses) + [ elapsed_time]
                training_history.append(entry)

                ## Save training history
                header = 'step,loss_v,loss_FOC_d,loss_FOC_g,loss_FOC_r,loss_dv_dY,elapsed_time'

                np.savetxt(self.params["export_folder"] + '/training_history.csv',
                        training_history,
                        fmt=['%d'] + ['%.5e'] * len(test_losses) + ['%d'],
                        delimiter=",",
                        header=header,
                        comments='')
            

        ## Use best neural networks 
        self.v_nn.set_weights(best_v_nn.get_weights())
        self.i_g_nn.set_weights(best_i_g_nn.get_weights())
        self.i_d_nn.set_weights(best_i_d_nn.get_weights())
        self.i_r_nn.set_weights(best_i_r_nn.get_weights())
        ## Export last check point
        self.v_nn.save_weights( self.params["export_folder"] + '/v_nn_checkpoint_PostDamagePreTech')
        self.i_g_nn.save_weights( self.params["export_folder"] + '/i_g_nn_checkpoint_PostDamagePreTech')
        self.i_d_nn.save_weights( self.params["export_folder"] + '/i_d_nn_checkpoint_PostDamagePreTech')
        self.i_r_nn.save_weights( self.params["export_folder"] + '/i_r_nn_checkpoint_PostDamagePreTech')

        ## Save training history

        np.savetxt(self.params["export_folder"] + '/training_history.csv',
                training_history,
                fmt=['%d'] + ['%.5e'] * len(test_losses) + ['%d'],
                delimiter=",",
                header=header,
                comments='')
        ## Plot losses loss_v,loss_FOC_d,loss_FOC_g,loss_dv_dY

        loss_v_history                   = [history_record[1] for history_record in training_history]
        loss_FOC_d_history               = [history_record[2] for history_record in training_history]
        loss_FOC_g_history               = [history_record[3] for history_record in training_history]
        loss_FOC_r_history               = [history_record[4] for history_record in training_history]
        loss_dv_dY_history               = [history_record[5] for history_record in training_history]


        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.title("Test loss: value function")
        plt.plot(loss_v_history)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_v_history.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: FOC_d")
        plt.plot(loss_FOC_d_history)
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_d.png")
        plt.close()
 

        plt.figure()
        plt.title("Test loss: FOC_g")
        plt.plot(loss_FOC_g_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_g_history.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: FOC_r")
        plt.plot(loss_FOC_r_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_r_history.png")
        plt.close()

        
        plt.figure()
        plt.title("Test loss: dv_dY")
        plt.plot(loss_dv_dY_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_dv_dY_history.png")
        plt.close()
  

        return np.array(training_history)

    def export_parameters(self):

        ## Export parameters

        with open(self.params["export_folder"] + '/params.txt', 'a') as the_file:
            for key in self.params.keys():
                if "nn_config" not in key:
                    the_file.write( str(key) + ": " + str(self.params[key]) + '\n')
        nn_config_keys = [x for x in self.params.keys() if "nn_config" in x]

        for nn_config_key in nn_config_keys:
            with open(self.params["export_folder"] + '/params_' + nn_config_key + '.txt', 'a') as the_file:
                for key in self.params[nn_config_key].keys():
                    the_file.write( str(key) + ": " + str(self.params[nn_config_key][key]) + '\n')

 


if __name__ == '__main__':
    import os
    import sys 
    
    export_folder                    = sys.argv[1]
    batch_size                       = int(sys.argv[2])
    num_iterations                   = int(sys.argv[3])
    pretrained_path                  = sys.argv[4]
    if pretrained_path == 'None':
        pretrained_path = None
    logging_frequency                = int(sys.argv[5])
    learning_rates                   = [float(x) for x in sys.argv[6].split(",")]
    hidden_layer_activations         = sys.argv[7].split(",")
    output_layer_activations         = sys.argv[8].split(",")
    num_hidden_layers                = int(sys.argv[9])
    num_neurons                      = int(sys.argv[10])
    learning_rate_schedule_type      = sys.argv[11]
    export_folder_output             = sys.argv[12]
    tech_jump_intensity_scale        = float(sys.argv[13]) if len(sys.argv) > 13 else PARAMS.get("tech_jump_intensity_scale", 1.0)
    tech_jump_probability            = float(sys.argv[14]) if len(sys.argv) > 14 else PARAMS.get("π", 0.04)
    logxi_min                        = float(os.environ.get("LOGXI_MIN", sys.argv[15] if len(sys.argv) > 15 else PARAMS.get("logξ_min", -3.0)))
    logxi_max                        = float(os.environ.get("LOGXI_MAX", sys.argv[16] if len(sys.argv) > 16 else PARAMS.get("logξ_max", 5.0)))
    validation_batch_size            = int(os.environ.get("VALIDATION_BATCH_SIZE", max(1024, batch_size)))
    validation_batches               = int(os.environ.get("VALIDATION_BATCHES", 4))
    validation_control_weight        = float(os.environ.get("VALIDATION_CONTROL_WEIGHT", 5.0))
    gradient_clip_norm               = float(os.environ.get("GRADIENT_CLIP_NORM", 1.0))
    tensorboard                      = os.environ.get("ENABLE_TENSORBOARD", "0").lower() in {"1", "true", "yes", "on"}
    
    
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
    "pretrained_path" : pretrained_path, "learning_rate_schedule_type" : learning_rate_schedule_type,
    "tech_jump_intensity_scale": tech_jump_intensity_scale, "π": tech_jump_probability,
    "logξ_min": logxi_min, "logξ_max": logxi_max,
    "validation_batch_size": validation_batch_size, "validation_batches": validation_batches,
    "validation_control_weight": validation_control_weight,
    "gradient_clip_norm": gradient_clip_norm, "tensorboard": tensorboard}

    params["job_name"] = export_folder
    params["export_folder"]  = export_folder +  "/PostDamagePreTech"
    params["v_PostDamagePostTech_nn_path"]  = export_folder +  "/PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech"
    params["v_PreDamagePostTech_nn_path"]  = export_folder +  "/PreDamagePostTech/v_nn_checkpoint_PreDamagePostTech"
    params["v_PostDamagePreTech_nn_path"]  = export_folder +  "/PostDamagePreTech/v_nn_checkpoint_PostDamagePreTech"
    params["v_PostDamageIntermTech_nn_path"]  = export_folder +  "/PostDamageIntermTech/v_nn_checkpoint_PostDamageIntermTech"
    params["v_PreDamageIntermTech_nn_path"]  = export_folder +  "/PreDamageIntermTech/v_nn_checkpoint_PreDamageIntermTech"
    # The lower control bound is -1/theta, which keeps log(1 + theta*i) valid.
    if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
        params["i_g_nn_config"]["final_activation"] = investment_rate_activation(PARAMS["θ_g"])
        params["i_d_nn_config"]["final_activation"] = investment_rate_activation(PARAMS["θ_d"])

    test_model = PostDamagePreTechModel(params)
    test_model.export_parameters()
    test_model.train()
