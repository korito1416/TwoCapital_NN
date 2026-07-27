"""
Solve  post-damage-post-technology  model.

The value function we are solving is v = V + log N (so V = v - log N). v still depends on log N in the HJB.
"""

import numpy as np
import tensorflow as tf
import pathlib
import time
from feedforward_subnet import (
    FeedForwardSubNet,
    large_sample_validation,
    sample_state_columns,
    validation_score,
    regime_input_bounds,
)
from schedule_v2 import build_optimizers_v2, EarlyStopper, lbfgs_polish_v2
from params import PARAMS, investment_rate_activation
from pretrained_paths import legacy_nber_folder


class PostDamagePostTechModel:
    """Post-damage & post-technology HJB.
 
    """

    def __init__(self, params):
        # Econcomic Parameters described in the appendix
        self.params = PARAMS.copy()
        # Nerual network parameters
        self.params.update(params or {})

        # ensure optimizers are prepared (v2: separate value/control schedules)
        build_optimizers_v2(self.params)

        
        self.params['tensorboard'] = bool(self.params.get('tensorboard', True))
        
        # v3: SEMI-ANALYTIC controls. The investment rates i_d, i_g are computed
        # in CLOSED FORM from the value-net derivatives inside pde_rhs (see
        # benchmarks/two_capital_deterministic/two_capital_model.consumption /
        # controls). There are NO control networks; v_nn is the ONLY trained net.
        self.v_nn    = FeedForwardSubNet(self.params['v_nn_config'])


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
        (rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, marg_norm, FOC_g, FOC_d)
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
        
        X = tf.concat([logK, Z, Y,  λ3, A_g_prime_prime *tf.ones(tf.shape(Y)) ,logξ, logξ], 1)
        # X = tf.concat([logK, R, Y, gamma_3, A_g_prime, log_xi, log_xi], 1)
        
        # Controls defined in section 3.4
        # v3: i_d, i_g are computed in CLOSED FORM below (after dv_dlogK, dv_dZ).
        v = self.v_nn(X)

        # State Variables Transformations
        ξ = tf.exp(logξ)
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


        ###########
        #### CLOSED-FORM (semi-analytic) controls  [v3]
        ####   Replaces the (under-identified) i_g_nn / i_d_nn networks.
        ####   q_d, q_g are the marginal-value elasticities; the FOCs
        ####     FOC_d = -delta/c + q_d * Gamma_d * theta_d / (1+theta_d*i_d) = 0
        ####     FOC_g = -delta/c + q_g * Gamma_g * theta_g / (1+theta_g*i_g) = 0
        ####   together with the resource constraint
        ####     c = (A_d - i_d)(1-Z) + (A_g'' - i_g) Z
        ####   solve algebraically for (c, i_d, i_g) -- exactly the closed form in
        ####   benchmarks/two_capital_deterministic/two_capital_model.py, with the
        ####   FULL-model slopes q_d = dv_dlogK - Z*dv_dZ, q_g = dv_dlogK + (1-Z)*dv_dZ.
        ###########
        eps_q = tf.constant(1e-3, tf.float32)
        # Admissibility clamp (mirrors the FD slope clamp): keep q_d>0, q_g>0 by
        # clipping the share-slope dv_dZ. q_d = dv_dlogK - Z*dv_dZ >= eps_q gives an
        # UPPER bound on dv_dZ; q_g = dv_dlogK + (1-Z)*dv_dZ >= eps_q gives a LOWER
        # bound on dv_dZ. (Z in [Z_min,Z_max] subset (0,1) so the divisors are >0.)
        dvZ_hi = (dv_dlogK - eps_q) / tf.maximum(Z, 1e-6)               # from q_d >= eps_q
        dvZ_lo = (eps_q - dv_dlogK) / tf.maximum(1.0 - Z, 1e-6)         # from q_g >= eps_q
        # clip into [dvZ_lo, dvZ_hi]; if the box is empty (dvZ_lo > dvZ_hi) clip-by-value
        # below would invert it, so take the midpoint as a safe fallback in that case.
        dvZ_hi_safe = tf.maximum(dvZ_hi, dvZ_lo)
        dv_dZ = tf.clip_by_value(dv_dZ, dvZ_lo, dvZ_hi_safe)

        q_d = dv_dlogK - Z * dv_dZ
        q_g = dv_dlogK + (1.0 - Z) * dv_dZ
        # numerical floor (the clamp above already enforces >= eps_q analytically)
        q_d = tf.maximum(q_d, eps_q)
        q_g = tf.maximum(q_g, eps_q)

        Abar = (1.0 - Z) * A_d + Z * A_g_prime_prime
        c = δ * (Abar + (1.0 - Z) / θ_d + Z / θ_g) \
            / (δ + (1.0 - Z) * Γ_d * q_d + Z * Γ_g * q_g)
        i_d = Γ_d * c * q_d / δ - 1.0 / θ_d
        i_g = Γ_g * c * q_g / δ - 1.0 / θ_g


        ###################
        ###### drift distortions
        ###################
        
        h_d = - 1.0 /  ξ * ((dv_dlogK - Z * dv_dZ ) * (1-Z) * σ_d )
        h_g = - 1.0 /  ξ * ((dv_dlogK + (1-Z) * dv_dZ ) * Z * σ_g )

        # We are solving for v = V + log N (so V = v - log N), so the damage term in the HJB is modified accordingly
        # dV/dY = dv_dY - d(log N)/dY and d(log N)/dY = λ1 + λ2 * Y  
        h_y = - 1.0 /  ξ * ( dv_dY  - (λ1  + λ2 * Y  + λ3 * (Y - y_upper) )   ) *    η *  A_d * (1-Z) * K     *  ϛ


        ######################
        #### consumption and flow
        #######################
        pv   =   δ * v

        # v3: c is ALREADY computed in closed form above (it equals the resource
        # constraint (A_d - i_d)(1-Z) + (A_g'' - i_g)Z by construction of the FOCs).
        # Do NOT recompute it here.
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

        rhs = flow \
            + v_logK_term * dv_dlogK  + v_logKlogK_term * d2v_dlogK2 \
            + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
            +   h_d *  (dv_dlogK - Z * dv_dZ)*(1-Z)*σ_d +    h_g * (dv_dlogK + (1-Z) * dv_dZ)*Z*σ_g    \
            + v_logK_Z_term * d2v_dlogKdZ \
            +  dv_dY  * v_y_term  + v_yy_term * d2v_dY2  + 0.5 * ξ * ( tf.pow( h_d, 2)+ tf.pow(h_g, 2)+tf.pow( h_y, 2))  \
            + (-1.0) * v_logN_term 
 

        ####################
        #### FOCs
        ####################
        marginal_util_c = δ / inside_log
 
        
        FOC_d = -marginal_util_c + Γ_d * θ_d / ( inside_log_i_d ) * (dv_dlogK - Z * dv_dZ)
        FOC_g = -marginal_util_c + Γ_g * θ_g / ( inside_log_i_g )   * (dv_dlogK +  (1.0 - Z) * dv_dZ)

        ####################
        #### Loss preconditioner weight (always-on, frozen via stop_gradient)
        ####################
        eps = tf.constant(5e-3, tf.float32)
        precond_w = tf.abs(v_logK_term) + tf.abs(v_Z_term) / tf.maximum(Z * (1.0 - Z), 1e-3) \
            + v_logKlogK_term + v_ZZ_term / tf.maximum(Z * (1.0 - Z), 1e-3) \
            + tf.abs(v_y_term) + eps
        precond_w = tf.stop_gradient(precond_w)

        return rhs, pv, dv_dY, c, 1.0 + θ_g * i_g , 1.0 + θ_d * i_d,  FOC_d, FOC_g , precond_w


    @tf.function
    def objective_fn(self, logK, Z, Y, logR, λ3, logξ,  compute_control = False, training = True):

        ## v3: SEMI-ANALYTIC controls. There is ONLY the value net to train, so the
        ## objective is the preconditioned HJB residual + the dv/dY>0 monotonicity
        ## penalty + the feasibility penalty. The `compute_control` flag is retained
        ## in the signature for API parity but no longer selects a separate branch
        ## (with closed-form controls FOC_d, FOC_g are ~0 by construction).

        rhs, pv, dv_dY, c, inside_log_i_g , inside_log_i_d ,  FOC_d, FOC_g , precond_w = self.pde_rhs(logK, Z, Y, logR, λ3, logξ)

        if training:
            ## Always-on smooth feasibility penalty. softplus(-x) activates smoothly
            ## as the argument approaches 0 from above. inside_log_i_g/inside_log_i_d
            ## are the RAW 1.0 + θ_* i_* returned by pde_rhs; c is the raw consumption.
            λ_pen = tf.constant(self.params.get("constraint_penalty_weight", 10.0), tf.float32)
            penalty = λ_pen * (
                tf.reduce_mean(tf.math.softplus(-(inside_log_i_d)))
                + tf.reduce_mean(tf.math.softplus(-(inside_log_i_g)))
                + tf.reduce_mean(tf.math.softplus(-c))
            )

            ## loss associated with dv/dY > 0
            loss_dv_dY = dv_dY  * tf.reshape( tf.cast(Y > self.params['y_upper'], tf.float32 ),  [-1, 1]) \
                * tf.reshape( tf.cast( dv_dY > 0, tf.float32 ),  [-1, 1]) + 10e-8

            ## Preconditioned HJB residual + dv/dY penalty + feasibility penalty.
            ## FOC_d/FOC_g are ~0 by construction (closed-form controls) and are
            ## intentionally dropped from the trained loss (they add only
            ## zero-gradient clutter w.r.t. v_nn).
            loss = tf.sqrt(tf.reduce_mean(tf.square( (rhs - pv) / precond_w )))  \
                    + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY  ))) + penalty

            return loss

        else:

            ## loss associated with dv/dY > 0
            loss_dv_dY = dv_dY * tf.reshape( tf.cast(Y > self.params['y_upper'], tf.float32 ),  [-1, 1]) \
                * tf.reshape( tf.cast( dv_dY > 0.0, tf.float32 ),  [-1, 1])  + 10e-8

            ## Keep the 4-tuple (loss_v, FOC_d, FOC_g, loss_dv_dY) for compatibility
            ## with validation_score / large_sample_validation. FOC_d, FOC_g are ~0.
            return tf.sqrt(tf.reduce_mean(tf.square((rhs - pv)  ))), tf.sqrt(tf.reduce_mean(tf.square(FOC_d))), tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))   ,  tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY )))

    def grad(self, logK, Z, Y, logR, λ3, logξ, compute_control = False, training = True):

        ## v3: ONLY the value net is trainable (controls are closed-form). The
        ## compute_control flag is ignored (kept for signature/API parity).
        with tf.GradientTape(persistent=True) as tape:
            objective = self.objective_fn(logK, Z, Y, logR, λ3, logξ, compute_control, training)

        grad = tape.gradient(objective, self.v_nn.trainable_variables)
        del tape

        return grad , objective

    @tf.function
    def train_step(self):
        logK, Z, Y, logR, λ3, logξ = self.sample()

        ## v3: single GradientTape over v_nn only; one (value) optimizer.
        grad, loss_v_train = self.grad(logK, Z, Y, logR, λ3, logξ, compute_control= False, training=True)
        self.params["optimizers"][0].apply_gradients(zip(grad, self.v_nn.trainable_variables))

        return loss_v_train


    def train(self):

        start_time = time.time()
        training_history = []

        # Prepare to store best neural networks and initialize networks
        min_loss = float("inf")
        
        n_inputs = 7

        # v3: ONLY the value net exists (controls are closed-form).
        best_v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        best_v_nn.build((None, n_inputs))
        self.v_nn.build((None, n_inputs))

        best_v_nn.set_weights(self.v_nn.get_weights())

        # v2: a true from-scratch base must NOT warm-start from the NBER checkpoint
        # (BN removed + layer names changed => v2 checkpoints are NOT NBER-load-
        # compatible). FROM_SCRATCH=1 -> train_from_scratch -> skip NBER entirely.
        # v3: only v_nn is loaded (no control nets).
        NBER_folder = None if self.params.get("train_from_scratch") else legacy_nber_folder(required=False)
        if NBER_folder is not None:
            self.v_nn.load_weights( NBER_folder + "/post_tech_post_damage/v_nn_checkpoint_post_tech_post_damage" )

        ## Load pretrained weights
        if self.params['pretrained_path'] is not None:
            self.v_nn.load_weights( self.params["pretrained_path"]  + '/PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech')

        # Preserve the loaded checkpoint if fine-tuning becomes nonfinite.
        best_v_nn.set_weights(self.v_nn.get_weights())
        min_loss = validation_score(
            large_sample_validation(self),
            self.params.get("validation_control_weight", 1.0),
        )

        # v2: patience-based early stopping with best-weights restore.
        # v3: a single (value) net is tracked.
        early = EarlyStopper(
            nets=[self.v_nn],
            best_nets=[best_v_nn],
            patience=int(self.params.get("early_stop_patience", 40)),
        )

        # begin sgd iteration
        # begin sgd iteration
        for step in range(self.params["num_iterations"]):
            loss_v_train = self.train_step()
            if step % self.params["logging_frequency"] == 0:
                test_losses = large_sample_validation(self)
                logK, Z, Y, logR, λ3, logξ = self.sample()
                ## Update normalization constants
                # rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, marginal_utility_of_consumption_norm, FOC_g, FOC_d, y_test, h_y = self.pde_rhs(logK, Z, Y, logR, λ3, logξ)
                # self.flow_pv_norm = (1.0 - self.params['norm_weight']) * self.flow_pv_norm + self.params['norm_weight'] * pv
                # self.marginal_utility_of_consumption_norm = (1.0 - self.params['norm_weight']) * self.marginal_utility_of_consumption_norm + self.params['norm_weight'] * marginal_utility_of_consumption_norm

                ## Store best neural networks (EarlyStopper mirrors live->best on
                ## improvement and signals when to stop on patience/nonfinite).
                score = validation_score(
                    test_losses,
                    self.params.get("validation_control_weight", 1.0),
                )
                if score < min_loss:
                    min_loss = score
                stop = early.update(score, step)
                if stop:
                    print(f"Stopping at step {step}: early-stop (best step {early.best_step}, best score {early.best:.5e}).")
                    break


                ## Generate checkpoints for tensorboard
                if self.params['tensorboard']:
                    grad_v_nn,loss_v_train = self.grad(logK, Z, Y, logR, λ3, logξ, compute_control=False, training=True)

                    with self.test_writer.as_default():
                        ## Export learning rate (v3: single value optimizer)
                        optimizer_idx = 0
                        if "sgd" in self.params['learning_rate_schedule_type']:
                            tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx]._decayed_lr(tf.float32), step=step)
                        elif "piecewiseconstant" in self.params['learning_rate_schedule_type']:
                            optimizer = self.params["optimizers"][optimizer_idx]
                            current_lr = optimizer.learning_rate(step) if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule) else optimizer.lr
                            tf.summary.scalar(f'learning_rate_{optimizer_idx}', current_lr, step=step)
                        else:
                            tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx].lr, step=step)

                        ## Export losses
                        tf.summary.scalar('loss_value_function', test_losses[0], step=step)
                        tf.summary.scalar('loss_FOC_d', test_losses[1], step=step)
                        tf.summary.scalar('loss_FOC_g', test_losses[2], step=step)
                        tf.summary.scalar('loss_dv_dY', test_losses[3], step=step)

                        tf.summary.scalar('loss_value_train', loss_v_train, step=step)


                        ## Export weights and gradients (value net only)
                        for layer in self.v_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step)

                        for g in range(len(self.v_nn.trainable_variables)):
                            tf.summary.histogram(self.v_nn.trainable_variables[g].name + '_grads', grad_v_nn[g], step=step)


                elapsed_time = time.time() - start_time

                ## Appending to training history
                entry = [step] + list(test_losses) + [ elapsed_time]
                training_history.append(entry)

                ## Save training history
                header = 'step,loss_v,loss_FOC_d,loss_FOC_g,loss_dv_dY,elapsed_time'

                np.savetxt(self.params["export_folder"] + '/training_history.csv',
                        training_history,
                        fmt=['%d'] + ['%.5e'] * len(test_losses) + ['%d'],
                        delimiter=",",
                        header=header,
                        comments='')
            

        ## Use best neural networks (restore best weights tracked by EarlyStopper)
        early.restore()

        ## Optional end-of-training L-BFGS polish on the RAW combined residual.
        lbfgs_iters = int(self.params.get("lbfgs_polish_iters", 0))
        if lbfgs_iters > 0:
            try:
                lbfgs_polish_v2(self, maxiter=lbfgs_iters)
            except Exception as exc:
                print(f"L-BFGS polish skipped ({exc}).")

        ## Export last check point (v3: value net only; controls are closed-form)
        self.v_nn.save_weights( self.params["export_folder"] + '/v_nn_checkpoint_PostDamagePostTech')


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
        loss_dv_dY_history               = [history_record[4] for history_record in training_history]


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
    # v2 training knobs
    phase                            = os.environ.get("PHASE", "base").lower()
    train_from_scratch               = os.environ.get("FROM_SCRATCH", "0").lower() in {"1", "true", "yes", "on"}
    early_stop_patience              = int(os.environ.get("EARLY_STOP_PATIENCE", 40))
    model_seed                       = int(os.environ.get("MODEL_SEED", 0))
    lbfgs_polish_iters               = int(os.environ.get("LBFGS_POLISH_ITERS", 0))


    hidden_layer_activations   = [None if x == "None" else x for x in hidden_layer_activations]
    output_layer_activations   = [None if x == "None" else x for x in output_layer_activations]

    # v2: per-column input bounds for the fixed input normalization.
    _input_bounds = regime_input_bounds("PostDamagePostTech")


    # v3: ONLY the value net is built (controls are semi-analytic / closed-form).
    # The control-net activation CLI positions (hidden/output [1] and [2]) are kept
    # in the argv contract for parity with the other regimes but are now unused.
    v_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_nn"}
    v_nn_config["final_activation"] = output_layer_activations[0]

    # v2: attach fixed input-normalization bounds and seed to the value-net config.
    v_nn_config["input_bounds"] = _input_bounds
    v_nn_config["seed"] = model_seed


    params = {"batch_size" : batch_size, "learning_rates":learning_rates,
    "v_nn_config" : v_nn_config,
    "num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True,
    "pretrained_path" : pretrained_path, "learning_rate_schedule_type" : learning_rate_schedule_type,
    "tech_jump_intensity_scale": tech_jump_intensity_scale, "π": tech_jump_probability,
    "logξ_min": logxi_min, "logξ_max": logxi_max,
    "validation_batch_size": validation_batch_size, "validation_batches": validation_batches,
    "validation_control_weight": validation_control_weight,
    "gradient_clip_norm": gradient_clip_norm, "tensorboard": tensorboard,
    "phase": phase, "train_from_scratch": train_from_scratch,
    "early_stop_patience": early_stop_patience,
    "lbfgs_polish_iters": lbfgs_polish_iters}

    params["export_folder"]  = export_folder +  "/PostDamagePostTech"

    test_model = PostDamagePostTechModel(params)
    test_model.export_parameters()
    test_model.train()
