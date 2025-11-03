"""
Solve  post-damage-post-technology  model.

The value function we are solving is v = V - log N. v still depends on log N in the HJB.
"""

import numpy as np
import tensorflow as tf
import pathlib
import time
from feedforward_subnet import FeedForwardSubNet, setup_optimizers
from params import PARAMS


class PostDamagePostTechModel:
    """Post-damage & post-technology HJB.
 
    """

    def __init__(self, params):
        # Econcomic Parameters described in the appendix
        self.params = PARAMS.copy()
        # Nerual network parameters
        self.params.update(params or {})

        # ensure optimizers are prepared
        setup_optimizers(self.params)

        
        self.params['tensorboard'] = True
        
        self.v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        self.i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        self.i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])


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
            ## Create objects to generate checkpoints for tensorboard
            pathlib.Path(self.params["export_folder"] + '/logs/train/').mkdir(parents=True, exist_ok=True) 
            pathlib.Path(self.params["export_folder"] + '/logs/test/').mkdir(parents=True, exist_ok=True) 
            self.train_writer = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/train/')
            self.test_writer  = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/test/')
 
 
 
    def sample(self):
        # Sampling all variables.

        offsets = tf.random.uniform((self.params['batch_size'],1), 0.0, 1.0)
        logK = tf.random.shuffle(self.params["state_intervals"]["logK"][:-1] + self.params["state_intervals"]["logK_interval_size"] * offsets)

        offsets = tf.random.uniform((self.params['batch_size'],1), 0.0, 1.0)
        Z = tf.random.shuffle(self.params["state_intervals"]["Z"][:-1] + self.params["state_intervals"]["Z_interval_size"] * offsets)

        offsets = tf.random.uniform((self.params['batch_size'],1), 0.0, 1.0)
        Y = tf.random.shuffle(self.params["state_intervals"]["Y"][:-1] + self.params["state_intervals"]["Y_interval_size"] * offsets)

        offsets = tf.random.uniform((self.params['batch_size'],1), 0.0, 1.0)
        logR = tf.random.shuffle(self.params["state_intervals"]["logR"][:-1] + self.params["state_intervals"]["logR_interval_size"] * offsets)

        offsets = tf.random.uniform((self.params['batch_size'],1), 0.0, 1.0)
        λ3 = tf.random.shuffle(self.params["state_intervals"]["λ3"][: -1] + self.params["state_intervals"]["λ3_interval_size"] * offsets)
 
        offsets = tf.random.uniform((self.params['batch_size'],1), 0.0, 1.0)
        logξ = tf.random.shuffle(self.params["state_intervals"]["logξ"][:-1] + self.params["state_intervals"]["logξ_interval_size"] * offsets)
 
        return logK, Z, Y, logR, λ3, logξ 

 
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
        v = self.v_nn(X)
        i_g = self.i_g_nn(X)
        i_d = self.i_d_nn(X)
        
        # State Variables Transformations
        ξ = tf.exp(logξ)
        K = tf.exp(logK)


        ###########
        #### Calculate derivatives
        ###########
        
        dv_dlogK                 = tf.reshape(tf.gradients(v, logK, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        d2v_dlogK2                = tf.reshape(tf.gradients(dv_dlogK, logK, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        d2v_dlogKdZ               = tf.reshape(tf.gradients(dv_dlogK, Z, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        dv_dZ                    = tf.reshape(tf.gradients(v, Z, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        d2v_dZ2                   = tf.reshape(tf.gradients(dv_dZ, Z, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        dv_dY                    = tf.reshape(tf.gradients(v, Y, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        d2v_dY2                   = tf.reshape(tf.gradients(dv_dY, Y, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

         
        ###################
        ###### drift distortions
        ###################
        
        h_d = - 1.0 /  ξ * ((dv_dlogK - Z * dv_dZ ) * (1-Z) * σ_d )
        h_g = - 1.0 /  ξ * ((dv_dlogK + (1-Z) * dv_dZ ) * Z * σ_g )

        # We are solving for v = V - log N, so the damage term in the HJB is modified accordingly
        # dV/dY = dv_dY - d(log N)/dY and d(log N)/dY = λ1 + λ2 * Y  
        h_y = - 1.0 /  ξ * ( dv_dY  - (λ1  + λ2 * Y   )   ) *    η *  A_d * (1-Z) * K     *  ϛ


        ######################
        #### consumption and flow
        #######################
        pv   =   δ * v

        c = ( A_d  - i_d) * (1 - Z) + (A_g_prime_prime - i_g) * Z
        inside_log = tf.reshape(tf.math.maximum(c, 1e-8), (self.params['batch_size'], 1))
        flow = δ * (tf.math.log(inside_log) + logK)


        # drift and drift-corrections (simplified, retains original structure)
        v_logKlogK_term = (σ_d**2 * (1 - Z)**2 + σ_g**2 * Z**2) / 2.0
        
        
        inside_log_i_d   = tf.reshape(tf.math.maximum(1.0 + θ_d * i_d, 1e-8), [self.params["batch_size"], 1])
        inside_log_i_g   = tf.reshape(tf.math.maximum(1.0 + θ_g * i_g, 1e-8), [self.params["batch_size"], 1])
 
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
  
        return rhs, pv, dv_dY, c, 1.0 + θ_g * i_g , 1.0 + θ_d * i_d,  FOC_d, FOC_g 


    @tf.function
    def objective_fn(self, logK, Z, Y, logR, λ3, logξ,  compute_control = False, training = True):

        ## This is the objective function that stochastic gradient descend will try to minimize
        ## It depends on which NN it is training. Controls and value functions have different
        ## objectives.
        
        rhs, pv, dv_dY, c, inside_log_i_g , inside_log_i_d ,  FOC_d, FOC_g = self.pde_rhs(logK, Z, Y, logR, λ3, logξ)

        epsilon = 10e-8
        
        negative_consumption_boolean = tf.reshape( tf.cast( c < 1e-8, tf.float32 ),  [self.params["batch_size"], 1])
        loss_c  = - c  * negative_consumption_boolean + epsilon
        
        negative_inside_log_i_g_boolean = tf.reshape( tf.cast( inside_log_i_g < 1e-8, tf.float32 ),  [self.params["batch_size"], 1])
        loss_inside_log_i_g             = - inside_log_i_g  * negative_inside_log_i_g_boolean + epsilon
        
        negative_inside_log_i_d_boolean = tf.reshape( tf.cast( inside_log_i_d < 1e-8, tf.float32 ),  [self.params["batch_size"], 1])
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
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_d  ))) 
                        
            else:

                ## loss associated with dv/dY > 0
                loss_dv_dY = dv_dY  * tf.reshape( tf.cast(Y > self.params['y_upper'], tf.float32 ),  [self.params["batch_size"], 1]) \
                    * tf.reshape( tf.cast( dv_dY > 0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-8

                    
                loss = tf.sqrt(tf.reduce_mean(tf.square(  rhs - pv    )))  \
                       + tf.sqrt(tf.reduce_mean(tf.square(FOC_g ))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(FOC_d  ))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY  )))  
                    
                return loss

        else:

            ## loss associated with dv/dY > 0
            loss_dv_dY = dv_dY * tf.reshape( tf.cast(Y > self.params['y_upper'], tf.float32 ),  [self.params["batch_size"], 1]) \
                * tf.reshape( tf.cast( dv_dY > 0.0, tf.float32 ),  [self.params["batch_size"], 1])  + 10e-8

            return tf.sqrt(tf.reduce_mean(tf.square((rhs - pv)  ))), tf.sqrt(tf.reduce_mean(tf.square(FOC_d))), tf.sqrt(tf.reduce_mean(tf.square(FOC_g)))   ,  tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY ))) 

    def grad(self, logK, Z, Y, logR, λ3, logξ, compute_control = False, training = True):

        if compute_control:
            with tf.GradientTape(persistent=True) as tape:
                objective = self.objective_fn(logK, Z, Y, logR, λ3, logξ, compute_control, training)

            trainable_variables = self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables 

            grad = tape.gradient(objective, trainable_variables)

            del tape

            return grad, objective
        else:
            with tf.GradientTape(persistent=True) as tape:
                objective = self.objective_fn(logK, Z, Y, logR, λ3, logξ, compute_control, training)
            
            grad = tape.gradient(objective, self.v_nn.trainable_variables)
            del tape

            return grad , objective

    @tf.function
    def train_step(self):
        logK, Z, Y, logR, λ3, logξ = self.sample()
 

        ## First, train value function
        
        grad, loss_v_train = self.grad(logK, Z, Y, logR, λ3, logξ, compute_control= False, training=True)
        self.params["optimizers"][0].apply_gradients(zip(grad, self.v_nn.trainable_variables))

        ## Second, train controls
        grad, loss_c_train = self.grad(logK, Z, Y, logR, λ3, logξ, compute_control= True, training=True)
        self.params["optimizers"][1].apply_gradients(zip(grad, self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables ))

        return loss_v_train, loss_c_train
    
    
    def train(self):

        start_time = time.time()
        training_history = []

        # Prepare to store best neural networks and initialize networks
        min_loss = float("inf")
        
        n_inputs = 7

        best_v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        best_v_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.v_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        best_i_g_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_g_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
        best_i_d_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_d_nn.build( (self.params["batch_size"], n_inputs) )


        best_v_nn.set_weights(self.v_nn.get_weights())
        best_i_g_nn.set_weights(self.i_g_nn.get_weights())
        best_i_d_nn.set_weights(self.i_d_nn.get_weights())
 
        NBER_folder = "/project/lhansen/Cap_NN_oldVersion/November_version_NewParameters/output/Novem_NewParaters_0.01_LR_piecewiseconstant_10e-5,10e-5,10e-5,10e-5_128_neurons_32_#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations2000000"
        self.v_nn.load_weights( NBER_folder + "/post_tech_post_damage/v_nn_checkpoint_post_tech_post_damage" )
        self.i_g_nn.load_weights( NBER_folder  + "/post_tech_post_damage/i_g_nn_checkpoint_post_tech_post_damage")
        self.i_d_nn.load_weights( NBER_folder + "/post_tech_post_damage/i_d_nn_checkpoint_post_tech_post_damage" )
 
        ## Load pretrained weights
        if self.params['pretrained_path'] is not None:
            self.v_nn.load_weights( self.params["pretrained_path"]  + '/PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech')
            self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/PostDamagePostTech/i_g_nn_checkpoint_PostDamagePostTech')
            self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/PostDamagePostTech/i_d_nn_checkpoint_PostDamagePostTech')
 
 
      

        # begin sgd iteration
        # begin sgd iteration
        for step in range(self.params["num_iterations"]):
            loss_v_train, loss_c_train = self.train_step()
            if step % self.params["logging_frequency"] == 0:
                ## Sample test data
                logK, Z, Y, logR, λ3, logξ = self.sample()

                ## Compute test loss
                test_losses = self.objective_fn(logK, Z, Y, logR, λ3, logξ, training=False) 
                ## Update normalization constants
                # rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, marginal_utility_of_consumption_norm, FOC_g, FOC_d, y_test, h_y = self.pde_rhs(logK, Z, Y, logR, λ3, logξ)
                # self.flow_pv_norm = (1.0 - self.params['norm_weight']) * self.flow_pv_norm + self.params['norm_weight'] * pv
                # self.marginal_utility_of_consumption_norm = (1.0 - self.params['norm_weight']) * self.marginal_utility_of_consumption_norm + self.params['norm_weight'] * marginal_utility_of_consumption_norm

                ## Store best neural networks
                if (test_losses[0] < min_loss):
                    min_loss = test_losses[0]

                    best_v_nn.set_weights(self.v_nn.get_weights())
                    best_i_g_nn.set_weights(self.i_g_nn.get_weights())
                    best_i_d_nn.set_weights(self.i_d_nn.get_weights())

 
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
                        tf.summary.scalar('loss_dv_dY', test_losses[3], step=step)
                        
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
            

        ## Use best neural networks 
        self.v_nn.set_weights(best_v_nn.get_weights())
        self.i_g_nn.set_weights(best_i_g_nn.get_weights())
        self.i_d_nn.set_weights(best_i_d_nn.get_weights())

        ## Export last check point
        self.v_nn.save_weights( self.params["export_folder"] + '/v_nn_checkpoint_PostDamagePostTech')
        self.i_g_nn.save_weights( self.params["export_folder"] + '/i_g_nn_checkpoint_PostDamagePostTech')
        self.i_d_nn.save_weights( self.params["export_folder"] + '/i_d_nn_checkpoint_PostDamagePostTech')


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
    
    
    hidden_layer_activations   = [None if x == "None" else x for x in hidden_layer_activations]
    output_layer_activations   = [None if x == "None" else x for x in output_layer_activations]
    
    
    v_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_nn"}
    v_nn_config["final_activation"] = output_layer_activations[0]

    i_g_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[1], "dim" : 1, "nn_name" : "i_g_nn"}
    i_g_nn_config["final_activation"] = output_layer_activations[1]

    i_d_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[2], "dim" : 1, "nn_name" : "i_d_nn"}
    i_d_nn_config["final_activation"] = output_layer_activations[2]
    
    
    params = {"batch_size" : batch_size, "learning_rates":learning_rates,
    "v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config, 
    "num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, 
    "pretrained_path" : pretrained_path, "learning_rate_schedule_type" : learning_rate_schedule_type}

    params["export_folder"]  = export_folder +  "/PostDamagePostTech"
 
    ## i_g and i_d activations come after params because we amy want to use phi_g and phi_d
    phi_g = 16.7
    phi_d = 16.7
    if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
        params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_g) / (tf.exp(2 * x) + 1.0)
        params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_d) / (tf.exp(2 * x) + 1.0)

    test_model = PostDamagePostTechModel(params)
    test_model.export_parameters()
    test_model.train()


