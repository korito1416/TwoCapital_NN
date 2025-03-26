######################################################################
######################################################################
##########       This file defines the model class          ########## 
######################################################################
######################################################################

#note line 1425-1436: distorted probability

import numpy as np
import tensorflow as tf
import time
import logging
from tensorflow import keras
import json 
import pathlib
import matplotlib.pyplot as plt
import os 
import pandas as pd


tf.random.set_seed(11117)


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__(name = config["nn_name"] + ".init_layer")
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5),
                name = config["nn_name"] + ".bn." + str(_)
            )
            for _ in range(len(config["num_hiddens"]) + 1)]
        
        if config['activation'] is not None and "relu" in config['activation']:
            initializer = tf.keras.initializers.HeNormal(seed=0)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers = [tf.keras.layers.Dense(config["num_hiddens"][i],
                                                   use_bias=config['use_bias'],
                                                   activation=config['activation'],
                                                   kernel_initializer = initializer,
                                                   name = config["nn_name"] + ".dense." + str(i))
                             for i in range(len(config["num_hiddens"]))]
        # final output should be gradient of size dim
        try:
            if config['final_activation'] is None:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
            elif "relu" in config['final_activation']:
                initializer = tf.keras.initializers.HeNormal(seed=0)
            else:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
        except:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers.append(tf.keras.layers.Dense(config["dim"], 
        kernel_initializer = initializer, 
        activation=config['final_activation'], use_bias = True, name = config["nn_name"] + ".output" ))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        x_inputs = []
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x_inputs.append(x)
        x = tf.keras.layers.Add()(x_inputs)
        x = self.dense_layers[-1](x)
        return x
    
class model:
    def __init__(self, params):
        
        ## Load parameters
        self.params  = params 
        
        ## Table 1: Economic Parameters
        self.params["δ"] = 0.01
        self.params["α_d"] =  -0.035; self.params["Γ_d"] =  0.06; self.params["θ_d"] =  16.7; self.params["σ_d"] = 0.01
        self.params["α_g"] =  -0.035; self.params["Γ_g"] =  0.06; self.params["θ_g"] =  16.7; self.params["σ_g"] = 0.01

        self.params["A_d"] = 0.12; self.params["A_g"] = 0.113
  
        self.params['A_g_prime_list'] = [0.120,0.128,0.136] # Comment out to parellize the computing
        self.params["A_g_prime_length"] =  len(self.params['A_g_prime_list'])
        self.params["A_g_prime_max"] = self.params['A_g_prime_list'][-1]
        self.params["A_g_prime_min"] = self.params['A_g_prime_list'][0]
        # self.params["A_g_prime_list"]     =  [self.params["A_g"] * self.params["lambda_A_g_prime"]**(1+i) for i in range(self.params["A_g_prime_length"]) ]
        
        self.params["λ"] = 1.063; self.params["n"] = 1 
        self.params["ζ"] = 0.0; self.params["ψ_0"] = 0.10573 ;self.params["ψ_1"] = 0.5; self.params["σ_κ"] = 0.0078
        self.params["ϱ"] = 1120
        
        
        ## Table 2: Climate Dynamics and Damages Parameters
        self.params["θ̄"] = 1.86 / 1000
        self.params["η"] = 0.316
        self.params["ς"] = 1.2 * 1.86 / 1000
        self.params["λ_1"] = 0.00017675
        self.params["λ_2"] = 2 * 0.0022
        
        self.params["λ_3_min"] = 0.0
        self.params["λ_3_max"] = 1.0/3.0    
        self.params["λ_3_length"] = 20
        self.params["λ_3_list"]   = np.linspace(self.params["λ_3_min"], self.params["λ_3_max"], self.params["λ_3_length"]).tolist()
        
        self.params["r_1"] = 1.5
        self.params["r_2"] = 2.5
        self.params["y̲"] = 1.5
        self.params["ȳ"] = 2.0
        
        
        
        ## Table 3: State Variable Initial Values and Ranges
        self.params["K_0"] = 880.0
        self.params["Z_0"] = 0.5
        self.params["Y_0"] = 1.1
        self.params["R_0"] = 11.2 
        self.params["logK_min"] = 4.0
        self.params["logK_max"] = 7.0
        self.params["Z_min"] = 0.01
        self.params["Z_max"] = 0.99
        self.params["Y_min"] = 10e-3
        self.params["Y_max"] = 3.0
        self.params["logR_min"] = 1.0
        self.params["logR_max"] = 6.0
        self.params['logξ_min'] = -3.0
        self.params['logξ_max'] = 5.0
        
        
        if 'tensorboard' not in params.keys():
            print("Tensorboard option not detected; setting to False by default.")
            self.params['tensorboard'] = False 
        print("Tensorboard boolean =", self.params['tensorboard'] )


        ## Create tensors to store normalizing constants 
        consumption_guess =  ( np.exp(self.params["logK_max"]) + np.exp(self.params["logK_min"]) ) / 2 * 0.1 ## assume consuming 10% of capital

        ## Normalization term. 
        self.flow_pv_norm                          =  tf.ones(shape = (self.params['batch_size'],1) ) #* self.params['δ'] * np.log(consumption_guess)
        self.marginal_utility_of_consumption_norm  =  tf.ones(shape = (self.params['batch_size'],1) ) #* self.params['δ'] / consumption_guess

        ## Create neural networks
        self.v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        self.i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        self.i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
        
        if "pre_tech" in self.params["model_type"]:
            print("Pre tech model detected. Building a neural network for i_r")
            self.i_r_nn  = FeedForwardSubNet(self.params['i_r_nn_config'])
        
        ################################################################
        ######## Load Trained Networks for Jump term calculations
        ###############################################################
        if "pre_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
 
            ## Load post_tech_post_damage model  
            self.v_post_tech_post_damage_nn = FeedForwardSubNet(self.params['v_nn_config'])
            ## 5 inputs here: logK, Z, Y,    λ_3, A_g , logξ 
            self.v_post_tech_post_damage_nn.build( (self.params["batch_size"], 6) )  
            self.v_post_tech_post_damage_nn.load_weights( self.params["v_post_tech_post_damage_nn_path"]  + '/v_nn_checkpoint_post_tech_post_damage')


        elif "pre_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
            
            ## Load post tech post damage model
            self.v_post_tech_post_damage_nn = FeedForwardSubNet(self.params['v_nn_config'])
            self.v_post_tech_post_damage_nn.build( (self.params["batch_size"], 6) )  ## need to build network; pre_tech has four state variables so we don't need to do anything (remember λ_3 is a pseudo state variable)
            ## 5 inputs here: logK, Z, Y, λ_3,   A_g ,  logξ   
            self.v_post_tech_post_damage_nn.load_weights( self.params["v_post_tech_post_damage_nn_path"]  + '/v_nn_checkpoint_post_tech_post_damage')


        elif "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
            
            # print(self.params["model_type"])
            self.v_pre_tech_post_damage_nn = FeedForwardSubNet(self.params['v_nn_config'])
            self.v_pre_tech_post_damage_nn.build( (self.params["batch_size"], 6) ) 
            ## 6 inputs here: logK, Z, Y, logR, λ_3, logξ 
            self.v_pre_tech_post_damage_nn.load_weights( self.params["v_pre_tech_post_damage_nn_path"]  + '/v_nn_checkpoint_pre_tech_post_damage')


            self.v_post_tech_pre_damage_nn = FeedForwardSubNet(self.params['v_nn_config'])
            self.v_post_tech_pre_damage_nn.build( (self.params["batch_size"], 5) ) 
            ## 5 inputs here: logK, Z, Y,  A_g, logξ  
            self.v_post_tech_pre_damage_nn.load_weights( self.params["v_post_tech_pre_damage_nn_path"]  + '/v_nn_checkpoint_post_tech_pre_damage')



        ## Create folder 
        pathlib.Path(self.params["export_folder"]).mkdir(parents=True, exist_ok=True) 

        ## Create ranges for sampling later 
        self.params["state_intervals"] = {}
 
        self.params["state_intervals"]["logK"]     =  tf.reshape(tf.linspace(self.params['logK_min'], self.params['logK_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["logK_interval_size"] =  self.params["state_intervals"]["logK"][1] -  self.params["state_intervals"]["logK"][0]

        self.params["state_intervals"]["Z"]        =  tf.reshape(tf.linspace(self.params['Z_min'], self.params['Z_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["Z_interval_size"] =  self.params["state_intervals"]["Z"][1] -  self.params["state_intervals"]["Z"][0]

        self.params["state_intervals"]["Y"]        =  tf.reshape(tf.linspace(self.params['Y_min'], self.params['Y_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["Y_interval_size"] =  self.params["state_intervals"]["Y"][1] -  self.params["state_intervals"]["Y"][0]
        
        self.params["state_intervals"]["logR"]        =  tf.reshape(tf.linspace(self.params['logR_min'], self.params['logR_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["logR_interval_size"] =  self.params["state_intervals"]["logR"][1] -  self.params["state_intervals"]["logR"][0]

        # if "post_damage" in self.params["model_type"]:
        self.params["state_intervals"]["λ_3"] = tf.reshape(tf.linspace(self.params['λ_3_min'], self.params['λ_3_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["λ_3_interval_size"] =  self.params["state_intervals"]["λ_3"][1] -  self.params["state_intervals"]["λ_3"][0]


        # if "post_tech" in self.params["model_type"]:
        self.params["state_intervals"]["A_g_prime"] = tf.reshape(tf.linspace(self.params['A_g_prime_min'], self.params['A_g_prime_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["A_g_prime_interval_size"] =  self.params["state_intervals"]["A_g_prime"][1] -  self.params["state_intervals"]["A_g_prime"][0]
 
        self.params["state_intervals"]["logξ"] = tf.reshape(tf.linspace(self.params['logξ_min'], self.params['logξ_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["logξ_interval_size"] =  self.params["state_intervals"]["logξ"][1] -  self.params["state_intervals"]["logξ"][0]


        ## Create objects to generate checkpoints for tensorboard
        pathlib.Path(self.params["export_folder"] + '/logs/train/').mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self.params["export_folder"] + '/logs/test/').mkdir(parents=True, exist_ok=True) 
        self.train_writer = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/train/')
        self.test_writer  = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/test/')


    def sample(self):
        '''
        Sampling all state variables. Not all variables are used in Calculation. 
        '''
        
        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        logK         = tf.random.shuffle(self.params["state_intervals"]["logK"][:-1] + self.params["state_intervals"]["logK_interval_size"] * offsets)

        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        Z            = tf.random.shuffle(self.params["state_intervals"]["Z"][:-1] + self.params["state_intervals"]["Z_interval_size"] * offsets)

        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        Y            = tf.random.shuffle(self.params["state_intervals"]["Y"][:-1] + self.params["state_intervals"]["Y_interval_size"] * offsets)

        
        offsets            = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        logR            = tf.random.shuffle(self.params["state_intervals"]["logR"][:-1] + self.params["state_intervals"]["logR_interval_size"] * offsets)
        
        ## Sample λ_3
        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        λ_3      = tf.random.shuffle(self.params["state_intervals"]["λ_3"][:-1] + self.params["state_intervals"]["λ_3_interval_size"] * offsets)
 
        ## Sample A_g_prime
        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        A_g_prime      = tf.random.shuffle(self.params["state_intervals"]["A_g_prime"][:-1] + self.params["state_intervals"]["A_g_prime_interval_size"] * offsets)

        ## Sample logξ 
        offsets = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        logξ = tf.random.shuffle(self.params["state_intervals"]["logξ"][:-1] +  self.params["state_intervals"]["logξ_interval_size"] * offsets)
        
        return logK, Z, Y,  logR,  λ_3 , A_g_prime, logξ
 

    @tf.function
    def pde_rhs(self, logK, Z, Y, logR, λ_3, A_g_prime, logξ):
        '''
        This is the RHS of the HJB equation
        '''

        δ    = self.params["δ"]
        α_d  = self.params["α_d"]
        Γ_d  = self.params["Γ_d"]
        θ_d  = self.params["θ_d"]
        σ_d  = self.params["σ_d"]

        α_g  = self.params["α_g"]
        Γ_g  = self.params["Γ_g"]
        θ_g  = self.params["θ_g"]
        σ_g  = self.params["σ_g"]

        A_d  = self.params["A_d"]
        A_g  = self.params["A_g"]

        
        λ    = self.params["λ"]
        n    = self.params["n"]
        ζ    = self.params["ζ"]
        ψ_0   = self.params["ψ_0"]
        ψ_1   = self.params["ψ_1"]
        σ_κ  = self.params["σ_κ"]
        ϱ    = self.params["ϱ"]

        # Table 2: Climate Dynamics and Damages Parameters
        θ̄   = self.params["θ̄"]
        η    = self.params["η"]
        ς    = self.params["ς"]

        λ_1   = self.params["λ_1"]
        λ_2   = self.params["λ_2"]
        

        r_1   = self.params["r_1"]
        r_2   = self.params["r_2"]
        y̲   = self.params["y̲"]
        ȳ    = self.params["ȳ"]

        
        
        
        ## Transform inputs
        if "post_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
            X = tf.concat([logK, Z, Y,       λ_3, A_g_prime, logξ], 1)
 
            
        if "pre_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
            X = tf.concat([logK, Z, Y, logR, λ_3,            logξ], 1)
            



        if "pre_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            X = tf.concat([logK, Z, Y,          A_g_prime, logξ ], 1)
 

        if "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
            X = tf.concat([logK, Z, Y, logR,                logξ], 1)
            

 
        
        ## Evalute neural networks 
        v            = self.v_nn(X)
        i_g          = self.i_g_nn(X)
        i_d          = self.i_d_nn(X)
        if "pre_tech" in self.params["model_type"]: 
            i_r         = self.i_r_nn(X)


        ## Calculate some variables for proceeding calculation. 
        ξ  = tf.exp(logξ) 
        K = tf.reshape(tf.exp(logK), [self.params['batch_size'], 1])

        #########################
        #### Calculate Partial Derivatives
        #########################
        dv_dlogK                 = tf.reshape(tf.gradients(v, logK, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddlogK                = tf.reshape(tf.gradients(dv_dlogK, logK, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        
        dv_dlogKdZ               = tf.reshape(tf.gradients(dv_dlogK, Z, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        dv_dZ                    = tf.reshape(tf.gradients(v, Z, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddZ                   = tf.reshape(tf.gradients(dv_dZ, Z, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        dv_dY                    = tf.reshape(tf.gradients(v, Y, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddY                   = tf.reshape(tf.gradients(dv_dY, Y, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        if "pre_tech" in self.params["model_type"]:
            ## Compute terms related to logR
            dv_dlogR                 = tf.reshape(tf.gradients(v, logR, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
            dv_ddlogR                 = tf.reshape(tf.gradients(dv_dlogR, logR, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
 
 
        #########################
        #### Compute h distortion 
        #########################
        h_d     = - 1.0 / ξ * ((dv_dlogK - Z * dv_dZ ) * (1-Z) * σ_d)
        h_g     = - 1.0 / ξ * ((dv_dlogK + (1-Z) * dv_dZ ) * Z * σ_g)
        h_y     = - 1.0 / ξ * dv_dY   * (  η  *  A_d  * (1-Z) * K ) *  ς
        if "pre_tech" in self.params["model_type"]:
            h_r = - 1.0 / ξ *  σ_κ  * dv_dlogR
   
        #########################
        #### 
        #########################
        pv   = δ * v

        if "pre_tech" in self.params["model_type"]:
            ## Before tech jump, productivity is A_g and planner invests in R&D
            c        = (self.params["A_d"] - i_d) * (1 - Z) + (self.params["A_g"] - i_g) * Z - i_r
        else:
            ## After tech jump, no Z&D investment and productivity is A_g_prime
            c        = (self.params["A_d"] - i_d) * (1 - Z) + (A_g_prime - i_g) * Z 


        # inside_log   =   tf.reshape( tf.math.maximum( c , 0.000000001), [self.params["batch_size"], 1])
        inside_log  =   tf.reshape( tf.math.maximum( c , 0.000000001), [self.params["batch_size"], 1])
        flow           = δ * (tf.math.log( inside_log )  +  logK )
        
        # if "pre_damage" in self.params["model_type"]:
        # if "post_damage" in self.params["model_type"]:
        #     logN = λ_1*Y  + 0.5*λ_2* tf.pow(Y, 2)+ λ_3* ( ȳ -Y)**2
        # else: 
        #     logN = λ_1*Y  + 0.5*λ_2* tf.pow(Y, 2)
        
        logN = λ_1*Y  + 0.5*λ_2* tf.pow(Y, 2)
        
        inside_log_i_d   =   tf.reshape( tf.math.maximum( 1 + θ_d * i_d , 0.0001), [self.params["batch_size"], 1])
        inside_log_i_g   =   tf.reshape( tf.math.maximum( 1 + θ_g * i_g , 0.0001), [self.params["batch_size"], 1])
        φ_d = α_d +  Γ_d  * tf.math.log( inside_log_i_d )
        φ_g = α_g +  Γ_g  * tf.math.log( inside_log_i_g )
            
        v_kk_term      = ( tf.pow(σ_d,2) * tf.pow(1-Z,2) + tf.pow(σ_g,2) * tf.pow(Z,2))/2.0
        v_k_term       = φ_d * (1 - Z) + φ_g * Z  - v_kk_term
 
 
        v_z_term       = ( φ_g  -  tf.pow(σ_g, 2)*Z - φ_d + tf.pow(σ_d,2) *  (1-Z )   ) * Z * (1 - Z)
        v_zz_term      = 0.5 * tf.pow(Z, 2) * tf.pow( 1- Z, 2) *  ( tf.pow(σ_g,2) + tf.pow(σ_d, 2))

        v_logK_z_term  = -Z * tf.pow(1-Z, 2) * tf.pow(σ_d, 2) + tf.pow(Z, 2) * (1.0 - Z) * tf.pow(σ_g, 2)

        v_y_term       = θ̄ * (η *  A_d * (1-Z) * K)
        v_yy_term      = 0.5 * tf.pow( ς,2) * tf.pow(η * A_d * (1-Z) * K, 2)
        
        if "pre_tech" in self.params["model_type"]:
            φ_logR = -  ζ  +  ψ_0  * tf.exp(- ψ_1  * (  tf.math.log(i_r) + logK -  logR)  )  
            v_logR_term     = φ_logR  - 0.5 * tf.pow(σ_κ, 2)
            v_logRlogR_term = 0.5 * tf.pow(σ_κ, 2)
        
        
        #########################################################
        ######## RHS without Jump terms #########################
        #########################################################
        #    -pv
        rhs = flow - δ*logN   \
                + v_k_term * dv_dlogK + v_kk_term * dv_ddlogK  \
                +v_z_term * dv_dZ + v_zz_term * dv_ddZ \
                + v_logK_z_term * dv_dlogKdZ \
                +v_y_term * dv_dY+ v_yy_term * dv_ddY \
                - 0.5* ξ *( tf.pow(h_d,2) + tf.pow(h_g,2)  +tf.pow(h_y,2) )
 
        if "pre_tech" in self.params["model_type"]:
            rhs +=   v_logR_term * dv_dlogR+ v_logRlogR_term * dv_ddlogR -  0.5* ξ * tf.pow(h_r,2) 

 
        #####################################
        ######## Jump Terms
        ###################################
         
        ## post tech and pre damage model
        if "post_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
            ## Damage Jump Intensity J_damage
            J_damage = r_1*( tf.exp(  r_2  / 2 * tf.pow(Y - y̲,2) ) - 1  ) * tf.cast(Y > y̲, tf.float32 )
            for k in range(self.params["λ_3_length"]):
                # X = tf.concat([logK, Z, Y,       λ_3, A_g_prime, logξ], 1)
                X_post_tech_post_damage    = tf.concat([logK, Z, Y,    tf.ones(tf.shape(Y)) * self.params["λ_3_list"][k], A_g_prime, logξ], 1)
                v_post_tech_post_damage    =  self.v_post_tech_post_damage_nn(X_post_tech_post_damage) 
                 
                g_damage       = tf.exp(-1.0/ ξ * (v_post_tech_post_damage - v))
 
                rhs +=  J_damage / self.params['λ_3_length'] *  (
                    g_damage * (v_post_tech_post_damage - v) \
                    +  ξ * (1.0 - g_damage  + g_damage  * tf.math.log(g_damage)  ))  

        elif "pre_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]: 
            J_tech = tf.exp(logR)
            for j in range(self.params["A_g_prime_length"]):
                X_post_tech_post_damage    = tf.concat([logK, Z, Y, λ_3, tf.ones(tf.shape(Y)) * self.params["A_g_prime_list"][j], logξ   ], 1)
                v_post_tech_post_damage    =  self.v_post_tech_post_damage_nn(X_post_tech_post_damage) 
               
                g_tech  = tf.exp(-1.0/ ξ * (v_post_tech_post_damage - v))

                rhs += ( J_tech/ self.params['A_g_prime_length'] ) / ϱ \
                       *  (g_tech * ( v_post_tech_post_damage - v ) +  
                          ξ * (1.0 - g_tech + g_tech  *tf.math.log(g_tech) )) 
                    
        elif "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
            J_damage = r_1*( tf.exp(  r_2  / 2 * tf.pow(Y - y̲,2) ) - 1  ) * tf.cast(Y > y̲, tf.float32 )
            J_tech = tf.exp(logR)
            
            for k in range(self.params["λ_3_length"]):
                # X = tf.concat([logK, Z, Y, logR, λ_3,            logξ], 1)
                X_pre_tech_post_damage = tf.concat([logK, Z, Y, logR, tf.ones(tf.shape(Y)) * self.params["λ_3_list"][k], logξ], 1)
                v_pre_tech_post_damage = self.v_pre_tech_post_damage_nn(X_pre_tech_post_damage)
                g_damage  = tf.exp(-1.0/ ξ * (v_pre_tech_post_damage - v))
            
                rhs +=  J_damage / self.params['λ_3_length'] *  (
                    g_damage * (v_pre_tech_post_damage - v) \
                    +  ξ * (1.0 - g_damage  + g_damage  * tf.math.log(g_damage)  ))  
 
            for j in range(self.params["A_g_prime_length"]):
                # X = tf.concat([logK, Z, Y,          A_g_prime, logξ ], 1)
                X_post_tech_pre_damage = tf.concat([logK, Z, Y,  tf.ones(tf.shape(Y)) * self.params["A_g_prime_list"][j], logξ ],  1)
                v_post_tech_pre_damage   = self.v_post_tech_pre_damage_nn(X_post_tech_pre_damage)
                 
                g_tech    =  tf.exp(-1.0/ ξ * (v_post_tech_pre_damage - v))
                
                rhs += ( J_tech/ self.params['A_g_prime_length'] ) / ϱ \
                       *  (g_tech * ( v_post_tech_pre_damage - v ) +  
                          ξ * (1.0 - g_tech + g_tech  *tf.math.log(g_tech) )) 


        ###################################################
        ######### FOCs w.r.t controls lar
        ###################################################
   
        marginal_util_c_over_k = δ/ inside_log

        FOC_g   = - marginal_util_c_over_k  +  Γ_g * θ_g / ( inside_log_i_g )   * (dv_dlogK   - Z  * dv_dZ)
        FOC_d   = - marginal_util_c_over_k  +  Γ_d * θ_d / ( inside_log_i_d )   * (dv_dlogK + (1.0 - Z) * dv_dZ)
        

        if "pre_tech" in self.params["model_type"]:
            FOC_r   = - marginal_util_c_over_k \
                       +  ψ_0  *  ψ_1  / i_r   *  tf.exp( self.params["ψ_1"] * (tf.math.log(i_r) +logK -  logR) )  * dv_dlogR   # + 1e-4

            return rhs, pv,  FOC_g, FOC_d, FOC_r, c,inside_log_i_g,inside_log_i_d,i_r,dv_dY,dv_dlogR
        else:
            return rhs, pv,  FOC_g, FOC_d        ,c,inside_log_i_g,inside_log_i_d    ,dv_dY 

    @tf.function
    def objective_fn(self, logK, Z, Y,  logR  , λ_3,  A_g_prime, logξ,  compute_control, training):
 
        ## This is the objective function that stochastic gradient descend will try to minimize
        ## It depends on which NN it is training. Controls and value functions have different
        ## objectives.

        if "pre_tech" in self.params["model_type"]:
            rhs, pv, FOC_g, FOC_d, FOC_r,c,inside_log_i_g,inside_log_i_d,i_r,dv_dY,dv_dlogR  = self.pde_rhs( logK, Z, Y, logR, λ_3, A_g_prime, logξ)
        else:
            rhs, pv, FOC_g, FOC_d       ,c,inside_log_i_g,inside_log_i_d, dv_dY      = self.pde_rhs(logK, Z, Y, logR, λ_3, A_g_prime, logξ)

        epsilon = 10e-4
        negative_consumption_boolean = tf.reshape( tf.cast( c < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
        loss_c  = - c  * negative_consumption_boolean + epsilon
        
        negative_inside_log_i_g_boolean = tf.reshape( tf.cast( inside_log_i_g < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
        loss_inside_log_i_g             = - inside_log_i_g  * negative_inside_log_i_g_boolean + epsilon
        
        negative_inside_log_i_d_boolean = tf.reshape( tf.cast( inside_log_i_d < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
        loss_inside_log_i_d             = - inside_log_i_d  * negative_inside_log_i_d_boolean + epsilon

        if "pre_tech" in self.params["model_type"]:
            ## i_r cannot be negative
            negative_i_r_boolean            = tf.reshape( tf.cast( i_r < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
            loss_i_r                        = - i_r  * negative_i_r_boolean + epsilon
                
        if training:    
            ## Take care of nonsensical controls first

            loss_c_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_c / self.marginal_utility_of_consumption_norm)))        
            loss_inside_log_i_g_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_g / self.marginal_utility_of_consumption_norm)))
            loss_inside_log_i_d_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_d / self.marginal_utility_of_consumption_norm)))
            control_constraints = tf.reduce_sum(negative_consumption_boolean) + tf.reduce_sum(negative_inside_log_i_g_boolean) + tf.reduce_sum(negative_inside_log_i_d_boolean)
            loss_constraints    = loss_c_mse + loss_inside_log_i_g_mse + loss_inside_log_i_d_mse  

            if "pre_tech" in self.params["model_type"]:

                control_constraints     = control_constraints + tf.reduce_sum(negative_i_r_boolean)
                loss_i_r_mse            = tf.sqrt(tf.reduce_mean(tf.square(loss_i_r / self.marginal_utility_of_consumption_norm)))
                loss_constraints        = loss_constraints + loss_i_r_mse
            
            if control_constraints > 0:
                return loss_constraints  

            if compute_control:
                loss_dv_dY=   dv_dY  * tf.reshape( tf.cast(Y < self.params['ȳ'], tf.float32 ),  [self.params["batch_size"], 1]) \
                    * tf.reshape( tf.cast( dv_dY > 0.0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4
                ## Optimizing all three together
                if "pre_tech" in self.params["model_type"]:
                    return -tf.reduce_mean( (rhs - pv ) / self.flow_pv_norm ) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm)))  + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_r / self.marginal_utility_of_consumption_norm))) 
                else:
                    return -tf.reduce_mean( (rhs - pv ) / self.flow_pv_norm ) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm)))  + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm)))  
            else:
                ## loss associated with dv/dY > 0
                loss_dv_dY = dv_dY  * tf.reshape( tf.cast(Y < self.params['ȳ'], tf.float32 ),  [self.params["batch_size"], 1]) \
                    * tf.reshape( tf.cast( dv_dY > 0.0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4


                if "pre_tech" in self.params["model_type"]:
 
                    loss_dv_dlogR = - dv_dlogR  * tf.reshape( tf.cast( dv_dlogR < 0.0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4

                    loss = tf.sqrt(tf.reduce_mean(tf.square( (rhs - pv) / self.flow_pv_norm ))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(FOC_r / self.marginal_utility_of_consumption_norm))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dlogR / self.marginal_utility_of_consumption_norm)))\
                        # + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY / self.marginal_utility_of_consumption_norm))) 

                else:
                    loss = tf.sqrt(tf.reduce_mean(tf.square( (rhs - pv) / self.flow_pv_norm ))) \
                        + tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm)))\
                        + tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) \
                        # + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY / self.marginal_utility_of_consumption_norm)))  
                return loss  

        else:

            ## loss associated with dv/dY > 0
            loss_dv_dY = dv_dY * tf.reshape( tf.cast(Y < self.params['ȳ'], tf.float32 ),  [self.params["batch_size"], 1]) \
                * tf.reshape( tf.cast( dv_dY > 0.0, tf.float32 ),  [self.params["batch_size"], 1])  + 10e-4
            
            if "pre_tech" in self.params["model_type"]:

                loss_dv_dlogR = - dv_dlogR   * tf.reshape( tf.cast( dv_dlogR < 0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4

                return tf.sqrt(tf.reduce_mean(tf.square( (rhs - pv)  / self.flow_pv_norm  ))),\
                    tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(FOC_r / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_c / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_g / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_d / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_i_r / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dlogR / self.marginal_utility_of_consumption_norm))),\
                    tf.sqrt(tf.reduce_mean(tf.square(c  ))),\
                    tf.sqrt(tf.reduce_mean(tf.square(inside_log_i_g))),\
                    tf.sqrt(tf.reduce_mean(tf.square(inside_log_i_d)))   
            else:
                return tf.sqrt(tf.reduce_mean(tf.square((rhs - pv) / self.flow_pv_norm ))), \
                    tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) ,\
                    tf.sqrt(tf.reduce_mean(tf.square(loss_c / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_g / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_d / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY/tf.stop_gradient(self.marginal_utility_of_consumption_norm )))),\
                    tf.sqrt(tf.reduce_mean(tf.square(c  ))),\
                    tf.sqrt(tf.reduce_mean(tf.square(inside_log_i_g))),\
                    tf.sqrt(tf.reduce_mean(tf.square(inside_log_i_d)))   
                             
    def grad(self, logK, Z, Y,logR , λ_3, A_g_prime, logξ,   compute_control , training ):
        
        if compute_control:
            with tf.GradientTape(persistent=True) as tape:
                objective  = self.objective_fn(logK, Z, Y, logR, λ_3, A_g_prime, logξ,   compute_control, training)

            if "pre_tech" in self.params["model_type"]:
                trainable_variables = self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_r_nn.trainable_variables
            else:
                trainable_variables = self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables 

            grad = tape.gradient(objective, trainable_variables)

            del tape

            return grad, objective 
        else:
            with tf.GradientTape(persistent=True) as tape:
                objective  = self.objective_fn(logK, Z, Y,logR, λ_3, A_g_prime, logξ,  compute_control, training)
            
            grad = tape.gradient(objective, self.v_nn.trainable_variables)
            del tape

            return grad , objective 

    @tf.function
    def train_step(self):
        logK, Z, Y,  logR,  λ_3 , A_g_prime, logξ = self.sample()
  
        ## First, train value function
        
        grad, loss_v_train = self.grad(logK, Z, Y,  logR,  λ_3 , A_g_prime, logξ,  False,  True) # compute_control=, training=
        self.params["optimizers"][0].apply_gradients(zip(grad, self.v_nn.trainable_variables))

        ## Second, train controls
        grad, loss_c_train  = self.grad(logK, Z, Y,  logR,  λ_3 , A_g_prime, logξ,  True,  True)

        if "pre_tech" in self.params["model_type"]:
            self.params["optimizers"][1].apply_gradients(zip(grad, self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_r_nn.trainable_variables ))
        else:
            self.params["optimizers"][1].apply_gradients(zip(grad, self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables ))

        return loss_v_train, loss_c_train 
    
    
    def train(self):

        start_time = time.time()
        training_history = []

        # Prepare to store best neural networks and initialize networks
        min_loss = float("inf")
        
        if "post_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            n_inputs = 6
    
        elif "post_damage" in self.params["model_type"] and "pre_tech" in self.params["model_type"]:
            n_inputs = 6
        
        elif "pre_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            n_inputs = 5

        elif "pre_damage" in self.params["model_type"] and "pre_tech" in self.params["model_type"]:
            n_inputs = 5

        best_v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        best_v_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.v_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        best_i_g_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_g_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
        best_i_d_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_d_nn.build( (self.params["batch_size"], n_inputs) )

        if "pre_tech" in self.params["model_type"]:
            best_i_r_nn  = FeedForwardSubNet(self.params['i_r_nn_config'])
            best_i_r_nn.build( (self.params["batch_size"], n_inputs) ) 
            self.i_r_nn.build( (self.params["batch_size"], n_inputs) )

        best_v_nn.set_weights(self.v_nn.get_weights())
        best_i_g_nn.set_weights(self.i_g_nn.get_weights())
        best_i_d_nn.set_weights(self.i_d_nn.get_weights())

        if "pre_tech" in self.params["model_type"]:
            best_i_r_nn.set_weights(self.i_r_nn.get_weights())
        
        ## Load pretrained weights
        if self.params['pretrained_path'] is not None:
            if "post_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
                print("Loading pretrained model for post-tech post-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_post_tech_post_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_post_tech_post_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_post_tech_post_damage')

            if "pre_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
                print("Loading pretrained model for pre-tech post-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_pre_tech_post_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_pre_tech_post_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_pre_tech_post_damage')
                self.i_r_nn.load_weights( self.params["pretrained_path"]  + '/i_r_nn_checkpoint_pre_tech_post_damage')

            if "post_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
                print("Loading pretrained model for post-tech pre-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_post_tech_pre_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_post_tech_pre_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_post_tech_pre_damage')

            if "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
                print("Loading pretrained model for pre-tech pre-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_pre_tech_pre_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_pre_tech_pre_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_pre_tech_pre_damage')
                self.i_r_nn.load_weights( self.params["pretrained_path"]  + '/i_r_nn_checkpoint_pre_tech_pre_damage')

        print("Training with " + self.params["model_type"])

        # begin sgd iteration
        # begin sgd iteration
        for step in range(self.params["num_iterations"]):
            if step % self.params["logging_frequency"] == 0:
                ## Sample test data
                logK, Z, Y, logR, λ_3, A_g_prime, logξ = self.sample()
                ## Compute test loss
                test_losses = self.objective_fn(logK, Z, Y, logR, λ_3, A_g_prime, logξ, False, False)  #compute_control, training 

                ## Store best neural networks
                if (test_losses[0] < min_loss):
                    min_loss = test_losses[0]

                    best_v_nn.set_weights(self.v_nn.get_weights())
                    best_i_g_nn.set_weights(self.i_g_nn.get_weights())
                    best_i_d_nn.set_weights(self.i_d_nn.get_weights())

                    if "pre_tech" in self.params["model_type"]:
                        best_i_r_nn.set_weights(self.i_r_nn.get_weights())

                ## Generate checkpoints for tensorboard
                if self.params['tensorboard']:
                    grad_v_nn,loss_v_train = self.grad(logK, Z, Y,logR, λ_3, A_g_prime, logξ,     False,  True)
                    grad_controls,loss_c_train = self.grad(logK, Z, Y,logR, λ_3, A_g_prime, logξ,     True,  True)

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

                        if "pre_tech" in self.params["model_type"]:
                            tf.summary.scalar('loss_value_function', test_losses[0], step=step)
                            tf.summary.scalar('loss_FOC_g', test_losses[1], step=step)
                            tf.summary.scalar('loss_FOC_d', test_losses[2], step=step)
                            tf.summary.scalar('loss_FOC_r', test_losses[3], step=step)
                            tf.summary.scalar('loss_c', test_losses[4], step=step)
                            tf.summary.scalar('loss_inside_log_i_g', test_losses[5], step=step)
                            tf.summary.scalar('loss_inside_log_i_d', test_losses[6], step=step)
                            tf.summary.scalar('loss_i_r', test_losses[7], step=step)
                            tf.summary.scalar('loss_dv_dY', test_losses[8], step=step)
                            tf.summary.scalar('loss_dv_dlogR', test_losses[9], step=step)
                        else:
                            tf.summary.scalar('loss_value_function', test_losses[0], step=step)
                            tf.summary.scalar('loss_FOC_g', test_losses[1], step=step)
                            tf.summary.scalar('loss_FOC_d', test_losses[2], step=step)
                            tf.summary.scalar('loss_c', test_losses[3], step=step)
                            tf.summary.scalar('loss_inside_log_i_g', test_losses[4], step=step)
                            tf.summary.scalar('loss_inside_log_i_d', test_losses[5], step=step)
                            tf.summary.scalar('loss_dv_dY', test_losses[6], step=step)

                            
                            
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

                        if "pre_tech" in self.params["model_type"]:
                            for layer in self.i_r_nn.layers:
                                for W in layer.weights:
                                    tf.summary.histogram(W.name + '_weights', W, step=step)

                            for g in range(len(self.i_r_nn.trainable_variables)):
                                tf.summary.histogram(self.i_r_nn.trainable_variables[g].name + '_grads', grad_controls[len(self.i_d_nn.trainable_variables) + len(self.i_g_nn.trainable_variables) + g], step=step)

                elapsed_time = time.time() - start_time

                ## Appending to training history
                entry = [step] + list(test_losses) + [ elapsed_time]
                training_history.append(entry)



 
                ## Save training history
                if "pre_tech" in self.params["model_type"]:
                    header = 'step,loss_value_function,loss_FOC_g,loss_FOC_d,loss_FOC_r,loss_c,loss_inside_log_i_g,loss_inside_log_i_d,loss_i_r,loss_dv_dY,loss_dv_dlogR,c,inside_log_i_g,inside_log_i_d,elapsed_time'
                else:
                    header = 'step,loss_value_function,loss_FOC_g,loss_FOC_d,loss_c,loss_inside_log_i_g,loss_inside_log_i_d,loss_dv_dY,c,inside_log_i_g,inside_log_i_d,elapsed_time'

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
        if "pre_tech" in self.params["model_type"]:
            self.i_r_nn.set_weights(best_i_r_nn.get_weights())


        ## Export last check point
        self.v_nn.save_weights( self.params["export_folder"] + '/v_nn_checkpoint_' + self.params["model_type"])
        self.i_g_nn.save_weights( self.params["export_folder"] + '/i_g_nn_checkpoint_' + self.params["model_type"])
        self.i_d_nn.save_weights( self.params["export_folder"] + '/i_d_nn_checkpoint_' + self.params["model_type"])
        if "pre_tech" in self.params["model_type"]:
            self.i_r_nn.save_weights( self.params["export_folder"] + '/i_r_nn_checkpoint_' + self.params["model_type"])


        ## Save training history
        if "pre_tech" in self.params["model_type"]:
            header = 'step,loss_value_function,loss_FOC_g,loss_FOC_d,loss_FOC_r,loss_c,loss_inside_log_i_g,loss_inside_log_i_d,loss_i_r,loss_dv_dY,loss_dv_dlogR,c,inside_log_i_g,inside_log_i_d,elapsed_time'
        else:
            header = 'step,loss_value_function,loss_FOC_g,loss_FOC_d,loss_c,loss_inside_log_i_g,loss_inside_log_i_d,loss_dv_dY,c,inside_log_i_g,inside_log_i_d,elapsed_time'

        np.savetxt(self.params["export_folder"] + '/training_history.csv',
                training_history,
                fmt=['%d'] + ['%.5e'] * len(test_losses) + ['%d'],
                delimiter=",",
                header=header,
                comments='')
        
        ## Plot losses
        
        
        if "pre_tech" in self.params["model_type"]:
            
            loss_v_history                   = [history_record[1] for history_record in training_history]
            loss_FOC_g_history               = [history_record[2] for history_record in training_history]
            loss_FOC_d_history               = [history_record[3] for history_record in training_history]
            loss_FOC_r_history               = [history_record[4] for history_record in training_history]
            loss_c_history                   = [history_record[5] for history_record in training_history]
            loss_inside_log_i_g_history      = [history_record[6] for history_record in training_history]
            loss_inside_log_i_d_history      = [history_record[7] for history_record in training_history]
            loss_i_r_history                 = [history_record[8] for history_record in training_history]
            loss_dv_dY_history               = [history_record[9] for history_record in training_history]
            loss_dv_dlogR_history            = [history_record[10] for history_record in training_history]

             
        else:
            loss_v_history                   = [history_record[1] for history_record in training_history]
            loss_FOC_g_history               = [history_record[2] for history_record in training_history]
            loss_FOC_d_history               = [history_record[3] for history_record in training_history]
            loss_c_history                   = [history_record[4] for history_record in training_history]
            loss_inside_log_i_g_history      = [history_record[5] for history_record in training_history]
            loss_inside_log_i_d_history      = [history_record[6] for history_record in training_history]
            loss_dv_dY_history               = [history_record[7] for history_record in training_history]

          
        plt.figure()
        plt.title("Test loss: value function")
        plt.plot(loss_v_history)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_v_history.png")
        plt.close()


        plt.figure()
        plt.title("Test loss: dvdY")
        plt.plot(loss_dv_dY_history)
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_dv_dY.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: c")
        plt.plot(loss_c_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_c.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: inside_log_i_g")
        plt.plot(loss_inside_log_i_g_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_inside_log_i_g.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: inside_log_i_d")
        plt.plot(loss_inside_log_i_d_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_inside_log_i_d.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: FOC_g")
        plt.plot(loss_FOC_g_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_g_history.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: FOC_d")
        plt.plot(loss_FOC_d_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_d_history.png")
        plt.close()



   

        if "pre_tech" in self.params["model_type"]:
            plt.figure()
            plt.title("Test loss: FOC_r")
            plt.plot(loss_FOC_r_history)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig( self.params["export_folder"] + "/loss_FOC_r_history.png")
            plt.close()

            plt.figure()
            plt.title("Test loss: i_r")
            plt.plot(loss_i_r_history)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig( self.params["export_folder"] + "/loss_i_r_history.png")
            plt.close()
 

            plt.figure()
            plt.title("Test loss: dvdlogR")
            plt.plot(loss_dv_dlogR_history)
            plt.xscale('log')
            plt.savefig( self.params["export_folder"] + "/loss_dv_dlogR_history.png")
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

                
    def simulate_path(self, Year, dt, logξ, export_folder):
        '''
        Simulate pre-damage-pre-technology path without brownian terms. 
        '''
        δ    = self.params["δ"]
        α_d  = self.params["α_d"]
        Γ_d  = self.params["Γ_d"]
        θ_d  = self.params["θ_d"]
        σ_d  = self.params["σ_d"]

        α_g  = self.params["α_g"]
        Γ_g  = self.params["Γ_g"]
        θ_g  = self.params["θ_g"]
        σ_g  = self.params["σ_g"]

        Γ    = self.params["Γ"]
        A_d  = self.params["A_d"]
        A_g  = self.params["A_g"]

        λ    = self.params["λ"]
        n    = self.params["n"]
        ζ    = self.params["ζ"]
        ψ_0   = self.params["ψ_0"]
        ψ_1   = self.params["ψ_1"]
        σ_κ  = self.params["σ_κ"]
        ϱ    = self.params["ϱ"]

        # Table 2: Climate Dynamics and Damages Parameters
        θ̄   = self.params["θ̄"]
        η    = self.params["η"]
        ς    = self.params["ς"]
        λ_1   = self.params["λ_1"]
        λ_2   = self.params["λ_2"]
        r_1   = self.params["r_1"]
        r_2   = self.params["r_2"]
        y̲   = self.params["y̲"]
        ȳ    = self.params["ȳ"]


        ## Create folder
        pathlib.Path(export_folder).mkdir(parents=True, exist_ok=True) 
        
        # Starting Values for simulation
        K_0 = self.params["K_0"] 
        Z_0 = self.params["Z_0"]
        Y_0 = self.params["Y_0"] 
        R_0 = self.params["R_0"] 
        ξ  = tf.exp(logξ)

        ###############################################################
        ######### Calculate the initial states to start the simulation
        ################################################################
        
        state    = tf.convert_to_tensor([[ tf.math.log(K_0), Z_0 , Y_0, tf.math.log(R_0), logξ ]] )
        state    = tf.reshape(state, (1,5))

        state_list       = [state]
        i_g_list      = [self.i_g_nn(state)]
        i_d_list      = [self.i_d_nn(state)]
        i_r_list      = [self.i_r_nn(state)]
        v             = self.v_nn(state)
  
        g_damage_t = []
        g_tech_t = []
        for k in range(self.params["λ_3_length"]):
            state_pre_tech_post_damage    = tf.convert_to_tensor( [[ tf.math.log(K_0),  Z_0 , Y_0,  tf.math.log(R_0),
                                                                        self.params["λ_3_list"][k], logξ]] )
            state_pre_tech_post_damage        = tf.reshape(state_pre_tech_post_damage, (1,6))
            v_pre_tech_post_damage  = self.v_pre_tech_post_damage_nn(state_pre_tech_post_damage)
            g_damage       = tf.exp(-1.0/ ξ * (v_pre_tech_post_damage - v))
            g_damage_t.append(g_damage)
        g_damage_list         = [g_damage_t]

        for j in range(self.params["A_g_prime_length"]):
            state_post_tech_pre_damage         = tf.convert_to_tensor( [[ tf.math.log(K_0),  Z_0 , Y_0,
                                                                             self.params["A_g_prime_list"][j], logξ ]] )
            v_post_tech_pre_damage   = self.v_post_tech_pre_damage_nn(state_post_tech_pre_damage)
            g_tech    = tf.exp(-1.0/ ξ  * (v_post_tech_pre_damage - v))
            g_tech_t.append(g_tech)
        g_tech_list            = [g_tech_t]

        #########################################################
        ########### Iteration Starts. 
        #########################################################
        for t in range(int(Year/dt)-1):
            i_g                         = self.i_g_nn(state_list[t])
            i_d                         = self.i_d_nn(state_list[t])
            i_r                         = self.i_r_nn(state_list[t])
            v                           = self.v_nn(state_list[t])
            ## State Variables 
            logK      = state_list[t][0,0]
            Z         = state_list[t][0,1] 
            Y         = state_list[t][0,2]
            logR      = state_list[t][0,3]
            K         = tf.exp(logK)
            ## Store the optimal decisions 
            i_g_list.append(i_g)
            i_d_list.append(i_d)
            i_r_list.append(i_r)
            ###################################################
            ########### Jump Distorions
            ###################################################
            g_damage_t   = []
            for k in range(self.params["λ_3_length"]):
                state_pre_tech_post_damage    = tf.convert_to_tensor( [[ logK, Z, Y, logR, 
                    self.params["λ_3_list"][k],   logξ ]] )
                state_pre_tech_post_damage    = tf.reshape(state_pre_tech_post_damage, (1,6))
                v_pre_tech_post_damage        = self.v_pre_tech_post_damage_nn(state_pre_tech_post_damage)
                g_damage       = tf.exp(-1.0/ ξ  * (v_pre_tech_post_damage - v))
                g_damage_t.append(g_damage)
            g_damage_list.append(g_damage_t)
 
            g_tech_t   = []
            for j in range(self.params["A_g_prime_length"]):
                    # [ tf.math.log(K_0), Z_0 , Y_0, tf.math.log(R_0), logξ ]
                state_post_tech_pre_damage    = tf.convert_to_tensor( [[ logK, Z, Y,
                    self.params["A_g_prime_list"][j], logξ]] )
                state_post_tech_pre_damage     = tf.reshape(state_post_tech_pre_damage, (1,5))
                v_post_tech_pre_damage         = self.v_post_tech_pre_damage_nn(state_post_tech_pre_damage)
                g_tech       = tf.exp(-1.0/ξ * (v_post_tech_pre_damage - v))
                g_tech_t.append(g_tech)
            g_tech_list.append(g_tech_t)


            ##################################################
            ##### State Variables Transition
            ##################################################
            
            inside_log_i_d   =   tf.math.maximum( 1 + θ_d * i_d , 0.0001)
            inside_log_i_g   =   tf.math.maximum( 1 + θ_g * i_g , 0.0001)
            
            φ_d = α_d +  Γ_d  * tf.math.log( inside_log_i_d )
            φ_g = α_g +  Γ_g  * tf.math.log( inside_log_i_g )
            
            v_kk_term      = ( tf.pow(σ_d,2) * tf.pow(1-Z,2) + tf.pow(σ_g,2) * tf.pow(Z,2))/2.0
            v_k_term       = φ_d * (1 - Z) + φ_g * Z  - v_kk_term
        
            v_z_term       = ( φ_g  -  tf.pow(σ_g, 2)*Z - φ_d + tf.pow(σ_d,2) *  (1-Z )   ) * Z * (1 - Z)
            # v_zz_term      = 0.5 * tf.pow(Z, 2) * tf.pow( 1- Z, 2) *  ( tf.pow(σ_g,2) + tf.pow(σ_d, 2))

            v_y_term       = θ̄ * (η *  A_d * (1-Z) * K)
            # v_yy_term      = 0.5 * tf.pow( ς,2) * tf.pow(η * A_d * (1-Z) * K, 2)
            
            φ_logR = -  ζ  +  ψ_0  * tf.exp(- ψ_1  * (  tf.math.log(i_r) + logK -  logR)  )  
            v_logR_term     = φ_logR  - 0.5 * tf.pow(σ_κ, 2)
            # v_logRlogR_term = 0.5 * tf.pow(σ_κ, 2)
 
 
            state_prepre = tf.reshape(tf.convert_to_tensor([
                [logK, Z, Y, logR, logξ]
            ], dtype=tf.float32), (1,5))

            
            state_prepre = tf.reshape(tf.convert_to_tensor([
                [logK, Z, Y, logR, logξ]
            ], dtype=tf.float32), (1,5))
             
            with tf.GradientTape() as tape:
                # Compute 'v' within the GradientTape context
                v = self.v_nn(state_prepre)

       
            dv_dlogK, dv_dZ, dv_dY, dv_dlogR = tape.gradient(v, [logK, Z, Y, logR])

            # Handle 'None' gradients and reshape
            dv_dlogK = tf.reshape(dv_dlogK, [1, 1]) if dv_dlogK is not None else tf.reshape(tf.zeros_like(logK), [1, 1])
            dv_dZ     = tf.reshape(dv_dZ, [1, 1]) if dv_dZ is not None else tf.reshape(tf.zeros_like(Z), [1, 1])
            dv_dY     = tf.reshape(dv_dY, [1, 1]) if dv_dY is not None else tf.reshape(tf.zeros_like(Y), [1, 1])
            dv_dlogR  = tf.reshape(dv_dlogR, [1, 1]) if dv_dlogR is not None else tf.reshape(tf.zeros_like(logR), [1, 1])

 
 
            '''
            dv_dlogK  = tf.reshape(tf.gradients(v, logK, unconnected_gradients='zero')[0], [1, 1])
            dv_dZ     = tf.reshape(tf.gradients(v, Z, unconnected_gradients='zero')[0], [1, 1])
            dv_dY     = tf.reshape(tf.gradients(v, Y, unconnected_gradients='zero')[0], [1, 1])
            dv_dlogR  = tf.reshape(tf.gradients(v, logR, unconnected_gradients='zero')[0], [1, 1])
            '''
            
            distortion_logK  = - 1.0 / ξ * dv_dlogK.numpy()[0,0] * ( ( Z * σ_g)**2  + ((1-Z) * σ_d)**2 )
            distortion_Z     = - 1.0 / ξ * dv_dZ.numpy()[0,0]   *   (  ((1-Z)* Z * σ_g)**2 +  ((1-Z)* Z * σ_d)**2 )
            distortion_y     = - 1.0 / ξ * dv_dY.numpy()[0,0]   * (  η  *  A_d  * (1-Z) * K *  ς)**2 
            distortion_r     = - 1.0 / ξ * σ_κ  * σ_κ  * dv_dlogR.numpy()[0,0]

            h_y = - 1.0 / ξ * dv_dY.numpy()[0,0]   * (  η  *  A_d  * (1-Z) * K *  ς) 
            
            new_logK       = logK + (v_k_term + distortion_logK )* dt 
            new_Z          = Z + (v_z_term    + distortion_Z)* dt  
            new_logR       = logR + (v_logR_term + distortion_r) * dt  
            new_Y          = Y + (v_y_term    + distortion_y) * dt 
            
            new_logK = tf.reshape(new_logK, [1, 1])
            new_Z = tf.reshape(new_Z, [1, 1])
            new_Y = tf.reshape(new_Y, [1, 1])
            new_logR = tf.reshape(new_logR, [1, 1])
            new_logξ = tf.reshape(logξ, [1, 1])

            # Concatenate along axis=1
            state = tf.concat([new_logK, new_Z, new_Y, new_logR, new_logξ], axis=1)
             
            state_list.append(state)


        state_matrix = tf.concat(state_list, axis = 0)
 
        data_dict = {}
        time_vec = np.linspace(0,Year,int(Year/dt))
        data_dict['Year'] = time_vec
        data_dict['log_K_simulation'] = [state.numpy()[0,0] for state in state_list]
        data_dict['Z_simulation'] = [state.numpy()[0,1] for state in state_list]
        data_dict['Y_simulation'] = [state.numpy()[0,2] for state in state_list]
        data_dict['logR_simulation'] = [state.numpy()[0,3] for state in state_list]

        data_dict['i_g_simulation'] = [i_g.numpy()[0,0] for i_g in i_g_list]
        data_dict['i_d_simulation'] = [i_d.numpy()[0,0] for i_d in i_d_list]
        data_dict['i_r_simulation'] = [np.exp(-i_r.numpy()[0,0]) for i_r in i_r_list]

        data_dict['Year'] = np.array(data_dict['Year'])
        data_dict['log_K_simulation'] = np.array(data_dict['log_K_simulation'])
        data_dict['Z_simulation'] = np.array(data_dict['Z_simulation'])
        data_dict['log_K_simulation'] = np.array(data_dict['log_K_simulation'])
        data_dict['Y_simulation'] = np.array(data_dict['Y_simulation'])
        data_dict['logR_simulation'] = np.array(data_dict['logR_simulation'])
        data_dict['i_g_simulation'] = np.array(data_dict['i_g_simulation'])
        data_dict['i_d_simulation'] = np.array(data_dict['i_d_simulation'])
        data_dict['i_r_simulation'] = np.array(data_dict['i_r_simulation'])

        data_dict["I_g"] =  np.exp(data_dict['log_K_simulation']) * data_dict['Z_simulation'] * data_dict['i_g_simulation']
        data_dict["I_d"] =  np.exp(data_dict['log_K_simulation']) * (1.0 - data_dict['Z_simulation'])  * data_dict['i_d_simulation']
        data_dict["I_r"] =  np.exp(data_dict['log_K_simulation']) * data_dict['i_r_simulation']
       
        data_dict["I_r/Y"] =  np.exp(data_dict['log_K_simulation']) * data_dict['i_r_simulation'] /  \
        (np.exp(data_dict['log_K_simulation']) * data_dict['Z_simulation'] * A_d + \
        np.exp(data_dict['log_K_simulation']) * (1.0 - data_dict['Z_simulation']) * A_g)
        data_dict["I_r/Y"] = np.array(data_dict["I_r/Y"])
        
        data_dict["E"] = η * np.exp(data_dict['log_K_simulation']) * (1.0 - data_dict['Z_simulation'] ) * A_d
        data_dict["E"] = np.array(data_dict["E"])
        data_dict['K_g'] = np.exp(data_dict['log_K_simulation']) * data_dict['Z_simulation']
        data_dict['K_d'] = np.exp(data_dict['log_K_simulation']) * (1.0 - data_dict['Z_simulation'])
 
        data_dict['damage_jump_intensity'] =  r_1 * ( np.exp( r_2 / 2 * np.power( data_dict['Y_simulation'] - y̲ ,2) ) - 1  ) * (data_dict['Y_simulation']  > y̲ )


        ########################################
        ####### Distorted Probability
        ########################################

        data_dict['g_damage_sum'] = 0

        for i in range(self.params["λ_3_length"]):

            data_dict['g_damage_' + str(i) + '_simulation'] = [g_damage_t[i].numpy()[0,0] for g_damage_t in g_damage_list]
            data_dict['g_damage_' + str(i) + '_simulation'] =  np.array(data_dict['g_damage_' + str(i) + '_simulation'])
            data_dict['g_damage_sum'] += data_dict['g_damage_' + str(i) + '_simulation']
 
        for i in range(self.params["λ_3_length"]):

            data_dict['g_damage_' + str(i) + '_simulation_norm'] = data_dict['g_damage_' + str(i) + '_simulation']/  data_dict['g_damage_sum']

 
        data_dict['distorted_dmg_jump_intensity'] = data_dict['g_damage_sum']*data_dict['damage_jump_intensity'] / self.params["λ_3_length"]
        data_dict['distorted_dmg_jump_prob'] = 1-np.exp(-np.cumsum(data_dict['distorted_dmg_jump_intensity'] * (dt)))

 

        R        =  np.exp(data_dict['logR_simulation'])
        data_dict['g_tech_sum'] = 0
        for i in range(self.params["A_g_prime_length"]):
            data_dict['g_tech_' + str(i) + '_simulation'] = [g_tech_t[i].numpy()[0,0] for g_tech_t in g_tech_list]
            data_dict['g_tech_' + str(i) + '_simulation'] =  np.array(data_dict['g_tech_' + str(i) + '_simulation'])
            data_dict['g_tech_sum'] += data_dict['g_tech_' + str(i) + '_simulation']

        for i in range(self.params["A_g_prime_length"]):
            data_dict['g_tech_' + str(i) + '_simulation_norm'] = data_dict['g_tech_' + str(i) + '_simulation']/  data_dict['g_tech_sum']
 
        data_dict['distorted_tech_jump_intensity'] = data_dict['g_tech_sum']* R * (dt) / ϱ / self.params["A_g_prime_length"]
        data_dict['distorted_tech_jump_prob'] = 1-np.exp(-np.cumsum(data_dict['distorted_tech_jump_intensity']))

        data_dict['distorted_dmg_jump_intensity']= (1-data_dict['distorted_dmg_jump_prob'])*(1-data_dict['distorted_tech_jump_prob'])*data_dict['distorted_dmg_jump_intensity']
        data_dict['distorted_tech_jump_prob']= (1-data_dict['distorted_dmg_jump_prob'])*(1-data_dict['distorted_tech_jump_prob'])*data_dict['distorted_tech_jump_prob']

        data_dict['h_y_simulation'] = h_y
        data_dict['h_y_simulation'] = np.array(data_dict['h_y_simulation'])

        with open(export_folder + '/data_dict.txt', 'a') as the_file:
            for key in data_dict.keys():
                if "nn_config" not in key:
                    the_file.write( str(key) + ": " + str(data_dict[key]) + '\n')


        ################################################################
        #########.  Plot Trajectories
        ################################################################

        ## logK
        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,0] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$log K$')
        plt.savefig(export_folder + "/logK_simulation.png")
        np.savetxt(export_folder + "/log_K_simulation.txt", np.array([state.numpy()[0,0] for state in state_list]))
        plt.close()

        ## Z
        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,1] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$Z$')
        plt.savefig(export_folder +  "/Z_simulation.png")
        np.savetxt(export_folder + "/Z_simulation.txt", np.array([state.numpy()[0,1] for state in state_list]))
        plt.close()

        ## Y 
        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,2] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$Y$')
        plt.savefig(export_folder + "/Y_simulation.png")
        np.savetxt(export_folder + "/Y_simulation.txt", np.array([state.numpy()[0,2] for state in state_list]))
        plt.close()

        ## logR
        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,3] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$log R$')
        plt.savefig(export_folder + "/logR_simulation.png")
        np.savetxt(export_folder +  "/logR_simulation.txt", np.array([state.numpy()[0,3] for state in state_list]))
        plt.close()

        ## i_g
        plt.figure()
        plt.plot(time_vec,[i_g.numpy()[0,0] for i_g in i_g_list])
        plt.xlabel("Years")
        plt.title(r'$i_g$')
        plt.savefig(export_folder + "/i_g_simulation.png")
        np.savetxt(export_folder + "/i_g_simulation.txt", np.array([i_g.numpy()[0,0] for i_g in i_g_list]))
        plt.close()

        ## i_d
        plt.figure()
        plt.plot(time_vec,[i_d.numpy()[0,0] for i_d in i_d_list])
        plt.xlabel("Years")
        plt.title(r'$i_d$')
        plt.savefig(export_folder + "/i_d_simulation.png")
        np.savetxt(export_folder + "/i_d_simulation.txt", np.array([i_d.numpy()[0,0] for i_d in i_d_list]))
        plt.close()

        ## i_r
        plt.figure()
        plt.plot(time_vec,[i_r.numpy()[0,0] for i_r in i_r_list])
        plt.xlabel("Years")
        plt.title(r'$i_r$')
        plt.savefig(export_folder + "/i_r_simulation.png")
        np.savetxt(export_folder + "/i_r_simulation.txt", np.array([ i_r.numpy()[0,0] for i_r in i_r_list]))
        plt.close()

  
        ################################################################
        ##########  Make plots Part 2
        ################################################################ 

        ## Emissions
        plt.figure()
        plt.plot(time_vec,data_dict["E"], label = r"$\xi = {:.3f}$".format( ξ ), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title("Emissions")
        # plt.ylim(7.5,11.5)
        plt.legend(loc='upper left')
        plt.savefig(export_folder + "/Ems_Comp_IMSI_2023.png")
        np.savetxt(export_folder +  "/Emissions.txt", data_dict["E"])
        plt.close()


        ## logR
        plt.figure()
        # plt.plot(data_dict_1["logR"], label = r"$\ξ = 0.15$")
        plt.plot(time_vec,data_dict["I_r"], label = r"$\xi = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title(r"Green Investment ($I_r$)")
        plt.legend(loc='upper left')
        plt.savefig(export_folder + '/Ir_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/Ir.txt", data_dict["logR"])
        plt.close()
        
        ## I_d
        plt.figure()
        plt.plot(time_vec,data_dict["I_d"], label = r"$\xi = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title(r"Dirty Investment ($I_d$)")
        plt.legend(loc='lower left')
        plt.savefig(export_folder + '/Id_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/Id.txt", data_dict["I_d"])
        plt.close()


        ## I_r/Y
        plt.figure()
        plt.plot(time_vec,data_dict["I_r/Y"]*100, label = r"$\xi = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title(r"R&D Investment as % of Output ($I_{\kappa}/Y$)")
        plt.legend(loc='upper right')
        plt.savefig(export_folder + '/RD_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/RD.txt", data_dict["I_I/Y"])
        plt.close()

        ## distorted_dmg_jump_prob
        plt.figure()
        plt.plot(time_vec,data_dict["distorted_dmg_jump_prob"], label = r"$\xi = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title("Distorted Probability of a Damage Jump")
        plt.ylim(0,1)
        plt.legend(loc='lower right')
        plt.savefig(export_folder + '/DmgJumpProb_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/DmgJumpProb.txt", data_dict["distorted_dmg_jump_prob"])        
        plt.close()


        ## distorted_tech_jump_prob
        plt.figure()
        plt.plot(time_vec,data_dict["distorted_tech_jump_prob"], label = r"$\xi = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title("Distorted Probability of a Technology Jump")
        plt.ylim(0,1)
        plt.legend(loc='lower right')
        plt.savefig(export_folder + '/TechJumpProb_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/TechJumpProb.txt", data_dict["distorted_tech_jump_prob"])         
        plt.close()

        ## distorted_dmg_jump_intensity
        plt.figure()
        plt.plot(
            time_vec,
            data_dict["distorted_dmg_jump_intensity"],
            label=r"$\xi = {:.3f}$".format(np.exp(logξ)),
            color='tab:red',  # Using a different color for distinction
            linewidth=2
        )
        plt.xlabel('Years')
        plt.title("Distorted Intensity of a Damage Jump")
        plt.ylim(0, max(data_dict["distorted_dmg_jump_intensity"]) * 1.1)  # Adjust y-limit based on data
        plt.legend(loc='upper right')
        plt.savefig(f"{export_folder}/DmgJumpIntensity_Comp_IMSI_2023.png")
        np.savetxt(f"{export_folder}/DmgJumpIntensity.txt", data_dict["distorted_dmg_jump_intensity"])
        plt.close()



        ## distorted_tech_jump_intensity
        plt.figure()
        plt.plot(
            time_vec,
            data_dict["distorted_tech_jump_intensity"],
            label=r"$\xi = {:.3f}$".format(np.exp(logξ)),
            color='tab:green',  # Using a different color for distinction
            linewidth=2
        )
        plt.xlabel('Years')
        plt.title("Distorted Intensity of a Technology Jump")
        plt.ylim(0, max(data_dict["distorted_tech_jump_intensity"]) * 1.1)  # Adjust y-limit based on data
        plt.legend(loc='upper right')
        plt.savefig(f"{export_folder}/TechJumpIntensity_Comp_IMSI_2023.png")
        np.savetxt(f"{export_folder}/TechJumpIntensity.txt", data_dict["distorted_tech_jump_intensity"])
        plt.close()

 
        ## Plot bar chart
        baseline = np.ones(self.params["λ_3_length"]) / self.params["λ_3_length"]
        distorted = np.ones(self.params["λ_3_length"]) / self.params["λ_3_length"]
        bin_edges = np.linspace(0, 1/3, 6)
        x1       = np.linspace(0,1/3,self.params["λ_3_length"])
        for i in range(self.params["λ_3_length"]):
            distorted[i] = data_dict['g_damage_'+str(i)+'_simulation_norm'][-1] 

        print("Climate Models: {}"  .format(distorted))

        plt.hist(x1, weights=baseline, bins=bin_edges,label='Baseline', color = 'C3', α=0.5, ec="darkgrey")
        plt.hist(x1, weights=distorted, bins=bin_edges, label='Distorted', color = 'C0', α=0.5, ec="darkgrey")
        plt.title("Distorted Probability of Damage Models")
        plt.xlabel(r"$\lambda_3$")
        plt.legend()
        plt.xlim([0,1/3])
        plt.ylim([0, 0.6])
        plt.savefig(export_folder + '/Dmg_Dist_IMSI_2023.png')
        plt.close()



        ## Plot bar chart
        baseline = np.ones(self.params["A_g_prime_length"]) / self.params["A_g_prime_length"]
        x1       =  np.array(self.params["A_g_prime_list"])
        distorted = np.ones(self.params["A_g_prime_length"]) / self.params["A_g_prime_length"]
        for i in range(self.params["A_g_prime_length"]):
            distorted[i] = data_dict['g_tech_'+str(i)+'_simulation_norm'][-1]

        # bin_edges = np.linspace(0.11, 0.14, 4)
        print("Tech Models: {}"  .format(distorted))
        plt.hist(x1, weights=baseline, label='Baseline', color = 'C3', α=0.5, ec="darkgrey")#,bins=bin_edges)
        plt.hist(x1, weights=distorted,  label='Distorted', color = 'C0', α=0.5, ec="darkgrey")#,bins=bin_edges)
        plt.title("Distorted Probability of Technology Models")
        plt.xlabel(r"$A'_g$")
        plt.legend()
        plt.xlim([self.params["A_g_prime_min"],self.params["A_g_prime_max"]])
        plt.ylim([0, 1.0])
        plt.savefig(export_folder + '/Tech_Dist_IMSI_2023.png')
        plt.close()

        theta_ell = (pd.read_csv("./TCRE144.csv", header=None).to_numpy()[:,0]/1000).astype(np.float32)
        ## h_y is the last one in iteration. 
        pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
        print("Distortion: {}"  .format(ς * h_y))
        plt.hist(1000*theta_ell, weights=pi_c_o, bins = np.linspace(0.8,3.,16), label = "Baseline", color = 'C3', α=0.5, density=True, ec="darkgrey")
        plt.hist(1000*(theta_ell + ς * h_y), weights=pi_c_o, label = "Distorted", bins = np.linspace(0.8,3.,16), color = 'C0', α=0.5,density=True,  ec="darkgrey")
        plt.title("Distorted Probability of Climate Models")
        plt.xlabel("Climate Sensitivity")
        plt.ylim(0, 1.5)
        plt.xlim(0.8, 3)
        plt.legend()
        plt.savefig(export_folder + '/Climate_Dist_IMSI_2023.png')
        plt.close()

'''
    def simulate_path_post_tech_post_jump(self, Year, dt, λ_3, A_g_prime, logξ, export_folder):

        ## Create folder
        pathlib.Path(export_folder+"/"+self.params["model_type"]).mkdir(parents=True, exist_ok=True) 
        
        ## Initial state 
        init_logK     = tf.math.log(739.0)
        init_Z        = 0.5  
        init_Y        = 1.1

        state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, λ_3, logξ, A_g_prime]] )
        state         = tf.reshape(state, (1,6))

        state_list       = [state]

        i_g_list      = [self.i_g_nn(state)]
        i_d_list      = [self.i_d_nn(state)]

        time_vec = np.linspace(0,Year,int(Year/dt))

        for t in range(int(Year/dt)-1):


            ## Find investments
            i_g                         = self.i_g_nn(state_list[t])
            i_d                         = self.i_d_nn(state_list[t])

            logK      = state_list[t][0,0]; Z = state_list[t][0,1]; 
            Y         = state_list[t][0,2]; 
            K         = tf.exp(logK)

            i_g_list.append(i_g)
            i_d_list.append(i_d)

            ## Transition
            v_kk_term      = ( tf.pow(σ_d,2) * tf.pow(1-Z,2) + tf.pow(σ_g,2) * tf.pow(Z,2))/2.0

            inside_log_i_d   =     tf.math. maximum( 1 + self.params["θ_d"] * i_d , 0.0001) 
            inside_log_i_g   =     tf.math. maximum( 1 + self.params["θ_g"] * i_g , 0.0001)

            v_k_term       = ( self.params["α_d"] + self.params["Γ"] * tf.math.log( inside_log_i_d ) ) * (1 - Z) + ( self.params["α_g"] +  self.params["Γ"] * tf.math.log( inside_log_i_g) ) * Z  - v_kk_term
            v_z_term       = ( self.params["α_g"] + self.params["Γ"] * tf.math.log( inside_log_i_g )  - ( self.params["α_d"] + self.params["Γ"] * tf.math.log( inside_log_i_d) ) + tf.pow(σ_d,2) *  (1-Z ) - 
                            tf.pow(σ_g, 2) *  Z ) * \
            Z * (1 - Z)
            
    
            new_logK       = logK + v_k_term * dt 
            new_Z          = Z + v_z_term * dt  
            new_Y          = Y + self.params["θ̄"] * ( η * self.params["A_d"] * (1-Z) * tf.exp( logK )) * dt 

            state          = tf.concat([new_logK, new_Z, tf.reshape(new_Y, [1,1]), tf.reshape(λ_3, [1,1]),  tf.reshape(logξ, [1,1]),  tf.reshape(A_g_prime, [1,1]) ], axis=1)
            state_list.append(state)

        ################################################################
        ################################################################
        ################################################################
        ## Make plots
        ################################################################
        ################################################################
        ################################################################


        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,0] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$\log K$')
        plt.savefig(export_folder +"/"+self.params["model_type"]+ "/logK_simulation.png")
        np.savetxt(export_folder +"/"+self.params["model_type"]+ "/log_K_simulation.txt", np.array([state.numpy()[0,0] for state in state_list]))
        plt.close()


        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,1] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$Z$')
        plt.savefig(export_folder +"/"+self.params["model_type"]+  "/Z_simulation.png")
        np.savetxt(export_folder +"/"+self.params["model_type"]+ "/Z_simulation.txt", np.array([state.numpy()[0,1] for state in state_list]))
        plt.close()

        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,2] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$T$')
        plt.savefig(export_folder +"/"+self.params["model_type"]+ "/T_simulation.png")
        np.savetxt(export_folder +"/"+self.params["model_type"] + "/T_simulation.txt", np.array([state.numpy()[0,2] for state in state_list]))
        plt.close()


        plt.figure()
        plt.plot(time_vec,[i_g.numpy()[0,0] for i_g in i_g_list])
        plt.xlabel("Years")
        plt.title(r'$i_g$')
        plt.savefig(export_folder +"/"+self.params["model_type"]+ "/i_g_simulation.png")
        np.savetxt(export_folder +"/"+self.params["model_type"]+ "/i_g_simulation.txt", np.array([i_g.numpy()[0,0] for i_g in i_g_list]))
        plt.close()

        theta_ell = (pd.read_csv("./model144.csv", header=None).to_numpy()[:,0]/1000).astype(np.float32)

        plt.figure()
        plt.plot(time_vec,[i_d.numpy()[0,0] for i_d in i_d_list])
        plt.xlabel("Years")
        plt.title(r'$i_d$')
        plt.savefig(export_folder +"/"+self.params["model_type"]+ "/i_d_simulation.png")
        np.savetxt(export_folder +"/"+self.params["model_type"]+ "/i_d_simulation.txt", np.array([i_d.numpy()[0,0] for i_d in i_d_list]))
        plt.close()

        Y      =  np.array([state.numpy()[0,2] for state in state_list])
        Z      =  np.array([state.numpy()[0,1] for state in state_list])
  
  
  
    def Fk_simulate_path(self, Year, dt, logξ, logξ_baseline, export_folder):

        ## Create folder
        pathlib.Path(export_folder).mkdir(parents=True, exist_ok=True) 
        
        ## Initial state 
        init_logK     = tf.math.log(739.0)
        init_Z        = 0.5  
        init_logR      = tf.math.log(11.2)
        init_Y        = 1.1

        # state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, logξ,  init_logR]] )
        # state         = tf.reshape(state, (1,5))



        if self.params["channel_type"] == "full":
            state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,  init_logR, logξ,  logξ,  logξ]] )
        elif self.params["channel_type"] == "capital":

            state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,  init_logR, logξ,  logξ_baseline,  logξ_baseline]] )
        elif self.params["channel_type"] == "climate":

            state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,  init_logR, logξ_baseline,  logξ,  logξ_baseline]] )
        # elif self.params["channel_type"] == "damage":

        #     state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, logξ_baseline,  logξ_baseline,  logξ,  logξ_baseline,  init_logR]] )
        elif self.params["channel_type"] == "technology":

            state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,  init_logR, logξ_baseline,  logξ_baseline,  logξ]] )
        elif self.params["channel_type"] == "baseline":

            state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,  init_logR, logξ_baseline,  logξ_baseline,  logξ_baseline]] )

        state         = tf.reshape(state, (1,7))


        # state_pre     = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, logξ, A_g_prime]] )
        # state_pre     = tf.reshape(state_pre, (1,5)) 


        # if self.params["channel_type"] == "full":
        #     state_pre_damage_post_tech         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, logξ,  logξ]] )
        # elif self.params["channel_type"] == "capital":

        #     state_pre_damage_post_tech         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, logξ,  logξ_baseline]] )
        # elif self.params["channel_type"] == "climate":

        #     state_pre_damage_post_tech         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, logξ_baseline,  logξ]] )

        # else:

        #     state_pre_damage_post_tech         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, logξ_baseline,  logξ_baseline]] )


        state_list       = [state]
        # state_pre_list   = [state_pre]

        i_g_list      = [self.i_g_nn(state)]
        i_d_list      = [self.i_d_nn(state)]
        i_r_list      = [self.i_r_nn(state)]

        # v_post_tech_pre_damage        = self.v_post_tech_pre_damage_nn(state_pre)
        v                             = self.v_nn(state)

        f_ms = []
        v_m_vals = [] 

        g_js = []
        g_j_logs = []
        v_j_vals = []
        v_diff_j_vals = []

        for k in range(self.params["λ_3_length"]):


            if self.params["channel_type"] == "full":
                                                                                                            #cap     # tempe  # tech    
       
                state_pre_tech_post_damage    = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, init_logR,
                                                                         self.params["λ_3_list"][k], logξ, logξ, logξ]] )

                state_pre_tech_post_damage        = tf.reshape(state_pre_tech_post_damage, (1,8))
                v_m                           = self.v_pre_tech_post_damage_nn(state_pre_tech_post_damage)
                v_m_vals.append( v_m )
                f_m       = tf.exp(-1.0/ np.exp(logξ) * (v_m - v))
                f_ms.append(f_m)


        f_ms_list         = [f_ms]


        for j in range(self.params["A_g_prime_length"]):
            


            if self.params["channel_type"] == "full":
                state_post_tech_pre_damage         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,
                                                                             self.params["A_g_prime_list"][j], logξ,  logξ]] )
            elif self.params["channel_type"] == "capital":

                state_post_tech_pre_damage         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,
                                                                             self.params["A_g_prime_list"][j], logξ,  logξ_baseline]] )
            elif self.params["channel_type"] == "climate":

                state_post_tech_pre_damage         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,
                                                                             self.params["A_g_prime_list"][j], logξ_baseline,  logξ]] )

            else:

                state_post_tech_pre_damage         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y,
                                                                             self.params["A_g_prime_list"][j], logξ_baseline,  logξ_baseline]] )


            if self.params["channel_type"] == "full" or self.params["channel_type"] == "technology":


                v_j                    = self.v_post_tech_pre_damage_nn(state_post_tech_pre_damage)
                v_j_vals.append( v_j )
                v_diff_temp                        = v_j - v
                g_j                           = tf.exp(-1.0/  np.exp(logξ) * (v_j - v))
                
                g_js.append(g_j)
                
            else:

            
                v_j                    = self.v_post_tech_pre_damage_nn(state_post_tech_pre_damage)
                v_j_vals.append( v_j )
                v_diff_temp                        = v_j - v
                g_j                           = tf.exp(-1.0/  np.exp(logξ_baseline) * (v_j - v))
                
                g_js.append(g_j)
                

        g_js_list            = [g_js]

        time_vec = np.linspace(0,Year,int(Year/dt))

        for t in range(int(Year/dt)-1):


            ## Find investments
            i_g                         = self.i_g_nn(state_list[t])
            i_d                         = self.i_d_nn(state_list[t])
            i_r                         = self.i_r_nn(state_list[t])
            v                           = self.v_nn(state_list[t])

            f_ms              = []

            for k in range(self.params["λ_3_length"]):

                if self.params["channel_type"] == "full":

                    state_pre_tech_post_damage    = tf.convert_to_tensor( [[ state_list[t][0,0], state_list[t][0,1], state_list[t][0,2], state_list[t][0,3], 
                        self.params["λ_3_list"][k],  state_list[t][0,4],  state_list[t][0,5],  state_list[t][0,6]]] )
                    state_pre_tech_post_damage        = tf.reshape(state_pre_tech_post_damage, (1,8))
                    
                    v_m                           = self.v_pre_tech_post_damage_nn(state_pre_tech_post_damage)
                    f_m       = tf.exp(-1.0/  np.exp(logξ) * (v_m - v))
                    f_ms.append(f_m)
                    

 

            f_ms_list.append(f_ms)

            g_js              = []

            for j in range(self.params["A_g_prime_length"]):

                if self.params["channel_type"] == "full" or self.params["channel_type"] == "technology":

                    state_post_tech_pre_damage    = tf.convert_to_tensor( [[ state_list[t][0,0], state_list[t][0,1], state_list[t][0,2],
                        self.params["A_g_prime_list"][j], state_list[t][0,4],  state_list[t][0,5]]] )
                    state_post_tech_pre_damage        = tf.reshape(state_post_tech_pre_damage, (1,6))
                    
                    v_j                           = self.v_post_tech_pre_damage_nn(state_post_tech_pre_damage)
                    g_j       = tf.exp(-1.0/  np.exp(logξ) * (v_j - v))
                    g_js.append(g_j)

                else:

                    state_post_tech_pre_damage    = tf.convert_to_tensor( [[ state_list[t][0,0], state_list[t][0,1], state_list[t][0,2],
                        self.params["A_g_prime_list"][j], state_list[t][0,4],  state_list[t][0,5]]] )
                    state_post_tech_pre_damage        = tf.reshape(state_post_tech_pre_damage, (1,6))
                    
                    v_j                           = self.v_post_tech_pre_damage_nn(state_post_tech_pre_damage)
                    g_j       = tf.exp(-1.0/  np.exp(logξ_baseline) * (v_j - v))
                    g_js.append(g_j)


            g_js_list.append(g_js)



            logK      = state_list[t][0,0]; Z = state_list[t][0,1]; 
            Y         = state_list[t][0,2]; logR = state_list[t][0,3]
            K         = tf.exp(logK)

            # g         = tf.exp(-1.0/  np.exp(logξ) * (v_post_tech_pre_damage  - v))
            # g_list.append(g)


            i_g_list.append(i_g)
            i_d_list.append(i_d)
            i_r_list.append(i_r)

            ## Transition
            v_kk_term      = ( tf.pow(σ_d,2) * tf.pow(1-Z,2) + tf.pow(σ_g,2) * tf.pow(Z,2))/2.0

            inside_log_i_d   =     tf.math. maximum( 1 + self.params["θ_d"] * i_d , 0.0001) 
            inside_log_i_g   =     tf.math. maximum( 1 + self.params["θ_g"] * i_g , 0.0001)

            v_k_term       = ( self.params["α_d"] + self.params["Γ"] * tf.math.log( inside_log_i_d ) ) * (1 - Z) + ( self.params["α_g"] +  self.params["Γ"] * tf.math.log( inside_log_i_g) ) * Z  - v_kk_term
            v_z_term       = ( self.params["α_g"] + self.params["Γ"] * tf.math.log( inside_log_i_g )  - ( self.params["α_d"] + self.params["Γ"] * tf.math.log( inside_log_i_d) ) + tf.pow(σ_d,2) *  (1-Z ) - 
                            tf.pow(σ_g, 2) *  Z ) * \
            Z * (1 - Z)
            
            v_logR_term     = - self.params["ζ"] + self.params["ψ_0"] * tf.exp(-i_r * self.params["ψ_1"]) * tf.exp( self.params["ψ_1"] * (logK -  logR) ) - 0.5 * tf.pow(self.params["σ_κ "], 2)
    
            new_logK       = logK + v_k_term * dt 
            new_Z          = Z + v_z_term * dt  
            new_logR    = logR + v_logR_term * dt  
            new_Y          = Y + self.params["θ̄"] * ( η * self.params["A_d"] * (1-Z) * tf.exp( logK )) * dt 

 

            if self.params["channel_type"] == "full":
                # state         = tf.convert_to_tensor( [[ init_logK,  init_Z, init_Y, logξ,  logξ,  logξ,  init_logR]] )
                state          = tf.concat([new_logK, new_Z, tf.reshape(new_Y, [1,1]), new_logR, tf.reshape(logξ, [1,1]), tf.reshape(logξ, [1,1]), tf.reshape(logξ, [1,1]) ], axis=1)

            elif self.params["channel_type"] == "capital":

                state          = tf.concat([new_logK, new_Z, tf.reshape(new_Y, [1,1]), new_logR, tf.reshape(logξ, [1,1]), tf.reshape(logξ_baseline, [1,1]), tf.reshape(logξ_baseline, [1,1]) ], axis=1)
            elif self.params["channel_type"] == "climate":

                state          = tf.concat([new_logK, new_Z, tf.reshape(new_Y, [1,1]), new_logR, tf.reshape(logξ_baseline, [1,1]), tf.reshape(logξ, [1,1]), tf.reshape(logξ_baseline, [1,1]) ], axis=1)
            # elif self.params["channel_type"] == "damage":

            #     state          = tf.concat([new_logK, new_Z, tf.reshape(new_Y, [1,1]), tf.reshape(logξ_baseline, [1,1]), tf.reshape(logξ_baseline, [1,1]), tf.reshape(logξ, [1,1]), new_logR ], axis=1)
            elif self.params["channel_type"] == "technology":

                state          = tf.concat([new_logK, new_Z, tf.reshape(new_Y, [1,1]), new_logR, tf.reshape(logξ_baseline, [1,1]), tf.reshape(logξ_baseline, [1,1]), tf.reshape(logξ, [1,1]) ], axis=1)
            elif self.params["channel_type"] == "baseline":

                state          = tf.concat([new_logK, new_Z, tf.reshape(new_Y, [1,1]), new_logR, tf.reshape(logξ_baseline, [1,1]), tf.reshape(logξ_baseline, [1,1]), tf.reshape(logξ_baseline, [1,1]) ], axis=1)

            state_list.append(state)


            # state_pre          = tf.concat([new_logK, new_Z, tf.reshape(new_Y, [1,1]),  tf.reshape(logξ, [1,1])], axis=1)
            # state_pre_list.append(state_pre)

        state_matrix = tf.concat(state_list, axis = 0)

        with tf.GradientTape() as tape:
            tape.watch(state_matrix)
            output_array = self.v_nn(state_matrix)
        derivative = tape.gradient(output_array, state_matrix) 

        dv_dlogK      = derivative[:,0] 
        dv_dlogZ      = derivative[:,1] 
        dv_dY      = derivative[:,2] 
        dv_dlogIg      = derivative[:,3] 


        Y      =  np.array([state.numpy()[0,2] for state in state_list])
        Z      =  np.array([state.numpy()[0,1] for state in state_list])
        K      =  np.exp(np.array([state.numpy()[0,0] for state in state_list]))

        # h      = -1.0 / tf.exp(logξ) * ((dv_dY.numpy() - \
        # (self.params["λ_1"] + self.params["λ_2"] * Y)) * self.params["ς"] * \
        #         η * self.params["A_d"] * (1 - Z) * K)


        if self.params["channel_type"]=="full" or self.params["channel_type"]=="climate": 
            h_y      = -1.0 / tf.exp(logξ) * ((dv_dY.numpy() - \
            (self.params["λ_1"] + self.params["λ_2"] * Y)) * self.params["ς"] * \
                    η * self.params["A_d"] * (1 - Z) * K)
        else: 

            h_y      = -1.0 / tf.exp(logξ_baseline) * ((dv_dY.numpy() - \
            (self.params["λ_1"] + self.params["λ_2"] * Y)) * self.params["ς"] * \
                    η * self.params["A_d"] * (1 - Z) * K)


        if self.params["channel_type"]=="full" or self.params["channel_type"]=="capital": 

            # h_k = - 1.0 / tf.exp(logξ) * ((dv_dlogK.numpy() - Z * dv_dlogZ.numpy() ) * (1-Z) * σ_d + (dv_dlogK.numpy() + (1-Z) * dv_dlogZ.numpy() ) * Z * σ_g )
            
            h_d = - 1.0 / tf.exp(logξ) * ((dv_dlogK.numpy() - Z * dv_dlogZ.numpy() ) * (1-Z) * σ_d)
            h_g = - 1.0 / tf.exp(logξ) * ((dv_dlogK.numpy() + (1-Z) * dv_dlogZ.numpy() ) * Z * σ_g)

 

        if self.params["channel_type"]=="full" or self.params["channel_type"]=="technology": 

            h_Z = - 1.0 / tf.exp(logξ) * self.params["σ_κ "] * dv_dlogIg.numpy()
 


        ϱ = self.params["ϱ"]
        # λ_3_length = 5
        A_d = 0.12
        A_g = 0.10
        beta = 1.86 / 1000
        η = 0.17
        r_1         = 1.5
        r_2         = 2.5
        y_lower_bar = 1.5

        data_dict = {}
        data_dict['Year'] = time_vec
        data_dict['log_K_simulation'] = [state.numpy()[0,0] for state in state_list]
        data_dict['Z_simulation'] = [state.numpy()[0,1] for state in state_list]
        data_dict['T_simulation'] = [state.numpy()[0,2] for state in state_list]
        data_dict['logR_simulation'] = [state.numpy()[0,3] for state in state_list]

        data_dict['i_g_simulation'] = [i_g.numpy()[0,0] for i_g in i_g_list]
        data_dict['i_d_simulation'] = [i_d.numpy()[0,0] for i_d in i_d_list]
        data_dict['i_r_simulation'] = [np.exp(-i_r.numpy()[0,0]) for i_r in i_r_list]

        data_dict['Year'] = np.array(data_dict['Year'])
        data_dict['log_K_simulation'] = np.array(data_dict['log_K_simulation'])
        data_dict['Z_simulation'] = np.array(data_dict['Z_simulation'])
        data_dict['log_K_simulation'] = np.array(data_dict['log_K_simulation'])
        data_dict['T_simulation'] = np.array(data_dict['T_simulation'])
        data_dict['logR_simulation'] = np.array(data_dict['logR_simulation'])
        data_dict['i_g_simulation'] = np.array(data_dict['i_g_simulation'])
        data_dict['i_d_simulation'] = np.array(data_dict['i_d_simulation'])
        data_dict['i_r_simulation'] = np.array(data_dict['i_r_simulation'])

        data_dict["logR"] =  np.exp(data_dict['log_K_simulation']) * data_dict['Z_simulation'] * data_dict['i_g_simulation']
        data_dict["I_d"] =  np.exp(data_dict['log_K_simulation']) * (1.0 - data_dict['Z_simulation'])  * data_dict['i_d_simulation']
        data_dict["I_I/Y"] =  np.exp(data_dict['log_K_simulation']) * data_dict['i_r_simulation'] /  \
        (np.exp(data_dict['log_K_simulation']) * data_dict['Z_simulation'] * A_d + \
        np.exp(data_dict['log_K_simulation']) * (1.0 - data_dict['Z_simulation']) * A_g)
        data_dict["I_I/Y"] = np.array(data_dict["I_I/Y"])
        data_dict["E"] = η * np.exp(data_dict['log_K_simulation']) * (1.0 - data_dict['Z_simulation'] ) * A_d
        data_dict["E"] = np.array(data_dict["E"])
        data_dict['K_g'] = np.exp(data_dict['log_K_simulation']) * data_dict['Z_simulation']
        data_dict['K_d'] = np.exp(data_dict['log_K_simulation']) * (1.0 - data_dict['Z_simulation'])


        data_dict['damage_jump_intensity'] =  r_1 * ( np.exp( r_2 / 2 * np.power( data_dict['T_simulation'] - y_lower_bar,2) ) - 1  ) * (data_dict['T_simulation']  > y_lower_bar )

        data_dict['f_m_avg'] = 0

        for i in range(self.params["λ_3_length"]):

            data_dict['f_m_' + str(i) + '_simulation'] = [f_ms[i].numpy()[0,0] for f_ms in f_ms_list]
            data_dict['f_m_' + str(i) + '_simulation'] =  np.array(data_dict['f_m_' + str(i) + '_simulation'])
            data_dict['f_m_avg'] += data_dict['f_m_' + str(i) + '_simulation']

        data_dict['f_m_sum'] = data_dict['f_m_avg']

        for i in range(self.params["λ_3_length"]):

            data_dict['f_m_' + str(i) + '_simulation_norm'] = data_dict['f_m_' + str(i) + '_simulation']/  data_dict['f_m_sum']


        data_dict['f_m_avg'] = data_dict['f_m_avg'] / self.params["λ_3_length"]

        data_dict['distorted_dmg_jump_intensity'] = data_dict['f_m_avg']*data_dict['damage_jump_intensity']
        data_dict['distorted_dmg_jump_prob'] = 1-np.exp(-np.cumsum(data_dict['distorted_dmg_jump_intensity'] * (dt)))

        logR        =  np.exp(data_dict['logR_simulation'])

        data_dict['g_j_avg'] = 0

        for i in range(self.params["A_g_prime_length"]):

            data_dict['g_j_' + str(i) + '_simulation'] = [g_js[i].numpy()[0,0] for g_js in g_js_list]
            data_dict['g_j_' + str(i) + '_simulation'] =  np.array(data_dict['g_j_' + str(i) + '_simulation'])
            data_dict['g_j_avg'] += data_dict['g_j_' + str(i) + '_simulation']

        data_dict['g_j_sum'] = data_dict['g_j_avg']

        for i in range(self.params["A_g_prime_length"]):

            data_dict['g_j_' + str(i) + '_simulation_norm'] = data_dict['g_j_' + str(i) + '_simulation']/  data_dict['g_j_sum']

        data_dict['g_j_avg'] = data_dict['g_j_avg'] / self.params["A_g_prime_length"]

        data_dict['distorted_tech_jump_intensity'] = data_dict['g_j_avg']* logR * (dt) / ϱ
        data_dict['distorted_tech_jump_prob'] = 1-np.exp(-np.cumsum(data_dict['distorted_tech_jump_intensity']))

        data_dict['h_y_simulation'] = h_y
        data_dict['h_y_simulation'] = np.array(data_dict['h_y_simulation'])

        with open(export_folder + '/data_dict.txt', 'a') as the_file:
            for key in data_dict.keys():
                if "nn_config" not in key:
                    the_file.write( str(key) + ": " + str(data_dict[key]) + '\n')


        ################################################################
        ################################################################
        ################################################################
        ## Make plots
        ################################################################
        ################################################################
        ################################################################


        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,0] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$\log K$')
        plt.savefig(export_folder + "/logK_simulation.png")
        np.savetxt(export_folder + "/log_K_simulation.txt", np.array([state.numpy()[0,0] for state in state_list]))
        plt.close()


        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,1] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$Z$')
        plt.savefig(export_folder +  "/Z_simulation.png")
        np.savetxt(export_folder + "/Z_simulation.txt", np.array([state.numpy()[0,1] for state in state_list]))
        plt.close()




        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,2] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$T$')
        plt.savefig(export_folder + "/T_simulation.png")
        np.savetxt(export_folder + "/T_simulation.txt", np.array([state.numpy()[0,2] for state in state_list]))
        plt.close()


        plt.figure()
        plt.plot(time_vec,[state.numpy()[0,3] for state in state_list])
        plt.xlabel("Years")
        plt.title(r'$\log logR$')
        plt.savefig(export_folder + "/logR_simulation.png")
        np.savetxt(export_folder +  "/logR_simulation.txt", np.array([state.numpy()[0,3] for state in state_list]))
        plt.close()


        plt.figure()
        plt.plot(time_vec,[i_g.numpy()[0,0] for i_g in i_g_list])
        plt.xlabel("Years")
        plt.title(r'$i_g$')
        plt.savefig(export_folder + "/i_g_simulation.png")
        np.savetxt(export_folder + "/i_g_simulation.txt", np.array([i_g.numpy()[0,0] for i_g in i_g_list]))
        plt.close()


        plt.figure()
        plt.plot(time_vec,[i_d.numpy()[0,0] for i_d in i_d_list])
        plt.xlabel("Years")
        plt.title(r'$i_d$')
        plt.savefig(export_folder + "/i_d_simulation.png")
        np.savetxt(export_folder + "/i_d_simulation.txt", np.array([i_d.numpy()[0,0] for i_d in i_d_list]))
        plt.close()


        plt.figure()
        plt.plot(time_vec,[np.exp(-i_r.numpy()[0,0]) for i_r in i_r_list])
        plt.xlabel("Years")
        plt.title(r'$i_r$')
        plt.savefig(export_folder + "/i_r_simulation.png")
        np.savetxt(export_folder + "/i_r_simulation.txt", np.array([ np.exp(-i_r.numpy()[0,0]) for i_r in i_r_list]))
        plt.close()




        plt.figure()
        plt.plot(time_vec,h_y)
        plt.xlabel("Years")
        plt.title(r'$h$: Temperature')
        plt.savefig(export_folder + "/h_y_simulation.png")
        np.savetxt(export_folder +  "/h_y_simulation.txt", h_y)
        plt.close()



        plt.figure()
        plt.plot(time_vec,h_d)
        plt.xlabel("Years")
        plt.title(r'$h$: Dirty Capital')
        plt.savefig(export_folder + "/h_d_simulation.png")
        np.savetxt(export_folder +  "/h_d_simulation.txt", h_d)
        plt.close()

        plt.figure()
        plt.plot(time_vec,h_g)
        plt.xlabel("Years")
        plt.title(r'$h$: Green Capital')
        plt.savefig(export_folder + "/h_g_simulation.png")
        np.savetxt(export_folder +  "/h_g_simulation.txt", h_g)
        plt.close()

        plt.figure()
        plt.plot(time_vec,h_Z)
        plt.xlabel("Years")
        plt.title(r'$h$: Technology')
        plt.savefig(export_folder + "/h_Z_simulation.png")
        np.savetxt(export_folder +  "/h_Z_simulation.txt", h_Z)
        plt.close()


        plt.figure()
        plt.plot(time_vec,dv_dY.numpy())
        plt.xlabel("Years")
        plt.title(r'$dv dY$')
        plt.savefig(export_folder + "/dv_dY_simulation.png")
        np.savetxt(export_folder + "/dv_dY_simulation.txt", dv_dY.numpy())
        plt.close()


        for i in range(self.params["λ_3_length"]):

            plt.figure()
            plt.plot(time_vec,[f_ms[i].numpy()[0,0] for f_ms in f_ms_list])
            plt.xlabel("Years")
            plt.title("f_m " + str(i+1))
            plt.savefig(export_folder + "/f_m_" + str(i+1) + "_simulation.png")
            np.savetxt(export_folder + "/f_m_" + str(i+1) + "_simulation.txt", [f_ms[i].numpy()[0,0] for f_ms in f_ms_list])
            plt.close()

        for i in range(self.params["A_g_prime_length"]):

            plt.figure()
            plt.plot(time_vec,[g_js[i].numpy()[0,0] for g_js in g_js_list])
            plt.xlabel("Years")
            plt.title("g_j " + str(i+1))
            plt.savefig(export_folder + "/g_j_" + str(i+1) + "_simulation.png")
            np.savetxt(export_folder + "/g_j_" + str(i+1) + "_simulation.txt", [g_js[i].numpy()[0,0] for g_js in g_js_list])
            plt.close()


        theta_ell = (pd.read_csv("./model144.csv", header=None).to_numpy()[:,0]/1000).astype(np.float32)

        ################################################################
        ################################################################
        ################################################################
        ## Make plots Part 2
        ################################################################
        ################################################################
        ################################################################


        plt.figure()
        plt.plot(time_vec,data_dict["E"], label = r"$\ξ = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title("Emissions")
        # plt.ylim(7.5,11.5)
        plt.legend(loc='upper left')
        plt.savefig(export_folder + "/Ems_Comp_IMSI_2023.png")
        np.savetxt(export_folder +  "/Emissions.txt", data_dict["E"])
        plt.close()



        plt.figure()
        # plt.plot(data_dict_1["logR"], label = r"$\ξ = 0.15$")
        plt.plot(time_vec,data_dict["logR"], label = r"$\ξ = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title(r"Green Investment ($logR$)")
        plt.legend(loc='upper left')
        plt.savefig(export_folder + '/Ig_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/Ig.txt", data_dict["logR"])
        plt.close()

        plt.figure()
        plt.plot(time_vec,data_dict["I_d"], label = r"$\ξ = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title(r"Dirty Investment ($I_d$)")
        plt.legend(loc='lower left')
        plt.savefig(export_folder + '/Id_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/Id.txt", data_dict["I_d"])
        plt.close()



        plt.figure()
        plt.plot(time_vec,data_dict["I_I/Y"]*100, label = r"$\ξ = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title(r"Z&D Investment as % of Output ($I_{\kappa}/Y$)")
        plt.legend(loc='upper right')
        plt.savefig(export_folder + '/ZD_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/ZD.txt", data_dict["I_I/Y"])
        plt.close()


        plt.figure()
        plt.plot(time_vec,data_dict["distorted_dmg_jump_prob"], label = r"$\ξ = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title("Distorted Probability of a Damage Jump")
        plt.ylim(0,1)
        plt.legend(loc='lower right')
        plt.savefig(export_folder + '/DmgJumpProb_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/DmgJumpProb.txt", data_dict["distorted_dmg_jump_prob"])        
        plt.close()


        plt.figure()
        plt.plot(
            time_vec,
            data_dict["distorted_dmg_jump_intensity"],
            label=r"$\ξ = {:.3f}$".format(np.exp(logξ)),
            color='tab:red',  # Using a different color for distinction
            linewidth=2
        )
        plt.xlabel('Years')
        plt.title("Distorted Intensity of a Damage Jump")
        plt.ylim(0, max(data_dict["distorted_dmg_jump_intensity"]) * 1.1)  # Adjust y-limit based on data
        plt.legend(loc='upper right')
        plt.savefig(f"{export_folder}/DmgJumpIntensity_Comp_IMSI_2023.png")
        np.savetxt(f"{export_folder}/DmgJumpIntensity.txt", data_dict["distorted_dmg_jump_intensity"])
        plt.close()


        plt.figure()
        plt.plot(time_vec,data_dict["distorted_tech_jump_prob"], label = r"$\ξ = {:.3f}$".format(np.exp(logξ)), color = 'tab:blue', linewidth='2')
        plt.xlabel('Years')
        plt.title("Distorted Probability of a Technology Jump")
        plt.ylim(0,1)
        plt.legend(loc='lower right')
        plt.savefig(export_folder + '/TechJumpProb_Comp_IMSI_2023.png')
        np.savetxt(export_folder +  "/TechJumpProb.txt", data_dict["distorted_tech_jump_prob"])         
        plt.close()
         
        # Plotting Distorted Intensity of a Technology Jump
        plt.figure()
        plt.plot(
            time_vec,
            data_dict["distorted_tech_jump_intensity"],
            label=r"$\ξ = {:.3f}$".format(np.exp(logξ)),
            color='tab:green',  # Using a different color for distinction
            linewidth=2
        )
        plt.xlabel('Years')
        plt.title("Distorted Intensity of a Technology Jump")
        plt.ylim(0, max(data_dict["distorted_tech_jump_intensity"]) * 1.1)  # Adjust y-limit based on data
        plt.legend(loc='upper right')
        plt.savefig(f"{export_folder}/TechJumpIntensity_Comp_IMSI_2023.png")
        np.savetxt(f"{export_folder}/TechJumpIntensity.txt", data_dict["distorted_tech_jump_intensity"])
        plt.close()


        ## Plot bar chart
        baseline = np.ones(self.params["λ_3_length"]) / self.params["λ_3_length"]
        distorted = np.ones(self.params["λ_3_length"]) / self.params["λ_3_length"]
        bin_edges = np.linspace(0, 1/3, 6)
        x1       = np.linspace(0,1/3,self.params["λ_3_length"])
        for i in range(self.params["λ_3_length"]):
            distorted[i] = data_dict['f_m_'+str(i)+'_simulation_norm'][-1] 

        print("Climate Models: {}"  .format(distorted))

        plt.hist(x1, weights=baseline, bins=bin_edges,label='Baseline', color = 'C3', α=0.5, ec="darkgrey")
        plt.hist(x1, weights=distorted, bins=bin_edges, label='Distorted', color = 'C0', α=0.5, ec="darkgrey")
        plt.title("Distorted Probability of Damage Models")
        plt.xlabel(r"$\λ_3$")
        plt.legend()
        plt.xlim([0,1/3])
        plt.ylim([0, 0.6])
        plt.savefig(export_folder + '/Dmg_Dist_IMSI_2023.png')
        plt.close()



        ## Plot bar chart
        baseline = np.ones(self.params["A_g_prime_length"]) / self.params["A_g_prime_length"]
        x1       = np.linspace(self.params["A_g_prime_min"], self.params["A_g_prime_max"], self.params["A_g_prime_length"])
        distorted = np.ones(self.params["A_g_prime_length"]) / self.params["A_g_prime_length"]
        for i in range(self.params["A_g_prime_length"]):
            distorted[i] = data_dict['g_j_'+str(i)+'_simulation_norm'][-1]

        bin_edges = np.linspace(0.12, 0.13, 4)
        print("Tech Models: {}"  .format(distorted))
        plt.hist(x1, weights=baseline, label='Baseline', color = 'C3', α=0.5, ec="darkgrey",bins=bin_edges)
        plt.hist(x1, weights=distorted,  label='Distorted', color = 'C0', α=0.5, ec="darkgrey",bins=bin_edges)
        plt.title("Distorted Probability of Technology Models")
        plt.xlabel(r"$A'_g$")
        plt.legend()
        plt.xlim([self.params["A_g_prime_min"],self.params["A_g_prime_max"]])
        plt.ylim([0, 0.6])
        plt.savefig(export_folder + '/Tech_Dist_IMSI_2023.png')
        plt.close()


        pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
        θ̄ =  1.86 / 1000
        ς = 1.2 * 1.86 / 1000
        print("Distortion: {}"  .format(ς * data_dict['h_y_simulation'][-1]))
        plt.hist(1000*theta_ell, weights=pi_c_o, bins = np.linspace(0.8,3.,16), label = "Baseline", color = 'C3', α=0.5, density=True, ec="darkgrey")
        plt.hist(1000*(theta_ell + ς * data_dict['h_y_simulation'][-1]), weights=pi_c_o, label = "Distorted", bins = np.linspace(0.8,3.,16), color = 'C0', α=0.5,density=True,  ec="darkgrey")
        plt.title("Distorted Probability of Climate Models")
        plt.xlabel("Climate Sensitivity")
        plt.ylim(0, 1.5)
        plt.xlim(0.8, 3)
        plt.legend()
        plt.savefig(export_folder + '/Climate_Dist_IMSI_2023.png')
        plt.close()
   
    def analyze(self):

        ## Analyze results
        n_points = 100

        if "post_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            Y_vector       = np.unique(self.solution_fd['stateSpace'][:,2])
            point_Y        = Y_vector[np.abs(Y_vector - 2.5).argmin()]
            mid_point_logK = np.unique(self.solution_fd['stateSpace'][:,0])[ round(self.solution_fd['nK']/2)]

            idx            = (self.solution_fd['stateSpace'][:,2] == point_Y) & (self.solution_fd['stateSpace'][:,0] == mid_point_logK)
            X              = self.solution_fd['stateSpace'][idx]
            logξ             = tf.ones( (X.shape[0],1) ) * self.params["logξ"]

            X              = tf.cast(X ,dtype= "float32")
            λ_3        = tf.ones( (X.shape[0],1) ) * self.params["λ_3"]
            A_g_prime      = tf.ones( (X.shape[0],1) ) * self.params["A_d"]
            
            X              = tf.concat([X, λ_3, A_g_prime, logξ], axis=1)
            v = self.v_nn(X); i_g = self.i_g_nn(X); 
            i_d = self.i_d_nn(X)


            ## Generate plots
            f, ax = plt.subplots(1,3, figsize=(20,5))

            ax[0].plot(X[:,1], v, label = "Neural network")
            if self.params["n_dims"] < 4:
                ax[0].plot(X[:,1], self.solution_fd['V'].flatten(order = 'F')[idx], label = "Finite difference")
            ax[0].set_xlabel(r'$Z$')
            ax[0].set_title(r'$v$')
            ax[0].legend()

            ax[1].plot(X[:,1], i_g)
            if self.params["n_dims"] < 4:
                ax[1].plot(X[:,1], self.solution_fd['i_G'].flatten(order = 'F')[idx])
            ax[1].set_xlabel(r'$Z$')
            ax[1].set_title(r'$i_g$')

            ax[2].plot(X[:,1], i_d)
            if self.params["n_dims"] < 4:
                ax[2].plot(X[:,1], self.solution_fd['i_B'].flatten(order = 'F')[idx])
            ax[2].set_xlabel(r'$Z$')
            ax[2].set_title(r'$i_d$')

            plt.savefig(self.params["export_folder"] + "/compare_results_fd.png")
            plt.close()

            ## Vary λ_3
            for λ_3_idx in range(self.params["λ_3_length"]):

                idx            = (self.solution_fd['stateSpace'][:,2] == point_Y) & (self.solution_fd['stateSpace'][:,0] == mid_point_logK)
                X              = self.solution_fd['stateSpace'][idx]
                X              = tf.cast(X ,dtype= "float32")
                λ_3        = tf.ones( (X.shape[0],1) ) * self.params["λ_3_list"][λ_3_idx]
                A_g_prime      = tf.ones( (X.shape[0],1) ) * self.params["A_d"]

                X              = tf.concat([X, λ_3, A_g_prime, logξ], axis=1)
                v = self.v_nn(X); i_g = self.i_g_nn(X); 
                i_d = self.i_d_nn(X)


                ## Generate plots
                f, ax = plt.subplots(1,3, figsize=(20,5))

                ax[0].plot(X[:,1], v, label = "Neural network; λ_3 = " + str( round(self.params["λ_3_list"][λ_3_idx],2) ))
                if self.params["n_dims"] < 4:
                    ax[0].plot(X[:,1], self.solution_fd['V'].flatten(order = 'F')[idx], label = "Finite difference")
                ax[0].set_xlabel(r'$Z$')
                ax[0].set_title(r'$v$')
                ax[0].legend()

                ax[1].plot(X[:,1], i_g)
                if self.params["n_dims"] < 4:
                    ax[1].plot(X[:,1], self.solution_fd['i_G'].flatten(order = 'F')[idx])
                ax[1].set_xlabel(r'$Z$')
                ax[1].set_title(r'$i_g$')

                ax[2].plot(X[:,1], i_d)
                if self.params["n_dims"] < 4:
                    ax[2].plot(X[:,1], self.solution_fd['i_B'].flatten(order = 'F')[idx])
                ax[2].set_xlabel(r'$Z$')
                ax[2].set_title(r'$i_d$')

                plt.savefig(self.params["export_folder"] + "/compare_results_" + str(λ_3_idx) + ".png")
                plt.close()

            ## Vary ξ
            logξ_list                                             = [float(np.log(ξ)) for ξ in np.linspace(np.exp(self.params['logξ_min']) + 0.02, 
                                                                                                               np.exp(self.params['logξ_max']) - 0.02, 5)]
            ## Generate plots
            f, ax = plt.subplots(1,3, figsize=(20,5))

            for logξ_idx in range(len(logξ_list)):

                idx            = (self.solution_fd['stateSpace'][:,2] == point_Y) & (self.solution_fd['stateSpace'][:,0] == mid_point_logK)
                X              = self.solution_fd['stateSpace'][idx]
                X              = tf.cast(X ,dtype= "float32")
                λ_3        = tf.ones( (X.shape[0],1) ) * self.params["λ_3"]
                A_g_prime      = tf.ones( (X.shape[0],1) ) * self.params["A_d"]

                logξ         = tf.ones( (X.shape[0],1) ) * logξ_list[logξ_idx]
                X              = tf.concat([X, λ_3, A_g_prime, logξ], axis=1)
                
                v   = self.v_nn(X); i_g = self.i_g_nn(X); 
                i_d = self.i_d_nn(X)




                ax[0].plot(X[:,1], v, label = "Neural network; ξ = " + str( round( np.exp( logξ_list[logξ_idx] ),2) ))
                ax[0].set_xlabel(r'$Z$')
                ax[0].set_title(r'$v$')
                ax[0].legend()

                ax[1].plot(X[:,1], i_g)
                ax[1].set_xlabel(r'$Z$')
                ax[1].set_title(r'$i_g$')

                ax[2].plot(X[:,1], i_d)
                ax[2].set_xlabel(r'$Z$')
                ax[2].set_title(r'$i_d$')

                
            plt.savefig(self.params["export_folder"] + "/compare_results_ξ.png")
            plt.close()

        if "pre_damage" in self.params["model_type"] and "pre_tech" in self.params["model_type"]:

            logK      = (self.params["logK_max"] + self.params["logK_min"]) / 2 * np.ones(n_points).reshape(n_points, 1)
            Y         = 2.5 * np.ones(n_points).reshape(n_points, 1)
            logR   = (self.params["logR_max"] + self.params["logR_min"]) / 2 * np.ones(n_points).reshape(n_points, 1)

            Z         = np.linspace(self.params["Z_min"], self.params["Z_max"], n_points)


            logK = tf.reshape(tf.cast(logK ,dtype= "float32"), [n_points,1]) 
            Y    = tf.reshape(tf.cast(Y ,dtype= "float32"), [n_points,1]) 
            logR    = tf.reshape(tf.cast(logR ,dtype= "float32"), [n_points,1]) 

            Z   = tf.reshape(tf.cast(Z ,dtype= "float32"), [n_points,1]) 

            logξ             = tf.ones( (n_points, 1) ) * self.params["logξ"]
            X = tf.concat([logK, Z, Y, logξ, logR], 1)

            v = self.v_nn(X); i_g = self.i_g_nn(X); 
            i_d = self.i_d_nn(X); i_r = self.i_r_nn(X)
            

            f, ax = plt.subplots(1,4, figsize=(20,5))

            ax[0].plot(X[:,1], v, label = "Neural network")
            ax[0].set_xlabel(r'$Z$')
            ax[0].set_title(r'$v$')
            ax[0].legend()

            ax[1].plot(X[:,1], i_g)
            ax[1].set_xlabel(r'$Z$')
            ax[1].set_title(r'$i_g$')

            ax[2].plot(X[:,1], i_d)
            ax[2].set_xlabel(r'$Z$')
            ax[2].set_title(r'$i_d$')

            ax[3].plot(X[:,1], tf.exp(-i_r))
            ax[3].set_xlabel(r'$Z$')
            ax[3].set_title(r'$i_r$')

            plt.savefig(self.params["export_folder"] + "/compare_results.png")
            plt.close()
            
'''