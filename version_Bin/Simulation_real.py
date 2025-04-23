import numpy as np
import tensorflow as tf 
import sys 
import pathlib
import json  
import matplotlib.pyplot as plt
 
import  model_pseudostate_original as model


export_folder = "/project/lhansen/Cap_NN_oldVersion/November_version_NewParameters/output/Novem_NewParaters_0.01_LR_piecewiseconstant_10e-5,10e-5,10e-5,10e-5_128_neurons_32_#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations2000000"


batch_size                  =   128
num_iterations              = 2000000
logging_frequency           = 1000
learning_rates              = [10e-5, 10e-5, 10e-5, 10e-5]    # (10e-5 is 1e-4)
hidden_layer_activations    = ["swish","tanh","tanh","softplus"]
output_layer_activations    = ["softplus","custom","custom","softplus"]
num_hidden_layers           = 4
num_neurons                 = 32
learning_rate_schedule_type = "piecewiseconstant"
channel_type                = "full"
pretrained_path = None
tensorboard = True


## Take care of activation functions 
hidden_layer_activations   = [None if x == "None" else x for x in hidden_layer_activations]
output_layer_activations   = [None if x == "None" else x for x in output_layer_activations]

v_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_nn"}
v_nn_config["final_activation"] = output_layer_activations[0]

i_g_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[1], "dim" : 1, "nn_name" : "i_g_nn"}
i_g_nn_config["final_activation"] = output_layer_activations[1]

i_d_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[2], "dim" : 1, "nn_name" : "i_d_nn"}
i_d_nn_config["final_activation"] = output_layer_activations[2]

i_I_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[3], "dim" : 1, "nn_name" : "i_I_nn"}
i_I_nn_config["final_activation"] = output_layer_activations[3]


## Create params struct 
params = {"batch_size" : batch_size, "learning_rates":learning_rates,\
"v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config,"i_I_nn_config":i_I_nn_config,\
# "n_dims" : 3, "model_type" : "post_tech_post_damage" , \
"num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, "load_parameters" : None,\
"pretrained_path" : pretrained_path, 'tensorboard' : tensorboard, "learning_rate_schedule_type" : learning_rate_schedule_type, "channel_type": channel_type }

  
## i_g and i_d activations come after params because we amy want to use phi_g and phi_d
if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_g"]) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_d"]) / (tf.exp(2 * x) + 1.0)

# Define the paths: Load all for paths for simulation
export_folder = "/project/lhansen/Cap_NN_oldVersion/November_version_NewParameters/output/Novem_NewParaters_0.01_LR_piecewiseconstant_10e-5,10e-5,10e-5,10e-5_128_neurons_32_#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations2000000"

v_nn_checkpoint_post_tech_post_damage_path = export_folder+  "/post_tech_post_damage"+'/v_nn_checkpoint_post_tech_post_damage'
i_g_nn_checkpoint_post_tech_post_damage_path = export_folder+  "/post_tech_post_damage"+'/i_g_nn_checkpoint_post_tech_post_damage'
i_d_nn_checkpoint_post_tech_post_damage_path = export_folder+  "/post_tech_post_damage"+'/i_d_nn_checkpoint_post_tech_post_damage' 

v_nn_checkpoint_post_tech_pre_damage_path = export_folder+  "/post_tech_pre_damage"+'/v_nn_checkpoint_post_tech_pre_damage'
i_g_nn_checkpoint_post_tech_pre_damage_path = export_folder+  "/post_tech_pre_damage"+'/i_g_nn_checkpoint_post_tech_pre_damage'
i_d_nn_checkpoint_post_tech_pre_damage_path = export_folder+  "/post_tech_pre_damage"+'/i_d_nn_checkpoint_post_tech_pre_damage' 

v_nn_checkpoint_pre_tech_post_damage_path = export_folder+  "/pre_tech_post_damage"+'/v_nn_checkpoint_pre_tech_post_damage'
i_g_nn_checkpoint_pre_tech_post_damage_path = export_folder+  "/pre_tech_post_damage"+'/i_g_nn_checkpoint_pre_tech_post_damage'
i_d_nn_checkpoint_pre_tech_post_damage_path = export_folder+  "/pre_tech_post_damage"+'/i_d_nn_checkpoint_pre_tech_post_damage' 
i_I_nn_checkpoint_pre_tech_post_damage_path = export_folder+  "/pre_tech_post_damage"+'/i_I_nn_checkpoint_pre_tech_post_damage' 

v_nn_checkpoint_pre_tech_pre_damage_path = export_folder+  "/pre_tech_pre_damage"+'/v_nn_checkpoint_pre_tech_pre_damage'
i_g_nn_checkpoint_pre_tech_pre_damage_path = export_folder+  "/pre_tech_pre_damage"+'/i_g_nn_checkpoint_pre_tech_pre_damage'
i_d_nn_checkpoint_pre_tech_pre_damage_path = export_folder+  "/pre_tech_pre_damage"+'/i_d_nn_checkpoint_pre_tech_pre_damage' 
i_I_nn_checkpoint_pre_tech_pre_damage_path = export_folder+  "/pre_tech_pre_damage"+'/i_I_nn_checkpoint_pre_tech_pre_damage' 



def load_model(params, n_inputs, v_nn_checkpoint_path, i_g_nn_checkpoint_path, i_d_nn_checkpoint_path,i_I_nn_checkpoint_path=None):
    test_model = model.model(params)
    
    # n_inputs = 7 # if "post_tech_post_damage" in params["model_type"] else 8
    test_model.v_nn.build((params["batch_size"], n_inputs))
    test_model.i_g_nn.build((params["batch_size"], n_inputs))
    test_model.i_d_nn.build((params["batch_size"], n_inputs)) 
    
    test_model.v_nn.load_weights(v_nn_checkpoint_path)
    test_model.i_g_nn.load_weights(i_g_nn_checkpoint_path)
    test_model.i_d_nn.load_weights(i_d_nn_checkpoint_path) 
    
    if i_I_nn_checkpoint_path is not None:
        test_model.i_I_nn.build((params["batch_size"], n_inputs)) 
        test_model.i_I_nn.load_weights(i_I_nn_checkpoint_path) 
    return test_model


params["model_type"]="post_tech_post_damage"
params["n_dims"]= 3
params["export_folder"]  = export_folder +  "/post_tech_post_damage"
post_tech_post_damage_model = load_model(params,7, v_nn_checkpoint_post_tech_post_damage_path, i_g_nn_checkpoint_post_tech_post_damage_path, i_d_nn_checkpoint_post_tech_post_damage_path)

params["model_type"]="post_tech_pre_damage"
params["n_dims"]= 3
params["export_folder"]  = export_folder +  "/post_tech_pre_damage"
params["v_post_tech_post_damage_nn_path"] = export_folder + "/post_tech_post_damage"
post_tech_pre_damage_model = load_model(params,6, v_nn_checkpoint_post_tech_pre_damage_path, i_g_nn_checkpoint_post_tech_pre_damage_path, i_d_nn_checkpoint_post_tech_pre_damage_path)

params["model_type"]="pre_tech_post_damage"
params["n_dims"]= 4
params["v_post_tech_post_damage_nn_path"] = export_folder + "/post_tech_post_damage"
params["export_folder"]  = export_folder +  "/pre_tech_post_damage"
pre_tech_post_damage_model = load_model(params,8, v_nn_checkpoint_pre_tech_post_damage_path, i_g_nn_checkpoint_pre_tech_post_damage_path, i_d_nn_checkpoint_pre_tech_post_damage_path, i_I_nn_checkpoint_pre_tech_post_damage_path)


params["model_type"]="pre_tech_pre_damage"
params["n_dims"]= 4
params["v_pre_tech_post_damage_nn_path"]  = export_folder + "/pre_tech_post_damage"  
params["v_post_tech_pre_damage_nn_path"]  = export_folder + "/post_tech_pre_damage"
params["export_folder"]  = export_folder +  "/pre_tech_pre_damage"
pre_tech_pre_damage_model = load_model(params,7, v_nn_checkpoint_pre_tech_pre_damage_path, i_g_nn_checkpoint_pre_tech_pre_damage_path, i_d_nn_checkpoint_pre_tech_pre_damage_path, i_I_nn_checkpoint_pre_tech_pre_damage_path)

log_xi = tf.constant(-0.0, dtype=tf.float32) 

dt = 1/12
scale = tf.sqrt(dt)


def iterate_state(logK, R, Y, log_I_g,  A_g_prime, gamma_3, whether_Tech_jump, whether_Damage_jump):
    """
    Iterates the state variables to compute next period's state.
    
    Parameters:
        logK: Tensor or float representing the log capital stock.
        R: Tensor or float representing the rate/return.
        I_g: Tensor or float representing the log investment in technology.
        Y: Tensor or float representing the output.
        A_g_prime: Tensor or float representing the next state of the technology parameter.
        gamma_3: Tensor or float representing an additional state variable.
        whether_Tech_jump: Integer (0 or 1) indicating if a technology jump occurs.
        whether_Damage_jump: Integer (0 or 1) indicating if a damage jump occurs.
    
    Returns:
        next_logK, next_R, next_I_g, next_Y, next_A_g_prime, next_gamma_3:
        The updated state values.
    """
      
    ## Sampling the Weineer Processes
    increments = tf.random.normal(shape=[4], mean=0.0, stddev=1.0) * scale
    dW_g, dW_d, dW_Y, dW_log_I_g = tf.unstack(increments)
    
    
    
    sigma_K_d      = pre_tech_pre_damage_model.params["sigma_d"]* (1-R)
    sigma_K_g      = pre_tech_pre_damage_model.params["sigma_g"]* R

    sigma_R_d      = - pre_tech_pre_damage_model.params["sigma_d"]* R *(1-R)
    sigma_R_g      =   pre_tech_pre_damage_model.params["sigma_g"]* R *(1-R)

    sigma_I_g      =    pre_tech_pre_damage_model.params["sigma_I"]

    sigma_Y = pre_tech_pre_damage_model.params['varsigma'] *  pre_tech_pre_damage_model.params['eta'] * pre_tech_pre_damage_model.params['A_d'] * (1-R) * tf.exp(logK  )
        

    # Placeholder update rules for the (0,0) case: no jumps.
    if whether_Tech_jump == 0 and whether_Damage_jump == 0:
        
        model = pre_tech_pre_damage_model
        
      
        state_1d = tf.stack([
            tf.reshape(logK, []),
            tf.reshape(R, []),
            tf.reshape(Y, []),
            tf.reshape(log_I_g, []),
            tf.reshape(log_xi, []),
            tf.reshape(log_xi, []),
            tf.reshape(log_xi, [])
        ], axis=0)  # shape (7,)

        state = tf.reshape(state_1d, (1, 7))
        
        
        
        K=   tf.exp(logK  )
        
        ### Controls
        i_g        = model.i_g_nn(state)
        i_d        = model.i_d_nn(state)
        i_I        = model.i_I_nn(state)
            
        ### Increments
         
         
        v_kk_term = ( tf.pow(model.params["sigma_d"],2) * tf.pow(1-R,2)  + tf.pow(model.params["sigma_g"],2) * tf.pow(R,2))/2.0

        inside_log_i_d   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_d"] * i_d , 0.0001), [1, 1])
        inside_log_i_g   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_g"] * i_g , 0.0001), [1, 1])

        v_k_term       = ( model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( model.params["alpha_g"] +  model.params["Gamma"] * tf.math.log( inside_log_i_g) ) * R  - v_kk_term
        
        v_r_term       = ( model.params["alpha_g"] + model.params["Gamma"] * tf.math.log( inside_log_i_g )  - (model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(model.params["sigma_d"],2) *  (1-R ) - 
                        tf.pow(model.params["sigma_g"], 2) *  R ) *  R * (1 - R)
        
     
        v_y_term       = model.params["beta_f"] * (model.params["eta"] *  model.params["A_d"] * (1-R) * tf.exp(logK  ))

        
        v_I_g_term     = - model.params["zeta"] + model.params["psi_0"] * tf.exp(-i_I * model.params["psi_1"]) * tf.exp( model.params["psi_1"] * (logK -  log_I_g) ) - 0.5 * tf.pow(model.params["sigma_I"], 2)

    
        ## dW_K, dW_R, dW_Y, dW_log_I_g
        new_logK       = logK + v_k_term * dt +   sigma_K_d  * dW_d     +     sigma_K_g* dW_g
        
        new_R          = R + v_r_term * dt   +   sigma_R_d  * dW_d     +     sigma_R_g* dW_g
        
        new_log_I_g    = log_I_g + v_I_g_term * dt  + sigma_I_g * dW_log_I_g
        
        new_Y          = Y +  v_y_term * dt  + sigma_Y* dW_Y

        #### Sample Whether Jump
        I_d   = model.params['r_1'] * ( tf.exp( model.params['r_2'] / 2 * tf.pow(new_Y - model.params['y_lower_bar'],2) ) - 1  ) * \
            tf.cast(new_Y > model.params['y_lower_bar'], tf.float32 )
        
        p_jump = 1 - tf.exp(-I_d * dt)
        
        # Determine if a jump occurs
        jump_occur = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), p_jump)  # This is a boolean tensor
        # Update the whether jump indicator
        whether_Damage_jump+= tf.cast(jump_occur, tf.int32)
 
        gamma_3_list = np.linspace(model.params["gamma_3_min"], model.params["gamma_3_max"], model.params["gamma_3_length"]).tolist()
     
        # Convert gamma_3_list to a TensorFlow tensor for sampling
        gamma_3_tensor = tf.convert_to_tensor(gamma_3_list, dtype=tf.float32)
    
        # Sample an index uniformly among the gamma_3_list entries
        idx = tf.random.uniform(shape=[], minval=0, maxval=model.params["gamma_3_length"], dtype=tf.int32)
        sampled_val = gamma_3_tensor[idx]
    
        # If a jump occurs, use the sampled gamma_3 value; if not, return a default value (e.g., 0.0)
        gamma_3 = tf.cond(jump_occur, lambda: sampled_val, lambda: tf.constant(0.0))
        new_Y = tf.cond(jump_occur, lambda: tf.constant(2.0), lambda: new_Y)

        #### Tech jump
        I_tech = tf.exp(new_log_I_g) / model.params["varrho"]
        p_jump = 1 - tf.exp(-I_tech * dt)
         
        # Determine if a jump occurs
        jump_occur = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), p_jump)  # This is a boolean tensor
        # Update the whether jump indicator
        whether_Tech_jump+=  tf.cast(jump_occur, tf.int32)
           

        # Convert gamma_3_list to a TensorFlow tensor for sampling
        A_g_prime_tensor = tf.convert_to_tensor(model.params["A_g_prime_list"], dtype=tf.float32)
    
        # Sample an index uniformly among the gamma_3_list entries
        idx = tf.random.uniform(shape=[], minval=0, maxval=model.params["A_g_prime_length"], dtype=tf.int32)
        sampled_val = A_g_prime_tensor[idx]
    
        # If a jump occurs, use the sampled gamma_3 value; if not, return a default value (e.g., 0.0)
        A_g_prime = tf.cond(jump_occur, lambda: sampled_val, lambda: tf.constant(0.0))
        new_log_I_g = tf.cond(jump_occur, lambda: tf.constant(0.0), lambda: new_log_I_g)
        
        
    # Placeholder update rules for the (0,1) case: damage jump only.
    elif whether_Tech_jump == 0 and whether_Damage_jump == 1:

        
        model = pre_tech_post_damage_model        
        
   
        
        state_1d = tf.stack([
            tf.reshape(logK, []),
            tf.reshape(R, []),
            tf.reshape(Y, []),
            tf.reshape(log_I_g, []),
             tf.reshape(gamma_3, []),
            tf.reshape(log_xi, []),
            tf.reshape(log_xi, []),
            tf.reshape(log_xi, [])
        ], axis=0)   

        state = tf.reshape(state_1d, (1, 8))
        
        
        K=   tf.exp(logK  )
        
        ### Controls
        i_g        = model.i_g_nn(state)
        i_d        = model.i_d_nn(state)
        i_I        = model.i_I_nn(state)
            
        ### Increments
         
 
        
        v_kk_term = ( tf.pow(model.params["sigma_d"],2) * tf.pow(1-R,2)  + tf.pow(model.params["sigma_g"],2) * tf.pow(R,2))/2.0

        inside_log_i_d   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_d"] * i_d , 0.0001), [1, 1])
        inside_log_i_g   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_g"] * i_g , 0.0001), [1, 1])

        v_k_term       = ( model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( model.params["alpha_g"] +  model.params["Gamma"] * tf.math.log( inside_log_i_g) ) * R  - v_kk_term
        
        v_r_term       = ( model.params["alpha_g"] + model.params["Gamma"] * tf.math.log( inside_log_i_g )  - (model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(model.params["sigma_d"],2) *  (1-R ) - 
                        tf.pow(model.params["sigma_g"], 2) *  R ) *  R * (1 - R)
        
     
        v_y_term       = model.params["beta_f"] * (model.params["eta"] *  model.params["A_d"] * (1-R) * tf.exp(logK  ))

        
        v_I_g_term     = - model.params["zeta"] + model.params["psi_0"] * tf.exp(-i_I * model.params["psi_1"]) * tf.exp( model.params["psi_1"] * (logK -  log_I_g) ) - 0.5 * tf.pow(model.params["sigma_I"], 2)

    
        ## dW_K, dW_R, dW_Y, dW_log_I_g
        new_logK       = logK + v_k_term * dt +   sigma_K_d  * dW_d     +     sigma_K_g* dW_g
        
        new_R          = R + v_r_term * dt   +   sigma_R_d  * dW_d     +     sigma_R_g* dW_g
        
        new_log_I_g    = log_I_g + v_I_g_term * dt  + sigma_I_g * dW_log_I_g
        
        new_Y          = Y +  v_y_term * dt  + sigma_Y* dW_Y


        #### Tech jump
        I_tech = tf.exp(new_log_I_g) / model.params["varrho"]
        p_jump = 1 - tf.exp(-I_tech * dt)
         
        # Determine if a jump occurs
        jump_occur = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), p_jump)  # This is a boolean tensor
        # Update the whether jump indicator
        whether_Tech_jump+= tf.cast(jump_occur, tf.int32)
  
        # Convert gamma_3_list to a TensorFlow tensor for sampling
        A_g_prime_tensor = tf.convert_to_tensor(model.params["A_g_prime_list"], dtype=tf.float32)
    
        # Sample an index uniformly among the gamma_3_list entries
        idx = tf.random.uniform(shape=[], minval=0, maxval=model.params["A_g_prime_length"], dtype=tf.int32)
        sampled_val = A_g_prime_tensor[idx]
    
        # If a jump occurs, use the sampled gamma_3 value; if not, return a default value (e.g., 0.0)
        A_g_prime = tf.cond(jump_occur, lambda: sampled_val, lambda: tf.constant(0.0))
        new_log_I_g = tf.cond(jump_occur, lambda: tf.constant(0.0), lambda: new_log_I_g)
        
        
    # Placeholder update rules for the (1,0) case: technology jump only.
    elif whether_Tech_jump == 1 and whether_Damage_jump == 0:
        model = post_tech_pre_damage_model        
          
        state_1d = tf.stack([
            tf.reshape(logK, []),
            tf.reshape(R, []),
            tf.reshape(Y, []), 
             tf.reshape(A_g_prime, []),
            tf.reshape(log_xi, []),
            tf.reshape(log_xi, []) 
        ], axis=0)   
        
        state    = tf.reshape(state_1d, (1,6))
        
        K=   tf.exp(logK  )
        
        ### Controls
        i_g        = model.i_g_nn(state)
        i_d        = model.i_d_nn(state)
        
        
        
         
        v_kk_term = ( tf.pow(model.params["sigma_d"],2) * tf.pow(1-R,2)  + tf.pow(model.params["sigma_g"],2) * tf.pow(R,2))/2.0

        inside_log_i_d   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_d"] * i_d , 0.0001), [1, 1])
        inside_log_i_g   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_g"] * i_g , 0.0001), [1, 1])

        v_k_term       = ( model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( model.params["alpha_g"] +  model.params["Gamma"] * tf.math.log( inside_log_i_g) ) * R  - v_kk_term
        
        v_r_term       = ( model.params["alpha_g"] + model.params["Gamma"] * tf.math.log( inside_log_i_g )  - (model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(model.params["sigma_d"],2) *  (1-R ) - 
                        tf.pow(model.params["sigma_g"], 2) *  R ) *  R * (1 - R)
        
     
        v_y_term       = model.params["beta_f"] * (model.params["eta"] *  model.params["A_d"] * (1-R) * tf.exp(logK  ))

        
      
    
        ## dW_K, dW_R, dW_Y, dW_log_I_g
        new_logK       = logK + v_k_term * dt +   sigma_K_d  * dW_d     +     sigma_K_g* dW_g
        
        new_R          = R + v_r_term * dt   +   sigma_R_d  * dW_d     +     sigma_R_g* dW_g
        
        new_log_I_g    = tf.constant(0.0)
        
        new_Y          = Y +  v_y_term * dt  + sigma_Y* dW_Y

        #### Sample Whether Jump
        I_d   = model.params['r_1'] * ( tf.exp( model.params['r_2'] / 2 * tf.pow(new_Y - model.params['y_lower_bar'],2) ) - 1  ) * \
            tf.cast(new_Y > model.params['y_lower_bar'], tf.float32 )
        
        p_jump = 1 - tf.exp(-I_d * dt)
        
        # Determine if a jump occurs
        jump_occur = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), p_jump)  # This is a boolean tensor
        # Update the whether jump indicator
        whether_Damage_jump+=tf.cast(jump_occur, tf.int32) 
 
        gamma_3_list = np.linspace(model.params["gamma_3_min"], model.params["gamma_3_max"], model.params["gamma_3_length"]).tolist()
     
        # Convert gamma_3_list to a TensorFlow tensor for sampling
        gamma_3_tensor = tf.convert_to_tensor(gamma_3_list, dtype=tf.float32)
    
        # Sample an index uniformly among the gamma_3_list entries
        idx = tf.random.uniform(shape=[], minval=0, maxval=model.params["gamma_3_length"], dtype=tf.int32)
        sampled_val = gamma_3_tensor[idx]
    
        # If a jump occurs, use the sampled gamma_3 value; if not, return a default value (e.g., 0.0)
        gamma_3 = tf.cond(jump_occur, lambda: sampled_val, lambda: tf.constant(0.0))
        new_Y = tf.cond(jump_occur, lambda: tf.constant(2.0), lambda: new_Y)
   
            
    # Placeholder update rules for the (1,1) case: both technology and damage jump.
    elif whether_Tech_jump == 1 and whether_Damage_jump == 1:
        model = post_tech_post_damage_model        
 
        
        state_1d = tf.stack([
            tf.reshape(logK, []),
            tf.reshape(R, []),
            tf.reshape(Y, []),
            tf.reshape(gamma_3, []), 
            tf.reshape(A_g_prime, []), 
            tf.reshape(log_xi, []),
            tf.reshape(log_xi, [])
        ], axis=0)   

        state = tf.reshape(state_1d, (1, 7))
        
        ### Controls
        i_g        = model.i_g_nn(state)
        i_d        = model.i_d_nn(state)
         
        v_kk_term = ( tf.pow(model.params["sigma_d"],2) * tf.pow(1-R,2)  + tf.pow(model.params["sigma_g"],2) * tf.pow(R,2))/2.0

        inside_log_i_d   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_d"] * i_d , 0.0001), [1, 1])
        inside_log_i_g   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_g"] * i_g , 0.0001), [1, 1])

        v_k_term       = ( model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( model.params["alpha_g"] +  model.params["Gamma"] * tf.math.log( inside_log_i_g) ) * R  - v_kk_term
        
        v_r_term       = ( model.params["alpha_g"] + model.params["Gamma"] * tf.math.log( inside_log_i_g )  - (model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(model.params["sigma_d"],2) *  (1-R ) - 
                        tf.pow(model.params["sigma_g"], 2) *  R ) *  R * (1 - R)
        
     
        v_y_term       = model.params["beta_f"] * (model.params["eta"] *  model.params["A_d"] * (1-R) * tf.exp(logK  ))

        
      
    
        ## dW_K, dW_R, dW_Y, dW_log_I_g
        new_logK       = logK + v_k_term * dt +   sigma_K_d  * dW_d     +     sigma_K_g* dW_g
        
        new_R          = R + v_r_term * dt   +   sigma_R_d  * dW_d     +     sigma_R_g* dW_g
        
        new_log_I_g    = tf.constant(0.0)
        
        new_Y          = Y +  v_y_term * dt  + sigma_Y* dW_Y
        
    else:
        raise ValueError("Invalid jump condition values. They should be either 0 or 1.")

    return new_logK, new_R, new_log_I_g, new_Y, A_g_prime, gamma_3, whether_Tech_jump, whether_Damage_jump




