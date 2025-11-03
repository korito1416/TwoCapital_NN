 
import tensorflow as tf
from params import PARAMS
from PreDamagePreTech import PreDamagePreTechModel
from DeterministicSimulation  import load_PreDamagePreTech_model, simulate_path_PreDamagePreTech
 
# Example usage
import numpy as np 
import pathlib
import time
from feedforward_subnet import FeedForwardSubNet, setup_optimizers 
import os
import argparse 
 
#############################################
#############################################
#############################################
#############################################
#############################################
 


parser = argparse.ArgumentParser(description="seed sets")

parser.add_argument("--id", type=int, default=1)
parser.add_argument("--xi", type=float, required=True)  # <<< move this UP

args = parser.parse_args()  # <<< after all arguments are added

seed = args.id
ξ = args.xi
logξ = tf.math.log(ξ)





 
output_folder        =f"/project/lhansen/Cap_damage/TwoStageTechJump_SITE_Pretrain/SimulationResults/paths_ξ_{ξ}/"

os.makedirs(output_folder  , exist_ok=True)



export_folder = "/project/lhansen/Cap_damage/TwoStageTechJump_SITE_Pretrain/output_shorterRun/TechSearch_LR_warmup_cosine_10e-6,10e-6,10e-6,10e-6_128_neurons_32_#HiddenLayer_4_num_iterations1000000"


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
  
## i_g and i_d activations come after params because we amy want to use phi_g and phi_d
if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_g"]) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_d"]) / (tf.exp(2 * x) + 1.0)


## i_g and i_d activations come after params because we amy want to use phi_g and phi_d
phi_g = 16.7
phi_d = 16.7
if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_g) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_d) / (tf.exp(2 * x) + 1.0)

PARAMS.update(params)

# Define the paths: Load all for paths for simulation
# export_folder = "/project/lhansen/Cap_NN_oldVersion/November_version_NewParameters/output/Novem_NewParaters_0.01_LR_piecewiseconstant_10e-5,10e-5,10e-5,10e-5_128_neurons_32_#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations2000000"



def load_model(model, n_inputs, v_nn_checkpoint_path, i_g_nn_checkpoint_path, i_d_nn_checkpoint_path,i_r_nn_checkpoint_path=None):
    test_model = model(PARAMS)
    
    # n_inputs = 7 # if "post_tech_post_damage" in params["model_type"] else 8
    test_model.v_nn.build((params["batch_size"], n_inputs))
    test_model.i_g_nn.build((params["batch_size"], n_inputs))
    test_model.i_d_nn.build((params["batch_size"], n_inputs)) 
    
    test_model.v_nn.load_weights(v_nn_checkpoint_path)
    test_model.i_g_nn.load_weights(i_g_nn_checkpoint_path)
    test_model.i_d_nn.load_weights(i_d_nn_checkpoint_path) 
    
    if i_r_nn_checkpoint_path is not None:
        test_model.i_r_nn.build((params["batch_size"], n_inputs)) 
        test_model.i_r_nn.load_weights(i_r_nn_checkpoint_path) 
    return test_model



v_nn_checkpoint_PostDamagePostTech_path = export_folder+  "/PostDamagePostTech"+'/v_nn_checkpoint_PostDamagePostTech'
i_g_nn_checkpoint_PostDamagePostTech_path = export_folder+  "/PostDamagePostTech"+'/i_g_nn_checkpoint_PostDamagePostTech'
i_d_nn_checkpoint_PostDamagePostTech_path = export_folder+  "/PostDamagePostTech"+'/i_d_nn_checkpoint_PostDamagePostTech' 


v_nn_checkpoint_PostDamageIntermTech_path = export_folder+  "/PostDamageIntermTech"+'/v_nn_checkpoint_PostDamageIntermTech'
i_g_nn_checkpoint_PostDamageIntermTech_path = export_folder+  "/PostDamageIntermTech"+'/i_g_nn_checkpoint_PostDamageIntermTech'
i_d_nn_checkpoint_PostDamageIntermTech_path = export_folder+  "/PostDamageIntermTech"+'/i_d_nn_checkpoint_PostDamageIntermTech' 
i_r_nn_checkpoint_PostDamageIntermTech_path = export_folder+  "/PostDamageIntermTech"+'/i_r_nn_checkpoint_PostDamageIntermTech' 

v_nn_checkpoint_PostDamagePreTech_path = export_folder+  "/PostDamagePreTech"+'/v_nn_checkpoint_PostDamagePreTech'
i_g_nn_checkpoint_PostDamagePreTech_path = export_folder+  "/PostDamagePreTech"+'/i_g_nn_checkpoint_PostDamagePreTech'
i_d_nn_checkpoint_PostDamagePreTech_path = export_folder+  "/PostDamagePreTech"+'/i_d_nn_checkpoint_PostDamagePreTech' 
i_r_nn_checkpoint_PostDamagePreTech_path = export_folder+  "/PostDamagePreTech"+'/i_r_nn_checkpoint_PostDamagePreTech' 

v_nn_checkpoint_PreDamageIntermTech_path = export_folder+  "/PreDamageIntermTech"+'/v_nn_checkpoint_PreDamageIntermTech'
i_g_nn_checkpoint_PreDamageIntermTech_path = export_folder+  "/PreDamageIntermTech"+'/i_g_nn_checkpoint_PreDamageIntermTech'
i_d_nn_checkpoint_PreDamageIntermTech_path = export_folder+  "/PreDamageIntermTech"+'/i_d_nn_checkpoint_PreDamageIntermTech' 
i_r_nn_checkpoint_PreDamageIntermTech_path = export_folder+  "/PreDamageIntermTech"+'/i_r_nn_checkpoint_PreDamageIntermTech' 


v_nn_checkpoint_PreDamagePostTech_path = export_folder+  "/PreDamagePostTech"+'/v_nn_checkpoint_PreDamagePostTech'
i_g_nn_checkpoint_PreDamagePostTech_path = export_folder+  "/PreDamagePostTech"+'/i_g_nn_checkpoint_PreDamagePostTech'
i_d_nn_checkpoint_PreDamagePostTech_path = export_folder+  "/PreDamagePostTech"+'/i_d_nn_checkpoint_PreDamagePostTech' 

v_nn_checkpoint_PreDamagePreTech_path = export_folder+  "/PreDamagePreTech"+'/v_nn_checkpoint_PreDamagePreTech'
i_g_nn_checkpoint_PreDamagePreTech_path = export_folder+  "/PreDamagePreTech"+'/i_g_nn_checkpoint_PreDamagePreTech'
i_d_nn_checkpoint_PreDamagePreTech_path = export_folder+  "/PreDamagePreTech"+'/i_d_nn_checkpoint_PreDamagePreTech' 
i_r_nn_checkpoint_PreDamagePreTech_path = export_folder+  "/PreDamagePreTech"+'/i_r_nn_checkpoint_PreDamagePreTech' 




try:
    from PostDamagePostTech import PostDamagePostTechModel
    PostDamagePostTech_model = load_model(
        PostDamagePostTechModel,
        7,
        v_nn_checkpoint_PostDamageIntermTech_path,
        i_g_nn_checkpoint_PostDamageIntermTech_path,
        i_d_nn_checkpoint_PostDamageIntermTech_path,
    )
except Exception as e:
    print("Warning: could not load PostDamageIntermTechModel:", e)
    post_tech_interm_model = None



# PostDamageIntermTech
try:
    from PostDamageIntermTech import PostDamageIntermTechModel
    PostDamageIntermTech_model = load_model(
        PostDamageIntermTechModel,
        8,
        v_nn_checkpoint_PostDamageIntermTech_path,
        i_g_nn_checkpoint_PostDamageIntermTech_path,
        i_d_nn_checkpoint_PostDamageIntermTech_path,
        i_r_nn_checkpoint_PostDamageIntermTech_path,
    )
except Exception as e:
    print("Warning: could not load PostDamageIntermTechModel:", e)
    post_tech_interm_model = None

# PostDamagePreTech
try:
    from PostDamagePreTech import PostDamagePreTechModel
    PostDamagePreTech_model = load_model(
        PostDamagePreTechModel,
        8,
        v_nn_checkpoint_PostDamagePreTech_path,
        i_g_nn_checkpoint_PostDamagePreTech_path,
        i_d_nn_checkpoint_PostDamagePreTech_path,
        i_r_nn_checkpoint_PostDamagePreTech_path,
    )
except Exception as e:
    print("Warning: could not load PostDamagePreTechModel:", e)
    post_tech_pre_damage_model = None

# PreDamageIntermTech
try:
    from PreDamageIntermTech import PreDamageIntermTechModel
    PreDamageIntermTech_model = load_model(
        PreDamageIntermTechModel,
        7,
        v_nn_checkpoint_PreDamageIntermTech_path,
        i_g_nn_checkpoint_PreDamageIntermTech_path,
        i_d_nn_checkpoint_PreDamageIntermTech_path,
        i_r_nn_checkpoint_PreDamageIntermTech_path,
    )
except Exception as e:
    print("Warning: could not load PreDamageIntermTechModel:", e)
    pre_tech_interm_model = None

# PreDamagePostTech
try:
    from PreDamagePostTech import PreDamagePostTechModel
    PreDamagePostTech_model = load_model(
        PreDamagePostTechModel,
        6,
        v_nn_checkpoint_PreDamagePostTech_path,
        i_g_nn_checkpoint_PreDamagePostTech_path,
        i_d_nn_checkpoint_PreDamagePostTech_path,
    )
except Exception as e:
    print("Warning: could not load PreDamagePostTechModel:", e)
    pre_tech_post_damage_model = None

# PreDamagePreTech
try:
    from PreDamagePreTech import PreDamagePreTechModel
    # PreDamagePreTechModel was already imported at the top, reuse it if present
    PreDamagePreTech_model = load_model(
        PreDamagePreTechModel,
        7,
        v_nn_checkpoint_PreDamagePreTech_path,
        i_g_nn_checkpoint_PreDamagePreTech_path,
        i_d_nn_checkpoint_PreDamagePreTech_path,
        i_r_nn_checkpoint_PreDamagePreTech_path,
    )
except Exception as e:
    print("Warning: could not load PreDamagePreTechModel:", e)
    pre_tech_pre_damage_model = None



A_d = PARAMS['A_d']; A_g = PARAMS['A_g']
A_g_prime = PARAMS['A_g_prime']
A_g_prime_prime = PARAMS['A_g_prime_prime']
α_d = PARAMS['α_d']; Γ_d = PARAMS['Γ_d']; θ_d = PARAMS['θ_d']; σ_d = PARAMS['σ_d']
α_g = PARAMS['α_g']; Γ_g = PARAMS['Γ_g']; θ_g = PARAMS['θ_g']; σ_g = PARAMS['σ_g']
ζ   = PARAMS['ζ'];   ψ0  = PARAMS['ψ0'];  ψ1  = PARAMS['ψ1'];  σ_κ = PARAMS['σ_κ']
θ_bar = PARAMS['θ_bar']; η = PARAMS['η'];  ϛ = PARAMS['ϛ']
varrho = PARAMS['varrho']
π = PARAMS['π']
λ1 = PARAMS['λ1']
λ2 = PARAMS['λ2']
L = PARAMS['L']
λ3_values = PARAMS['λ3_values']
r1 = PARAMS['r1']
r2 = PARAMS['r2']
y_lower = PARAMS['y_lower']
y_upper = PARAMS['y_upper']


dt = 1/12
scale = tf.sqrt(dt)


def iterate_state(logK, Z, logR,  Y,  λ3, whether_Tech_jump, whether_Damage_jump):
 
      
    ## Sampling the Weineer Processes
    increments = tf.random.normal(shape=[4], mean=0.0, stddev=1.0) * scale
    dW_g, dW_d, dW_Y, dW_logR = tf.unstack(increments)
    
    
    
    sigma_K_d      = σ_d * (1-Z)
    sigma_K_g      = σ_g * Z

    sigma_Z_d      = - σ_d * Z *(1-Z)
    sigma_Z_g      =   σ_g * Z *(1-Z)

    sigma_logR      =   σ_κ

    sigma_Y = θ_bar * η * A_d *(1-Z) * tf.exp(logK  )
        

    # Placeholder update rules for the (0,0) case: no jumps.
    if whether_Tech_jump == 0 and whether_Damage_jump == 0:
        
        model = PreDamagePreTech_model
        
      
        state_1d = tf.stack([
            tf.reshape(logK, []),
            tf.reshape(Z, []),
            tf.reshape(Y, []),
            tf.reshape(logR, []),
            tf.reshape(logξ , []),
            tf.reshape(logξ , []),
            tf.reshape(logξ , [])
        ], axis=0)  # shape (7,)

        state = tf.reshape(state_1d, (1, 7))
        
        
        
        K=   tf.exp(logK  )
        
        ### Controls
        i_g        = model.i_g_nn(state)
        i_d        = model.i_d_nn(state)
        i_r        = tf.exp(-model.i_r_nn(state)) 
            
        ### Increments
          
        
        v_kk_term = (σ_d**2 * (1 - Z)**2 + σ_g**2 * Z**2) / 2.0
        inside_log_i_d   = tf.reshape(tf.math.maximum(1.0 + θ_d * i_d, 1e-8), [1, 1])
        inside_logR   = tf.reshape(tf.math.maximum(1.0 + θ_g * i_g, 1e-8), [1, 1])
  
        v_logK_term = (α_d + Γ_d * tf.math.log(inside_log_i_d)) * (1 - Z) \
                      + (α_g + Γ_g * tf.math.log(inside_logR)) * Z  \
                      - v_kk_term

        v_Z_term = (α_g + Γ_g * tf.math.log(inside_logR)   \
            - (α_d + Γ_d * tf.math.log(inside_log_i_d)) \
            - Z * σ_g**2\
            + (1-Z) * σ_d**2) * Z * (1 - Z)
        
 

        v_y_term =  θ_bar   * η * A_d * (1 - Z) * K 
                
                
        v_logR_term = - ζ + ψ0 * tf.exp( ψ1  *   ( tf.math.log(i_r) +logK -  logR) )  + 0.5 * σ_κ**2     
 
 
        ## dW_K, dW_R, dW_Y, dW_logR
        new_logK       = logK + v_logK_term * dt +   sigma_K_d  * dW_d     +     sigma_K_g* dW_g
        
        new_Z          = Z + v_Z_term * dt   +   sigma_Z_d  * dW_d     +     sigma_Z_g* dW_g
        
        new_logR    = logR + v_logR_term * dt  + sigma_logR * dW_logR
        
        new_Y          = Y +  v_y_term * dt  + sigma_Y* dW_Y
 
        #### Sample Whether Jump
        J_d = r1 * ( tf.exp( r2 / 2 * tf.pow(Y - y_lower ,2) ) - 1  ) *  tf.cast(Y > y_lower, tf.float32 )

        # I_d   = model.params['r_1'] * ( tf.exp( model.params['r_2'] / 2 * tf.pow(new_Y - model.params['y_lower_bar'],2) ) - 1  ) * \
        #     tf.cast(new_Y > model.params['y_lower_bar'], tf.float32 )
        
        p_jump = 1 - tf.exp(- J_d * dt)
        
        # Determine if a jump occurs
        jump_occur = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), p_jump)  # This is a boolean tensor
        # Update the whether jump indicator
        whether_Damage_jump+= tf.cast(jump_occur, tf.int32)
 
        λ3_list = np.linspace(model.params["λ3_min"], model.params["λ3_max"], model.params["L"]).tolist()
     
        # Convert λ3_list to a TensorFlow tensor for sampling
        λ3_tensor = tf.convert_to_tensor(λ3_list, dtype=tf.float32)
    
        # Sample an index uniformly among the λ3_list entries
        idx = tf.random.uniform(shape=[], minval=0, maxval=model.params["L"], dtype=tf.int32)
        sampled_val = λ3_tensor[idx]
    
        # If a jump occurs, use the sampled λ3 value; if not, return a default value (e.g., 0.0)
        λ3 = tf.cond(jump_occur, lambda: sampled_val, lambda: tf.constant(0.0))
        new_Y = tf.cond(jump_occur, lambda: tf.constant(model.params["y_bar"]), lambda: new_Y)

        #### Tech jump
        J_g_prime  = (1- π) * tf.exp(logR) / varrho ; J_g_prime_prime  =  π * tf.exp(logR) / varrho 
        
        
        
        J_tech = tf.exp(logR) / varrho 
        p_jump = 1 - tf.exp(-I_tech * dt)
         
        # Determine if a jump occurs
        jump_occur = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), p_jump)  # This is a boolean tensor
        
        # Initialize indicator (default no jump)
        whether_Tech_jump = tf.constant(0, dtype=tf.int32)

        # If jump occurs, draw again from probabilities [1-pi, pi]
        def jump_case():
            second_draw = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
            return tf.cond(
                second_draw < (1.0 - pi),
                lambda: tf.constant(1, dtype=tf.int32),
                lambda: tf.constant(2, dtype=tf.int32)
            )

        # Update the indicator
        whether_Tech_jump = tf.cond(jump_occur, jump_case, lambda: whether_Tech_jump)
        
    # Placeholder update rules for the (0,1) case: damage jump only.
    elif whether_Tech_jump == 0 and whether_Damage_jump == 1:

        
        model = pre_tech_post_damage_model        
        
   
        
        state_1d = tf.stack([
            tf.reshape(logK, []),
            tf.reshape(Z, []),
            tf.reshape(Y, []),
            tf.reshape(logR, []),
             tf.reshape(λ3, []),
            tf.reshape(logξ , []),
            tf.reshape(logξ , []),
            tf.reshape(logξ , [])
        ], axis=0)   

        state = tf.reshape(state_1d, (1, 8))
        
        
        K=   tf.exp(logK  )
        
        ### Controls
        i_g        = model.i_g_nn(state)
        i_d        = model.i_d_nn(state)
        i_I        = model.i_I_nn(state)
            
        ### Increments
         
 
        
        v_kk_term = ( tf.pow(model.params["sigma_d"],2) * tf.pow(1-Z,2)  + tf.pow(model.params["sigma_g"],2) * tf.pow(Z,2))/2.0

        inside_log_i_d   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_d"] * i_d , 0.0001), [1, 1])
        inside_logR   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_g"] * i_g , 0.0001), [1, 1])

        v_k_term       = ( model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( model.params["alpha_g"] +  model.params["Gamma"] * tf.math.log( inside_logR) ) * R  - v_kk_term
        
        v_r_term       = ( model.params["alpha_g"] + model.params["Gamma"] * tf.math.log( inside_logR )  - (model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(model.params["sigma_d"],2) *  (1-R ) - 
                        tf.pow(model.params["sigma_g"], 2) *  Z ) *  Z * (1 - Z)
        
     
        v_y_term       = model.params["beta_f"] * (model.params["eta"] *  model.params["A_d"] * (1-R) * tf.exp(logK  ))

        
        v_I_g_term     = - model.params["zeta"] + model.params["psi_0"] * tf.exp(-i_I * model.params["psi_1"]) * tf.exp( model.params["psi_1"] * (logK -  logR) ) - 0.5 * tf.pow(model.params["sigma_I"], 2)

    
        ## dW_K, dW_R, dW_Y, dW_logR
        new_logK       = logK + v_k_term * dt +   sigma_K_d  * dW_d     +     sigma_K_g* dW_g
        
        new_Z          = Z + v_r_term * dt   +   sigma_R_d  * dW_d     +     sigma_R_g* dW_g
        
        new_logR    = logR + v_I_g_term * dt  + sigma_I_g * dW_logR
        
        new_Y          = Y +  v_y_term * dt  + sigma_Y* dW_Y


        #### Tech jump
        I_tech = tf.exp(new_logR) / model.params["varrho"]
        p_jump = 1 - tf.exp(-I_tech * dt)
         
        # Determine if a jump occurs
        jump_occur = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), p_jump)  # This is a boolean tensor
        # Update the whether jump indicator
        whether_Tech_jump+= tf.cast(jump_occur, tf.int32)
  
        # Convert λ3_list to a TensorFlow tensor for sampling
        A_g_prime_tensor = tf.convert_to_tensor(model.params["A_g_prime_list"], dtype=tf.float32)
    
        # Sample an index uniformly among the λ3_list entries
        idx = tf.random.uniform(shape=[], minval=0, maxval=model.params["A_g_prime_length"], dtype=tf.int32)
        sampled_val = A_g_prime_tensor[idx]
    
        # If a jump occurs, use the sampled λ3 value; if not, return a default value (e.g., 0.0)
        A_g_prime = tf.cond(jump_occur, lambda: sampled_val, lambda: tf.constant(0.0))
        new_logR = tf.cond(jump_occur, lambda: tf.constant(0.0), lambda: new_logR)
        
      
    
    elif whether_Tech_jump == 0.5 and whether_Damage_jump == 1:
        
        raise ValueError("Invalid state: whether_Tech_jump cannot be 0.5 when whether_Damage_jump is 0.")
    
    
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
        inside_logR   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_g"] * i_g , 0.0001), [1, 1])

        v_k_term       = ( model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( model.params["alpha_g"] +  model.params["Gamma"] * tf.math.log( inside_logR) ) * R  - v_kk_term
        
        v_r_term       = ( model.params["alpha_g"] + model.params["Gamma"] * tf.math.log( inside_logR )  - (model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(model.params["sigma_d"],2) *  (1-R ) - 
                        tf.pow(model.params["sigma_g"], 2) *  R ) *  R * (1 - R)
        
     
        v_y_term       = model.params["beta_f"] * (model.params["eta"] *  model.params["A_d"] * (1-R) * tf.exp(logK  ))

        
      
    
        ## dW_K, dW_R, dW_Y, dW_logR
        new_logK       = logK + v_k_term * dt +   sigma_K_d  * dW_d     +     sigma_K_g* dW_g
        
        new_R          = R + v_r_term * dt   +   sigma_R_d  * dW_d     +     sigma_R_g* dW_g
        
        new_logR    = tf.constant(0.0)
        
        new_Y          = Y +  v_y_term * dt  + sigma_Y* dW_Y

        #### Sample Whether Jump
        I_d   = model.params['r_1'] * ( tf.exp( model.params['r_2'] / 2 * tf.pow(new_Y - model.params['y_lower_bar'],2) ) - 1  ) * \
            tf.cast(new_Y > model.params['y_lower_bar'], tf.float32 )
        
        p_jump = 1 - tf.exp(-I_d * dt)
        
        # Determine if a jump occurs
        jump_occur = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), p_jump)  # This is a boolean tensor
        # Update the whether jump indicator
        whether_Damage_jump+=tf.cast(jump_occur, tf.int32) 
 
        λ3_list = np.linspace(model.params["λ3_min"], model.params["λ3_max"], model.params["λ3_length"]).tolist()
     
        # Convert λ3_list to a TensorFlow tensor for sampling
        λ3_tensor = tf.convert_to_tensor(λ3_list, dtype=tf.float32)
    
        # Sample an index uniformly among the λ3_list entries
        idx = tf.random.uniform(shape=[], minval=0, maxval=model.params["λ3_length"], dtype=tf.int32)
        sampled_val = λ3_tensor[idx]
    
        # If a jump occurs, use the sampled λ3 value; if not, return a default value (e.g., 0.0)
        λ3 = tf.cond(jump_occur, lambda: sampled_val, lambda: tf.constant(0.0))
        new_Y = tf.cond(jump_occur, lambda: tf.constant(model.params["y_bar"]), lambda: new_Y)
   
    elif whether_Tech_jump == 0.5 and whether_Damage_jump == 1:
        raise ValueError("Invalid state: whether_Tech_jump cannot be 0.5 when whether_Damage_jump is 1.")       
    # Placeholder update rules for the (1,1) case: both technology and damage jump.
    elif whether_Tech_jump == 1 and whether_Damage_jump == 1:
        model = post_tech_post_damage_model        
 
        
        state_1d = tf.stack([
            tf.reshape(logK, []),
            tf.reshape(R, []),
            tf.reshape(Y, []),
            tf.reshape(λ3, []),
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
        inside_logR   =   tf.reshape( tf.math.maximum( 1 + model.params["phi_g"] * i_g , 0.0001), [1, 1])

        v_k_term       = ( model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( model.params["alpha_g"] +  model.params["Gamma"] * tf.math.log( inside_logR) ) * R  - v_kk_term
        
        v_r_term       = ( model.params["alpha_g"] + model.params["Gamma"] * tf.math.log( inside_logR )  - (model.params["alpha_d"] + model.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(model.params["sigma_d"],2) *  (1-R ) - 
                        tf.pow(model.params["sigma_g"], 2) *  R ) *  R * (1 - R)
        
     
        v_y_term       = model.params["beta_f"] * (model.params["eta"] *  model.params["A_d"] * (1-R) * tf.exp(logK  ))

        
      
    
        ## dW_K, dW_R, dW_Y, dW_logR
        new_logK       = logK + v_k_term * dt +   sigma_K_d  * dW_d     +     sigma_K_g* dW_g
        
        new_R          = R + v_r_term * dt   +   sigma_R_d  * dW_d     +     sigma_R_g* dW_g
        
        new_logR    = tf.constant(0.0)
        
        new_Y          = Y +  v_y_term * dt  + sigma_Y* dW_Y
        
    else:
        raise ValueError("Invalid jump condition values. They should be either 0 or 1.")

    return new_logK, new_Z, new_logR, new_Y,   λ3, whether_Tech_jump, whether_Damage_jump



dt = 1/12                   # one month per time step
scale = tf.sqrt(dt)         # scale for the Wiener increments
T = 60 * 12                 # total time steps for 60 years (720 steps)
n_paths = 100               # number of simulation paths

# --- Initial values ---
init_logK     = tf.math.log(880.0)              # log capital stock
init_Z        = tf.constant(0.7, dtype=tf.float32)
init_logR  = tf.math.log(11.2)
init_Y        = tf.constant(1.1, dtype=tf.float32) 
init_λ3  = tf.constant(0.0, dtype=tf.float32)
init_TechJump = tf.constant(0, dtype=tf.int32)
init_DmgJump  = tf.constant(0, dtype=tf.int32)


# --- Simulation storage lists ---
logK_sim        = []
Z_sim           = []
logR_sim     = []
Y_sim           = []
A_g_prime_sim   = [] 
λ3_sim     = [] 
tech_jump_sim   = []
damage_jump_sim = []
  
# --- Simulation loop over n_paths ---
for sim in range(n_paths):
    # Initialize the current state values for the simulation path
    logK_t      = init_logK
    Z_t         = init_Z
    logR_t   = init_logR
    Y_t         = init_Y 
    λ3_t   = init_λ3
    tech_jump_t = init_TechJump
    dmg_jump_t  = init_DmgJump
    
    # Lists to store the simulation path for each variable
    logK_path      = []
    Z_path         = []
    logR_path   = []
    Y_path         = [] 
    λ3_path   = []
    tech_jump_path = []
    damage_jump_path = []
    
    # Record the initial state (convert tensors to scalars)
    logK_path.append(float(logK_t.numpy()))
    Z_path.append(float(Z_t.numpy()))
    logR_path.append(float(logR_t.numpy()))
    Y_path.append(float(Y_t.numpy())) 
    
    # Run simulation for T time steps
    for t in range(T):
        (logK_t,
         Z_t,
         logR_t,
         Y_t, 
         λ3_t,
         tech_jump_t,
         dmg_jump_t) = iterate_state(
                          logK_t,
                          Z_t,
                          logR_t,
                          Y_t, 
                          λ3_t,
                          tech_jump_t,
                          dmg_jump_t)
                          
        # Record state variables at the current time step
        logK_path.append(float(logK_t.numpy()))
        Z_path.append(float(Z_t.numpy()))
        logR_path.append(float(logR_t.numpy()))
        Y_path.append(float(Y_t.numpy())) 
        λ3_path.append(float(λ3_t.numpy()))
        tech_jump_path.append(int(tech_jump_t.numpy()))
        damage_jump_path.append(int(dmg_jump_t.numpy()))
    
    # Convert the lists to NumPy arrays (with uniform scalar entries)
    logK_sim.append(np.array(logK_path, dtype=float))
    Z_sim.append(np.array(Z_path, dtype=float))
    logR_sim.append(np.array(logR_path, dtype=float))
    Y_sim.append(np.array(Y_path, dtype=float))
    tech_jump_sim.append(np.array(tech_jump_path, dtype=int))
    damage_jump_sim.append(np.array(damage_jump_path, dtype=int))  
    
    

 


logK_array    = np.stack(logK_sim, axis=0)
Z_array       = np.stack(Z_sim, axis=0)
logR_array = np.stack(logR_sim, axis=0)
Y_array       = np.stack(Y_sim, axis=0)

tech_jump_array = np.stack(tech_jump_sim, axis=0)
damage_jump_array= np.stack(damage_jump_sim, axis=0)

 
np.save(output_folder+f'logK_sim_{seed}.npy', logK_array)
np.save(output_folder+f'Z_array_{seed}.npy', Z_array)
np.save(output_folder+f'logR_array_{seed}.npy', logR_array)
np.save(output_folder+f'Y_array_{seed}.npy', Y_array)
np.save(output_folder+f'tech_jump_array_{seed}.npy', tech_jump_array)
np.save(output_folder+f'damage_jump_array_{seed}.npy', damage_jump_array)


 
 