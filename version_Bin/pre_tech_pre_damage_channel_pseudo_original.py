##################################################################
##################################################################
#### This program solves pre-tech post-damage models
####
##################################################################
##################################################################
##################################################################

import model_pseudostate_original as model
import numpy as np
import tensorflow as tf 
import sys 
import pathlib
import json 

#############################################
#############################################
#############################################
#############################################
#############################################
export_folder                    = sys.argv[1]
post_tech_pre_damage_export_folder = sys.argv[2]
pre_tech_post_damage_export_folder = sys.argv[3]
steps                            = int(round(float(sys.argv[4])))
log_xi_min                       = float(sys.argv[5])
log_xi_max                       = float(sys.argv[6])
batch_size                       = int(sys.argv[7])
num_iterations                   = int(sys.argv[8])
pretrained_path                  = sys.argv[9]
logging_frequency                = int(sys.argv[10])
learning_rates                   = [float(x) for x in sys.argv[11].split(",")]
hidden_layer_activations         = sys.argv[12].split(",")
output_layer_activations         = sys.argv[13].split(",")
num_hidden_layers                = int(sys.argv[14])
num_neurons                      = int(sys.argv[15])
learning_rate_schedule_type      = sys.argv[16]
delta                            = float(sys.argv[17])

## Take care of tensorboard
if len(sys.argv) > 18:
    tensorboard = sys.argv[18] == "True"
else:
    tensorboard = False 

export_folder_output             = sys.argv[19]
log_xi_baseline_min              = float(sys.argv[20])
log_xi_baseline_max              = float(sys.argv[21])
channel_type                     = sys.argv[22]
A_g_prime_min                    = float(sys.argv[23])
A_g_prime_max                    = float(sys.argv[24])

A_g_prime_length                 = int(sys.argv[25])
gamma_3_length                   = int(sys.argv[26])


## Take care of pretrained path
if pretrained_path == "None":
    pretrained_path = None



#############################################
## Part 2
## Solve pre tech pre damage model
#############################################

## This model has three state variables

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
params = {"batch_size" : batch_size, "learning_rates":learning_rates,
"v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config,  "i_I_nn_config" : i_I_nn_config,\
"model_type" : "pre_tech_pre_damage" , "n_dims": 4,\
"num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, "load_parameters" : None,
"pretrained_path" : pretrained_path, 'tensorboard' : tensorboard, "learning_rate_schedule_type" : learning_rate_schedule_type , "channel_type": channel_type }
 
phi_g = 16.7
phi_d = 16.7
if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_g) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_d) / (tf.exp(2 * x) + 1.0)



## This model has four state variables. 
## Add in paramters associated with this 4d model 

 
params["v_pre_tech_post_damage_nn_path"]  = export_folder + "/pre_tech_post_damage"  
params["v_post_tech_pre_damage_nn_path"]  = export_folder + "/post_tech_pre_damage"
params["export_folder"]                   = export_folder + "/pre_tech_pre_damage"
 
 
##########
test_model = model.model(params)
test_model.export_parameters()
test_model.train()
# test_model.analyze()

log_xi_list = [float(np.log(0.075)),float(np.log(0.1)), float(np.log(0.3)), float(5.0) ]


# if channel_type == "baseline":
#     for log_xi_baseline_idx in range(len(log_xi_baseline_list)):
#     # test_model.simulate_path(60, 1.0 / 12.0, log_xi_list[log_xi_idx], export_folder + "/output/pre_damage_pre_tech/log_xi_idx_" + str(log_xi_idx))
#         test_model.simulate_path(60, 1.0 / 12.0, log_xi_min, log_xi_baseline_list[log_xi_baseline_idx], export_folder + "/final_output/pre_tech_pre_damage/log_xi_idx_" + str(log_xi_baseline_idx))
# else:
for log_xi_idx in range(len(log_xi_list)):
    # test_model.simulate_path(60, 1.0 / 12.0, log_xi_list[log_xi_idx], export_folder + "/output/pre_damage_pre_tech/log_xi_idx_" + str(log_xi_idx))
    test_model.simulate_path(60, 1.0 / 12.0, log_xi_list[log_xi_idx], log_xi_baseline_min, export_folder + "/final_output/pre_tech_pre_damage/log_xi_idx_" + str(log_xi_idx))
