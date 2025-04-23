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

## Load parameters
export_folder                    = sys.argv[1]
log_xi_min                       = float(sys.argv[2])
log_xi_max                       = float(sys.argv[3])
batch_size                       = int(sys.argv[4])
num_iterations                   = int(sys.argv[5])
pretrained_path                  = sys.argv[6]
logging_frequency                = int(sys.argv[7])
learning_rates                   = [float(x) for x in sys.argv[8].split(",")]
hidden_layer_activations         = sys.argv[9].split(",")
output_layer_activations         = sys.argv[10].split(",")
num_hidden_layers                = int(sys.argv[11])
num_neurons                      = int(sys.argv[12])
learning_rate_schedule_type      = sys.argv[13]
delta                            = float(sys.argv[14])

if len(sys.argv) > 15:
    tensorboard = sys.argv[15] == "True"
else:
    tensorboard = False 

export_folder_output             = sys.argv[16]
log_xi_baseline_min              = float(sys.argv[17])
log_xi_baseline_max              = float(sys.argv[18])
channel_type                     = sys.argv[19]

A_g_prime_min                    = float(sys.argv[20])
A_g_prime_max                    = float(sys.argv[21])

A_g_prime_length                 = int(sys.argv[22])
gamma_3_length                   = int(sys.argv[23])



## Take care of pretrained path
if pretrained_path == "None":
    pretrained_path = None
else:
    pretrained_path = pretrained_path 
## Take care of activation functions 
hidden_layer_activations   = [None if x == "None" else x for x in hidden_layer_activations]
output_layer_activations   = [None if x == "None" else x for x in output_layer_activations]

#############################################
## Part 1
## Solve post tech post damage model
#############################################

## This model has three state variables

v_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_nn"}
v_nn_config["final_activation"] = output_layer_activations[0]

i_g_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[1], "dim" : 1, "nn_name" : "i_g_nn"}
i_g_nn_config["final_activation"] = output_layer_activations[1]

i_d_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[2], "dim" : 1, "nn_name" : "i_d_nn"}
i_d_nn_config["final_activation"] = output_layer_activations[2]

i_I_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[3], "dim" : 1, "nn_name" : "i_I_nn"}
i_I_nn_config["final_activation"] = output_layer_activations[3]


## Create params struct 
params = {"batch_size" : batch_size,  "learning_rates":learning_rates, \
"v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config, "i_I_nn_config" :i_I_nn_config,\
"model_type" : "pre_tech_post_damage" ,"n_dims":4,\
"num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, "load_parameters" : None,
"pretrained_path" : pretrained_path, 'tensorboard' : tensorboard, "learning_rate_schedule_type" : learning_rate_schedule_type , "channel_type": channel_type }

phi_g = 16.7
phi_d = 16.7
## i_g and i_d activations come after params because we amy want to use phi_g and phi_d
if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_g) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ phi_d) / (tf.exp(2 * x) + 1.0)


#############################################
## Part 2
## Solve pre tech post damage model
#############################################

## This model has four state variables. 
## Add in paramters associated with this 4d model 
  


params["v_post_tech_post_damage_nn_path"] = export_folder + "/post_tech_post_damage"
params["export_folder"]         = export_folder + "/pre_tech_post_damage" 
params["pretrained_path"]       = pretrained_path


test_model = model.model(params)
test_model.export_parameters()
test_model.train()
# test_model.analyze()
