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


'''
export_folder = 'test_grad'
log_xi_min = -3.0
log_xi_max = -1.5
batch_size = 16
num_iterations = 10
A_g_prime = 0.15
pretrained_path = "None"
logging_frequency = 1
'''



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

# print(A_g_prime_num)        


## Create params struct 
params = {"batch_size" : batch_size, "R_min" : 0.01, \
"R_max" : 0.99, "logK_min" : 4.0,\
"logK_max" : 7.0, "Y_min" : 10e-3, "Y_max" : 3.0, \
"log_I_g_max" : 6.0, "log_I_g_min": 1.0, \
"sigma_d" : 0.15 , "sigma_g" : 0.15, "A_d" : 0.12, "A_g_prime_min" : A_g_prime_min, "A_g_prime_max" : A_g_prime_max, "A_g_prime_length" : A_g_prime_length, \
"gamma_1" : 0.00017675, "gamma_2" : 2 * 0.0022, "gamma_3_min" : 0.0, "gamma_3_max" : 1.0/3.0, "gamma_3_length" : gamma_3_length,  \
"y_bar" : 2.0, "beta_f" : 1.86 / 1000, "eta" : 0.17, \
"varsigma" : 1.2 * 1.86 / 1000, "phi_d" : 100.0,  "phi_g" : 100.0, "Gamma" : 0.025,  \
"alpha_d" : -0.0236, "alpha_g" : -0.0236, "delta" : delta, \
"v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config, \
"n_dims" : 3, "model_type" : "post_tech_post_damage" , \
"num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, "load_parameters" : None, "norm_weight" : 0.9,
"log_xi_min" : log_xi_min, "log_xi_max" : log_xi_max, "log_xi_baseline_min" : log_xi_baseline_min, "log_xi_baseline_max" : log_xi_baseline_max, "pretrained_path" : pretrained_path, 'tensorboard' : tensorboard, "learning_rate_schedule_type" : learning_rate_schedule_type, "channel_type": channel_type }

# params["A_g_prime"]       = np.linspace(params["A_g_prime_min"], params["A_g_prime_max"], params["A_g_prime_length"])[params["A_g_prime_num"]]


if params["learning_rate_schedule_type"] == "None":
    lr_schedulers = learning_rates
    params["optimizers"] = [tf.keras.optimizers.Adam( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "piecewiseconstant":
    boundaries            = [int(round(x)) for x in np.linspace(0,num_iterations,5)][1:-1]
    values_list           = [[learning_rate / np.power(2,x) for x in range(len(boundaries)+1)] for learning_rate in learning_rates]
    lr_schedulers         = [ tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values) for values in values_list]
    params["optimizers"] = [tf.keras.optimizers.Adam( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "sgd+piecewiseconstant":
    boundaries            = [int(round(x)) for x in np.linspace(0,num_iterations,5)][1:-1]
    values_list           = [[learning_rate / np.power(2,x) for x in range(len(boundaries)+1)] for learning_rate in learning_rates]
    lr_schedulers         = [ tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values) for values in values_list]
    params["optimizers"] = [tf.keras.optimizers.legacy.SGD( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "sgd":
    lr_schedulers = learning_rates
    params["optimizers"] = [tf.keras.optimizers.legacy.SGD( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]

params["load_solution"]  = "model_results.json"
params["export_folder"]  = export_folder +  "/post_tech_post_damage"
params["model_type"]            = "post_tech_post_damage"

## i_g and i_d activations come after params because we amy want to use phi_g and phi_d
if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_g"]) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_d"]) / (tf.exp(2 * x) + 1.0)

#######################################################
##### Check if post-tech post-damage is already trained
#######################################################
if not (pathlib.Path(params["export_folder"]+ "/v_nn_checkpoint_post_tech_post_damage"+".index").is_file()):
    ## Model has not yet been trained
    print("Post-tech post-jump model has not yet been trained. Trainning now... ")
    test_model = model.model(params)
    test_model.export_parameters()
    test_model.train()
    # test_model.analyze()

    # log_xi_list                  = [float(np.log(xi)) for xi in np.linspace(np.exp(log_xi_min) + 0.02, np.exp(log_xi_max) - 0.02, 10)]

    # for log_xi_idx in range(len(log_xi_list)):
    #     test_model.simulate_path_post_tech_post_jump(60, 1.0 / 12.0, 
    #                                                     test_model.params["gamma_3"], 
    #                                                     log_xi_list[log_xi_idx], export_folder_output + "/output/post_tech_post_damage/log_xi_idx_" + str(log_xi_idx))
else:
    print("Post-tech post-jump model has been trained. ")
    test_model = model.model(params)

    # log_xi_list                  = [float(np.log(xi)) for xi in np.linspace(np.exp(log_xi_min) + 0.02, np.exp(log_xi_max) - 0.02, 10)]

    # for log_xi_idx in range(len(log_xi_list)):
    #     test_model.simulate_path_post_tech_post_jump(60, 1.0 / 12.0, 
    #                                                     test_model.params["gamma_3"], 
    #                                                     log_xi_list[log_xi_idx], np.exp(10.25), export_folder_output + "/output/post_tech_post_damage/log_xi_idx_" + str(log_xi_idx))
        
