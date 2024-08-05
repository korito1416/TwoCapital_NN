#!/bin/sh


channel_type="baseline"
batch_size=64


num_neurons="32"

prefix="/project/lhansen/Capital_NN/TwoCapital_NN/output"
work_dir="/project/lhansen/Capital_NN/TwoCapital_NN"

A_g_prime_min=0.15
A_g_prime_max=0.15


foldername="Pseudo_model_1m2m_${channel_type}_shortrange_${batch_size}_neurons_${num_neurons}_Agmin_${A_g_prime_min}_Agmax_${A_g_prime_max}"



job_name="${prefix}/${foldername}"
jobout_name="${foldername}"
pre_tech_pre_damage_export_folder="${job_name}/pre_tech_pre_damage"
pre_tech_post_damage_export_folder="${job_name}/pre_tech_post_damage"
post_tech_pre_damage_export_folder="${job_name}/post_tech_pre_damage"
post_tech_post_damage_export_folder="${job_name}/post_tech_post_damage"

pretrained_pre_tech_pre_damage_export_folder="None"
pretrained_pre_tech_post_damage_export_folder="None"
pretrained_post_tech_pre_damage_export_folder="None"
pretrained_post_tech_post_damage_export_folder="None"


log_xi_min="5"
log_xi_max="5.1"

log_xi_baseline_min="5.0"
log_xi_baseline_max="5.1"



num_iterations1="500000"
num_iterations2="500000"


logging_frequency="1000"
learning_rates="10e-5,10e-5,10e-5,10e-5"
hidden_layer_activations="swish,tanh,tanh,softplus"
output_layer_activations="None,custom,custom,softplus"
num_hidden_layers="4"





learning_rate_schedule_type="None"
delta="0.025"
tensorboard='True'
A_g_prime_length=10
gamma_3_length=5





run_script="${work_dir}/run_test.sh"
cat > "${run_script}" <<EOF
#!/bin/bash

#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G  # NOTE DO NOT USE THE --mem= OPTION

# Load the default version of GNU parallel.
module load parallel
# module load tensorflow/2.1
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

python version_Bin/pre_tech_pre_damage_channel_pseudo_original.py $job_name $post_tech_pre_damage_export_folder $pre_tech_post_damage_export_folder -10 $log_xi_min $log_xi_max $batch_size $num_iterations2  $pretrained_pre_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $A_g_prime_min $A_g_prime_max $A_g_prime_length $gamma_3_length

EOF

sbatch "${run_script}"

