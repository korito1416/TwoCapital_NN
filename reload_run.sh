#!/bin/sh




#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G 

module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

prefix="/project/lhansen/Cap_damage/TwoCapital_NN/output"
 
channel_type="full"
range_type=0.1
batch_size="128"
SLURM_NTASKS=2




# batch_size="128"
# batch_size="256"
# batch_size="512"
# batch_size="1024"

num_neurons="32"

log_xi_min="-3.0"
log_xi_max="5.0"
log_xi_baseline_min="-3.0"
log_xi_baseline_max="5.0"
num_iterations1="2000000"
num_iterations2="2000000"




# A_g_prime_num="0.15"
logging_frequency="1000"
learning_rates="10e-7,10e-7,10e-7,10e-7"
# learning_rates="10e-5,10e-5,10e-5,10e-5"
# learning_rates="40e-7,40e-7,40e-7,40e-7"
# learning_rates="40e-5,40e-5,40e-5,40e-5"
hidden_layer_activations="swish,tanh,tanh,softplus"
output_layer_activations="softplus,custom,custom,softplus"
num_hidden_layers="4"
learning_rate_schedule_type="piecewiseconstant"   
# learning_rate_schedule_type="None"
delta="0.01"
tensorboard='True'
A_g_prime_length=3
gamma_3_length=20


if [ "${range_type}" -eq 1 ]; then

echo "range_type=${range_type}"
A_g_prime_min=0.10
A_g_prime_max=0.20
# foldername="Pseudo_model_1m2m_${channel_type}_widerange_${batch_size}"
foldername="Pseudo_model_1m2m_${channel_type}_widerange_${batch_size}_neurons_${num_neurons}"

else

echo "range_type=${range_type}"

# A_g_prime_min=0.19
# A_g_prime_max=0.22
A_g_prime_min=0.12
A_g_prime_max=0.13

# foldername= One_denom_pretrained_dVdY_Delta_0.01_LR_piecewiseconstant_10e-5,10e-5,10e-5,10e-5_128_neurons_32_#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations2000000
foldername="One_denom_pretrained_dVdY_Delta_${delta}_LR_${learning_rate_schedule_type}_${learning_rates}_${batch_size}_neurons_${num_neurons}_#HiddenLayer_${num_hidden_layers}_logxi_${log_xi_min}_logximax_${log_xi_max}_num_iterations${num_iterations1}"

fi


# foldername="model_test_${channel_type}_range_[$A_g_prime_min,$A_g_prime_max]"

job_name="${prefix}/${foldername}"
jobout_name="${foldername}"
pre_tech_pre_damage_export_folder="${job_name}/pre_tech_pre_damage"
pre_tech_post_damage_export_folder="${job_name}/pre_tech_post_damage"
post_tech_pre_damage_export_folder="${job_name}/post_tech_pre_damage"
post_tech_post_damage_export_folder="${job_name}/post_tech_post_damage"

pretrained_pre_tech_pre_damage_export_folder="None"
# pretrained_pre_tech_pre_damage_export_folder="${job_name}/pre_damage_pre_tech"
pretrained_pre_tech_post_damage_export_folder="None"
pretrained_post_tech_pre_damage_export_folder="None"
pretrained_post_tech_post_damage_export_folder="None"




job_file="pre_tech_pre_damage_reload.job"
job_output_file="pre_tech_pre_damage_reload.out"


echo "#!/bin/bash
#SBATCH --job-name=runtd
#SBATCH --output=${job_output_file}
#SBATCH --error=run.err
#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=21G
module load tensorflow/2.1
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

python version_Bin/reload_net.py $job_name $post_tech_pre_damage_export_folder $pre_tech_post_damage_export_folder -10 $log_xi_min $log_xi_max $batch_size $num_iterations2  $pretrained_pre_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $A_g_prime_min $A_g_prime_max $A_g_prime_length $gamma_3_length" > $job_file

sbatch $job_file
 



