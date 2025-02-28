#!/bin/sh




#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=21G 

module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

# prefix="/scratch/midway3/pengyu"
prefix="/project/lhansen/Capital_NN/TwoCapital_NN/output"



# channel_type="full"
# channel_type="baseline"
# channel_type="capital"
# channel_type="technology"
# channel_type="damage"
# channel_type="climate"
channel_type="full"
range_type=0.1
batch_size="128"
SLURM_NTASKS=2
# echo "${channel_type}, ${range_type}, ${batch_size}"

# channel_type="full"
# range_type=0.1
# Agprime_range_list="0.1,0.2"
# logxi_list="-2.3,-2.3,-2.3,-2.3"
# batch_size="128"
# SLURM_NTASKS=2
# sbatch -J rmlogxi_addc_${Agprime_range_list}_xi_${logxi_list}_n_${num_neurons}_b_${batch_size} parallel_Agprime_PseudoState.sbatch $channel_type $range_type $batch_size $SLURM_NTASKS



# A_g_prime_min=0.10
# A_g_prime_max=0.20
# foldername="model_longer_${channel_type}_widerange"

# A_g_prime_min=0.14
# A_g_prime_max=0.16
# foldername="model_longer_${channel_type}_shortrange"



# if [ "${range_type}" -eq 1 ]; then

# echo "range_type=${range_type}"
# A_g_prime_min=0.10
# A_g_prime_max=0.20
# foldername="model_1m2m_${channel_type}_widerange_512"

# else

# echo "range_type=${range_type}"

# A_g_prime_min=0.14
# A_g_prime_max=0.16
# foldername="model_1m2m_${channel_type}_shortrange_512"

# fi



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


# log_xi_min="-2.0"
# log_xi_max="-1.5"
# log_xi_baseline_min="-2.0"
# log_xi_baseline_max="-1.5"
# num_iterations1="2000000"
# num_iterations2="2000000"

# log_xi_min="1.0"
# log_xi_max="1.5"
# log_xi_baseline_min="1.0"
# log_xi_baseline_max="1.5"
# num_iterations1="2000000"
# num_iterations2="2000000"

# log_xi_min="-1.0"
# log_xi_max="-0.5"
# log_xi_baseline_min="-1.0"
# log_xi_baseline_max="-0.5"
# num_iterations1="2000000"
# num_iterations2="2000000"

# log_xi_min="-3.0"
# log_xi_max="-2.5"
# log_xi_baseline_min="-3.0"
# log_xi_baseline_max="-2.5"
# num_iterations1="2000000"
# num_iterations2="2000000"

# log_xi_min="5.0"
# log_xi_max="5.1"
# log_xi_baseline_min="5.0"
# log_xi_baseline_max="5.1"
# num_iterations1="2000000"
# num_iterations2="2000000"

# log_xi_min="10"
# log_xi_max="10.1"
# log_xi_baseline_min="10"
# log_xi_baseline_max="10.1"
# num_iterations1="3000000"
# num_iterations2="3000000"



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

# foldername="Pseudo_model_1m2m_${channel_type}_shortrange_${batch_size}"
foldername="Ag_prime_3states_${channel_type}_shortrange_${batch_size}_neurons_${num_neurons}_Agmin_${A_g_prime_min}_Agmax_${A_g_prime_max}_logxi_${log_xi_min}_logximax_${log_xi_max}_logxibaseline_${log_xi_baseline_min}_num_iterations${num_iterations1}"

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



# A_g_prime_num="0.15"
logging_frequency="1000"
learning_rates="10e-5,10e-5,10e-5,10e-5"
hidden_layer_activations="swish,tanh,tanh,softplus"
output_layer_activations="softplus,custom,custom,softplus"
num_hidden_layers="4"



learning_rate_schedule_type="None"
delta="0.025"
tensorboard='True'
A_g_prime_length=3
gamma_3_length=5

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
 



