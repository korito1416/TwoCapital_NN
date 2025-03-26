#!/bin/sh

#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

# Load the default version of GNU parallel.
module load parallel
# module load tensorflow/2.1
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

# prefix="/scratch/midway3/pengyu"
prefix="/project/lhansen/Cap_damage/TwoCapital_NN/output"

# channel_type="full"
# channel_type="baseline"
# channel_type="capital"
# channel_type="technology"
# channel_type="damage"
# channel_type="climate"
channel_type=$1
range_type=$2
batch_size=$3
SLURM_NTASKS=$4
echo "${channel_type}, ${range_type}, ${batch_size}"

# A_g_prime_min=0.10
# A_g_prime_max=0.20
# foldername="model_longer_${channel_type}_widerange"

# A_g_prime_min=0.14
# A_g_prime_max=0.16
# foldername="model_longer_${channel_type}_shortrange"

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


# num_iterations1="100000"
# num_iterations2="100000"

# A_g_prime_num="0.15"
logging_frequency="1000"
# learning_rates="40e-5,40e-5,40e-5,40e-5"
learning_rates="10e-5,10e-5,10e-5,10e-5"
# learning_rates="40e-7,40e-7,40e-7,40e-7"
# learning_rates="10e-7,10e-7,10e-7,10e-7"
hidden_layer_activations="swish,tanh,tanh,softplus"
output_layer_activations="softplus,custom,custom,softplus"
num_hidden_layers="4"

learning_rate_schedule_type="piecewiseconstant"
# learning_rate_schedule_type="None"
delta="0.01"
tensorboard='True'
A_g_prime_length=1
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

# foldername=One_denom_pretrained_Delta_0.01_LR_None_10e-5,10e-5,10e-5,10e-5_128_neurons_32_#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations2000000
foldername="One_denom_pretrained_Delta_${delta}_LR_${learning_rate_schedule_type}_${learning_rates}_${batch_size}_neurons_${num_neurons}_#HiddenLayer_${num_hidden_layers}_logxi_${log_xi_min}_logximax_${log_xi_max}_num_iterations${num_iterations1}"

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

# pretrain_folder="/project/lhansen/Capital_NN/TwoCapital_NN/output/Four_distortion8_domain_full_shortrange_128_neurons_32_Agmin_0.12_Agmax_0.16_logxi_-3.0_logxibaseline_-3.0_num_iterations2000000"
# pretrained_pre_tech_pre_damage_export_folder="${pretrain_folder}/pre_tech_pre_damage"
# pretrained_pre_tech_post_damage_export_folder="${pretrain_folder}/pre_tech_post_damage"
# pretrained_post_tech_pre_damage_export_folder="${pretrain_folder}/post_tech_pre_damage"
# pretrained_post_tech_post_damage_export_folder="${pretrain_folder}/post_tech_post_damage"

mkdir -p ./job-outs/${jobout_name}/

if [ -f ./job-outs/${jobout_name}/runtask1.log ]; then
    rm ./job-outs/${jobout_name}/runtask1.log
fi

if [ -f ./job-outs/${jobout_name}/runtask2.log ]; then
    rm ./job-outs/${jobout_name}/runtask2.log
fi

if [ -f ./job-outs/${jobout_name}/runtask3.log ]; then
    rm ./job-outs/${jobout_name}/runtask3.log
fi

mkdir -p ./bash/${jobout_name}/

touch ./bash/${jobout_name}/runtask_dmg_tech_prepre.sh
chmod +x ./bash/${jobout_name}/runtask_dmg_tech_prepre.sh

# Create the training script for the last model
tee ./bash/${jobout_name}/runtask_dmg_tech_prepre.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

echo "pre_tech_pre_damage"
python version_Bin/pre_tech_pre_damage_channel_pseudo_original.py $job_name $post_tech_pre_damage_export_folder $pre_tech_post_damage_export_folder -10 $log_xi_min $log_xi_max $batch_size $num_iterations2  $pretrained_pre_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $A_g_prime_min $A_g_prime_max $A_g_prime_length $gamma_3_length

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF

# Directly run the last training script
./bash/${jobout_name}/runtask_dmg_tech_prepre.sh > ./job-outs/${jobout_name}/dmg_tech_prepre.out 2> ./job-outs/${jobout_name}/dmg_tech_prepre.err
