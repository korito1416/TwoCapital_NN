#!/bin/sh

#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=21G

# Load the default version of GNU parallel.
module load parallel
# module load tensorflow/2.1
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

# prefix="/scratch/midway3/pengyu"
# prefix="/project/lhansen/Cap_damage/TwoCapital_NN/output_Scheduler_BestFinal"
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
A_g_prime_length=$5

echo "${channel_type}, ${range_type}, ${batch_size}"


# A_g_prime_min=0.10
# A_g_prime_max=0.20
# foldername="model_longer_${channel_type}_widerange"

# A_g_prime_min=0.14
# A_g_prime_max=0.16
# foldername="model_longer_${channel_type}_shortrange"
 


num_neurons="32"

log_xi_min="-3.0"
log_xi_max="5.0"
log_xi_baseline_min="-3.0"
log_xi_baseline_max="5.0"
num_iterations1="2000000"
# num_iterations1="1"
num_iterations2="2000000"

# num_iterations1="1000000"
# num_iterations2="1000000"
# num_iterations1="200000"
# num_iterations2="200000" 

# num_iterations1="200"
# num_iterations2="20" 

# A_g_prime_num="0.15"
logging_frequency="1000"

learning_rates="40e-3,40e-3,40e-3,40e-3"
learning_rates="60e-3,60e-3,60e-3,60e-3"
learning_rates="80e-3,80e-3,80e-3,80e-3"
learning_rates="10e-4,10e-4,10e-4,10e-4"
learning_rates="40e-4,40e-4,40e-4,40e-4"
# learning_rates="70e-4,70e-4,70e-4,70e-4"
learning_rates="10e-5,10e-5,10e-5,10e-5"
learning_rates="60e-5,60e-5,60e-5,60e-5"
# learning_rates="40e-6,40e-6,40e-6,40e-6"
# learning_rates="10e-7,10e-7,10e-7,10e-7"
# learning_rates="40e-7,40e-7,40e-7,40e-7"
# learning_rates="10e-8,10e-8,10e-8,10e-8"
hidden_layer_activations="swish,tanh,tanh,softplus"
output_layer_activations="softplus,custom,custom,softplus"
num_hidden_layers="4"


learning_rate_schedule_type="piecewiseconstant"
# learning_rate_schedule_type="None"
delta="0.01"
tensorboard='True'
# A_g_prime_length=3
gamma_3_length=20

scheduler_num='4'
scheduler_num='6'
# scheduler_num='8'
# scheduler_num='10'

scheduler_size='2'
scheduler_size='4'
# scheduler_size='6'
# scheduler_size='8'
# scheduler_size='10'

# lambda_A_g_prime=1.063
# lambda_A_g_prime=1.3292


# original parameter settings
Z_0=0.7
A_d=0.12
A_g=0.112857
A_g_prime_list="0.120,0.128,0.136"
eta=0.316


# Z_0=0.3
# A_d=0.12
# A_g=0.112857
# A_g_prime_list="0.120,0.128,0.136"
# eta=0.135



# Z_0=0.5
# A_d=0.13
# A_g=0.1
# A_g_prime_list="0.128,0.136,0.144"
# eta=0.175


# Z_0=0.7
# A_d=0.13
# A_g=0.1086
# A_g_prime_list="0.130,0.139,0.147"
# eta=0.291


# Z_0=0.3
# A_d=0.13
# A_g=0.08
# A_g_prime_list="0.130,0.139,0.147"
# eta=0.125


echo "range_type=${range_type}"

# foldername="model_test_${channel_type}_range_[$A_g_prime_min,$A_g_prime_max]"
# foldername=Cap_damage/TwoCapital_NN/output/One_denom_Delta_0.01_LR_None_40e-7,40e-7,40e-7,40e-7_128_neurons_32_#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations2000000
# foldername="A_g_prime_1.5_${A_g_prime_length}_Delta_${delta}_LR_${learning_rate_schedule_type}_${learning_rates}_${batch_size}_neurons_${num_neurons}_#HiddenLayer_${num_hidden_layers}_logxi_${log_xi_min}_logximax_${log_xi_max}_num_iterations${num_iterations1}"

foldername="Scheduler_num_${scheduler_num}_size_${scheduler_size}_${eta}_Ag_${A_g}_AgPrime_${A_g_prime_list}_eta_${eta}_LR_${learning_rate_schedule_type}_${learning_rates}_num_iterations${num_iterations2}"



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


# pretrain_folder="/project/lhansen/Cap_damage/TwoCapital_NN/output/NewModel_Ag_0.112857_AgPrime_0.120,0.128,0.136_eta_0.316_LR_piecewiseconstant_40e-7,40e-7,40e-7,40e-7_num_iterations200000"
# pretrain_folder="/project/lhansen/Capital_NN/TwoCapital_NN/output/Four_distortion8_domain_full_shortrange_128_neurons_32_Agmin_0.12_Agmax_0.16_logxi_-3.0_logxibaseline_-3.0_num_iterations2000000"
# pretrain_folder="/project/lhansen/Cap_damage/TwoCapital_NN/output_Scheduler_trainPostTech/Scheduler_num_8_size_4_0.316_Ag_0.112857_AgPrime_0.120,0.128,0.136_eta_0.316_LR_piecewiseconstant_10e-4,10e-4,10e-4,10e-4_num_iterations2000000"
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

touch ./bash/${jobout_name}/runtask_dmg_post_tech_post.sh
chmod +x ./bash/${jobout_name}/runtask_dmg_post_tech_post.sh

# tee -a ./bash/${job_name}/runtask_dmg_post_tech_post.sh <<EOF
tee ./bash/${jobout_name}/runtask_dmg_post_tech_post.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python version_Bin/post_tech_post_damage_channel_pseudo_original.py $job_name $log_xi_min $log_xi_max $batch_size $num_iterations1  $pretrained_post_tech_post_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $Z_0  $A_d $A_g $A_g_prime_list $eta  $gamma_3_length $scheduler_num $scheduler_size

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF



touch ./bash/${jobout_name}/runtask_dmg_tech_prepost.sh
chmod +x ./bash/${jobout_name}/runtask_dmg_tech_prepost.sh

# tee -a ./bash/${job_name}/runtask_dmg_tech_prepost.sh <<EOF
tee ./bash/${jobout_name}/runtask_dmg_tech_prepost.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

if [ "\${1}" -eq 0 ]; then
    echo "pre_tech_post_damage_channel_original"
    python version_Bin/pre_tech_post_damage_channel_pseudo_original.py $job_name $log_xi_min $log_xi_max $batch_size $num_iterations1  $pretrained_pre_tech_post_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $Z_0  $A_d $A_g $A_g_prime_list $eta $gamma_3_length $scheduler_num $scheduler_size
else
    echo "post_tech_pre_damage_channel_original"
    python version_Bin/post_tech_pre_damage_channel_pseudo_original.py $job_name $pre_tech_post_damage_export_folder $log_xi_min $log_xi_max $batch_size $num_iterations1  $pretrained_post_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $Z_0  $A_d $A_g $A_g_prime_list $eta $gamma_3_length $scheduler_num $scheduler_size
fi

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF


touch ./bash/${jobout_name}/runtask_dmg_tech_prepre.sh
chmod +x ./bash/${jobout_name}/runtask_dmg_tech_prepre.sh

# tee -a ./bash/${job_name}/runtask_dmg_tech_prepre.sh <<EOF
tee ./bash/${jobout_name}/runtask_dmg_tech_prepre.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

echo "pre_tech_pre_damage"
python version_Bin/pre_tech_pre_damage_channel_pseudo_original.py $job_name $post_tech_pre_damage_export_folder $pre_tech_post_damage_export_folder -10 $log_xi_min $log_xi_max $batch_size $num_iterations2  $pretrained_pre_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $Z_0  $A_d $A_g $A_g_prime_list $eta $gamma_3_length $scheduler_num $scheduler_size




echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF



# When running a large number of tasks simultaneously, it may be necessary to increase the user process limit.
ulimit -u 10000




# This specifies the options used to run srun. The "-N1 -n1" options are used to allocate a single core to each task.
srun="srun --exclusive -N1 -n1"

# This specifies the options used to run GNU parallel:
#   --delay of 0.2 prevents overloading the controlling node.
#   -j is the number of tasks run simultaneously.
#   The combination of --joblog and --resume create a task log that can be used to monitor progress.

parallel1="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask1.log --resume"
parallel2="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask2.log --resume"
parallel3="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask3.log --resume"

# Run a script, runtask.sh, using GNU parallel and srun. Parallel will run the runtask script for the numbers 1 through 128. To illustrate, the first job will run like this:

# srun --exclusive -N1 -n1 ./runtask.sh arg1:1 > runtask.1

# $parallel "$srun ./runtask.sh arg1:{1} > ./job-outs/test/runtask_{1}" ::: {1..18}
# $parallel "$srun ./runtask.sh {1} {2} > ./job-outs/test/runtask_{1}_{2}" ::: {1..5} ::: {10..15}
# $parallel "$srun ./runtask.sh {1} {2} --output="./job-outs/test/runtask_{1}_{2}.out" --error="./job-outs/test/runtask_{1}_{2}.err"" ::: {1..5} ::: {10..15}
$parallel1 -n0 "$srun ./bash/${jobout_name}/runtask_dmg_post_tech_post.sh > ./job-outs/${jobout_name}/dmg_post_tech_post.out 2> ./job-outs/${jobout_name}/dmg_post_tech_post.err" ::: {1}
$parallel2 "$srun ./bash/${jobout_name}/runtask_dmg_tech_prepost.sh {1} > ./job-outs/${jobout_name}/dmg_tech_prepost_{1}.out 2> ./job-outs/${jobout_name}/dmg_tech_prepost_{1}.err" ::: {0..1}
$parallel3 -n0 "$srun ./bash/${jobout_name}/runtask_dmg_tech_prepre.sh > ./job-outs/${jobout_name}/dmg_tech_prepre.out 2> ./job-outs/${jobout_name}/dmg_tech_prepre.err" ::: {1}

# Note that if your program does not take any input, use the -n0 option to call the parallel command:

#   $parallel -n0 "$srun ./run_noinput_task.sh > output.{1}" ::: {1..128}