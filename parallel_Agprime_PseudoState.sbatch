#!/bin/sh



#SBATCH --time=0-36:00:00
#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos=ssd
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

# prefix="/scratch/midway3/pengyu"
prefix="/project/lhansen/Capital_NN/TwoCapital_NN/output"







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


# log_xi_min="-1.0"
# log_xi_max="-0.5"
# log_xi_baseline_min="-1.0"
# log_xi_baseline_max="-0.5"
# num_iterations1="2000000"
# num_iterations2="2000000"


log_xi_min="10"
log_xi_max="10.1"
log_xi_baseline_min="10"
log_xi_baseline_max="10.1"
num_iterations1="500000"
num_iterations2="500000"
# num_iterations1="1000000"
# num_iterations2="1000000"


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
A_g_prime_min=0.15
A_g_prime_max=0.15

# foldername="Pseudo_model_1m2m_${channel_type}_shortrange_${batch_size}"
foldername="Pseudo_model_1m2m_${channel_type}_shortrange_${batch_size}_neurons_${num_neurons}_Agmin_${A_g_prime_min}_Agmax_${A_g_prime_max}_logxi_${log_xi_min}_logxibaseline_${log_xi_baseline_min}"

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
output_layer_activations="None,custom,custom,softplus"
num_hidden_layers="4"





learning_rate_schedule_type="None"
delta="0.025"
tensorboard='True'
A_g_prime_length=10
gamma_3_length=5



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

python version_Bin/post_tech_post_damage_channel_pseudo_original.py $job_name $log_xi_min $log_xi_max $batch_size $num_iterations1  $pretrained_post_tech_post_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $A_g_prime_min $A_g_prime_max $A_g_prime_length $gamma_3_length

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
    python version_Bin/pre_tech_post_damage_channel_pseudo_original.py $job_name $log_xi_min $log_xi_max $batch_size $num_iterations1  $pretrained_pre_tech_post_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $A_g_prime_min $A_g_prime_max $A_g_prime_length $gamma_3_length
else
    echo "post_tech_pre_damage_channel_original"
    python version_Bin/post_tech_pre_damage_channel_pseudo_original.py $job_name $pre_tech_post_damage_export_folder $log_xi_min $log_xi_max $batch_size $num_iterations1  $pretrained_post_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $A_g_prime_min $A_g_prime_max $A_g_prime_length $gamma_3_length
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
python version_Bin/pre_tech_pre_damage_channel_pseudo_original.py $job_name $post_tech_pre_damage_export_folder $pre_tech_post_damage_export_folder -10 $log_xi_min $log_xi_max $batch_size $num_iterations2  $pretrained_pre_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard  $foldername $log_xi_baseline_min $log_xi_baseline_max $channel_type $A_g_prime_min $A_g_prime_max $A_g_prime_length $gamma_3_length




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