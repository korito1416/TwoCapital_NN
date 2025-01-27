#!/bin/sh


#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G

# Load the default version of GNU parallel.
module load parallel
# module load tensorflow/2.1
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

batch_size=$1
SLURM_NTASKS=$2 
 
num_neurons="32"
 
num_iterations1="2000000"
num_iterations2="2000000"
 
num_iterations1="200000"
num_iterations2="200000"

# A_g_prime_num="0.15"
logging_frequency="1000"
learning_rates="10e-5,10e-5,10e-5,10e-5"
# learning_rates="40e-7,40e-7,40e-7,40e-7"
# learning_rates="10e-4,10e-4,10e-4,10e-4"
# learning_rates="10e-3,10e-3,10e-3,10e-3"
# learning_rates="10e-7,10e-7,10e-7,10e-7"
# learning_rates="10e-8,10e-8,10e-8,10e-8"
hidden_layer_activations="swish,tanh,tanh,softplus"
output_layer_activations="softplus,custom,custom,softplus"
# output_layer_activations="softplus,softplus,softplus,softplus"
# output_layer_activations="softplus,tanh,tanh,softplus"
num_hidden_layers="4"
learning_rate_schedule_type="piecewiseconstant"
# learning_rate_schedule_type="None"
tensorboard='True'


prefix="/project/lhansen/Cap_damage/FourStates_NN/output"
foldername="WhyNotConverge_LR_${learning_rate_schedule_type}_${learning_rates}_num_iterations${num_iterations1}_output_layer_activations_${output_layer_activations}"



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


# pretrain_folder="/project/lhansen/Cap_damage/TwoCapital_NN/output/Z_00.7_Ad_0.13_Ag_0.1086_AgPrime_0.130,0.139,0.147_eta_0.291_LR_piecewiseconstant_10e-5,10e-5,10e-5,10e-5_num_iterations2000000"
# # pretrain_folder="/project/lhansen/Cap_damage/TwoCapital_NN/output/A_g_prime_3_Delta_0.01_LR_piecewiseconstant_10e-5,10e-5,10e-5,10e-5_128_neurons_32_#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations600000"
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

python Model/post_tech_post_damage.py $job_name $batch_size $num_iterations1  $pretrained_post_tech_post_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername 

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
    python Model/pre_tech_post_damage.py $job_name  $batch_size $num_iterations1  $pretrained_pre_tech_post_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type    $foldername  
else
    echo "post_tech_pre_damage_channel_original"
    python Model/post_tech_pre_damage.py $job_name $pre_tech_post_damage_export_folder  $batch_size $num_iterations1  $pretrained_post_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type   $foldername 
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
python Model/pre_tech_pre_damage.py $job_name $post_tech_pre_damage_export_folder $pre_tech_post_damage_export_folder  $batch_size $num_iterations2  $pretrained_pre_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername 




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