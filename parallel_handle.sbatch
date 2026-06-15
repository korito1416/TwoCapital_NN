#!/bin/sh


#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake 
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

# prefix="/scratch/midway3/pengyu" /project/lhansen/Cap_damage/TwoStageTechJump/output
prefix="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"

pretrain_folder="None"
  
num_neurons="32"

batch_size="128"
SLURM_NTASKS=2
# num_iterations1="2000000"
# num_iterations2="2000000"

num_iterations1="1000000"
num_iterations2="1000000"

# num_iterations1="1"
# num_iterations2="1"

# num_iterations1="100000"
# num_iterations2="5000"
# num_iterations1="5000"

# A_g_prime_num="0.15"
logging_frequency="1000"

# Post Tech learning rates 
learning_rates_PostTech="10e-6,40e-4"

  
learning_rates="10e-5,10e-5,10e-5,10e-5"
learning_rates="40e-5,40e-5,40e-5,40e-5"
learning_rates="10e-6,10e-6,10e-6,10e-6"
learning_rates="40e-6,40e-6,40e-6,40e-6" 


learning_rates="10e-6,10e-4"
learning_rates="10e-6,40e-4"
learning_rates="10e-6,10e-5"
learning_rates="10e-6,40e-5"
learning_rates="10e-6,40e-6"




hidden_layer_activations="swish,tanh,tanh,softplus"
output_layer_activations="softplus,custom,custom,softplus"
num_hidden_layers="4"


learning_rate_schedule_type="piecewiseconstant"
#   'None', 'piecewiseconstant', 'sgd+piecewiseconstant', 'sgd',
#   'cosine', 'cosine_restarts', 'exp_decay', 'poly_decay',
#   'inverse_time', 'warmup_cosine', 'cyclical', 'onecycle'

learning_rate_schedule_type="warmup_cosine"  
# delta="0.01"
# tensorboard='True'
# A_g_prime_length=3
# gamma_3_length=20
 


# echo "range_type=${range_type}"

foldername="TwoStageTech_LR_${learning_rate_schedule_type}_${learning_rates}_${batch_size}_neurons_${num_neurons}_#HiddenLayer_${num_hidden_layers}_num_iterations${num_iterations1}"


job_name="${prefix}/${foldername}"
jobout_name="${foldername}"

PostDamagePostTech_folder="${job_name}/PostDamagePostTech" 
PostDamagePreTech_folder="${job_name}/PostDamagePreTech" 
PreDamagePostTech_folder="${job_name}/PreDamagePostTech" 
PreDamagePreTech_folder="${job_name}/PreDamagePreTech" 
PostDamageIntermTech_folder="${job_name}/PostDamageIntermTech" 
PreDamageIntermTech_folder="${job_name}/PreDamageIntermTech" 


pretrained_folder="None" 
 
# pretrained_folder="/project/lhansen/Cap_damage/TwoStageTechJump_SITE_Pretrain/output_shorterRun/TechSearch_LR_warmup_cosine_10e-6,10e-6,10e-6,10e-6_128_neurons_32_#HiddenLayer_4_num_iterations1000000"

 



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

if [ -f ./job-outs/${jobout_name}/runtask4.log ]; then
    rm ./job-outs/${jobout_name}/runtask4.log
fi

if [ -f ./job-outs/${jobout_name}/runtask5.log ]; then
    rm ./job-outs/${jobout_name}/runtask5.log
fi
if [ -f ./job-outs/${jobout_name}/runtask6.log ]; then
    rm ./job-outs/${jobout_name}/runtask6.log
fi

mkdir -p ./bash/${jobout_name}/

###############################################
###### stage 1 : PostDamagePostTech
############################################# 
touch ./bash/${jobout_name}/runtask_PostDamagePostTech.sh
chmod +x ./bash/${jobout_name}/runtask_PostDamagePostTech.sh

tee ./bash/${jobout_name}/runtask_PostDamagePostTech.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python models/PostDamagePostTech.py $job_name $batch_size $num_iterations1  $pretrained_folder $logging_frequency $learning_rates_PostTech  $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername    

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
 

###############################################
###### stage 2 : PostDamageIntermTech
############################################# 
touch ./bash/${jobout_name}/runtask_PostDamageIntermTech.sh
chmod +x ./bash/${jobout_name}/runtask_PostDamageIntermTech.sh

tee ./bash/${jobout_name}/runtask_PostDamageIntermTech.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python models/PostDamageIntermTech.py $job_name $batch_size $num_iterations2  $pretrained_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername    

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF







###############################################
###### stage 3 : PostDamagePreTech
############################################# 
touch ./bash/${jobout_name}/runtask_PostDamagePreTech.sh
chmod +x ./bash/${jobout_name}/runtask_PostDamagePreTech.sh

tee ./bash/${jobout_name}/runtask_PostDamagePreTech.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python models/PostDamagePreTech.py $job_name $batch_size $num_iterations2  $pretrained_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername    

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF








###############################################
###### stage 4 : PreDamagePostTech
############################################# 
touch ./bash/${jobout_name}/runtask_PreDamagePostTech.sh
chmod +x ./bash/${jobout_name}/runtask_PreDamagePostTech.sh

tee ./bash/${jobout_name}/runtask_PreDamagePostTech.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python models/PreDamagePostTech.py $job_name $batch_size $num_iterations1  $pretrained_folder $logging_frequency $learning_rates_PostTech $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername    

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF



###############################################
###### stage 5 : PreDamageIntermTech
############################################# 
touch ./bash/${jobout_name}/runtask_PreDamageIntermTech.sh
chmod +x ./bash/${jobout_name}/runtask_PreDamageIntermTech.sh

tee ./bash/${jobout_name}/runtask_PreDamageIntermTech.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python models/PreDamageIntermTech.py $job_name $batch_size $num_iterations2  $pretrained_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername    

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF


###############################################
###### stage 65 : PreDamagePreTech
############################################# 
touch ./bash/${jobout_name}/runtask_PreDamagePreTech.sh
chmod +x ./bash/${jobout_name}/runtask_PreDamagePreTech.sh

tee ./bash/${jobout_name}/runtask_PreDamagePreTech.sh <<EOF
#! /bin/bash

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python models/PreDamagePreTech.py $job_name $batch_size $num_iterations2  $pretrained_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername    

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
 
parallel1="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask1.log --resume"
parallel2="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask2.log --resume"
parallel3="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask3.log --resume"
parallel4="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask4.log --resume"
parallel5="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask5.log --resume"
parallel6="parallel --delay 0.2 -j $SLURM_NTASKS --joblog ./job-outs/${jobout_name}/runtask6.log --resume"
# Execute stages in sequence 
$parallel1 -n0 "$srun bash/${jobout_name}/runtask_PostDamagePostTech.sh > ./job-outs/${jobout_name}/PostDamagePostTech.out 2> ./job-outs/${jobout_name}/dmg_PostDamagePostTech.err" ::: {1}
$parallel2 -n0 "$srun bash/${jobout_name}/runtask_PostDamageIntermTech.sh > ./job-outs/${jobout_name}/PostDamageIntermTech.out 2> ./job-outs/${jobout_name}/dmg_PostDamageIntermTech.err" ::: {1}
$parallel3 -n0 "$srun bash/${jobout_name}/runtask_PostDamagePreTech.sh > ./job-outs/${jobout_name}/PostDamagePreTech.out 2> ./job-outs/${jobout_name}/dmg_PostDamagePreTech.err" ::: {1}
$parallel4 -n0 "$srun bash/${jobout_name}/runtask_PreDamagePostTech.sh > ./job-outs/${jobout_name}/PreDamagePostTech.out 2> ./job-outs/${jobout_name}/dmg_PreDamagePostTech.err" ::: {1}
$parallel5 -n0 "$srun bash/${jobout_name}/runtask_PreDamageIntermTech.sh > ./job-outs/${jobout_name}/PreDamageIntermTech.out 2> ./job-outs/${jobout_name}/dmg_PreDamageIntermTech.err" ::: {1}
$parallel6 -n0 "$srun bash/${jobout_name}/runtask_PreDamagePreTech.sh > ./job-outs/${jobout_name}/PreDamagePreTech.out 2> ./job-outs/${jobout_name}/dmg_PreDamagePreTech.err" ::: {1}
 