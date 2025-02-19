channel_type="full"
range_type=0.1
Agprime_range_list="0.1,0.2"
logxi_list="-2.3,-2.3,-2.3,-2.3"
batch_size="128"
SLURM_NTASKS=2
A_g_prime_length=3


sbatch -J NNTwoCapital parallel_Agprime_PseudoState.sbatch $channel_type $range_type $batch_size $SLURM_NTASKS $A_g_prime_length

# Retrain
# sbatch -J rmlogxi_addc_${Agprime_range_list}_xi_${logxi_list}_n_${num_neurons}_b_${batch_size} parallel_Pre_pre_model_training.sbatch $channel_type $range_type $batch_size $SLURM_NTASKS $A_g_prime_length



# sbatch --output=/dev/null --error=/dev/null -J rmlogxi_addc_${Agprime_range_list}_xi_${logxi_list}_n_${num_neurons}_b_${batch_size} parallel_Agprime_PseudoState.sbatch $channel_type $range_type $batch_size $SLURM_NTASKS
