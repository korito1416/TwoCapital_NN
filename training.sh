


batch_size="64"
SLURM_NTASKS=2


sbatch -J rmlogxi_addc_SLURM_NTASKS_${SLURM_NTASKS}_batch_size_${batch_size} ParallelTraining.sbatch  $batch_size $SLURM_NTASKS  

# Retrain
# sbatch -J rmlogxi_addc_${Agprime_range_list}_xi_${logxi_list}_n_${num_neurons}_b_${batch_size} parallel_Pre_pre_model_training.sbatch $channel_type $range_type $batch_size $SLURM_NTASKS $A_g_prime_length
 
# sbatch --output=/dev/null --error=/dev/null -J rmlogxi_addc_${Agprime_range_list}_xi_${logxi_list}_n_${num_neurons}_b_${batch_size} parallel_Agprime_PseudoState.sbatch $channel_type $range_type $batch_size $SLURM_NTASKS
