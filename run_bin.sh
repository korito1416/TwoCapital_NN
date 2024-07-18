channel_type="full"
range_type=0.1
Agprime_range_list="0.1,0.2"
logxi_list="-2.3,-2.3,-2.3,-2.3"
batch_size="32"
SLURM_NTASKS=1
sbatch -J rmlogxi_addc_${Agprime_range_list}_xi_${logxi_list}_n_${num_neurons}_b_${batch_size} parallel_Agprime_PseudoState.sbatch $channel_type $range_type $batch_size $SLURM_NTASKS



# sbatch --output=/dev/null --error=/dev/null -J rmlogxi_addc_${Agprime_range_list}_xi_${logxi_list}_n_${num_neurons}_b_${batch_size} parallel_Agprime_PseudoState.sbatch $channel_type $range_type $batch_size $SLURM_NTASKS
