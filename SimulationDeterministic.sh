#!/bin/bash
 
prefix="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
foldername="TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
job_name="${prefix}/${foldername}"
jobout_name="${foldername}"
 

# Create directories if they don't exist
mkdir -p "./bash/${jobout_name}"
mkdir -p "./job-outs/${jobout_name}"

job_file="./bash/${jobout_name}/SimulationDeterministic.job"
echo "#!/bin/bash
#SBATCH --job-name=DeterministricSimulation
#SBATCH --output=./job-outs/${jobout_name}/SimulationDeterministic.out
#SBATCH --error=./job-outs/${jobout_name}/SimulationDeterministic.err
#SBATCH --time=0-1:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
module load tensorflow/2.1
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05
python3 models/SimulationDeterministic.py $job_name" > "$job_file"
sbatch "$job_file"