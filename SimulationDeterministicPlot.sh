#!/bin/bash

prefix="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
foldername="TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"

job_name="${prefix}/${foldername}"
jobout_name="${foldername}"
# echo "Export folder: $pre_tech_pre_damage_export_folder"

# Create directories if they don't exist
mkdir -p "./logging/scripts/${jobout_name}"
mkdir -p "./logging/${jobout_name}"

job_file="./logging/scripts/${jobout_name}/SimDtmPlots.job"
echo "#!/bin/bash
#SBATCH --job-name=DeterministricSimulation
#SBATCH --output=./logging/${jobout_name}/SimDtmPlots.out
#SBATCH --error=./logging/${jobout_name}/SimDtmPlots.err
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
python3 models/SimulationDeterministicPlot.py $job_name" > "$job_file"
sbatch "$job_file"