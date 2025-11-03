

idarray=($(seq 1 100))

hmc_python_name="SimulationRandom.py"

xi=0.05
# xi=0.1
# xi=0.3
# xi=148.4 

for id in "${idarray[@]}"; do
  action_name="Simulation_withJump"
  mkdir -p ./job-outs/${action_name}_xi_${xi}/id_${id}/
  mkdir -p ./bash/${action_name}_xi_${xi}/id_${id}/

  cat > ./bash/${action_name}_xi_${xi}/id_${id}/run.sh <<EOF
#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --job-name=id_${id}_xi_${xi}
#SBATCH --output=./job-outs/${action_name}_xi_${xi}/id_${id}/run.out
#SBATCH --error=./job-outs/${action_name}_xi_${xi}/id_${id}/run.err
#SBATCH --time=5:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

# --- Clean & load a TF-compatible environment (NumPy 1.x) ---
module unload python
module unload cuda
# If you want GPU libs present (even if not used), mirror training:
module load cuda/11.2
module load python/anaconda-2021.05

# Avoid user-site packages shadowing the module's NumPy/TF
export PYTHONNOUSERSITE=1

# Optional: reproducibility vs. speed tradeoff
export TF_ENABLE_ONEDNN_OPTS=0

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

# Sanity print versions into the log header
python - <<'PY'
import sys, numpy as np
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("TensorFlow import ERROR:", e)
PY

# Run simulation
python -u /project/lhansen/Cap_damage/TwoStageTechJump_SITE_Pretrain/models/$hmc_python_name --id ${id} --xi ${xi}

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))
eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"
EOF

  chmod +x ./bash/${action_name}_xi_${xi}/id_${id}/run.sh
  sbatch ./bash/${action_name}_xi_${xi}/id_${id}/run.sh
done
