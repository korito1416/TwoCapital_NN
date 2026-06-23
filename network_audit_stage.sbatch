#!/bin/bash

#SBATCH --time=0-02:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=21G
#SBATCH --job-name=network_audit

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05

EXPORT_FOLDER="${EXPORT_FOLDER:?EXPORT_FOLDER must be set}"
REFERENCE_FOLDER="${REFERENCE_FOLDER:-}"
SAMPLE_SIZE="${SAMPLE_SIZE:-16384}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
OUTPUT_NAME="${OUTPUT_NAME:-training_error_audit_${SAMPLE_SIZE}.csv}"
INCLUDE_ALL_STAGES="${INCLUDE_ALL_STAGES:-1}"

args=(
    --export-folder "$EXPORT_FOLDER"
    --sample-size "$SAMPLE_SIZE"
    --chunk-size "$CHUNK_SIZE"
    --output "${EXPORT_FOLDER}/${OUTPUT_NAME}"
)

if [ "$INCLUDE_ALL_STAGES" = "1" ]; then
    args+=(--include-all-stages)
fi

if [ -n "$REFERENCE_FOLDER" ]; then
    args+=(--reference-folder "$REFERENCE_FOLDER")
fi

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 \
    python audit_trained_networks.py "${args[@]}" \
    | tee "${EXPORT_FOLDER}/${OUTPUT_NAME%.csv}.txt"

python summarize_training_histories.py \
    --folder "$EXPORT_FOLDER" \
    --output "${EXPORT_FOLDER}/training_history_summary.csv"
