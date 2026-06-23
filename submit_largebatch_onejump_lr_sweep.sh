#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

SOURCE_ROOT="${SOURCE_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_largebatch_001}"
STAGE_SCRIPT="${STAGE_SCRIPT:-one_tech_jump_stage.sbatch}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-network_audit_stage.sbatch}"
MODEL_DIR="${MODEL_DIR:-models}"

BATCH_SIZE="${BATCH_SIZE:-1024}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-2048}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
NUM_NEURONS="${NUM_NEURONS:-32}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}"
SCHEDULE_TYPE="${SCHEDULE_TYPE:-warmup_cosine}"
AUDIT_SAMPLE_SIZE="${AUDIT_SAMPLE_SIZE:-16384}"
AUDIT_CHUNK_SIZE="${AUDIT_CHUNK_SIZE:-1024}"
TECH_JUMP_INTENSITY_SCALE="${TECH_JUMP_INTENSITY_SCALE:-1.0}"
TECH_JUMP_PROBABILITY="${TECH_JUMP_PROBABILITY:-1.0}"
POST_TECH_LR="${POST_TECH_LR:-10e-6,40e-4}"

SOURCE_NAME="OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
SOURCE_ONE="${SOURCE_ROOT}/${SOURCE_NAME}"
TWO_STAGE_SOURCE="${SOURCE_ROOT}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"

LEARNING_RATE_GRID=(
    "10e-6,10e-4"
    "10e-6,40e-5"
    "10e-6,10e-5"
    "10e-6,40e-6"
    "40e-6,40e-6,40e-6,40e-6"
    "10e-7,10e-7,10e-7,10e-7"
    "20e-7,20e-7,20e-7,20e-7"
    "40e-7,40e-7,40e-7,40e-7"
)

copy_stage_seed() {
    local source="$1"
    local target="$2"
    local stage="$3"

    if [ ! -f "${source}/${stage}/checkpoint" ]; then
        echo "Missing source checkpoint: ${source}/${stage}" >&2
        exit 2
    fi
    cp -a "${source}/${stage}" "${target}/${stage}"
}

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local target_folder="$3"
    local active_lr="$4"
    local dependency="${5:-}"
    local dependency_args=()

    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    PREFIX="$OUTPUT_ROOT" \
    PRETRAINED_FOLDER="$target_folder" \
    MODEL_DIR="$MODEL_DIR" \
    FOLDERNAME="$foldername" \
    JOB_NAME="$target_folder" \
    BATCH_SIZE="$BATCH_SIZE" \
    VALIDATION_BATCH_SIZE="$VALIDATION_BATCH_SIZE" \
    VALIDATION_BATCHES="$VALIDATION_BATCHES" \
    VALIDATION_CONTROL_WEIGHT=5.0 \
    GRADIENT_CLIP_NORM=1.0 \
    ENABLE_TENSORBOARD=0 \
    NUM_NEURONS="$NUM_NEURONS" \
    NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
    NUM_ITERATIONS_POSTTECH="$NUM_ITERATIONS" \
    NUM_ITERATIONS_INTERMTECH="$NUM_ITERATIONS" \
    LOGGING_FREQUENCY="$LOGGING_FREQUENCY" \
    LEARNING_RATES_POSTTECH="$POST_TECH_LR" \
    LEARNING_RATES_INTERMTECH="$active_lr" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" \
    TECH_JUMP_PROBABILITY="$TECH_JUMP_PROBABILITY" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="onePi1LR_${stage}" \
        --output="./job-outs/LargeBatch/${foldername}/${stage}-%j.out" \
        --error="./job-outs/LargeBatch/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

mkdir -p "$OUTPUT_ROOT" "./job-outs/LargeBatch"
summary="${OUTPUT_ROOT}/onejump_largebatch_lr_sweep_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "learning_rates,stage,job_id,dependency,target_folder" > "$summary"

for lr in "${LEARNING_RATE_GRID[@]}"; do
    foldername="${SOURCE_NAME}_LargeBatch${BATCH_SIZE}_Continue_LR_${lr}_iters${NUM_ITERATIONS}"
    target_folder="${OUTPUT_ROOT}/${foldername}"

    if [ -e "${target_folder}/run_manifest.txt" ]; then
        echo "Refusing to overwrite existing sweep folder: ${target_folder}" >&2
        exit 2
    fi

    mkdir -p "$target_folder" "./job-outs/LargeBatch/${foldername}"

    copy_stage_seed "$TWO_STAGE_SOURCE" "$target_folder" PostDamagePostTech
    copy_stage_seed "$TWO_STAGE_SOURCE" "$target_folder" PreDamagePostTech
    copy_stage_seed "$SOURCE_ONE" "$target_folder" PostDamagePreTech
    copy_stage_seed "$SOURCE_ONE" "$target_folder" PreDamagePreTech

    cat > "${target_folder}/run_manifest.txt" <<EOF
Run: one-jump pi=1 large-batch learning-rate sweep
Created: $(date)
Source one-jump folder: ${SOURCE_ONE}
Inherited post-tech source: ${TWO_STAGE_SOURCE}
Output folder: ${target_folder}
Training batch size: ${BATCH_SIZE}
Validation: ${VALIDATION_BATCHES} independent batches of ${VALIDATION_BATCH_SIZE}
Iterations per trained regime: ${NUM_ITERATIONS}
Learning-rate schedule: ${SCHEDULE_TYPE}
Pre-tech learning rates: ${lr}
Post-tech learning rates recorded but not trained: ${POST_TECH_LR}
Technology intensity scale: ${TECH_JUMP_INTENSITY_SCALE}
Technology jump probability pi: ${TECH_JUMP_PROBABILITY}
Regimes present: PostDamagePostTech, PreDamagePostTech, PostDamagePreTech, PreDamagePreTech
Intermediate technology regimes present: no
Inherited without retraining: PostDamagePostTech, PreDamagePostTech
Large-batch continuation training: PostDamagePreTech, PreDamagePreTech
Checkpoint protection: retain inherited checkpoint unless large-sample validation improves
EOF

    jid_post_pre=$(submit_stage PostDamagePreTech "$foldername" "$target_folder" "$lr")
    echo "\"${lr}\",\"PostDamagePreTech\",\"${jid_post_pre}\",\"\",\"${target_folder}\"" >> "$summary"

    jid_pre_pre=$(submit_stage PreDamagePreTech "$foldername" "$target_folder" "$lr" "afterok:${jid_post_pre}")
    echo "\"${lr}\",\"PreDamagePreTech\",\"${jid_pre_pre}\",\"afterok:${jid_post_pre}\",\"${target_folder}\"" >> "$summary"

    audit_jid=$(EXPORT_FOLDER="$target_folder" \
        REFERENCE_FOLDER="" \
        INCLUDE_ALL_STAGES=0 \
        SAMPLE_SIZE="$AUDIT_SAMPLE_SIZE" \
        CHUNK_SIZE="$AUDIT_CHUNK_SIZE" \
        sbatch --parsable \
        --dependency="afterok:${jid_pre_pre}" \
        --job-name="audit_onePi1LR" \
        --output="./job-outs/LargeBatch/${foldername}/TrainingAudit-%j.out" \
        --error="./job-outs/LargeBatch/${foldername}/TrainingAudit-%j.err" \
        --export=ALL \
        "$AUDIT_SCRIPT")
    echo "\"${lr}\",\"TrainingAudit\",\"${audit_jid}\",\"afterok:${jid_pre_pre}\",\"${target_folder}\"" >> "$summary"

    cat >> "${target_folder}/run_manifest.txt" <<EOF
PostDamagePreTech job: ${jid_post_pre}
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre})
Training-error audit job: ${audit_jid} (afterok:${jid_pre_pre})
Final audit CSV: ${target_folder}/training_error_audit_${AUDIT_SAMPLE_SIZE}.csv
Training-history summary: ${target_folder}/training_history_summary.csv
EOF

    printf 'Submitted one-jump LR=%s: PostDamagePreTech=%s PreDamagePreTech=%s audit=%s\n' \
        "$lr" "$jid_post_pre" "$jid_pre_pre" "$audit_jid"
done

printf 'Wrote job summary: %s\n' "$summary"
