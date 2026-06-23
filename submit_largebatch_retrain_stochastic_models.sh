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
MODEL_FILTER="${MODEL_FILTER:-}"

BASELINE_TWO_STAGE_SOURCE="${SOURCE_ROOT}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"

# label|source folder|intensity|pi|post-tech LR|other-regime LR
MODELS=(
    "OneTechPi1S1|OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|1.0|1.0|10e-6,40e-4|10e-6,40e-4"
    "TwoStageBase|TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|1.0|0.04|10e-6,40e-4|10e-6,10e-4"
    "TwoStageS2|TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000|2.0|0.04|10e-6,40e-4|10e-6,40e-4"
)

ALL_STAGES=(
    PostDamagePostTech
    PostDamageIntermTech
    PreDamagePostTech
    PostDamagePreTech
    PreDamageIntermTech
    PreDamagePreTech
)

ONE_JUMP_STAGES=(
    PostDamagePostTech
    PreDamagePostTech
    PostDamagePreTech
    PreDamagePreTech
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
    local intensity="$4"
    local pi="$5"
    local post_lr="$6"
    local active_lr="$7"
    local dependency="${8:-}"
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
    LEARNING_RATES_POSTTECH="$post_lr" \
    LEARNING_RATES_INTERMTECH="$active_lr" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$intensity" \
    TECH_JUMP_PROBABILITY="$pi" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="largeB_${stage}" \
        --output="./job-outs/LargeBatch/${foldername}/${stage}-%j.out" \
        --error="./job-outs/LargeBatch/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

mkdir -p "$OUTPUT_ROOT" "./job-outs/LargeBatch"
summary="${OUTPUT_ROOT}/largebatch_retraining_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "label,stage,job_id,dependency,target_folder" > "$summary"

for spec in "${MODELS[@]}"; do
    IFS='|' read -r label source_name intensity pi post_lr active_lr <<< "$spec"
    if [ -n "$MODEL_FILTER" ] && [ "$label" != "$MODEL_FILTER" ]; then
        continue
    fi
    source_folder="${SOURCE_ROOT}/${source_name}"
    foldername="${source_name}_LargeBatch${BATCH_SIZE}_Continue_iters${NUM_ITERATIONS}"
    target_folder="${OUTPUT_ROOT}/${foldername}"

    if [ -e "${target_folder}/run_manifest.txt" ]; then
        echo "Refusing to overwrite existing retraining folder: ${target_folder}" >&2
        exit 2
    fi
    mkdir -p "$target_folder" "./job-outs/LargeBatch/${foldername}"

    if [ "$label" = "OneTechPi1S1" ]; then
        for stage in "${ONE_JUMP_STAGES[@]}"; do
            seed_source="$source_folder"
            if [[ "$stage" == *PostTech ]]; then
                seed_source="$BASELINE_TWO_STAGE_SOURCE"
            fi
            copy_stage_seed "$seed_source" "$target_folder" "$stage"
        done
    else
        for stage in "${ALL_STAGES[@]}"; do
            copy_stage_seed "$source_folder" "$target_folder" "$stage"
        done
    fi

    cat > "${target_folder}/run_manifest.txt" <<EOF
Run: large-batch continuation training for stochastic-simulation model
Created: $(date)
Label: ${label}
Source folder: ${source_folder}
Output folder: ${target_folder}
Architecture: inherited original feedforward networks
Training batch size: ${BATCH_SIZE}
Original training batch size: 128
Validation: ${VALIDATION_BATCHES} independent batches of ${VALIDATION_BATCH_SIZE}
Iterations per regime: ${NUM_ITERATIONS}
Learning-rate schedule: ${SCHEDULE_TYPE}
Post-tech learning rates: ${post_lr}
Other-regime learning rates: ${active_lr}
Technology intensity scale: ${intensity}
Technology jump probability pi: ${pi}
Checkpoint protection: retain inherited checkpoint unless large-sample validation improves
EOF
    if [ "$label" = "OneTechPi1S1" ]; then
        cat >> "${target_folder}/run_manifest.txt" <<EOF
Regimes present: PostDamagePostTech, PreDamagePostTech, PostDamagePreTech, PreDamagePreTech
Intermediate technology regimes present: no
Inherited without retraining from ${BASELINE_TWO_STAGE_SOURCE}: PostDamagePostTech, PreDamagePostTech
Large-batch continuation training: PostDamagePreTech, PreDamagePreTech
EOF
        audit_reference=""
        include_all_stages=0
        jid_post_pre=$(submit_stage PostDamagePreTech "$foldername" "$target_folder" "$intensity" "$pi" "$post_lr" "$active_lr")
        echo "\"${label}\",\"PostDamagePreTech\",\"${jid_post_pre}\",\"\",\"${target_folder}\"" >> "$summary"

        jid_pre_pre=$(submit_stage PreDamagePreTech "$foldername" "$target_folder" "$intensity" "$pi" "$post_lr" "$active_lr" "afterok:${jid_post_pre}")
        echo "\"${label}\",\"PreDamagePreTech\",\"${jid_pre_pre}\",\"afterok:${jid_post_pre}\",\"${target_folder}\"" >> "$summary"
    else
        cat >> "${target_folder}/run_manifest.txt" <<EOF
Regimes present: PostDamagePostTech, PostDamageIntermTech, PreDamagePostTech, PostDamagePreTech, PreDamageIntermTech, PreDamagePreTech
All six regimes receive large-batch continuation training: yes
EOF
        jid_post_post=$(submit_stage PostDamagePostTech "$foldername" "$target_folder" "$intensity" "$pi" "$post_lr" "$active_lr")
        echo "\"${label}\",\"PostDamagePostTech\",\"${jid_post_post}\",\"\",\"${target_folder}\"" >> "$summary"

        jid_post_interm=$(submit_stage PostDamageIntermTech "$foldername" "$target_folder" "$intensity" "$pi" "$post_lr" "$active_lr" "afterok:${jid_post_post}")
        echo "\"${label}\",\"PostDamageIntermTech\",\"${jid_post_interm}\",\"afterok:${jid_post_post}\",\"${target_folder}\"" >> "$summary"

        jid_pre_post=$(submit_stage PreDamagePostTech "$foldername" "$target_folder" "$intensity" "$pi" "$post_lr" "$active_lr" "afterok:${jid_post_post}")
        echo "\"${label}\",\"PreDamagePostTech\",\"${jid_pre_post}\",\"afterok:${jid_post_post}\",\"${target_folder}\"" >> "$summary"

        jid_post_pre=$(submit_stage PostDamagePreTech "$foldername" "$target_folder" "$intensity" "$pi" "$post_lr" "$active_lr" "afterok:${jid_post_interm}")
        echo "\"${label}\",\"PostDamagePreTech\",\"${jid_post_pre}\",\"afterok:${jid_post_interm}\",\"${target_folder}\"" >> "$summary"

        jid_pre_interm=$(submit_stage PreDamageIntermTech "$foldername" "$target_folder" "$intensity" "$pi" "$post_lr" "$active_lr" "afterok:${jid_post_interm}:${jid_pre_post}")
        echo "\"${label}\",\"PreDamageIntermTech\",\"${jid_pre_interm}\",\"afterok:${jid_post_interm}:${jid_pre_post}\",\"${target_folder}\"" >> "$summary"

        jid_pre_pre=$(submit_stage PreDamagePreTech "$foldername" "$target_folder" "$intensity" "$pi" "$post_lr" "$active_lr" "afterok:${jid_post_pre}:${jid_pre_interm}")
        echo "\"${label}\",\"PreDamagePreTech\",\"${jid_pre_pre}\",\"afterok:${jid_post_pre}:${jid_pre_interm}\",\"${target_folder}\"" >> "$summary"

        audit_reference="$source_folder"
        include_all_stages=1
    fi
    audit_jid=$(EXPORT_FOLDER="$target_folder" \
        REFERENCE_FOLDER="$audit_reference" \
        INCLUDE_ALL_STAGES="$include_all_stages" \
        SAMPLE_SIZE="$AUDIT_SAMPLE_SIZE" \
        CHUNK_SIZE="$AUDIT_CHUNK_SIZE" \
        sbatch --parsable \
        --dependency="afterok:${jid_pre_pre}" \
        --job-name="audit_${label}" \
        --output="./job-outs/LargeBatch/${foldername}/TrainingAudit-%j.out" \
        --error="./job-outs/LargeBatch/${foldername}/TrainingAudit-%j.err" \
        --export=ALL \
        "$AUDIT_SCRIPT")
    echo "\"${label}\",\"TrainingAudit\",\"${audit_jid}\",\"afterok:${jid_pre_pre}\",\"${target_folder}\"" >> "$summary"

    if [ "$label" = "OneTechPi1S1" ]; then
        cat >> "${target_folder}/run_manifest.txt" <<EOF
PostDamagePreTech job: ${jid_post_pre}
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre})
EOF
        regime_count=4
    else
        cat >> "${target_folder}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PostDamageIntermTech job: ${jid_post_interm} (afterok:${jid_post_post})
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_interm})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm}:${jid_pre_post})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre}:${jid_pre_interm})
EOF
        regime_count=6
    fi
    cat >> "${target_folder}/run_manifest.txt" <<EOF
Training-error audit job: ${audit_jid} (afterok:${jid_pre_pre})
Final audit CSV: ${target_folder}/training_error_audit_${AUDIT_SAMPLE_SIZE}.csv
Training-history summary: ${target_folder}/training_history_summary.csv
EOF

    printf '%s: %s-regime final=%s audit=%s\n' "$label" "$regime_count" "$jid_pre_pre" "$audit_jid"
done

printf 'Wrote job summary: %s\n' "$summary"
