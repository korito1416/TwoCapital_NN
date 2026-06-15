#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_dgm_001}"
STAGE_SCRIPT="${STAGE_SCRIPT:-dgm_stage.sbatch}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
SCHEDULE_TYPE="${SCHEDULE_TYPE:-warmup_cosine}"

# tag|width|layers|batch|value LR|control LR
CONFIGS=(
    "w32_l3_b256_lr1e-5_1e-4|32|3|256|1e-5|1e-4"
    "w48_l3_b256_lr5e-6_5e-5|48|3|256|5e-6|5e-5"
    "w32_l4_b512_lr2e-6_2e-5|32|4|512|2e-6|2e-5"
)

mkdir -p "$OUTPUT_ROOT" job-outs-dgm

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local width="$3"
    local layers="$4"
    local batch="$5"
    local learning_rates="$6"
    local dependency="${7:-}"
    local dependency_args=()

    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    OUTPUT_ROOT="$OUTPUT_ROOT" \
    FOLDERNAME="$foldername" \
    JOB_NAME="${OUTPUT_ROOT}/${foldername}" \
    PRETRAINED_FOLDER=None \
    BATCH_SIZE="$batch" \
    NUM_NEURONS="$width" \
    NUM_HIDDEN_LAYERS="$layers" \
    NUM_ITERATIONS="$NUM_ITERATIONS" \
    LOGGING_FREQUENCY="$LOGGING_FREQUENCY" \
    LEARNING_RATES="$learning_rates" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE=1.0 \
    TECH_JUMP_PROBABILITY=0.04 \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="dgm_${stage}" \
        --output="./job-outs-dgm/${foldername}/${stage}-%j.out" \
        --error="./job-outs-dgm/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r tag width layers batch value_lr control_lr <<< "$config"
    foldername="DGM_TwoStage_${tag}_iters${NUM_ITERATIONS}"
    export_folder="${OUTPUT_ROOT}/${foldername}"
    learning_rates="${value_lr},${control_lr}"

    if [ -e "${export_folder}/run_manifest.txt" ]; then
        echo "Refusing to resubmit existing DGM variant: ${foldername}" >&2
        exit 2
    fi

    mkdir -p "$export_folder" "./job-outs-dgm/${foldername}"
    cat > "${export_folder}/run_manifest.txt" <<EOF
Architecture: Deep Galerkin Method gated network
Source architecture: alialaradi/DeepGalerkinMethod
Created: $(date)
Model directory: models_dgm
Output root: ${OUTPUT_ROOT}
Width: ${width}
DGM layers: ${layers}
Batch size: ${batch}
Value learning rate: ${value_lr}
Control learning rate: ${control_lr}
Schedule: ${SCHEDULE_TYPE}
Iterations per regime: ${NUM_ITERATIONS}
Tech jump probability pi: 0.04
Tech intensity scale: 1.0
Training initialization: random DGM weights
Baseline function distillation: 20000 steps before HJB training
EOF

    jid_post_post=$(submit_stage PostDamagePostTech "$foldername" "$width" "$layers" "$batch" "$learning_rates")
    jid_post_interm=$(submit_stage PostDamageIntermTech "$foldername" "$width" "$layers" "$batch" "$learning_rates" "afterok:${jid_post_post}")
    jid_pre_post=$(submit_stage PreDamagePostTech "$foldername" "$width" "$layers" "$batch" "$learning_rates" "afterok:${jid_post_post}")
    jid_post_pre=$(submit_stage PostDamagePreTech "$foldername" "$width" "$layers" "$batch" "$learning_rates" "afterok:${jid_post_interm}")
    jid_pre_interm=$(submit_stage PreDamageIntermTech "$foldername" "$width" "$layers" "$batch" "$learning_rates" "afterok:${jid_post_interm}:${jid_pre_post}")
    jid_pre_pre=$(submit_stage PreDamagePreTech "$foldername" "$width" "$layers" "$batch" "$learning_rates" "afterok:${jid_post_pre}:${jid_pre_interm}")

    cat >> "${export_folder}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PostDamageIntermTech job: ${jid_post_interm} (afterok:${jid_post_post})
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_interm})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm}:${jid_pre_post})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre}:${jid_pre_interm})
EOF

    printf '%s: %s %s %s %s %s %s\n' \
        "$foldername" "$jid_post_post" "$jid_post_interm" "$jid_pre_post" \
        "$jid_post_pre" "$jid_pre_interm" "$jid_pre_pre"
done
