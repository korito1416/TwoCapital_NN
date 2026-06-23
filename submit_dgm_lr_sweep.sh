#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_dgm_001}"
STAGE_SCRIPT="${STAGE_SCRIPT:-dgm_stage.sbatch}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1000}"
SCHEDULE_TYPE="${SCHEDULE_TYPE:-warmup_cosine}"
WIDTH="${WIDTH:-32}"
LAYERS="${LAYERS:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TECH_JUMP_INTENSITY_SCALE="${TECH_JUMP_INTENSITY_SCALE:-1.0}"
TECH_JUMP_PROBABILITY="${TECH_JUMP_PROBABILITY:-0.04}"
DGM_TEACHER_FOLDER="${DGM_TEACHER_FOLDER:-/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000}"

# tag|value LR|control LR
CONFIGS=(
    "lr1e-5_4e-4|1e-5|4e-4"
    "lr1e-5_4e-5|1e-5|4e-5"
    "lr5e-6_1e-4|5e-6|1e-4"
    "lr2e-6_1e-4|2e-6|1e-4"
    "lr1e-6_1e-5|1e-6|1e-5"
    "lr5e-7_5e-6|5e-7|5e-6"
)

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local learning_rates="$3"
    local dependency="${4:-}"
    local dependency_args=()

    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    OUTPUT_ROOT="$OUTPUT_ROOT" \
    FOLDERNAME="$foldername" \
    JOB_NAME="${OUTPUT_ROOT}/${foldername}" \
    PRETRAINED_FOLDER=None \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_NEURONS="$WIDTH" \
    NUM_HIDDEN_LAYERS="$LAYERS" \
    NUM_ITERATIONS="$NUM_ITERATIONS" \
    LOGGING_FREQUENCY="$LOGGING_FREQUENCY" \
    LEARNING_RATES="$learning_rates" \
    LEARNING_RATE_SCHEDULE_TYPE="$SCHEDULE_TYPE" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" \
    TECH_JUMP_PROBABILITY="$TECH_JUMP_PROBABILITY" \
    DGM_TEACHER_FOLDER="$DGM_TEACHER_FOLDER" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="dgmLR_${stage}" \
        --output="./job-outs-dgm/${foldername}/${stage}-%j.out" \
        --error="./job-outs-dgm/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

mkdir -p "$OUTPUT_ROOT" job-outs-dgm
summary="${OUTPUT_ROOT}/dgm_lr_sweep_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "tag,value_lr,control_lr,stage,job_id,dependency,target_folder" > "$summary"

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r tag value_lr control_lr <<< "$config"
    foldername="DGM_TwoStage_w${WIDTH}_l${LAYERS}_b${BATCH_SIZE}_${tag}_iters${NUM_ITERATIONS}"
    export_folder="${OUTPUT_ROOT}/${foldername}"
    learning_rates="${value_lr},${control_lr}"

    if [ -e "${export_folder}/run_manifest.txt" ]; then
        echo "Refusing to resubmit existing DGM LR variant: ${foldername}" >&2
        exit 2
    fi

    mkdir -p "$export_folder" "./job-outs-dgm/${foldername}"
    cat > "${export_folder}/run_manifest.txt" <<EOF
Architecture: Deep Galerkin Method gated network
Source architecture: alialaradi/DeepGalerkinMethod
Run: DGM two-stage learning-rate sweep
Created: $(date)
Model directory: models_dgm
Output root: ${OUTPUT_ROOT}
Width: ${WIDTH}
DGM layers: ${LAYERS}
Batch size: ${BATCH_SIZE}
Value learning rate: ${value_lr}
Control learning rate: ${control_lr}
Schedule: ${SCHEDULE_TYPE}
Iterations per regime: ${NUM_ITERATIONS}
Tech jump probability pi: ${TECH_JUMP_PROBABILITY}
Tech intensity scale: ${TECH_JUMP_INTENSITY_SCALE}
Training initialization: random DGM weights
Teacher folder: ${DGM_TEACHER_FOLDER}
Baseline function distillation: 20000 steps before HJB training
Regimes present: PostDamagePostTech, PostDamageIntermTech, PreDamagePostTech, PostDamagePreTech, PreDamageIntermTech, PreDamagePreTech
EOF

    jid_post_post=$(submit_stage PostDamagePostTech "$foldername" "$learning_rates")
    echo "\"${tag}\",\"${value_lr}\",\"${control_lr}\",\"PostDamagePostTech\",\"${jid_post_post}\",\"\",\"${export_folder}\"" >> "$summary"

    jid_post_interm=$(submit_stage PostDamageIntermTech "$foldername" "$learning_rates" "afterok:${jid_post_post}")
    echo "\"${tag}\",\"${value_lr}\",\"${control_lr}\",\"PostDamageIntermTech\",\"${jid_post_interm}\",\"afterok:${jid_post_post}\",\"${export_folder}\"" >> "$summary"

    jid_pre_post=$(submit_stage PreDamagePostTech "$foldername" "$learning_rates" "afterok:${jid_post_post}")
    echo "\"${tag}\",\"${value_lr}\",\"${control_lr}\",\"PreDamagePostTech\",\"${jid_pre_post}\",\"afterok:${jid_post_post}\",\"${export_folder}\"" >> "$summary"

    jid_post_pre=$(submit_stage PostDamagePreTech "$foldername" "$learning_rates" "afterok:${jid_post_interm}")
    echo "\"${tag}\",\"${value_lr}\",\"${control_lr}\",\"PostDamagePreTech\",\"${jid_post_pre}\",\"afterok:${jid_post_interm}\",\"${export_folder}\"" >> "$summary"

    jid_pre_interm=$(submit_stage PreDamageIntermTech "$foldername" "$learning_rates" "afterok:${jid_post_interm}:${jid_pre_post}")
    echo "\"${tag}\",\"${value_lr}\",\"${control_lr}\",\"PreDamageIntermTech\",\"${jid_pre_interm}\",\"afterok:${jid_post_interm}:${jid_pre_post}\",\"${export_folder}\"" >> "$summary"

    jid_pre_pre=$(submit_stage PreDamagePreTech "$foldername" "$learning_rates" "afterok:${jid_post_pre}:${jid_pre_interm}")
    echo "\"${tag}\",\"${value_lr}\",\"${control_lr}\",\"PreDamagePreTech\",\"${jid_pre_pre}\",\"afterok:${jid_post_pre}:${jid_pre_interm}\",\"${export_folder}\"" >> "$summary"

    cat >> "${export_folder}/run_manifest.txt" <<EOF
PostDamagePostTech job: ${jid_post_post}
PostDamageIntermTech job: ${jid_post_interm} (afterok:${jid_post_post})
PreDamagePostTech job: ${jid_pre_post} (afterok:${jid_post_post})
PostDamagePreTech job: ${jid_post_pre} (afterok:${jid_post_interm})
PreDamageIntermTech job: ${jid_pre_interm} (afterok:${jid_post_interm}:${jid_pre_post})
PreDamagePreTech job: ${jid_pre_pre} (afterok:${jid_post_pre}:${jid_pre_interm})
EOF

    printf 'Submitted DGM LR %s: final=%s\n' "$learning_rates" "$jid_pre_pre"
done

printf 'Wrote job summary: %s\n' "$summary"
