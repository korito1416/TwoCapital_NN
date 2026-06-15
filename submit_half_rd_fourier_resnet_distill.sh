#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_fourier_resnet_distill"
PRETRAINED_FOLDER="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
STAGE_SCRIPT="half_rd_full_stage.sbatch"
MODEL_DIR="models_fourier_resnet"
BATCH_SIZE="128"
NUM_ITERATIONS1="1000000"
NUM_ITERATIONS2="1000000"
TECH_JUMP_INTENSITY_SCALE="0.5"
HIDDEN_LAYER_ACTIVATIONS="swish,tanh,tanh,softplus"
OUTPUT_LAYER_ACTIVATIONS="softplus,custom,custom,softplus"

# tag|scheduler|neurons|blocks|posttech_lr|active_lr|distill_steps|distill_lr
VARIANTS=(
    "polish32|warmup_cosine|32|4|5e-6,5e-5|5e-6,5e-5|50000|1e-3"
    "wide64|warmup_cosine|64|4|10e-6,10e-4|10e-6,10e-4|75000|1e-3"
    "deep64_onecycle|onecycle|64|6|5e-5,5e-5|5e-5,5e-5|75000|5e-4"
)

submit_stage() {
    local stage="$1"
    local foldername="$2"
    local dependency="${3:-}"
    local dependency_args=()

    if [ -n "$dependency" ]; then
        dependency_args=(--dependency="$dependency")
    fi

    PREFIX="$PREFIX" \
    PRETRAINED_FOLDER="$PRETRAINED_FOLDER" \
    MODEL_DIR="$MODEL_DIR" \
    FOLDERNAME="$foldername" \
    JOB_NAME="${PREFIX}/${foldername}" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_NEURONS="$num_neurons" \
    NUM_HIDDEN_LAYERS="$num_hidden_layers" \
    NUM_ITERATIONS1="$NUM_ITERATIONS1" \
    NUM_ITERATIONS2="$NUM_ITERATIONS2" \
    LOGGING_FREQUENCY="1000" \
    HIDDEN_LAYER_ACTIVATIONS="$HIDDEN_LAYER_ACTIVATIONS" \
    OUTPUT_LAYER_ACTIVATIONS="$OUTPUT_LAYER_ACTIVATIONS" \
    LEARNING_RATES_POSTTECH="$learning_rates_posttech" \
    LEARNING_RATES_ACTIVE="$learning_rates_active" \
    LEARNING_RATE_SCHEDULE_TYPE="$schedule_type" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" \
    DISTILL_ITERATIONS="$distill_iterations" \
    DISTILL_LEARNING_RATE="$distill_learning_rate" \
    DISTILL_LOGGING_FREQUENCY="1000" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="frd_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

submit_variant() {
    local tag="$1"
    schedule_type="$2"
    num_neurons="$3"
    num_hidden_layers="$4"
    learning_rates_posttech="$5"
    learning_rates_active="$6"
    distill_iterations="$7"
    distill_learning_rate="$8"

    local foldername="TwoStageTech_RDIntensityScale_0p5_fourier_resnet_distill_${tag}_LR_${schedule_type}_${learning_rates_active}_${BATCH_SIZE}_neurons_${num_neurons}_#HiddenLayer_${num_hidden_layers}_num_iterations${NUM_ITERATIONS2}"
    local job_name="${PREFIX}/${foldername}"
    mkdir -p "$job_name" "./job-outs/${foldername}"

    cat > "${job_name}/run_manifest.txt" <<EOF
Run: half technology jump intensity Fourier ResNet with teacher distillation
Created: $(date)
Tech jump intensity scale: ${TECH_JUMP_INTENSITY_SCALE}
Model directory: ${MODEL_DIR}
Pretrained teacher source: ${PRETRAINED_FOLDER}
Variant tag: ${tag}
Architecture: Fourier-feature LayerNorm residual network; original BatchNorm MLP used only as frozen teacher.
Learning-rate schedule: ${schedule_type}
Post-tech learning rates: ${learning_rates_posttech}
Active-regime learning rates: ${learning_rates_active}
Post-tech iterations: ${NUM_ITERATIONS1}
Active-regime iterations: ${NUM_ITERATIONS2}
Batch size: ${BATCH_SIZE}
Neurons: ${num_neurons}
Residual blocks: ${num_hidden_layers}
Distillation iterations per stage: ${distill_iterations}
Distillation learning rate: ${distill_learning_rate}
EOF

    jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername")
    jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "afterok:${jid_post_post}")
    jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "afterok:${jid_post_interm}")
    jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "afterok:${jid_post_post}")
    jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "afterok:${jid_post_interm}:${jid_pre_post}")
    jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "afterok:${jid_post_pre}:${jid_pre_interm}")

    printf 'Submitted Fourier ResNet distill variant %s\n' "$foldername"
    printf '  PostDamagePostTech:  %s\n' "$jid_post_post"
    printf '  PostDamageIntermTech:%s\n' "$jid_post_interm"
    printf '  PostDamagePreTech:   %s\n' "$jid_post_pre"
    printf '  PreDamagePostTech:   %s\n' "$jid_pre_post"
    printf '  PreDamageIntermTech: %s\n' "$jid_pre_interm"
    printf '  PreDamagePreTech:    %s\n' "$jid_pre_pre"
}

for variant in "${VARIANTS[@]}"; do
    IFS="|" read -r tag schedule_type num_neurons num_hidden_layers learning_rates_posttech learning_rates_active distill_iterations distill_learning_rate <<< "$variant"
    submit_variant "$tag" "$schedule_type" "$num_neurons" "$num_hidden_layers" "$learning_rates_posttech" "$learning_rates_active" "$distill_iterations" "$distill_learning_rate"
done
