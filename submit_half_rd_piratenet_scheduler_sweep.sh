#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_piratenet_scheduler"
STAGE_SCRIPT="half_rd_full_stage.sbatch"
MODEL_DIR="models_piratenet"
PRETRAINED_FOLDER="None"
BATCH_SIZE="128"
NUM_ITERATIONS1="1000000"
NUM_ITERATIONS2="1000000"
TECH_JUMP_INTENSITY_SCALE="0.5"

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
    LEARNING_RATES_POSTTECH="$learning_rates_posttech" \
    LEARNING_RATES_ACTIVE="$learning_rates_active" \
    LEARNING_RATE_SCHEDULE_TYPE="$schedule_type" \
    TECH_JUMP_INTENSITY_SCALE="$TECH_JUMP_INTENSITY_SCALE" \
    sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="pirsch_${stage}" \
        --output="./job-outs/${foldername}/${stage}-%j.out" \
        --error="./job-outs/${foldername}/${stage}-%j.err" \
        --export=ALL,STAGE="$stage" \
        "$STAGE_SCRIPT"
}

submit_variant() {
    schedule_type="$1"
    num_neurons="$2"
    num_hidden_layers="$3"
    learning_rates_active="$4"
    learning_rates_posttech="$5"
    note="$6"

    foldername="TwoStageTech_RDIntensityScale_0p5_piratenet_sched_${schedule_type}_LR_${learning_rates_active}_${BATCH_SIZE}_neurons_${num_neurons}_#HiddenLayer_${num_hidden_layers}_num_iterations${NUM_ITERATIONS2}"
    job_name="${PREFIX}/${foldername}"
    mkdir -p "$job_name" "./job-outs/${foldername}"

    cat > "${job_name}/run_manifest.txt" <<EOF
Run: half technology jump intensity PirateNet scheduler sweep
Created: $(date)
Tech jump intensity scale: ${TECH_JUMP_INTENSITY_SCALE}
Model directory: ${MODEL_DIR}
Pretrained source: ${PRETRAINED_FOLDER}
Post-tech learning rates: ${learning_rates_posttech}
Active-regime learning rates: ${learning_rates_active}
Learning-rate schedule: ${schedule_type}
Post-tech iterations: ${NUM_ITERATIONS1}
Active-regime iterations: ${NUM_ITERATIONS2}
Neurons: ${num_neurons}
Hidden layers / Pirate blocks: ${num_hidden_layers}
Architecture: Fourier-feature adaptive residual network.
Note: ${note}
EOF

    jid_post_post=$(submit_stage "PostDamagePostTech" "$foldername")
    jid_post_interm=$(submit_stage "PostDamageIntermTech" "$foldername" "afterok:${jid_post_post}")
    jid_post_pre=$(submit_stage "PostDamagePreTech" "$foldername" "afterok:${jid_post_interm}")
    jid_pre_post=$(submit_stage "PreDamagePostTech" "$foldername" "afterok:${jid_post_post}")
    jid_pre_interm=$(submit_stage "PreDamageIntermTech" "$foldername" "afterok:${jid_post_interm}:${jid_pre_post}")
    jid_pre_pre=$(submit_stage "PreDamagePreTech" "$foldername" "afterok:${jid_post_pre}:${jid_pre_interm}")

    printf 'Submitted PirateNet scheduler variant %s\n' "$foldername"
    printf '  PostDamagePostTech:  %s\n' "$jid_post_post"
    printf '  PostDamageIntermTech:%s\n' "$jid_post_interm"
    printf '  PostDamagePreTech:   %s\n' "$jid_post_pre"
    printf '  PreDamagePostTech:   %s\n' "$jid_pre_post"
    printf '  PreDamageIntermTech: %s\n' "$jid_pre_interm"
    printf '  PreDamagePreTech:    %s\n' "$jid_pre_pre"
}

for schedule in piecewiseconstant cosine cosine_restarts onecycle cyclical; do
    submit_variant "$schedule" "64" "6" \
        "10e-5,10e-5,10e-5,10e-5" \
        "10e-6,40e-4" \
        "Scheduler comparison at the same width-64 LR scale as the first warmup-cosine PirateNet run."

    submit_variant "$schedule" "128" "6" \
        "40e-6,40e-6,40e-6,40e-6" \
        "10e-6,40e-4" \
        "Scheduler comparison at the same width-128 LR scale as the first warmup-cosine PirateNet run."
done

submit_variant "None" "64" "6" \
    "40e-6,40e-6,40e-6,40e-6" \
    "10e-6,40e-5" \
    "Constant Adam polish-style run with a lower fixed control LR to reduce fixed-step instability."

submit_variant "None" "128" "6" \
    "20e-6,20e-6,20e-6,20e-6" \
    "10e-6,40e-5" \
    "Constant Adam polish-style run with a lower fixed control LR to reduce fixed-step instability."
