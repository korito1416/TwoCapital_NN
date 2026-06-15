#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
CURRENT_HALF_FOLDER="${PREFIX}/TwoStageTech_RDIntensityScale_0p5_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
STAGE_SCRIPT="half_rd_stage.sbatch"

CURRENT_POST_DAMAGE_INTERM_JOB_ID="${CURRENT_POST_DAMAGE_INTERM_JOB_ID:-50360884}"
CURRENT_POST_DAMAGE_PRE_JOB_ID="${CURRENT_POST_DAMAGE_PRE_JOB_ID:-50360885}"
CURRENT_PRE_DAMAGE_INTERM_JOB_ID="${CURRENT_PRE_DAMAGE_INTERM_JOB_ID:-50360886}"
CURRENT_PRE_DAMAGE_PRE_JOB_ID="${CURRENT_PRE_DAMAGE_PRE_JOB_ID:-50360887}"

submit_variant() {
    local foldername="$1"
    local learning_rates="$2"
    local schedule_type="$3"
    local num_iterations="$4"
    local note="$5"

    local job_name="${PREFIX}/${foldername}"
    mkdir -p "${job_name}" "./job-outs/${foldername}"

    for stage in PostDamagePostTech PreDamagePostTech; do
        if [ ! -d "${job_name}/${stage}" ]; then
            cp -a "${CURRENT_HALF_FOLDER}/${stage}" "${job_name}/${stage}"
        fi
    done

    cat > "${job_name}/run_manifest.txt" <<EOF
Run: half RD/technology jump intensity LR continuation
Created: $(date)
Tech jump intensity scale: 0.5
Pretrained source: ${CURRENT_HALF_FOLDER}
Learning rates: ${learning_rates}
Learning-rate schedule: ${schedule_type}
Iterations per active regime: ${num_iterations}
Copied unchanged regimes: PostDamagePostTech, PreDamagePostTech
Dependency source jobs: ${CURRENT_POST_DAMAGE_INTERM_JOB_ID}, ${CURRENT_POST_DAMAGE_PRE_JOB_ID}, ${CURRENT_PRE_DAMAGE_INTERM_JOB_ID}, ${CURRENT_PRE_DAMAGE_PRE_JOB_ID}
Note: ${note}
EOF

    local jid_interm
    jid_interm=$(PRETRAINED_FOLDER="${CURRENT_HALF_FOLDER}" FOLDERNAME="${foldername}" LEARNING_RATES_ACTIVE="${learning_rates}" LEARNING_RATE_SCHEDULE_TYPE="${schedule_type}" NUM_ITERATIONS="${num_iterations}" TECH_JUMP_INTENSITY_SCALE=0.5 \
        sbatch --parsable \
        --dependency=afterok:${CURRENT_POST_DAMAGE_INTERM_JOB_ID} \
        --job-name=lrvar_PostDamageIntermTech \
        --output="./job-outs/${foldername}/PostDamageIntermTech-%j.out" \
        --error="./job-outs/${foldername}/PostDamageIntermTech-%j.err" \
        --export=ALL,STAGE=PostDamageIntermTech \
        "$STAGE_SCRIPT")

    local jid_post_pre
    jid_post_pre=$(PRETRAINED_FOLDER="${CURRENT_HALF_FOLDER}" FOLDERNAME="${foldername}" LEARNING_RATES_ACTIVE="${learning_rates}" LEARNING_RATE_SCHEDULE_TYPE="${schedule_type}" NUM_ITERATIONS="${num_iterations}" TECH_JUMP_INTENSITY_SCALE=0.5 \
        sbatch --parsable \
        --dependency=afterok:${CURRENT_POST_DAMAGE_PRE_JOB_ID}:${jid_interm} \
        --job-name=lrvar_PostDamagePreTech \
        --output="./job-outs/${foldername}/PostDamagePreTech-%j.out" \
        --error="./job-outs/${foldername}/PostDamagePreTech-%j.err" \
        --export=ALL,STAGE=PostDamagePreTech \
        "$STAGE_SCRIPT")

    local jid_pre_interm
    jid_pre_interm=$(PRETRAINED_FOLDER="${CURRENT_HALF_FOLDER}" FOLDERNAME="${foldername}" LEARNING_RATES_ACTIVE="${learning_rates}" LEARNING_RATE_SCHEDULE_TYPE="${schedule_type}" NUM_ITERATIONS="${num_iterations}" TECH_JUMP_INTENSITY_SCALE=0.5 \
        sbatch --parsable \
        --dependency=afterok:${CURRENT_PRE_DAMAGE_INTERM_JOB_ID}:${jid_interm} \
        --job-name=lrvar_PreDamageIntermTech \
        --output="./job-outs/${foldername}/PreDamageIntermTech-%j.out" \
        --error="./job-outs/${foldername}/PreDamageIntermTech-%j.err" \
        --export=ALL,STAGE=PreDamageIntermTech \
        "$STAGE_SCRIPT")

    local jid_pre_pre
    jid_pre_pre=$(PRETRAINED_FOLDER="${CURRENT_HALF_FOLDER}" FOLDERNAME="${foldername}" LEARNING_RATES_ACTIVE="${learning_rates}" LEARNING_RATE_SCHEDULE_TYPE="${schedule_type}" NUM_ITERATIONS="${num_iterations}" TECH_JUMP_INTENSITY_SCALE=0.5 \
        sbatch --parsable \
        --dependency=afterok:${CURRENT_PRE_DAMAGE_PRE_JOB_ID}:${jid_post_pre}:${jid_pre_interm} \
        --job-name=lrvar_PreDamagePreTech \
        --output="./job-outs/${foldername}/PreDamagePreTech-%j.out" \
        --error="./job-outs/${foldername}/PreDamagePreTech-%j.err" \
        --export=ALL,STAGE=PreDamagePreTech \
        "$STAGE_SCRIPT")

    printf 'Submitted LR variant: %s\n' "$foldername"
    printf '  PostDamageIntermTech: %s (afterok:%s)\n' "$jid_interm" "$CURRENT_POST_DAMAGE_INTERM_JOB_ID"
    printf '  PostDamagePreTech:    %s (afterok:%s:%s)\n' "$jid_post_pre" "$CURRENT_POST_DAMAGE_PRE_JOB_ID" "$jid_interm"
    printf '  PreDamageIntermTech:  %s (afterok:%s:%s)\n' "$jid_pre_interm" "$CURRENT_PRE_DAMAGE_INTERM_JOB_ID" "$jid_interm"
    printf '  PreDamagePreTech:     %s (afterok:%s:%s:%s)\n' "$jid_pre_pre" "$CURRENT_PRE_DAMAGE_PRE_JOB_ID" "$jid_post_pre" "$jid_pre_interm"
    printf '  Output folder: %s\n' "$job_name"
}

submit_variant \
    "TwoStageTech_RDIntensityScale_0p5_finetune_LR_warmup_cosine_5e-6,2e-4_128_neurons_32_#HiddenLayer_4_num_iterations500000" \
    "5e-6,2e-4" \
    "warmup_cosine" \
    "500000" \
    "Balanced continuation: smaller value LR than the first half-intensity run, still enough control LR to keep FOC errors moving."

submit_variant \
    "TwoStageTech_RDIntensityScale_0p5_finetune_LR_None_2e-6,5e-5_128_neurons_32_#HiddenLayer_4_num_iterations500000" \
    "2e-6,5e-5" \
    "None" \
    "500000" \
    "Polish continuation: constant small Adam steps after the main run so the schedule does not decay to zero."
