#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

PREFIX="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_001"
PRETRAINED_FOLDER="${PREFIX}/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
FOLDERNAME="TwoStageTech_RDIntensityScale_0p5_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
JOB_NAME="${PREFIX}/${FOLDERNAME}"
STAGE_SCRIPT="half_rd_stage.sbatch"

mkdir -p "${JOB_NAME}" "./job-outs/${FOLDERNAME}"

for stage in PostDamagePostTech PreDamagePostTech; do
    if [ ! -d "${JOB_NAME}/${stage}" ]; then
        cp -a "${PRETRAINED_FOLDER}/${stage}" "${JOB_NAME}/${stage}"
    fi
done

cat > "${JOB_NAME}/run_manifest.txt" <<EOF
Run: half RD/technology jump intensity scale
Created: $(date)
Tech jump intensity scale: 0.5
Interpretation: J_g = 0.5 * exp(logR) / varrho, with branch probabilities applied afterward.
Pretrained source: ${PRETRAINED_FOLDER}
Learning rates: warmup_cosine active regimes 10e-6,10e-4
Copied unchanged regimes: PostDamagePostTech, PreDamagePostTech
Submitted stages: PostDamageIntermTech, PostDamagePreTech, PreDamageIntermTech, PreDamagePreTech
EOF

jid_interm=$(sbatch --parsable \
    --job-name=halfRD_PostDamageIntermTech \
    --output="./job-outs/${FOLDERNAME}/PostDamageIntermTech-%j.out" \
    --error="./job-outs/${FOLDERNAME}/PostDamageIntermTech-%j.err" \
    --export=ALL,STAGE=PostDamageIntermTech \
    "$STAGE_SCRIPT")

jid_post_pre=$(sbatch --parsable \
    --dependency=afterok:${jid_interm} \
    --job-name=halfRD_PostDamagePreTech \
    --output="./job-outs/${FOLDERNAME}/PostDamagePreTech-%j.out" \
    --error="./job-outs/${FOLDERNAME}/PostDamagePreTech-%j.err" \
    --export=ALL,STAGE=PostDamagePreTech \
    "$STAGE_SCRIPT")

jid_pre_interm=$(sbatch --parsable \
    --dependency=afterok:${jid_interm} \
    --job-name=halfRD_PreDamageIntermTech \
    --output="./job-outs/${FOLDERNAME}/PreDamageIntermTech-%j.out" \
    --error="./job-outs/${FOLDERNAME}/PreDamageIntermTech-%j.err" \
    --export=ALL,STAGE=PreDamageIntermTech \
    "$STAGE_SCRIPT")

jid_pre_pre=$(sbatch --parsable \
    --dependency=afterok:${jid_post_pre}:${jid_pre_interm} \
    --job-name=halfRD_PreDamagePreTech \
    --output="./job-outs/${FOLDERNAME}/PreDamagePreTech-%j.out" \
    --error="./job-outs/${FOLDERNAME}/PreDamagePreTech-%j.err" \
    --export=ALL,STAGE=PreDamagePreTech \
    "$STAGE_SCRIPT")

printf 'Submitted half-RD-intensity job chain:\n'
printf '  PostDamageIntermTech: %s\n' "$jid_interm"
printf '  PostDamagePreTech:    %s (afterok:%s)\n' "$jid_post_pre" "$jid_interm"
printf '  PreDamageIntermTech:  %s (afterok:%s)\n' "$jid_pre_interm" "$jid_interm"
printf '  PreDamagePreTech:     %s (afterok:%s:%s)\n' "$jid_pre_pre" "$jid_post_pre" "$jid_pre_interm"
printf 'Output folder: %s\n' "$JOB_NAME"
