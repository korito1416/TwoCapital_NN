#!/bin/bash

set -euo pipefail

cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

# Reuse the existing low-LR continuation workflow, but submit a distinct
# tiny-LR sweep with lower learning rates and separate output/job names.
SOURCE_SCRIPT="submit_logxi001_extension_low_lr_sweep.sh"
TMP_SCRIPT="$(mktemp /tmp/logxi001_tiny_lr_sweep.XXXXXX.sh)"
trap 'rm -f "$TMP_SCRIPT"' EXIT

awk '
    /^LEARNING_RATE_GRID=\(/ {
        print "LEARNING_RATE_GRID=("
        print "    \"10e-9,10e-9,10e-9,10e-9\""
        print "    \"10e-8,10e-8,10e-8,10e-8\""
        print "    \"20e-8,20e-8,20e-8,20e-8\""
        print "    \"40e-8,40e-8,40e-8,40e-8\""
        print ")"
        in_grid = 1
        next
    }
    in_grid {
        if (/^\)/) {
            in_grid = 0
        }
        next
    }
    { print }
' "$SOURCE_SCRIPT" \
    | sed \
        -e 's/_lowLR_/_tinyLR_/g' \
        -e 's/low-learning-rate/tiny-learning-rate/g' \
        -e 's/low-LR/tiny-LR/g' \
        -e 's/lowLR/tinyLR/g' \
        -e 's/LowLR/TinyLR/g' \
        -e 's/logxiLow/logxiTiny/g' \
        -e 's/low_lr/tiny_lr/g' \
    > "$TMP_SCRIPT"

chmod +x "$TMP_SCRIPT"

if [ "${VALIDATE_ONLY:-0}" = "1" ]; then
    bash -n "$TMP_SCRIPT"
    exit 0
fi

exec bash "$TMP_SCRIPT"
