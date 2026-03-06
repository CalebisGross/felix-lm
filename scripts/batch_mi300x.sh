#!/bin/bash
# MI300X Batch Experiment Runner
# Runs all 20 experiments simultaneously
#
# Usage: bash scripts/batch_mi300x.sh [--epochs 2] [--device cuda]

set -e

EPOCHS="${1:-2}"
DEVICE="${2:-cuda}"
BATCH_SIZE=8
GRAD_ACCUM=4
EVAL_SCRIPT="scripts/evaluate.py"
TRAIN_SCRIPT="scripts/train.py"

CONFIGS=(
    m2_asymmetric m2_bottleneck m2_crossattn m2_shared_deep m2_nosup
    m2_geometric m2_hadamard m2_antimerge m2_noise_merge m2_bottleneck_half
    m2_diverge_low m2_diverge_mid m2_diverge_high m2_frozen_init m2_asymmetric_bottleneck
    m2_asymmetric_diverge m2_geometric_diverge m2_progressive_unfreeze m2_stream_permute m2_asymmetric_v2
)

echo "========================================"
echo "Felix-LM MI300X Batch Runner"
echo "Epochs: $EPOCHS | Device: $DEVICE"
echo "Launching all ${#CONFIGS[@]} experiments simultaneously"
echo "========================================"

# Create log directory
mkdir -p logs

run_experiment() {
    local config="$1"
    echo "[START] $config"
    python "$TRAIN_SCRIPT" \
        --config "$config" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --eval-interval 500 \
        --save-interval 5000 \
        2>&1 | tee "logs/${config}.log"
    echo "[DONE]  $config"
}

# Launch all 20 experiments at once
for config in "${CONFIGS[@]}"; do
    run_experiment "$config" &
done

echo "All ${#CONFIGS[@]} experiments launched. Waiting for completion..."
wait
echo ""
echo "========================================"
echo "All training complete. Running evaluations..."
echo "========================================"

echo ""
echo "CONFIG                          | TEST PPL"
echo "-------------------------------|----------"
for config in "${CONFIGS[@]}"; do
    CKPT="checkpoints/${config}/best.pt"
    if [ -f "$CKPT" ]; then
        PPL=$(python "$EVAL_SCRIPT" --checkpoint "$CKPT" --split test --device "$DEVICE" --batch-size "$BATCH_SIZE" 2>&1 | grep "Test perplexity" | awk '{print $NF}')
        printf "%-32s| %s\n" "$config" "$PPL"
    else
        printf "%-32s| NO CHECKPOINT\n" "$config"
    fi
done

echo ""
echo "========================================"
echo "Batch complete!"
echo "========================================"
