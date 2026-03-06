#!/bin/bash
# MI300X Batch Experiment Runner
# Runs 20 experiments in 4 parallel waves of 5
#
# Usage: bash scripts/batch_mi300x.sh [--epochs 2] [--device cuda]

set -e

EPOCHS="${1:-2}"
DEVICE="${2:-cuda}"
BATCH_SIZE=8
GRAD_ACCUM=4
EVAL_SCRIPT="scripts/evaluate.py"
TRAIN_SCRIPT="scripts/train.py"

echo "========================================"
echo "Felix-LM MI300X Batch Runner"
echo "Epochs: $EPOCHS | Device: $DEVICE"
echo "Parallel: 5 processes per wave"
echo "========================================"

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

# Create log directory
mkdir -p logs

echo ""
echo "=== WAVE 1: Already Built ==="
run_experiment m2_asymmetric &
run_experiment m2_bottleneck &
run_experiment m2_crossattn &
run_experiment m2_shared_deep &
run_experiment m2_nosup &  # 3-epoch reference (use --epochs 3 override if needed)
wait
echo "=== WAVE 1 COMPLETE ==="

echo ""
echo "=== WAVE 2: Novel Architecture ==="
run_experiment m2_geometric &
run_experiment m2_hadamard &
run_experiment m2_antimerge &
run_experiment m2_noise_merge &
run_experiment m2_bottleneck_half &
wait
echo "=== WAVE 2 COMPLETE ==="

echo ""
echo "=== WAVE 3: Training Objectives + Combos ==="
run_experiment m2_diverge_low &
run_experiment m2_diverge_mid &
run_experiment m2_diverge_high &
run_experiment m2_frozen_init &
run_experiment m2_asymmetric_bottleneck &
wait
echo "=== WAVE 3 COMPLETE ==="

echo ""
echo "=== WAVE 4: Combos + Wild Cards ==="
run_experiment m2_asymmetric_diverge &
run_experiment m2_geometric_diverge &
run_experiment m2_progressive_unfreeze &
run_experiment m2_stream_permute &
run_experiment m2_asymmetric_v2 &
wait
echo "=== WAVE 4 COMPLETE ==="

echo ""
echo "========================================"
echo "All training complete. Running evaluations..."
echo "========================================"

# Evaluate all experiments sequentially
CONFIGS=(
    m2_asymmetric m2_bottleneck m2_crossattn m2_shared_deep m2_nosup
    m2_geometric m2_hadamard m2_antimerge m2_noise_merge m2_bottleneck_half
    m2_diverge_low m2_diverge_mid m2_diverge_high m2_frozen_init m2_asymmetric_bottleneck
    m2_asymmetric_diverge m2_geometric_diverge m2_progressive_unfreeze m2_stream_permute m2_asymmetric_v2
)

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
