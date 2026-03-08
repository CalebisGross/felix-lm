#!/bin/bash
# Run all scaled experiments on MI300X, then shut down.
# Usage: nohup bash scripts/run_scaled.sh &> training.log &
#
# MI300X math SDPA NaN bug: batch sizes >4 produce NaN with real data
# even though debug_nan.py (random data) shows higher max. Use batch=4.
# batch=4, grad_accum=16, eff_batch=64, ~7629 opt steps per 1B tokens.
set -e

echo "=== Starting scaled experiments $(date) ==="

# 100M experiments (batch=4 confirmed working on MI300X)
echo "--- m0_100m ---"
python scripts/train_scaled.py --config m0_100m --device cuda \
    --batch-size 4 --grad-accum 16 --lr 1e-4

echo "--- felix_100m ---"
python scripts/train_scaled.py --config felix_100m --device cuda \
    --batch-size 4 --grad-accum 16 --lr 1e-4

# 500M experiments (batch=4, may need batch=2 if OOM)
echo "--- m0_500m ---"
python scripts/train_scaled.py --config m0_500m --device cuda \
    --batch-size 4 --grad-accum 16 --lr 1e-4

echo "--- felix_500m ---"
python scripts/train_scaled.py --config felix_500m --device cuda \
    --batch-size 4 --grad-accum 16 --lr 1e-4

echo "=== All experiments complete $(date) ==="

# Auto-shutdown to stop charges
sudo shutdown -h now
