#!/bin/bash
# Run all scaled experiments on MI300X, then shut down.
# Usage: nohup bash scripts/run_scaled.sh &> training.log &
set -e

echo "=== Starting scaled experiments $(date) ==="

# 100M experiments
# M0: batch=64, grad_accum=4, effective=256
# MSPM: batch=16, grad_accum=16, effective=256 (linear attn needs smaller batch)
echo "--- m0_100m ---"
python scripts/train_scaled.py --config m0_100m --device cuda \
    --batch-size 64 --grad-accum 4

echo "--- felix_100m ---"
python scripts/train_scaled.py --config felix_100m --device cuda \
    --batch-size 16 --grad-accum 16

# 500M experiments
# M0: batch=32, grad_accum=8, effective=256
# MSPM: batch=8, grad_accum=32, effective=256 (4 streams * larger dim)
echo "--- m0_500m ---"
python scripts/train_scaled.py --config m0_500m --device cuda \
    --batch-size 32 --grad-accum 8

echo "--- felix_500m ---"
python scripts/train_scaled.py --config felix_500m --device cuda \
    --batch-size 8 --grad-accum 32

echo "=== All experiments complete $(date) ==="

# Auto-shutdown to stop charges
sudo shutdown -h now
