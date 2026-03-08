#!/bin/bash
# Run all scaled experiments on MI300X, then shut down.
# Usage: nohup bash scripts/run_scaled.sh &> training.log &
set -e

echo "=== Starting scaled experiments $(date) ==="

# 100M experiments
echo "--- m0_100m ---"
python scripts/train_scaled.py --config m0_100m --device cuda \
    --batch-size 64 --grad-accum 4 --lr 6e-4

echo "--- felix_100m ---"
python scripts/train_scaled.py --config felix_100m --device cuda \
    --batch-size 16 --grad-accum 16 --lr 6e-4

# 500M experiments
echo "--- m0_500m ---"
python scripts/train_scaled.py --config m0_500m --device cuda \
    --batch-size 32 --grad-accum 8 --lr 6e-4

echo "--- felix_500m ---"
python scripts/train_scaled.py --config felix_500m --device cuda \
    --batch-size 8 --grad-accum 32 --lr 6e-4

echo "=== All experiments complete $(date) ==="

# Auto-shutdown to stop charges
sudo shutdown -h now
