#!/bin/bash
# Run all scaled experiments on MI300X, then shut down.
# Usage: nohup bash scripts/run_scaled.sh &> training.log &
#
# Batch sizes found via debug_nan.py (max → safe with headroom):
#   m0_100m:    max=24, use 20 (grad_accum=12, eff=240)
#   felix_100m: max=19, use 16 (grad_accum=16, eff=256)
#   m0_500m:    max=9,  use 8  (grad_accum=32, eff=256)
#   felix_500m: max=9,  use 8  (grad_accum=32, eff=256)
set -e

echo "=== Starting scaled experiments $(date) ==="

# 100M experiments
echo "--- m0_100m ---"
python scripts/train_scaled.py --config m0_100m --device cuda \
    --batch-size 20 --grad-accum 12 --lr 6e-4

echo "--- felix_100m ---"
python scripts/train_scaled.py --config felix_100m --device cuda \
    --batch-size 16 --grad-accum 16 --lr 6e-4

# 500M experiments
echo "--- m0_500m ---"
python scripts/train_scaled.py --config m0_500m --device cuda \
    --batch-size 8 --grad-accum 32 --lr 6e-4

echo "--- felix_500m ---"
python scripts/train_scaled.py --config felix_500m --device cuda \
    --batch-size 8 --grad-accum 32 --lr 6e-4

echo "=== All experiments complete $(date) ==="

# Auto-shutdown to stop charges
sudo shutdown -h now
