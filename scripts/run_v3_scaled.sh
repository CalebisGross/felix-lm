#!/bin/bash
# Run v3 scaling experiments on MI300X.
# Usage: nohup bash scripts/run_v3_scaled.sh &> v3_training.log &
#
# Budget: ~$49 (~7.5 hours @ $6.50/hr)
#   100M: 1B tokens each (~2-3hr per run, ~5hr total)
#   500M: 250M tokens each (~1-1.5hr per run, ~2.5hr total)
#   Total: ~7.5hr = ~$49
#
# IMPORTANT: torch.compile + SDPA math backend must be used on MI300X.
# IMPORTANT: Must use ROCm torch build, not CUDA.
# IMPORTANT: Run debug_nan.py FIRST before launching this script.
set -e

source "$(dirname "$0")/../.venv/bin/activate"

echo "=== v3 Scaling Run $(date) ==="
echo "GPU: $(rocm-smi --showproductname 2>/dev/null | grep 'Card' || echo 'unknown')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# --- 100M experiments (1B tokens each) ---

echo "=== [1/4] v3_baseline_100m (control, 1B tokens) ==="
python scripts/train_scaled.py \
    --config v3_baseline_100m \
    --device cuda \
    --batch-size 16 --grad-accum 16 \
    --lr 3e-3 --beta2 0.99 \
    --compile \
    --eval-interval 1000 \
    --tokens-per-epoch 1000000000

echo ""
echo "=== [2/4] v3_100m_proj_r64 (spokes, 1B tokens) ==="
python scripts/train_scaled.py \
    --config v3_100m_proj_r64 \
    --device cuda \
    --batch-size 16 --grad-accum 16 \
    --lr 3e-3 --beta2 0.99 \
    --spoke-lr-mult 2.0 \
    --compile \
    --eval-interval 1000 \
    --tokens-per-epoch 1000000000

# --- 500M experiments (250M tokens each — directional) ---

echo ""
echo "=== [3/4] v3_baseline_500m (control, 250M tokens) ==="
python scripts/train_scaled.py \
    --config v3_baseline_500m \
    --device cuda \
    --batch-size 8 --grad-accum 32 \
    --lr 1e-3 --beta2 0.99 \
    --compile \
    --eval-interval 250 \
    --tokens-per-epoch 250000000

echo ""
echo "=== [4/4] v3_500m_proj_r64 (spokes, 250M tokens) ==="
python scripts/train_scaled.py \
    --config v3_500m_proj_r64 \
    --device cuda \
    --batch-size 8 --grad-accum 32 \
    --lr 1e-3 --beta2 0.99 \
    --spoke-lr-mult 2.0 \
    --compile \
    --eval-interval 250 \
    --tokens-per-epoch 250000000

echo ""
echo "=== All v3 experiments complete $(date) ==="

# Auto-shutdown to stop charges
sudo shutdown -h now
