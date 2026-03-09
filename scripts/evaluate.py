"""Evaluate a trained Felix-LM checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test
"""

import argparse
import math

import torch
from torch.utils.data import DataLoader

from felix_lm.model import FelixLM
from felix_lm.utils import count_parameters
from felix_lm.v2.config import FelixV2Config
from felix_lm.v2.model import FelixLMv2


def main():
    parser = argparse.ArgumentParser(description="Evaluate Felix-LM")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, FelixV2Config):
        model = FelixLMv2(config).to(device)
    else:
        model = FelixLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = count_parameters(model)
    if isinstance(config, FelixV2Config):
        nl, ns = config.num_layers, config.num_streams
        print(f"Model: v2 ({nl} layers, {ns} streams), {n_params:,} params")
    else:
        print(f"Model: {config.num_stages} stages, {n_params:,} params")
    print(f"Trained for {ckpt.get('step', '?')} steps")

    # Load data
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import WikiTextDataset

    ds = WikiTextDataset(args.split, seq_len=args.seq_len, cache_dir=args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Evaluate
    total_loss = 0.0
    total_tokens = 0
    per_stage_totals = None

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            result = model(input_ids, targets)

            total_loss += result["loss"].item() * input_ids.numel()
            total_tokens += input_ids.numel()

            if "per_stage_losses" in result:
                if per_stage_totals is None:
                    per_stage_totals = [0.0] * len(result["per_stage_losses"])
                for k, sl in enumerate(result["per_stage_losses"]):
                    per_stage_totals[k] += sl * input_ids.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))

    print(f"\n{args.split} set results:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity:   {ppl:.2f}")

    if per_stage_totals:
        print("  Per-stage losses:")
        for k, total in enumerate(per_stage_totals):
            stage_loss = total / total_tokens
            print(f"    Stage {k}: {stage_loss:.4f} (ppl {math.exp(min(stage_loss, 20)):.2f})")


if __name__ == "__main__":
    main()
