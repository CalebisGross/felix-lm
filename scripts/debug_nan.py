"""Find max batch size for each config on MI300X with math SDPA + bf16."""

import argparse

import torch

from felix_lm.model import FelixLM
from felix_lm.v2.config import FelixV2Config
from felix_lm.v2.model import FelixLMv2
from scripts.train_scaled import CONFIGS


def find_max_batch(config_name, max_try=64):
    """Binary search for max batch size that fits in memory."""
    config = CONFIGS[config_name]()
    is_v2 = isinstance(config, FelixV2Config)
    lo, hi, best = 1, max_try, 0

    while lo <= hi:
        mid = (lo + hi) // 2
        model = None
        optimizer = None
        try:
            torch.cuda.empty_cache()
            if is_v2:
                model = FelixLMv2(config).cuda()
            else:
                model = FelixLM(config).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
            x = torch.randint(0, 50257, (mid, 2048)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(x, x)
            result["loss"].backward()
            optimizer.step()
            del model, optimizer, x, result
            torch.cuda.empty_cache()
            model = None
            optimizer = None
            best = mid
            print(f"  {config_name} bs={mid}: OK")
            lo = mid + 1
        except torch.OutOfMemoryError:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            torch.cuda.empty_cache()
            print(f"  {config_name} bs={mid}: OOM")
            hi = mid - 1
        except Exception as e:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            torch.cuda.empty_cache()
            print(f"  {config_name} bs={mid}: ERROR: {e}")
            hi = mid - 1

    return best


ALL_V1 = ["m0_100m", "felix_100m", "m0_500m", "felix_500m"]
ALL_V2 = ["felix_v2_100m", "felix_v2_500m"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(CONFIGS.keys()),
        help="Test a single config (default: all v1)",
    )
    parser.add_argument("--v2", action="store_true", help="Test v2 configs")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.config:
        configs = [args.config]
    elif args.v2:
        configs = ALL_V2
    else:
        configs = ALL_V1

    for name in configs:
        print(f"\n=== {name} ===")
        bs = find_max_batch(name)
        print(f"  Max batch size: {bs}")


if __name__ == "__main__":
    main()
