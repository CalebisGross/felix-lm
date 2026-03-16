"""Find max batch size for each config on MI300X with math SDPA + bf16."""

import argparse

import torch

from felix_lm.model import FelixLM
from felix_lm.v2.config import FelixV2Config
from felix_lm.v2.model import FelixLMv2
from felix_lm.v3.config import FelixV3Config
from felix_lm.v3.model import FelixLMv3
from scripts.train_scaled import CONFIGS


def find_max_batch(config_name, max_try=64, steps=5):
    """Binary search for max batch size that fits in memory.

    Runs `steps` forward+backward passes to catch delayed OOMs.
    """
    config = CONFIGS[config_name]()
    is_v3 = isinstance(config, FelixV3Config)
    is_v2 = isinstance(config, FelixV2Config)
    lo, hi, best = 1, max_try, 0

    while lo <= hi:
        mid = (lo + hi) // 2
        model = None
        optimizer = None
        try:
            torch.cuda.empty_cache()
            if is_v3:
                model = FelixLMv3(config).cuda()
            elif is_v2:
                model = FelixLMv2(config).cuda()
            else:
                model = FelixLM(config).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

            # Run multiple steps to catch delayed OOMs
            for step in range(steps):
                x = torch.randint(0, 50257, (mid, 2048)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(x, x)
                result["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                del x, result

                if step == 0:
                    loss_val = result["loss"].item() if "loss" in result else 0
                    # Check for NaN on first step
                    if loss_val != loss_val:  # NaN check
                        raise ValueError("NaN loss on step 0!")

            del model, optimizer
            torch.cuda.empty_cache()
            model = None
            optimizer = None
            best = mid
            print(f"  {config_name} bs={mid}: OK ({steps} steps)")
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
ALL_V3 = [
    "v3_baseline_100m",
    "v3_100m_proj_r64",
    "v3_baseline_500m",
    "v3_500m_proj_r64",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(CONFIGS.keys()),
        help="Test a single config (default: all v3)",
    )
    parser.add_argument("--v1", action="store_true", help="Test v1 configs")
    parser.add_argument("--v2", action="store_true", help="Test v2 configs")
    parser.add_argument("--v3", action="store_true", help="Test v3 configs (default)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=5, help="Steps per batch size test")
    args = parser.parse_args()

    if args.config:
        configs = [args.config]
    elif args.v1:
        configs = ALL_V1
    elif args.v2:
        configs = ALL_V2
    else:
        configs = ALL_V3

    print(f"Testing {len(configs)} configs, {args.steps} steps each")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")
    print()

    results = {}
    for name in configs:
        print(f"=== {name} ===")
        bs = find_max_batch(name, steps=args.steps)
        results[name] = bs
        print(f"  -> Max batch size: {bs}\n")

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, bs in results.items():
        print(f"  {name}: max_batch={bs}")


if __name__ == "__main__":
    main()
