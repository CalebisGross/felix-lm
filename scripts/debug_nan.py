"""Find max batch size for each config on MI300X with math SDPA + bf16."""

import torch

from felix_lm.model import FelixLM
from scripts.train_scaled import CONFIGS


def find_max_batch(config_name, max_try=64):
    """Binary search for max batch size that fits in memory."""
    config = CONFIGS[config_name]()
    lo, hi, best = 1, max_try, 0

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            model = FelixLM(config).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
            x = torch.randint(0, 50257, (mid, 2048)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(x, x)
            result["loss"].backward()
            optimizer.step()
            del model, optimizer, x, result
            torch.cuda.empty_cache()
            best = mid
            print(f"  {config_name} bs={mid}: OK")
            lo = mid + 1
        except torch.OutOfMemoryError:
            del model, optimizer
            torch.cuda.empty_cache()
            print(f"  {config_name} bs={mid}: OOM")
            hi = mid - 1
        except Exception:
            # Model might not have been created
            torch.cuda.empty_cache()
            hi = mid - 1

    return best


def main():
    for name in ["m0_100m", "felix_100m", "m0_500m", "felix_500m"]:
        print(f"\n=== {name} ===")
        bs = find_max_batch(name)
        print(f"  Max batch size: {bs}")


if __name__ == "__main__":
    main()
