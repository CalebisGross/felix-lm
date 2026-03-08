"""Debug NaN at 100M scale. Run on MI300X to diagnose."""

import torch

from felix_lm.model import FelixLM
from scripts.train_scaled import CONFIGS


def main():
    config = CONFIGS["m0_100m"]()
    model = FelixLM(config).cuda()

    print("=== Init check ===")
    x = torch.randint(0, 50257, (4, 2048)).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(x, x)
    print(f"Batch=4 bf16 loss (no grad): {result['loss'].item():.3f}")

    # Train 30 steps: math backend (forced in attention.py) + bf16 + batch=4
    print("\n=== Training bf16, bs=4, lr=6e-4 ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

    for step in range(30):
        x = torch.randint(0, 50257, (4, 2048)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(x, x)
        loss = result["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step {step}: loss={loss.item()} -- DEAD")
            break

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step}: loss={loss.item():.3f}  grad_norm={gn.item():.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
