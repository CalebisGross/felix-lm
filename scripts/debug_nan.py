"""Debug NaN at 100M scale. Run on MI300X to diagnose."""

import torch

from felix_lm.model import FelixLM
from scripts.train_scaled import CONFIGS


def check_nans(model, name=""):
    """Check all parameters for NaN/Inf."""
    for pname, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f"  NaN in param: {pname}")
            return True
        if torch.isinf(p).any():
            print(f"  Inf in param: {pname}")
            return True
    return False


def main():
    config = CONFIGS["m0_100m"]()
    model = FelixLM(config).cuda()

    print("=== Init check ===")
    check_nans(model, "init")

    # Single sample forward
    x = torch.randint(0, 50257, (1, 2048)).cuda()
    result = model(x, x)
    print(f"Batch=1 loss: {result['loss'].item():.3f}")

    # Larger batch forward only (no backward)
    x = torch.randint(0, 50257, (32, 2048)).cuda()
    with torch.no_grad():
        result = model(x, x)
    print(f"Batch=32 loss (no grad): {result['loss'].item():.3f}")

    # Now train 20 steps at batch=32, fp32, lr=6e-4
    print("\n=== Training fp32, bs=32, lr=6e-4 ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

    for step in range(30):
        x = torch.randint(0, 50257, (32, 2048)).cuda()
        result = model(x, x)
        loss = result["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step {step}: loss={loss.item()} -- DEAD")
            check_nans(model, f"step {step}")
            break

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step}: loss={loss.item():.3f}  grad_norm={gn.item():.1f}")

    # Try again with lower LR
    print("\n=== Fresh model, fp32, bs=32, lr=1e-4 ===")
    del model, optimizer
    torch.cuda.empty_cache()

    model = FelixLM(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(30):
        x = torch.randint(0, 50257, (32, 2048)).cuda()
        result = model(x, x)
        loss = result["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step {step}: loss={loss.item()} -- DEAD")
            check_nans(model, f"step {step}")
            break

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step}: loss={loss.item():.3f}  grad_norm={gn.item():.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
