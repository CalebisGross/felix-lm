"""Training script for Felix-LM.

Usage:
    python scripts/train.py                          # Train M2 (default)
    python scripts/train.py --config m0_uniform      # Train M0 baseline
    python scripts/train.py --device cpu             # CPU training (slow)
    python scripts/train.py --no-wandb               # Disable wandb logging
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from felix_lm.config import (
    FelixConfig,
    make_m0_config,
    make_m2_best_config,
    make_m2_config,
    make_m2_fullcausal_config,
    make_m2_nosup_config,
)
from felix_lm.model import FelixLM
from felix_lm.utils import count_parameters

# --- Data ---


class WikiTextDataset(Dataset):
    """WikiText-103 chunked into fixed-length sequences."""

    def __init__(self, split: str, seq_len: int = 512, cache_dir: str = "./data"):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Load and tokenize
        cache_path = Path(cache_dir) / f"wikitext103_{split}_{seq_len}.pt"
        if cache_path.exists():
            print(f"Loading cached {split} data from {cache_path}")
            self.tokens = torch.load(cache_path, weights_only=True)
        else:
            print(f"Tokenizing WikiText-103 {split}...")
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
            # Tokenize in chunks to avoid OOM on large splits
            all_ids = []
            batch_size = 10000
            texts = [t for t in ds["text"] if t.strip()]
            for i in range(0, len(texts), batch_size):
                batch = "\n".join(texts[i : i + batch_size])
                all_ids.extend(tokenizer.encode(batch))
                if (i // batch_size) % 10 == 0:
                    done = min(i + batch_size, len(texts))
                    print(f"  Tokenized {done:,}/{len(texts):,} articles...")
            self.tokens = torch.tensor(all_ids, dtype=torch.long)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.tokens, cache_path)
            print(f"Cached {len(self.tokens):,} tokens to {cache_path}")

        # Chunk into sequences of seq_len + 1 (input + target)
        n_chunks = len(self.tokens) // (seq_len + 1)
        self.tokens = self.tokens[: n_chunks * (seq_len + 1)]
        self.tokens = self.tokens.view(n_chunks, seq_len + 1)
        print(f"  {split}: {len(self.tokens):,} sequences of length {seq_len}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        chunk = self.tokens[idx]
        return chunk[:-1], chunk[1:]  # input, target


# --- Training ---


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(config: FelixConfig, args):
    device = torch.device(args.device)
    use_mps = device.type == "mps"
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Model
    model = FelixLM(config).to(device)
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    n_params = count_parameters(model)
    print(f"\nModel: {config.num_stages} stages, {config.total_layers} layers, {n_params:,} params")

    # Autocast setup for mixed precision
    if use_mps:
        autocast_ctx = torch.autocast("mps", dtype=torch.float16)
        print("  Mixed precision: fp16 via MPS autocast")
    elif device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
        print("  Mixed precision: fp16 via CUDA autocast")
    else:
        autocast_ctx = torch.autocast("cpu", enabled=False)

    # Data — num_workers=0 on macOS (fork overhead), 2 on Linux
    num_workers = 0 if use_mps else 2
    pin_memory = not use_mps
    train_ds = WikiTextDataset("train", seq_len=args.seq_len, cache_dir=args.data_dir)
    val_ds = WikiTextDataset("validation", seq_len=args.seq_len, cache_dir=args.data_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    # Training params
    steps_per_epoch = len(train_loader) // args.grad_accum
    max_steps = args.epochs * steps_per_epoch
    ga = args.grad_accum
    print(f"\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"  grad_accum={ga}, {max_steps} total steps")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")

    # Wandb
    if args.use_wandb:
        import wandb

        wandb.init(
            project="felix-lm",
            config={
                "model": args.config_name,
                "params": n_params,
                "stages": config.num_stages,
                "layers": config.total_layers,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch_size": args.batch_size * args.grad_accum,
                "seq_len": args.seq_len,
                "lr": args.lr,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
            },
        )

    # Training loop
    global_step = 0
    best_val_ppl = float("inf")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        accum_steps = args.grad_accum
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad()
        accum_loss = 0.0
        last_result = None

        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward
            with autocast_ctx:
                result = model(input_ids, targets)
            loss = result["loss"] / accum_steps  # scale for accumulation
            loss.backward()

            accum_loss += result["loss"].item()
            last_result = result

            # Logging
            batch_tokens = input_ids.numel()
            epoch_loss += result["loss"].item() * batch_tokens
            epoch_tokens += batch_tokens

            # Optimizer step after accumulation
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Learning rate schedule
                lr = get_lr(global_step, args.warmup_steps, max_steps, args.lr, args.lr * 0.1)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                avg_accum_loss = accum_loss / accum_steps
                pbar.set_postfix(
                    loss=f"{avg_accum_loss:.3f}",
                    ppl=f"{math.exp(min(avg_accum_loss, 20)):.1f}",
                    lr=f"{lr:.2e}",
                )
                accum_loss = 0.0
                global_step += 1

                if args.use_wandb and global_step % args.log_interval == 0:
                    log_dict = {
                        "train/loss": avg_accum_loss,
                        "train/ppl": math.exp(min(avg_accum_loss, 20)),
                        "train/lr": lr,
                        "train/step": global_step,
                    }
                    if last_result and "per_stage_losses" in last_result:
                        for k, sl in enumerate(last_result["per_stage_losses"]):
                            log_dict[f"train/stage_{k}_loss"] = sl.item() if hasattr(sl, "item") else sl
                    if last_result and "stream_agreements" in last_result:
                        for k, ag in enumerate(last_result["stream_agreements"]):
                            log_dict[f"train/agreement_merge_{k}"] = ag.item() if hasattr(ag, "item") else ag

                    import wandb

                    wandb.log(log_dict, step=global_step)

                # Validation
                if global_step % args.eval_interval == 0:
                    val_ppl = evaluate(model, val_loader, device)
                    print(f"\n  Step {global_step}: val_ppl = {val_ppl:.2f}")

                    if args.use_wandb:
                        import wandb

                        wandb.log({"val/ppl": val_ppl}, step=global_step)

                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        torch.save(
                            {"model": model.state_dict(), "config": config, "step": global_step},
                            ckpt_dir / "best.pt",
                        )
                        print("  New best! Saved checkpoint.")

                    model.train()

                # Periodic checkpoint
                if global_step % args.save_interval == 0:
                    torch.save(
                        {"model": model.state_dict(), "config": config, "step": global_step},
                        ckpt_dir / f"step_{global_step}.pt",
                    )

        avg_loss = epoch_loss / epoch_tokens
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}, ppl: {math.exp(min(avg_loss, 20)):.2f}")

    # Final evaluation
    val_ppl = evaluate(model, val_loader, device)
    print(f"\nFinal validation perplexity: {val_ppl:.2f}")
    print(f"Best validation perplexity: {best_val_ppl:.2f}")

    if args.use_wandb:
        import wandb

        wandb.log({"val/final_ppl": val_ppl, "val/best_ppl": best_val_ppl})
        wandb.finish()


@torch.no_grad()
def evaluate(model, dataloader, device) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    use_mps = device.type == "mps"
    autocast_ctx = (
        torch.autocast("mps", dtype=torch.float16) if use_mps
        else torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with autocast_ctx:
            result = model(input_ids, targets)
        total_loss += result["loss"].item() * input_ids.numel()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 20))


def main():
    parser = argparse.ArgumentParser(description="Train Felix-LM")
    parser.add_argument(
        "--config",
        default="m2",
        choices=["m0", "m2", "m2_fullcausal", "m2_nosup", "m2_best"],
        help="Model config",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective_bs = batch_size * grad_accum)",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Checkpoint dir (default: ./checkpoints/<config>)"
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (speeds up MPS/CUDA)")
    args = parser.parse_args()

    args.use_wandb = not args.no_wandb
    args.config_name = args.config
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./checkpoints/{args.config}"

    configs = {
        "m2": make_m2_config,
        "m0": make_m0_config,
        "m2_fullcausal": make_m2_fullcausal_config,
        "m2_nosup": make_m2_nosup_config,
        "m2_best": make_m2_best_config,
    }
    config = configs[args.config]()

    train(config, args)


if __name__ == "__main__":
    main()
