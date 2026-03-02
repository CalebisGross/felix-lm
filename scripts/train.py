"""Training script for Felix-LM.

Usage:
    python scripts/train.py                          # Train M2 (default)
    python scripts/train.py --config m0_uniform      # Train M0 baseline
    python scripts/train.py --device cpu             # CPU training (slow)
    python scripts/train.py --no-wandb               # Disable wandb logging
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from felix_lm.config import FelixConfig, make_m0_config, make_m2_config
from felix_lm.diagnostics import collect_merge_diagnostics, cross_stream_agreement
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
            all_text = "\n".join([t for t in ds["text"] if t.strip()])
            encoded = tokenizer.encode(all_text)
            self.tokens = torch.tensor(encoded, dtype=torch.long)
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
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Model
    model = FelixLM(config).to(device)
    n_params = count_parameters(model)
    print(f"\nModel: {config.num_stages} stages, {config.total_layers} layers, {n_params:,} params")

    # Data
    train_ds = WikiTextDataset("train", seq_len=args.seq_len, cache_dir=args.data_dir)
    val_ds = WikiTextDataset("validation", seq_len=args.seq_len, cache_dir=args.data_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    # Training params
    steps_per_epoch = len(train_loader)
    max_steps = args.epochs * steps_per_epoch
    print(f"\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch, {max_steps} total steps")

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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Learning rate schedule
            lr = get_lr(global_step, args.warmup_steps, max_steps, args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            result = model(input_ids, targets)
            loss = result["loss"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Logging
            batch_tokens = input_ids.numel()
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                ppl=f"{math.exp(min(loss.item(), 20)):.1f}",
                lr=f"{lr:.2e}",
            )

            if args.use_wandb and global_step % args.log_interval == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/ppl": math.exp(min(loss.item(), 20)),
                    "train/lr": lr,
                    "train/step": global_step,
                }
                # Per-stage losses
                if "per_stage_losses" in result:
                    for k, sl in enumerate(result["per_stage_losses"]):
                        log_dict[f"train/stage_{k}_loss"] = sl
                # Stream agreements
                if "stream_agreements" in result:
                    for k, ag in enumerate(result["stream_agreements"]):
                        log_dict[f"train/agreement_merge_{k}"] = ag

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
                    print(f"  New best! Saved checkpoint.")

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

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        result = model(input_ids, targets)
        total_loss += result["loss"].item() * input_ids.numel()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 20))


def main():
    parser = argparse.ArgumentParser(description="Train Felix-LM")
    parser.add_argument("--config", default="m2", choices=["m0", "m2"], help="Model config")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    args.use_wandb = not args.no_wandb
    args.config_name = args.config

    if args.config == "m2":
        config = make_m2_config()
    elif args.config == "m0":
        config = make_m0_config()

    train(config, args)


if __name__ == "__main__":
    main()
