"""Scaled training script for Felix-LM (100M-500M params on Dolma).

Usage:
    python scripts/train_scaled.py --config felix_100m --device cuda
    python scripts/train_scaled.py --config m0_100m --device cuda
    python scripts/train_scaled.py --config felix_500m --device cuda
    python scripts/train_scaled.py --config m0_500m --device cuda
    python scripts/train_scaled.py --config felix_v2_100m --device cuda
    python scripts/train_scaled.py --config felix_v2_500m --device cuda
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from felix_lm.config import FelixConfig, StageConfig
from felix_lm.model import FelixLM
from felix_lm.utils import count_parameters
from felix_lm.v2.config import FelixV2Config
from felix_lm.v2.model import FelixLMv2

# --- Configs ---


def make_m0_100m_config() -> FelixConfig:
    """M0 standard transformer baseline at ~101M params.

    d_embed=512, 18 layers dim=512, 8 heads.
    Embedding: 50257*512 = 25.7M (tied). Layers: 18 * 16*512^2 = 75.5M.
    """
    return FelixConfig(
        vocab_size=50257,
        d_embed=512,
        stages=[
            StageConfig(
                num_streams=1,
                dim=512,
                num_layers=18,
                num_heads=8,
                attention_type="full_causal",
            ),
        ],
        use_deep_supervision=False,
        tie_embeddings=True,
        dropout=0.1,
    )


def make_felix_100m_config() -> FelixConfig:
    """MSPM at ~101M params. 4->2->1 streams, hetero attention, helical RoPE.

    d_embed=512. Stage 0: 4x128 (9L linear), Stage 1: 2x256 (9L sliding),
    Stage 2: 1x512 (11L full_causal). Proven design choices from 11M ablation.
    Embedding: 25.7M. Layers: ~74.5M. Merges: ~0.8M. Stream init: ~0.3M.
    """
    return FelixConfig(
        vocab_size=50257,
        d_embed=512,
        stages=[
            StageConfig(
                num_streams=4,
                dim=128,
                num_layers=9,
                num_heads=4,
                attention_type="linear",
            ),
            StageConfig(
                num_streams=2,
                dim=256,
                num_layers=9,
                num_heads=4,
                attention_type="sliding_window",
                window_size=128,
            ),
            StageConfig(
                num_streams=1,
                dim=512,
                num_layers=11,
                num_heads=8,
                attention_type="full_causal",
            ),
        ],
        rope_helical_turns=2,
        use_deep_supervision=False,
        tie_embeddings=True,
        dropout=0.1,
    )


def make_m0_500m_config() -> FelixConfig:
    """M0 standard transformer baseline at ~489M params.

    d_embed=1024, 26 layers dim=1024, 16 heads.
    """
    return FelixConfig(
        vocab_size=50257,
        d_embed=1024,
        stages=[
            StageConfig(
                num_streams=1,
                dim=1024,
                num_layers=26,
                num_heads=16,
                attention_type="full_causal",
            ),
        ],
        use_deep_supervision=False,
        tie_embeddings=True,
        dropout=0.1,
    )


def make_felix_500m_config() -> FelixConfig:
    """MSPM at ~486M params. 4->2->1 streams, hetero attention, helical RoPE.

    d_embed=1024. Stage 0: 4x512 (5L linear), Stage 1: 2x1024 (6L sliding w=256),
    Stage 2: 1x1024 (8L full_causal).
    """
    return FelixConfig(
        vocab_size=50257,
        d_embed=1024,
        stages=[
            StageConfig(
                num_streams=4,
                dim=512,
                num_layers=5,
                num_heads=8,
                attention_type="linear",
            ),
            StageConfig(
                num_streams=2,
                dim=1024,
                num_layers=6,
                num_heads=16,
                attention_type="sliding_window",
                window_size=256,
            ),
            StageConfig(
                num_streams=1,
                dim=1024,
                num_layers=8,
                num_heads=16,
                attention_type="full_causal",
            ),
        ],
        rope_helical_turns=2,
        use_deep_supervision=False,
        tie_embeddings=True,
        dropout=0.1,
    )


def make_felix_v2_100m_config() -> FelixV2Config:
    """Felix v2 at ~101M params. Adaptive convergence architecture.

    d_embed=512, d_stream=256, d_post=256, 4 streams, 14 layers + 2 refine.
    Embedding: 25.7M. Stream layers: ~58.7M. CentralPost: ~7.4M. Refine: ~8.4M.
    """
    return FelixV2Config(
        vocab_size=50257,
        d_embed=512,
        d_stream=256,
        d_post=256,
        num_streams=4,
        num_layers=14,
        num_heads=8,  # head_dim=32
        num_refine_layers=2,
        num_refine_heads=8,  # head_dim=64
        ffn_mult=4,
        dropout=0.1,
    )


def make_felix_v2_500m_config() -> FelixV2Config:
    """Felix v2 at ~484M params. Adaptive convergence architecture.

    d_embed=1024, d_stream=512, d_post=512, 4 streams, 21 layers + 2 refine.
    Embedding: 51.5M. Stream layers: ~352.4M. CentralPost: ~44.1M. Refine: ~33.6M.
    Gradient checkpointing enabled (4 streams * full causal needs it at 500M).
    """
    return FelixV2Config(
        vocab_size=50257,
        d_embed=1024,
        d_stream=512,
        d_post=512,
        num_streams=4,
        num_layers=21,
        num_heads=16,  # head_dim=32
        num_refine_layers=2,
        num_refine_heads=16,  # head_dim=64
        ffn_mult=4,
        dropout=0.1,
        gradient_checkpointing=True,
    )


CONFIGS = {
    "m0_100m": make_m0_100m_config,
    "felix_100m": make_felix_100m_config,
    "m0_500m": make_m0_500m_config,
    "felix_500m": make_felix_500m_config,
    "felix_v2_100m": make_felix_v2_100m_config,
    "felix_v2_500m": make_felix_v2_500m_config,
}


# --- Data ---


class DolmaDataset(IterableDataset):
    """Streaming Dolma dataset from HuggingFace.

    Streams tokenized chunks of seq_len+1 tokens from the Dolma3 Dolmino mix.
    Uses streaming to avoid downloading the full dataset.
    """

    def __init__(
        self,
        seq_len: int = 2048,
        max_tokens: int = 1_000_000_000,  # 1B tokens default
        split: str = "train",
    ):
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.split = split

    def __iter__(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        ds = load_dataset(
            "allenai/dolma3_dolmino_mix-100B-1125",
            split="train",
            streaming=True,
        )

        buffer = []
        tokens_seen = 0
        chunk_size = self.seq_len + 1

        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue
            ids = tokenizer.encode(text)
            buffer.extend(ids)

            while len(buffer) >= chunk_size:
                chunk = torch.tensor(buffer[:chunk_size], dtype=torch.long)
                buffer = buffer[chunk_size:]
                tokens_seen += chunk_size
                yield chunk[:-1], chunk[1:]

                if tokens_seen >= self.max_tokens:
                    return


class WikiTextValDataset(torch.utils.data.Dataset):
    """WikiText-103 validation set for eval (small, can fit in memory)."""

    def __init__(self, seq_len: int = 2048, cache_dir: str = "./data"):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cache_path = Path(cache_dir) / f"wikitext103_validation_{seq_len}.pt"

        if cache_path.exists():
            self.tokens = torch.load(cache_path, weights_only=True)
        else:
            print("Tokenizing WikiText-103 validation...")
            ds = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="validation",
                cache_dir=cache_dir,
            )
            texts = [t for t in ds["text"] if t.strip()]
            all_ids = tokenizer.encode("\n".join(texts))
            self.tokens = torch.tensor(all_ids, dtype=torch.long)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.tokens, cache_path)

        n_chunks = len(self.tokens) // (seq_len + 1)
        self.tokens = self.tokens[: n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        chunk = self.tokens[idx]
        return chunk[:-1], chunk[1:]


# --- LR Schedule ---


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# --- Training ---


def train(config, args):
    device = torch.device(args.device)
    is_v2 = isinstance(config, FelixV2Config)
    if is_v2:
        model = FelixLMv2(config).to(device)
    else:
        model = FelixLM(config).to(device)
    n_params = count_parameters(model)
    if is_v2:
        print(
            f"\nModel: v2 ({config.num_layers}+{config.num_refine_layers} layers, "
            f"{config.num_streams} streams), {n_params:,} params"
        )
    else:
        ns, tl = config.num_stages, config.total_layers
        print(f"\nModel: {ns} stages, {tl} layers, {n_params:,} params")

    # Mixed precision
    if args.dtype == "fp32":
        autocast_ctx = torch.autocast("cuda", enabled=False)
        print("  Precision: fp32")
    elif device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        print("  Mixed precision: bf16")
    elif device.type == "mps":
        autocast_ctx = torch.autocast("mps", dtype=torch.float16)
        print("  Mixed precision: fp16")
    else:
        autocast_ctx = torch.autocast("cpu", enabled=False)

    # Data
    tokens_per_epoch = args.tokens_per_epoch
    train_ds = DolmaDataset(seq_len=args.seq_len, max_tokens=tokens_per_epoch * args.epochs)
    val_ds = WikiTextValDataset(seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = tokens_per_epoch // (args.batch_size * args.seq_len * args.grad_accum)
    max_steps = steps_per_epoch * args.epochs
    opt_steps = max_steps // args.grad_accum
    print(f"\nTraining: {args.epochs} epochs, ~{steps_per_epoch} steps/epoch")
    print(f"  grad_accum={args.grad_accum}, ~{max_steps} total steps, {opt_steps} optimizer steps")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Tokens per epoch: {tokens_per_epoch:,}")
    # Auto-scale warmup: default 10% of optimizer steps
    if args.warmup_steps == 0:
        args.warmup_steps = max(1, opt_steps // 10)
    if args.warmup_steps > opt_steps // 2:
        old = args.warmup_steps
        args.warmup_steps = max(1, opt_steps // 10)
        print(f"  Warmup {old} > 50% of opt steps, auto-scaled to {args.warmup_steps}")
    pct = args.warmup_steps / opt_steps * 100
    print(f"  Warmup: {args.warmup_steps} optimizer steps ({pct:.0f}%)")

    # wandb
    if not args.no_wandb:
        import wandb

        wandb.init(
            project="felix-lm",
            name=args.config,
            config={
                "model_params": n_params,
                "config": args.config,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "seq_len": args.seq_len,
                "tokens_per_epoch": tokens_per_epoch,
                "epochs": args.epochs,
            },
        )

    # Training loop
    ckpt_dir = Path(f"checkpoints/{args.config}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_ppl = float("inf")
    global_step = 0
    lr = args.lr

    model.train()
    optimizer.zero_grad()
    pbar = tqdm(total=max_steps, desc="Training")

    for input_ids, targets in train_loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with autocast_ctx:
            result = model(input_ids, targets)
            loss = result["loss"] / args.grad_accum

        loss.backward()

        if (global_step + 1) % args.grad_accum == 0:
            lr = get_lr(
                global_step // args.grad_accum,
                args.warmup_steps,
                max_steps // args.grad_accum,
                args.lr,
                args.lr * 0.1,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        actual_loss = loss.item() * args.grad_accum
        ppl = math.exp(min(actual_loss, 20))

        pbar.update(1)
        pbar.set_postfix(loss=f"{actual_loss:.3f}", lr=f"{lr:.2e}", ppl=f"{ppl:.1f}")

        if not args.no_wandb and global_step % 10 == 0:
            import wandb

            log_dict = {
                "train/loss": actual_loss,
                "train/ppl": ppl,
                "train/lr": lr,
            }
            if is_v2 and "agreements" in result:
                for i, a in enumerate(result["agreements"]):
                    log_dict[f"v2/agreement_layer_{i}"] = a.item()
                for i, s in enumerate(result["merge_strengths"]):
                    log_dict[f"v2/merge_strength_layer_{i}"] = s.item()
            wandb.log(log_dict, step=global_step)

        # Eval
        if global_step % args.eval_interval == 0:
            val_ppl = evaluate(model, val_loader, device, autocast_ctx)
            print(f"\n  Step {global_step}: val_ppl = {val_ppl:.2f}")
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
                print("  New best! Saved checkpoint.")
            if not args.no_wandb:
                import wandb

                wandb.log({"val/ppl": val_ppl, "val/best_ppl": best_val_ppl}, step=global_step)
            model.train()

        if global_step >= max_steps:
            break

    pbar.close()

    # Final eval
    val_ppl = evaluate(model, val_loader, device, autocast_ctx)
    print(f"\nFinal val_ppl = {val_ppl:.2f} (best = {best_val_ppl:.2f})")
    torch.save(model.state_dict(), ckpt_dir / "last.pt")

    if not args.no_wandb:
        import wandb

        wandb.finish()


def evaluate(model, dataloader, device, autocast_ctx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with torch.no_grad(), autocast_ctx:
            result = model(input_ids, targets)
        total_loss += result["loss"].item() * targets.numel()
        total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 20))


def main():
    parser = argparse.ArgumentParser(description="Felix-LM scaled training")
    parser.add_argument("--config", type=str, default="felix_100m", choices=list(CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="Warmup optimizer steps (0=auto: 10%% of total)"
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--tokens-per-epoch", type=int, default=1_000_000_000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()

    config = CONFIGS[args.config]()
    train(config, args)


if __name__ == "__main__":
    main()
