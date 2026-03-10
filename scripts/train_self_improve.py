"""Agreement-based self-improvement for Felix-LM v2.

Hypothesis: inter-stream agreement alone (without answer verification) can improve
a language model. In arithmetic (felix-auto), agreement + verification gave +8% accuracy.
Here, agreement is the ONLY signal.

Usage:
    python scripts/train_self_improve.py --checkpoint checkpoints/felix_v2/best.pt
    python scripts/train_self_improve.py --checkpoint checkpoints/felix_v2/best.pt \\
        --n-cycles 5 --n-generate 10  # smoke test
"""

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from felix_lm.utils import count_parameters
from felix_lm.v2.config import FelixV2Config
from felix_lm.v2.model import FelixLMv2
from scripts.train import WikiTextDataset, evaluate


def load_checkpoint(path: str, device: torch.device) -> tuple[FelixLMv2, FelixV2Config]:
    """Load a v2 checkpoint and return (model, config)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    # Handle older checkpoints missing newer config fields
    if not hasattr(config, "attention_schedule"):
        config.attention_schedule = []
    model = FelixLMv2(config).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from {path} (step {ckpt.get('step', '?')})")
    print(
        f"  Config: {config.num_streams} streams, {config.num_layers} layers, "
        f"d_stream={config.d_stream}"
    )
    print(f"  Parameters: {count_parameters(model):,}")
    return model, config


def forward_with_agreement(model: FelixLMv2, token_ids: torch.Tensor, targets=None):
    """Forward pass that captures per-position agreement from all layers.

    Uses forward hooks on AdaptiveMergeLayer to get [B, T] agreement
    without modifying any model code.
    """
    raw_agreements = []
    hooks = []

    for layer in model.layers:

        def make_hook(store):
            def hook_fn(module, input, output):
                # output = (merged_streams, agreement[B,T], merge_strength[B,T])
                store.append(output[1].detach())

            return hook_fn

        h = layer.adaptive_merge.register_forward_hook(make_hook(raw_agreements))
        hooks.append(h)

    result = model(token_ids, targets)

    for h in hooks:
        h.remove()

    # Stack [L, B, T] -> mean over layers -> [B, T]
    if raw_agreements:
        result["per_position_agreement"] = torch.stack(raw_agreements).mean(dim=0)

    return result


@torch.no_grad()
def generate_with_agreement(
    model: FelixLMv2,
    prompts: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive generation with per-sequence agreement tracking.

    Args:
        model: Felix-LM v2 model in eval mode.
        prompts: [B, prompt_len] token ids.
        max_new_tokens: number of new tokens to generate.
        temperature: sampling temperature.
        top_k: top-k filtering.

    Returns:
        sequences: [B, prompt_len + max_new_tokens] full sequences.
        agreements: [B] mean agreement over generated positions.
    """
    model.eval()
    B, T = prompts.shape
    device = prompts.device

    # Build full sequence buffer
    sequences = torch.zeros(B, T + max_new_tokens, dtype=torch.long, device=device)
    sequences[:, :T] = prompts

    # Track agreement at each generated position
    gen_agreements = []  # list of [B] tensors

    for i in range(max_new_tokens):
        # Use last `model.config.total_layers * 32` tokens or full seq to avoid OOM
        # For 11M model with seq_len 512, full context is fine
        ctx_len = min(T + i, 512)
        input_ids = sequences[:, T + i - ctx_len : T + i]

        result = forward_with_agreement(model, input_ids)
        logits = result["logits"][:, -1, :]  # [B, V]

        # Agreement at the last position (the one we're predicting from)
        if "per_position_agreement" in result:
            gen_agreements.append(result["per_position_agreement"][:, -1])  # [B]

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            logits[logits < threshold] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        sequences[:, T + i] = next_token

    # Mean agreement per sequence over generated positions
    if gen_agreements:
        agreements = torch.stack(gen_agreements, dim=1).mean(dim=1)  # [B]
    else:
        agreements = torch.zeros(B, device=device)

    return sequences, agreements


def has_repetition(token_ids: list[int], n: int = 4, max_repeats: int = 3) -> bool:
    """Check if a sequence has excessive n-gram repetition."""
    if len(token_ids) < n:
        return False
    ngrams = Counter()
    for i in range(len(token_ids) - n + 1):
        ngram = tuple(token_ids[i : i + n])
        ngrams[ngram] += 1
        if ngrams[ngram] > max_repeats:
            return True
    return False


def score_sequences_by_loss(
    model: FelixLMv2, sequences: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Score sequences by per-sequence cross-entropy loss (lower = more natural).

    Args:
        model: frozen reference model for scoring.
        sequences: [N, T] token ids.

    Returns:
        losses: [N] per-sequence mean loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(len(sequences)):
            seq = sequences[i : i + 1]  # [1, T]
            input_ids = seq[:, :-1]
            targets = seq[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(input_ids, targets)
            losses.append(result["loss"].item())
    return torch.tensor(losses, device=device)


def self_improvement_cycle(
    model: FelixLMv2,
    optimizer: torch.optim.Optimizer,
    train_dataset: WikiTextDataset,
    val_loader: DataLoader,
    device: torch.device,
    args,
    ref_model: FelixLMv2 | None = None,
) -> dict:
    """Run one cycle of self-improvement.

    1. Sample random prompts from training data
    2. Generate completions
    3. Filter by agreement/loss + repetition
    4. Train on kept sequences
    5. Train on real data batches (prevent forgetting)

    Returns dict with cycle stats.
    """
    prompt_len = args.prompt_len
    max_new_tokens = args.max_new_tokens
    gen_batch_size = args.gen_batch_size

    # 1. Sample random prompts
    indices = torch.randint(0, len(train_dataset), (args.n_generate,))
    prompts = []
    for idx in indices:
        input_ids, _ = train_dataset[idx.item()]
        # Take first prompt_len tokens as prompt
        prompts.append(input_ids[:prompt_len])
    prompts = torch.stack(prompts).to(device)  # [n_generate, prompt_len]

    # 2. Generate completions in batches
    all_sequences = []
    all_agreements = []
    model.eval()

    for i in range(0, len(prompts), gen_batch_size):
        batch_prompts = prompts[i : i + gen_batch_size]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            seqs, agrees = generate_with_agreement(
                model, batch_prompts, max_new_tokens, args.temperature, args.top_k
            )
        all_sequences.append(seqs)
        all_agreements.append(agrees)

    all_sequences = torch.cat(all_sequences, dim=0)  # [n_generate, total_len]
    all_agreements = torch.cat(all_agreements, dim=0)  # [n_generate]

    # 3. Agreement stats (for logging)
    agree_stats = {
        "mean": all_agreements.mean().item(),
        "std": all_agreements.std().item(),
        "min": all_agreements.min().item(),
        "max": all_agreements.max().item(),
        "p25": all_agreements.quantile(0.25).item(),
        "p50": all_agreements.quantile(0.50).item(),
        "p75": all_agreements.quantile(0.75).item(),
    }

    # 4. Filter: by reference-model loss or by agreement
    if args.filter_by_loss and ref_model is not None:
        # Score all sequences with frozen reference model
        ref_losses = score_sequences_by_loss(ref_model, all_sequences, device)
        agree_stats["ref_loss_mean"] = ref_losses.mean().item()
        agree_stats["ref_loss_std"] = ref_losses.std().item()
        agree_stats["ref_loss_p25"] = ref_losses.quantile(0.25).item()
        agree_stats["ref_loss_p50"] = ref_losses.quantile(0.50).item()

        # Keep top N% by lowest loss (most natural)
        pct = args.threshold_percentile if args.threshold_percentile else 20.0
        loss_cutoff = ref_losses.quantile(pct / 100.0)  # lower = better
        kept_mask = ref_losses <= loss_cutoff
        agree_stats["loss_cutoff"] = loss_cutoff.item()
    elif args.threshold_percentile is not None:
        # Adaptive: keep top N% by agreement
        cutoff = all_agreements.quantile(1.0 - args.threshold_percentile / 100.0)
        kept_mask = all_agreements >= cutoff
        agree_stats["adaptive_cutoff"] = cutoff.item()
    else:
        kept_mask = all_agreements >= args.agreement_threshold

    # 5. Filter by repetition
    kept_indices = []
    for j in range(len(all_sequences)):
        if not kept_mask[j]:
            continue
        # Check repetition in generated portion only
        gen_tokens = all_sequences[j, prompt_len:].tolist()
        if not has_repetition(gen_tokens):
            kept_indices.append(j)

    kept = len(kept_indices)
    total = len(all_sequences)

    # 6. Train on kept sequences (if any)
    self_loss = 0.0
    if kept > 0:
        model.train()
        kept_seqs = all_sequences[kept_indices]  # [kept, total_len]

        # Train on these sequences (next-token prediction)
        # Process in mini-batches
        mini_batch = min(args.self_batch_size, kept)
        perm = torch.randperm(kept)
        n_batches = 0

        for bi in range(0, kept, mini_batch):
            batch_idx = perm[bi : bi + mini_batch]
            batch_seqs = kept_seqs[batch_idx]
            input_ids = batch_seqs[:, :-1].contiguous()
            targets = batch_seqs[:, 1:].contiguous()

            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(input_ids, targets)
                loss = result["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            self_loss += loss.item()
            n_batches += 1

        self_loss /= max(n_batches, 1)

    # 7. Train on real data (prevent forgetting)
    model.train()
    real_loss = 0.0
    real_indices = torch.randint(0, len(train_dataset), (args.real_batches * args.self_batch_size,))
    for bi in range(args.real_batches):
        batch_start = bi * args.self_batch_size
        batch_end = batch_start + args.self_batch_size
        batch_items = [train_dataset[real_indices[j].item()] for j in range(batch_start, batch_end)]
        input_ids = torch.stack([x[0] for x in batch_items]).to(device)
        targets = torch.stack([x[1] for x in batch_items]).to(device)

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(input_ids, targets)
            loss = result["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        real_loss += loss.item()

    real_loss /= max(args.real_batches, 1)

    return {
        "kept": kept,
        "total": total,
        "self_loss": self_loss,
        "real_loss": real_loss,
        "agree_stats": agree_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Agreement-based self-improvement for Felix-LM v2")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to supervised checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--n-cycles", type=int, default=40, help="Number of self-improvement cycles"
    )
    parser.add_argument(
        "--n-generate", type=int, default=200, help="Sequences to generate per cycle"
    )
    parser.add_argument("--gen-batch-size", type=int, default=4, help="Generation batch size")
    parser.add_argument(
        "--self-batch-size", type=int, default=4, help="Training batch size for self-improvement"
    )
    parser.add_argument(
        "--agreement-threshold", type=float, default=0.3, help="Min agreement to keep"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=None,
        help="Keep top N%% by agreement (overrides --agreement-threshold)",
    )
    parser.add_argument(
        "--filter-by-loss",
        action="store_true",
        help="Filter by reference model loss instead of agreement",
    )
    parser.add_argument(
        "--self-lr", type=float, default=1e-5, help="Learning rate for self-improvement"
    )
    parser.add_argument("--real-batches", type=int, default=5, help="Real data batches per cycle")
    parser.add_argument("--prompt-len", type=int, default=64, help="Prompt length in tokens")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--eval-interval", type=int, default=5, help="Eval every N cycles")
    parser.add_argument("--patience", type=int, default=4, help="Stop after N stale evals")
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Save directory (default: same as checkpoint)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)

    if args.save_dir is None:
        args.save_dir = str(ckpt_path.parent)

    # Tee output to log file
    import builtins

    log_path = Path(args.save_dir) / "self_improve.log"
    log_file = open(log_path, "w")
    _orig_print = builtins.print

    def tee_print(*a, **kw):
        _orig_print(*a, **kw)
        _orig_print(*a, **{**kw, "file": log_file, "flush": True})

    builtins.print = tee_print

    # Load model
    model, config = load_checkpoint(str(ckpt_path), device)

    # Create frozen reference model for loss-based filtering
    ref_model = None
    if args.filter_by_loss:
        print("Loading frozen reference model for loss-based filtering...")
        ref_model, _ = load_checkpoint(str(ckpt_path), device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # Load data
    print("\nLoading data...")
    train_dataset = WikiTextDataset("train", seq_len=512, cache_dir="./data")
    val_dataset = WikiTextDataset("validation", seq_len=512, cache_dir="./data")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Baseline evaluation
    print("\n=== Baseline Evaluation ===")
    baseline_ppl = evaluate(model, val_loader, device)
    print(f"Baseline val PPL: {baseline_ppl:.2f}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.self_lr, weight_decay=0.01)

    # Self-improvement loop
    print(f"\n=== Self-Improvement ({args.n_cycles} cycles) ===")
    if args.filter_by_loss:
        pct = args.threshold_percentile if args.threshold_percentile else 20.0
        print(f"  filter=ref_model_loss (top {pct}% lowest loss)")
    elif args.threshold_percentile is not None:
        print(f"  threshold_percentile=top {args.threshold_percentile}%")
    else:
        print(f"  agreement_threshold={args.agreement_threshold}")
    print(f"  self_lr={args.self_lr}")
    print(f"  n_generate={args.n_generate}")
    print(f"  real_batches={args.real_batches}")
    print(f"  temperature={args.temperature}, top_k={args.top_k}")

    best_ppl = baseline_ppl
    stale_count = 0
    total_kept = 0
    start_time = time.time()

    # Save initial state for revert
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for cycle in range(1, args.n_cycles + 1):
        cycle_start = time.time()
        stats = self_improvement_cycle(
            model, optimizer, train_dataset, val_loader, device, args, ref_model
        )
        cycle_time = time.time() - cycle_start
        total_kept += stats["kept"]

        # Log every cycle
        elapsed = time.time() - start_time
        print(
            f"  Cycle {cycle}: kept={stats['kept']}/{stats['total']} "
            f"self_loss={stats['self_loss']:.4f} real_loss={stats['real_loss']:.4f} "
            f"[{cycle_time:.0f}s, {elapsed:.0f}s total]"
        )

        # Log agreement stats on first cycle and every eval_interval
        if cycle == 1 or cycle % args.eval_interval == 0:
            a = stats["agree_stats"]
            cutoff_str = ""
            if "adaptive_cutoff" in a:
                cutoff_str = f" cutoff={a['adaptive_cutoff']:.3f}"
            print(
                f"    Agreement: mean={a['mean']:.3f} std={a['std']:.3f} "
                f"min={a['min']:.3f} max={a['max']:.3f} "
                f"p25={a['p25']:.3f} p50={a['p50']:.3f} p75={a['p75']:.3f}"
                f"{cutoff_str}"
            )
            if "ref_loss_mean" in a:
                print(
                    f"    Ref loss: mean={a['ref_loss_mean']:.3f} "
                    f"std={a['ref_loss_std']:.3f} "
                    f"p25={a['ref_loss_p25']:.3f} "
                    f"p50={a['ref_loss_p50']:.3f} "
                    f"cutoff={a['loss_cutoff']:.3f}"
                )

        # Periodic evaluation
        if cycle % args.eval_interval == 0:
            val_ppl = evaluate(model, val_loader, device)
            print(
                f"    Val PPL: {val_ppl:.2f} (baseline: {baseline_ppl:.2f}, best: {best_ppl:.2f})"
            )

            if val_ppl < best_ppl:
                best_ppl = val_ppl
                stale_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                # Save checkpoint
                save_path = Path(args.save_dir) / "best_self_improve.pt"
                torch.save(
                    {"model": model.state_dict(), "config": config, "cycle": cycle},
                    save_path,
                )
                print(f"    Saved best self-improve checkpoint to {save_path}")
            else:
                stale_count += 1
                print(f"    No improvement ({stale_count}/{args.patience})")

            # Safety: revert if PPL degrades too much
            if val_ppl > baseline_ppl * 1.05:
                print("    WARNING: PPL degraded >5% from baseline, reverting!")
                model.load_state_dict(best_state)
                break

            if stale_count >= args.patience:
                print(f"    Stalled for {stale_count} evals, stopping.")
                break

    # Revert to best
    model.load_state_dict(best_state)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_ppl = evaluate(model, val_loader, device)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    print(f"Final PPL:    {final_ppl:.2f}")
    print(f"Delta:        {final_ppl - baseline_ppl:+.2f}")
    print(f"Total kept:   {total_kept}")
    print(f"Time:         {time.time() - start_time:.0f}s")

    delta = final_ppl - baseline_ppl
    print(
        f"\nResults: baseline={baseline_ppl:.2f} final={final_ppl:.2f} "
        f"delta={delta:+.2f} kept={total_kept}"
    )

    log_file.close()
    builtins.print = _orig_print
    _orig_print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
