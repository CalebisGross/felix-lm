"""Compare two models beyond PPL — look at WHERE they differ, not just by how much.

Usage:
    python scripts/compare_models.py \
        --ckpt-a checkpoints/v3_none/best.pt \
        --ckpt-b checkpoints/v3_r32/best.pt \
        --device cuda
"""

import argparse
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from felix_lm.utils import count_parameters
from felix_lm.v3.model import FelixLMv3


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = FelixLMv3(config).to(device)
    # Handle compiled model state dicts
    state_dict = ckpt["model"]
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(cleaned)
    model.eval()
    return model, config


@torch.no_grad()
def analyze_predictions(model, input_ids, targets, tokenizer):
    """Get per-token predictions from a model."""
    result = model(input_ids, targets)
    logits = result["logits"]  # [B, T, V]

    # Per-token cross entropy
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="none",
    ).view(targets.shape)  # [B, T]

    # Per-token entropy of the prediction distribution
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [B, T]

    # Top-1 predictions
    preds = logits.argmax(dim=-1)  # [B, T]

    # Top-1 accuracy
    correct = (preds == targets).float()

    # Agreement info if available
    agreements = result.get("agreements", [])

    return {
        "ce": ce,
        "entropy": entropy,
        "preds": preds,
        "correct": correct,
        "agreements": agreements,
    }


def categorize_tokens(token_ids, tokenizer):
    """Categorize each token for stratified analysis."""
    categories = []
    for tid in token_ids:
        text = tokenizer.decode([tid.item()])
        if text.strip() == "":
            categories.append("whitespace")
        elif text.strip() in ".,;:!?()[]{}\"'-":
            categories.append("punctuation")
        elif text.strip().isdigit() or text.strip().replace(".", "").isdigit():
            categories.append("number")
        elif tid.item() < 1000:
            categories.append("common")  # top-1000 tokens by frequency
        elif tid.item() > 40000:
            categories.append("rare")  # bottom ~10K tokens
        else:
            categories.append("content")
    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True, help="Checkpoint A (e.g. baseline)")
    parser.add_argument("--ckpt-b", required=True, help="Checkpoint B (e.g. spokes)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--generate", action="store_true", help="Generate text samples")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Loading {args.label_a}: {args.ckpt_a}")
    model_a, config_a = load_model(args.ckpt_a, device)
    print(
        f"  Config: {config_a.num_layers}L, d={config_a.d_embed}, "
        f"spokes={'none' if config_a.gate_schedule == 'none' else f'{config_a.num_spokes}x r={config_a.spoke_rank}'}"  # noqa: E501
    )
    print(f"  Params: {count_parameters(model_a):,}")

    print(f"\nLoading {args.label_b}: {args.ckpt_b}")
    model_b, config_b = load_model(args.ckpt_b, device)
    print(
        f"  Config: {config_b.num_layers}L, d={config_b.d_embed}, "
        f"spokes={'none' if config_b.gate_schedule == 'none' else f'{config_b.num_spokes}x r={config_b.spoke_rank}'}"  # noqa: E501
    )
    print(f"  Params: {count_parameters(model_b):,}")

    # Load validation data
    # Import dataset from train script's module
    import sys

    from torch.utils.data import DataLoader

    sys.path.insert(0, ".")
    from scripts.train import WikiTextDataset

    val_ds = WikiTextDataset("validation", seq_len=args.seq_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Collect per-token statistics
    all_ce_a, all_ce_b = [], []
    all_entropy_a, all_entropy_b = [], []
    all_correct_a, all_correct_b = [], []
    all_agreements = []
    n_agree_both, n_agree_a_only, n_agree_b_only, n_agree_neither = 0, 0, 0, 0
    total_tokens = 0

    # Per-category stats
    cat_ce_a = defaultdict(list)
    cat_ce_b = defaultdict(list)

    print(f"\nAnalyzing {args.num_batches} batches...")
    with torch.autocast(args.device, dtype=torch.bfloat16):
        for batch_idx, (input_ids, targets) in enumerate(val_loader):
            if batch_idx >= args.num_batches:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            res_a = analyze_predictions(model_a, input_ids, targets, tokenizer)
            res_b = analyze_predictions(model_b, input_ids, targets, tokenizer)

            # Flatten
            ce_a = res_a["ce"].cpu()
            ce_b = res_b["ce"].cpu()
            ent_a = res_a["entropy"].cpu()
            ent_b = res_b["entropy"].cpu()
            cor_a = res_a["correct"].cpu()
            cor_b = res_b["correct"].cpu()
            preds_a = res_a["preds"].cpu()
            preds_b = res_b["preds"].cpu()
            targets.cpu()

            all_ce_a.append(ce_a)
            all_ce_b.append(ce_b)
            all_entropy_a.append(ent_a)
            all_entropy_b.append(ent_b)
            all_correct_a.append(cor_a)
            all_correct_b.append(cor_b)

            # Agreement between models (both predict same token)
            (preds_a == preds_b).float()

            # Categorize tokens
            for seq_idx in range(targets.shape[0]):
                cats = categorize_tokens(targets[seq_idx], tokenizer)
                for t_idx, cat in enumerate(cats):
                    cat_ce_a[cat].append(ce_a[seq_idx, t_idx].item())
                    cat_ce_b[cat].append(ce_b[seq_idx, t_idx].item())

            # Agreement: where do they agree/disagree with ground truth?
            both_right = (cor_a * cor_b).sum().item()
            a_only = (cor_a * (1 - cor_b)).sum().item()
            b_only = ((1 - cor_a) * cor_b).sum().item()
            neither = ((1 - cor_a) * (1 - cor_b)).sum().item()
            n_agree_both += both_right
            n_agree_a_only += a_only
            n_agree_b_only += b_only
            n_agree_neither += neither
            total_tokens += targets.numel()

            # Collect spoke agreements if model B has them
            if res_b["agreements"]:
                all_agreements.append([a.item() for a in res_b["agreements"]])

            if (batch_idx + 1) % 10 == 0:
                print(f"  {batch_idx + 1}/{args.num_batches} batches processed")

    # Aggregate
    all_ce_a = torch.cat([x.flatten() for x in all_ce_a])
    all_ce_b = torch.cat([x.flatten() for x in all_ce_b])
    all_entropy_a = torch.cat([x.flatten() for x in all_entropy_a])
    all_entropy_b = torch.cat([x.flatten() for x in all_entropy_b])
    all_correct_a = torch.cat([x.flatten() for x in all_correct_a])
    all_correct_b = torch.cat([x.flatten() for x in all_correct_b])

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Overall metrics
    ppl_a = math.exp(all_ce_a.mean().item())
    ppl_b = math.exp(all_ce_b.mean().item())
    print(f"\n{'Metric':<30} {args.label_a:>15} {args.label_b:>15} {'Delta':>10}")
    print("-" * 70)
    print(f"{'PPL':<30} {ppl_a:>15.2f} {ppl_b:>15.2f} {ppl_b - ppl_a:>+10.2f}")
    print(
        f"{'Mean CE (nats)':<30} {all_ce_a.mean():>15.4f} {all_ce_b.mean():>15.4f} "
        f"{all_ce_b.mean() - all_ce_a.mean():>+10.4f}"
    )
    print(
        f"{'Top-1 Accuracy':<30} {all_correct_a.mean():>15.4f} {all_correct_b.mean():>15.4f} "
        f"{all_correct_b.mean() - all_correct_a.mean():>+10.4f}"
    )
    print(
        f"{'Mean Entropy':<30} {all_entropy_a.mean():>15.4f} {all_entropy_b.mean():>15.4f} "
        f"{all_entropy_b.mean() - all_entropy_a.mean():>+10.4f}"
    )
    print(
        f"{'Median CE':<30} {all_ce_a.median():>15.4f} {all_ce_b.median():>15.4f} "
        f"{all_ce_b.median() - all_ce_a.median():>+10.4f}"
    )

    # Prediction agreement between models
    print("\n--- Model Agreement (Top-1 Prediction) ---")
    print(f"  Both correct:        {n_agree_both:>8.0f} ({n_agree_both / total_tokens * 100:.1f}%)")
    print(
        f"  {args.label_a} only correct:  {n_agree_a_only:>8.0f} ({n_agree_a_only / total_tokens * 100:.1f}%)"  # noqa: E501
    )
    print(
        f"  {args.label_b} only correct:  {n_agree_b_only:>8.0f} ({n_agree_b_only / total_tokens * 100:.1f}%)"  # noqa: E501
    )
    print(
        f"  Neither correct:     {n_agree_neither:>8.0f} ({n_agree_neither / total_tokens * 100:.1f}%)"  # noqa: E501
    )

    # Per-category breakdown
    print("\n--- Per-Category CE (lower = better) ---")
    print(
        f"{'Category':<15} {'Count':>8} {args.label_a + ' CE':>12} {args.label_b + ' CE':>12} {'Delta':>10}"  # noqa: E501
    )
    print("-" * 57)
    for cat in ["common", "content", "punctuation", "whitespace", "number", "rare"]:
        if cat in cat_ce_a and len(cat_ce_a[cat]) > 0:
            mean_a = sum(cat_ce_a[cat]) / len(cat_ce_a[cat])
            mean_b = sum(cat_ce_b[cat]) / len(cat_ce_b[cat])
            print(
                f"{cat:<15} {len(cat_ce_a[cat]):>8} {mean_a:>12.4f} {mean_b:>12.4f} {mean_b - mean_a:>+10.4f}"  # noqa: E501
            )

    # CE distribution: where does B beat A most?
    delta_ce = all_ce_b - all_ce_a  # negative = B better
    print(f"\n--- Where {args.label_b} Beats {args.label_a} ---")
    b_better = (delta_ce < -0.1).sum().item()
    a_better = (delta_ce > 0.1).sum().item()
    tied = total_tokens - b_better - a_better
    print(
        f"  {args.label_b} better (delta < -0.1): {b_better:>8} ({b_better / total_tokens * 100:.1f}%)"  # noqa: E501
    )
    print(
        f"  {args.label_a} better (delta > +0.1): {a_better:>8} ({a_better / total_tokens * 100:.1f}%)"  # noqa: E501
    )
    print(f"  Similar (|delta| < 0.1):    {tied:>8} ({tied / total_tokens * 100:.1f}%)")

    # Percentile analysis of deltas
    print(
        f"\n--- CE Delta Percentiles ({args.label_b} - {args.label_a}, negative = {args.label_b} better) ---"  # noqa: E501
    )
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = torch.quantile(delta_ce.float(), pct / 100).item()
        print(f"  p{pct:02d}: {val:>+.4f}")

    # Spoke agreement analysis (if model B has spokes)
    if all_agreements:
        print(f"\n--- Spoke Agreement ({args.label_b}) ---")
        import numpy as np

        agreements = np.array(all_agreements)
        for layer_idx in range(agreements.shape[1]):
            print(
                f"  Layer {layer_idx:>2}: mean={agreements[:, layer_idx].mean():.4f}, "
                f"std={agreements[:, layer_idx].std():.4f}"
            )

    # Generate text samples if requested
    if args.generate:
        print("\n--- Text Generation Samples ---")
        prompts = [
            "The theory of relativity states that",
            "In the beginning, there was",
            "The stock market crashed because",
            "Scientists discovered that the",
        ]
        for prompt in prompts:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            print(f"\nPrompt: {prompt}")

            for label, model in [(args.label_a, model_a), (args.label_b, model_b)]:
                gen_ids = ids.clone()
                for _ in range(50):
                    with torch.autocast(args.device, dtype=torch.bfloat16):
                        result = model(gen_ids)
                    next_token = result["logits"][0, -1].argmax()
                    gen_ids = torch.cat([gen_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                print(f"  [{label}]: {text}")


if __name__ == "__main__":
    main()
