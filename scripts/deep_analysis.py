"""Deep analysis of spoke vs baseline models.

Goes beyond PPL to examine:
1. Calibration — are confidences meaningful?
2. Rare token deep dive — what kinds of rare tokens improve?
3. Positional analysis — do spokes help more at long range?
4. Conditional perplexity — PPL on hard vs easy contexts
5. Representation diversity — how different are the internal representations?
6. Agreement-conditioned quality — do high-agreement layers predict better tokens?
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from felix_lm.utils import count_parameters
from felix_lm.v3.model import FelixLMv3


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = FelixLMv3(config).to(device)
    state_dict = ckpt["model"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model, config


def get_hidden_states(model, input_ids):
    """Get hidden states from each layer using hooks."""
    hiddens = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hiddens.append(output[0].detach())
        else:
            hiddens.append(output.detach())

    hooks = []
    for layer in model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return hiddens


@torch.no_grad()
def calibration_analysis(model, dataloader, device, label, num_batches=50):
    """Measure calibration: when model says P(token)=0.8, is it right 80% of the time?"""
    model.eval()
    # Bin confidences and track accuracy per bin
    n_bins = 20
    bin_correct = [0.0] * n_bins
    bin_total = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            result = model(input_ids, targets)

        probs = F.softmax(result["logits"], dim=-1)
        top_probs, top_preds = probs.max(dim=-1)
        correct = (top_preds == targets).float()

        for conf, cor in zip(top_probs.flatten().cpu(), correct.flatten().cpu()):
            bin_idx = min(int(conf.item() * n_bins), n_bins - 1)
            bin_correct[bin_idx] += cor.item()
            bin_total[bin_idx] += 1
            bin_conf_sum[bin_idx] += conf.item()

    print(f"\n--- Calibration: {label} ---")
    print(f"{'Conf Range':<15} {'Count':>8} {'Accuracy':>10} {'Avg Conf':>10} {'Gap':>10}")
    print("-" * 53)
    ece = 0.0
    total = sum(bin_total)
    for i in range(n_bins):
        if bin_total[i] > 0:
            acc = bin_correct[i] / bin_total[i]
            avg_conf = bin_conf_sum[i] / bin_total[i]
            gap = abs(acc - avg_conf)
            ece += gap * bin_total[i] / total
            lo = i / n_bins
            hi = (i + 1) / n_bins
            print(
                f"[{lo:.2f}, {hi:.2f})"
                f" {bin_total[i]:>8}"
                f" {acc:>10.4f}"
                f" {avg_conf:>10.4f}"
                f" {gap:>+10.4f}"
            )
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    return ece


@torch.no_grad()
def positional_analysis(model, dataloader, device, label, num_batches=50):
    """Do spokes help more at later positions (long-range dependencies)?"""
    model.eval()
    # Collect CE per position
    position_ce = defaultdict(list)

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            result = model(input_ids, targets)

        ce = F.cross_entropy(
            result["logits"].view(-1, result["logits"].size(-1)),
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)

        for pos in range(targets.shape[1]):
            position_ce[pos].append(ce[:, pos].mean().item())

    # Aggregate into buckets of 64 positions
    bucket_size = 64
    n_buckets = 512 // bucket_size
    buckets = []
    for b in range(n_buckets):
        vals = []
        for pos in range(b * bucket_size, (b + 1) * bucket_size):
            if pos in position_ce:
                vals.extend(position_ce[pos])
        if vals:
            buckets.append(sum(vals) / len(vals))
        else:
            buckets.append(0)

    return buckets


@torch.no_grad()
def context_difficulty_analysis(
    model_a, model_b, dataloader, device, label_a, label_b, num_batches=50
):
    """Split tokens by context difficulty and compare models."""
    # Use model A's CE to define "easy" vs "hard" contexts
    all_ce_a = []
    all_ce_b = []

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            res_a = model_a(input_ids, targets)
            res_b = model_b(input_ids, targets)

        ce_a = F.cross_entropy(
            res_a["logits"].view(-1, res_a["logits"].size(-1)),
            targets.view(-1),
            reduction="none",
        ).cpu()
        ce_b = F.cross_entropy(
            res_b["logits"].view(-1, res_b["logits"].size(-1)),
            targets.view(-1),
            reduction="none",
        ).cpu()

        all_ce_a.append(ce_a)
        all_ce_b.append(ce_b)

    all_ce_a = torch.cat(all_ce_a)
    all_ce_b = torch.cat(all_ce_b)

    # Split into quintiles by baseline difficulty
    quintiles = torch.quantile(all_ce_a.float(), torch.tensor([0.2, 0.4, 0.6, 0.8]))

    print("\n--- Context Difficulty Analysis ---")
    print(f"(Quintiles defined by {label_a}'s CE)")
    print(
        f"{'Quintile':<15} {'Count':>8} {label_a + ' CE':>12} {label_b + ' CE':>12} "
        f"{'Delta':>10} {'Rel %':>8}"
    )
    print("-" * 65)

    boundaries = [0] + quintiles.tolist() + [float("inf")]
    for i in range(5):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (all_ce_a >= lo) & (all_ce_a < hi)
        n = mask.sum().item()
        if n == 0:
            continue
        mean_a = all_ce_a[mask].mean().item()
        mean_b = all_ce_b[mask].mean().item()
        delta = mean_b - mean_a
        rel = delta / mean_a * 100
        labels = ["Easiest", "Easy", "Medium", "Hard", "Hardest"]
        print(
            f"{labels[i]:<15} {n:>8} {mean_a:>12.4f} {mean_b:>12.4f} {delta:>+10.4f} {rel:>+7.1f}%"
        )


@torch.no_grad()
def representation_analysis(model_a, model_b, dataloader, device, label_a, label_b, num_batches=10):
    """Compare internal representations between models."""
    print("\n--- Representation Analysis ---")

    # Collect hidden states from a few batches
    all_cosines = []

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        input_ids = input_ids.to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            hiddens_a = get_hidden_states(model_a, input_ids)
            hiddens_b = get_hidden_states(model_b, input_ids)

        # Compare representations at each layer
        cosines = []
        for h_a, h_b in zip(hiddens_a, hiddens_b):
            # Flatten to [B*T, d]
            h_a_flat = h_a.reshape(-1, h_a.shape[-1]).float()
            h_b_flat = h_b.reshape(-1, h_b.shape[-1]).float()
            cos = F.cosine_similarity(h_a_flat, h_b_flat, dim=-1).mean().item()
            cosines.append(cos)
        all_cosines.append(cosines)

    avg_cosines = np.mean(all_cosines, axis=0)

    print(f"Cosine similarity of hidden states ({label_a} vs {label_b}):")
    print(f"{'Layer':>6} {'Cosine Sim':>12}")
    print("-" * 18)
    for i, cos in enumerate(avg_cosines):
        bar = "#" * int(cos * 40)
        print(f"{i:>6} {cos:>12.4f}  {bar}")

    return avg_cosines


@torch.no_grad()
def gate_analysis(model, dataloader, device, num_batches=50):
    """Analyze learned gate values — which layers use spokes most?"""
    if model.spokes is None:
        print("\n--- Gate Analysis: No spokes ---")
        return

    print("\n--- Learned Gate Values ---")
    gate_vals = []
    for spoke in model.spokes:
        gate_vals.append(torch.sigmoid(spoke.gate_bias).item())

    print(f"{'Layer':>6} {'Gate (sigmoid)':>14} {'Interpretation':>20}")
    print("-" * 42)
    for i, g in enumerate(gate_vals):
        if g < 0.3:
            interp = "weak (explore)"
        elif g > 0.7:
            interp = "strong (converge)"
        else:
            interp = "moderate"
        bar = "#" * int(g * 30)
        print(f"{i:>6} {g:>14.4f}  {interp:<20} {bar}")


@torch.no_grad()
def token_surprise_analysis(
    model_a, model_b, dataloader, device, tokenizer, label_a, label_b, num_batches=30
):
    """Find specific tokens where the models disagree most."""
    surprises = []  # (delta_ce, token_text, context, ce_a, ce_b)

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            res_a = model_a(input_ids, targets)
            res_b = model_b(input_ids, targets)

        ce_a = (
            F.cross_entropy(
                res_a["logits"].view(-1, res_a["logits"].size(-1)),
                targets.view(-1),
                reduction="none",
            )
            .view(targets.shape)
            .cpu()
        )
        ce_b = (
            F.cross_entropy(
                res_b["logits"].view(-1, res_b["logits"].size(-1)),
                targets.view(-1),
                reduction="none",
            )
            .view(targets.shape)
            .cpu()
        )

        delta = ce_b - ce_a  # negative = B better

        # Find biggest disagreements
        for seq_idx in range(targets.shape[0]):
            for pos in range(max(5, 0), targets.shape[1]):
                d = delta[seq_idx, pos].item()
                if abs(d) > 2.0:  # significant disagreement
                    ctx_start = max(0, pos - 5)
                    context = tokenizer.decode(input_ids[seq_idx, ctx_start : pos + 1].cpu())
                    token = tokenizer.decode([targets[seq_idx, pos].item()])
                    surprises.append(
                        (
                            d,
                            token,
                            context,
                            ce_a[seq_idx, pos].item(),
                            ce_b[seq_idx, pos].item(),
                        )
                    )

    surprises.sort(key=lambda x: x[0])

    print("\n--- Biggest Disagreements ---")
    print(f"\nTop 15 where {label_b} is BETTER (lower CE):")
    print(f"{'Delta':>8} {label_a + ' CE':>10} {label_b + ' CE':>10}  Token -> Context")
    print("-" * 70)
    for d, token, context, ca, cb in surprises[:15]:
        ctx_clean = context.replace("\n", " ")[:40]
        print(f"{d:>+8.2f} {ca:>10.2f} {cb:>10.2f}  '{token}' -> ...{ctx_clean}")

    print(f"\nTop 15 where {label_a} is BETTER (lower CE):")
    print(f"{'Delta':>8} {label_a + ' CE':>10} {label_b + ' CE':>10}  Token -> Context")
    print("-" * 70)
    for d, token, context, ca, cb in surprises[-15:]:
        ctx_clean = context.replace("\n", " ")[:40]
        print(f"{d:>+8.2f} {ca:>10.2f} {cb:>10.2f}  '{token}' -> ...{ctx_clean}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--label-a", default="baseline")
    parser.add_argument("--label-b", default="spokes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-batches", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Loading {args.label_a}: {args.ckpt_a}")
    model_a, config_a = load_model(args.ckpt_a, device)
    print(f"  {count_parameters(model_a):,} params")

    print(f"Loading {args.label_b}: {args.ckpt_b}")
    model_b, config_b = load_model(args.ckpt_b, device)
    print(f"  {count_parameters(model_b):,} params")

    import sys

    from torch.utils.data import DataLoader

    sys.path.insert(0, ".")
    from scripts.train import WikiTextDataset

    val_ds = WikiTextDataset("validation", seq_len=512)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # 1. Calibration
    print("\n" + "=" * 70)
    print("1. CALIBRATION ANALYSIS")
    print("=" * 70)
    ece_a = calibration_analysis(model_a, val_loader, device, args.label_a, args.num_batches)
    ece_b = calibration_analysis(model_b, val_loader, device, args.label_b, args.num_batches)
    print(f"\nECE comparison: {args.label_a}={ece_a:.4f}, {args.label_b}={ece_b:.4f}")
    if ece_b < ece_a:
        print(f"  -> {args.label_b} is better calibrated (lower ECE)")
    else:
        print(f"  -> {args.label_a} is better calibrated (lower ECE)")

    # 2. Positional analysis
    print("\n" + "=" * 70)
    print("2. POSITIONAL ANALYSIS")
    print("=" * 70)
    pos_a = positional_analysis(model_a, val_loader, device, args.label_a, args.num_batches)
    pos_b = positional_analysis(model_b, val_loader, device, args.label_b, args.num_batches)
    print(f"\n{'Position':<12} {args.label_a + ' CE':>12} {args.label_b + ' CE':>12} {'Delta':>10}")
    print("-" * 44)
    for i, (a, b) in enumerate(zip(pos_a, pos_b)):
        lo = i * 64
        hi = (i + 1) * 64
        print(f"[{lo:>3}-{hi:>3}]     {a:>12.4f} {b:>12.4f} {b - a:>+10.4f}")

    # 3. Context difficulty
    print("\n" + "=" * 70)
    print("3. CONTEXT DIFFICULTY ANALYSIS")
    print("=" * 70)
    context_difficulty_analysis(
        model_a,
        model_b,
        val_loader,
        device,
        args.label_a,
        args.label_b,
        args.num_batches,
    )

    # 4. Gate analysis (if model B has spokes)
    print("\n" + "=" * 70)
    print("4. GATE ANALYSIS")
    print("=" * 70)
    # Unwrap compiled model if needed
    raw_b = model_b._orig_mod if hasattr(model_b, "_orig_mod") else model_b
    gate_analysis(raw_b, val_loader, device, args.num_batches)

    # 5. Representation analysis
    print("\n" + "=" * 70)
    print("5. REPRESENTATION DIVERGENCE")
    print("=" * 70)
    representation_analysis(
        model_a,
        model_b,
        val_loader,
        device,
        args.label_a,
        args.label_b,
        num_batches=10,
    )

    # 6. Token surprise analysis
    print("\n" + "=" * 70)
    print("6. TOKEN SURPRISE ANALYSIS")
    print("=" * 70)
    token_surprise_analysis(
        model_a,
        model_b,
        val_loader,
        device,
        tokenizer,
        args.label_a,
        args.label_b,
        num_batches=30,
    )


if __name__ == "__main__":
    main()
