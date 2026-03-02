"""Diagnostic metrics for Felix-LM (Section 6.3).

Tracks the health and behavior of the helical funnel:
- Cross-stream agreement: are streams specializing or converging?
- Gate activation statistics: what's being preserved vs suppressed?
- Effective dimensionality: how much compression at each merge?
"""

import torch
import torch.nn.functional as F

from felix_lm.merge import GatedMerge


def cross_stream_agreement(streams: list[torch.Tensor]) -> float:
    """Mean pairwise cosine similarity between streams (eq. 22).

    Returns scalar in [-1, 1]. Higher = more agreement/convergence.
    """
    if len(streams) < 2:
        return 1.0

    total = 0.0
    count = 0
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            sim = F.cosine_similarity(streams[i], streams[j], dim=-1).mean()
            total += sim.item()
            count += 1
    return total / count


def gate_statistics(gate_values: torch.Tensor) -> dict[str, float]:
    """Compute statistics on merge gate activations.

    Args:
        gate_values: [B, T, 2*d_k] sigmoid gate outputs.

    Returns:
        Dict with mean, std, min, max, and effective dimensionality.
    """
    return {
        "gate_mean": gate_values.mean().item(),
        "gate_std": gate_values.std().item(),
        "gate_min": gate_values.min().item(),
        "gate_max": gate_values.max().item(),
        "effective_dim": gate_values.sum(dim=-1).mean().item(),
    }


def effective_dimensionality(gate_values: torch.Tensor) -> float:
    """Effective dimensionality (Definition 2.2): d_eff = ||g||_1.

    Args:
        gate_values: [B, T, 2*d_k] sigmoid gate outputs.

    Returns:
        Mean effective dimensionality across batch and sequence.
    """
    return gate_values.sum(dim=-1).mean().item()


def collect_merge_diagnostics(
    merge_module: GatedMerge, h_s: torch.Tensor, h_s_prime: torch.Tensor
) -> dict[str, float]:
    """Collect all diagnostic metrics for a single merge operation."""
    with torch.no_grad():
        g = merge_module.get_gate_values(h_s, h_s_prime)
        stats = gate_statistics(g)
    return stats
