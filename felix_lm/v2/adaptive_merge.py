"""Adaptive merge: continuous agreement-driven stream convergence.

In Felix, agents don't merge on a schedule — they converge when they start
agreeing. This module implements that: at each layer, inter-stream agreement
determines how much streams are pulled toward their collective mean.

Low agreement  -> streams stay independent (divergent exploration)
High agreement -> streams converge (convergent synthesis)

The merge trajectory is learned and input-dependent.
"""

import torch
import torch.nn as nn

from felix_lm.exit_heads import compute_cross_stream_agreement
from felix_lm.transformer_block import RMSNorm


class AdaptiveMergeLayer(nn.Module):
    """Per-layer soft merge driven by inter-stream agreement.

    merge_strength = sigmoid(temperature * agreement + bias)
    stream_i = (1 - strength) * stream_i + strength * norm(mean(streams))

    Two learned parameters per layer control the agreement-to-strength mapping.
    """

    def __init__(self, d_stream: int, temp_init: float = 1.0, bias_init: float = -1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temp_init))
        self.bias = nn.Parameter(torch.tensor(bias_init))
        self.norm = RMSNorm(d_stream)

    def forward(
        self, streams: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            streams: list of N tensors, each [B, T, d_stream].

        Returns:
            merged_streams: list of N tensors, each [B, T, d_stream].
            agreement: [B, T] mean pairwise cosine similarity.
            merge_strength: [B, T] learned strength in [0, 1].
        """
        # Compute inter-stream agreement
        agreement = compute_cross_stream_agreement(streams)  # [B, T]

        # Map agreement to merge strength via learned temperature and bias
        merge_strength = torch.sigmoid(self.temperature * agreement + self.bias)  # [B, T]
        strength = merge_strength.unsqueeze(-1)  # [B, T, 1]

        # Compute normalized stream mean
        stacked = torch.stack(streams, dim=0)  # [N, B, T, d]
        stream_mean = self.norm(stacked.mean(dim=0))  # [B, T, d]

        # Interpolate each stream toward the mean
        merged = []
        for s in streams:
            merged.append((1 - strength) * s + strength * stream_mean)

        return merged, agreement, merge_strength
