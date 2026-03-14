"""Lightweight FelixV2Layer: minimal overhead stream + cross-stream communication.

Instead of separate CentralPost + AdaptiveMerge, uses a single lightweight
cross-stream exchange: each stream receives a gated residual from the
stream mean, providing inter-stream communication with minimal parameters.
"""

import torch
import torch.nn as nn

from felix_lm.rope import build_rope_cache
from felix_lm.transformer_block import RMSNorm, TransformerBlock


class LightFelixLayer(nn.Module):
    """Stream processing + lightweight cross-stream exchange."""

    def __init__(
        self,
        d_stream: int,
        num_streams: int,
        num_heads: int,
        attention_type: str,
        ffn_mult: int,
        layer_idx: int,
        total_layers: int,
        rope_base: float,
        rope_helical_turns: int,
        rope_depth_alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.d_stream = d_stream
        self.num_streams = num_streams
        self.num_heads = num_heads
        self.rope_base = rope_base
        self.rope_helical_turns = rope_helical_turns
        self.rope_depth_alpha = rope_depth_alpha

        # Independent transformer block per stream
        self.stream_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=d_stream,
                    num_heads=num_heads,
                    attention_type=attention_type,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(num_streams)
            ]
        )

        # Lightweight cross-stream exchange: per-stream gate + shared norm
        self.exchange_gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_streams)]
        )
        self.exchange_norm = RMSNorm(d_stream)

    def forward(
        self,
        streams: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            streams: list of N tensors, each [B, T, d_stream].

        Returns:
            streams: updated streams.
            agreement: [B, T] mean pairwise cosine similarity.
            exchange_strength: scalar mean gate value.
        """
        T = streams[0].shape[1]
        head_dim = self.d_stream // self.num_heads

        cos, sin = build_rope_cache(
            seq_len=T,
            dim=head_dim,
            global_layer_idx=self.layer_idx,
            total_layers=self.total_layers,
            helical_turns=self.rope_helical_turns,
            base=self.rope_base,
            depth_alpha=self.rope_depth_alpha,
            device=streams[0].device,
        )

        # 1. Independent transformer processing
        processed = [block(stream, cos, sin) for block, stream in zip(self.stream_blocks, streams)]

        # 2. Cross-stream exchange: gated residual from stream mean
        stacked = torch.stack(processed, dim=0)  # [N, B, T, d]
        stream_mean = self.exchange_norm(stacked.mean(dim=0))  # [B, T, d]

        exchanged = []
        gate_values = []
        for i, s in enumerate(processed):
            gate = torch.sigmoid(self.exchange_gates[i])  # scalar
            gate_values.append(gate)
            exchanged.append(s + gate * (stream_mean - s))

        # Compute agreement for monitoring (cosine similarity between streams)
        with torch.no_grad():
            s0_norm = torch.nn.functional.normalize(processed[0], dim=-1)
            s1_norm = torch.nn.functional.normalize(processed[1], dim=-1)
            agreement = (s0_norm * s1_norm).sum(dim=-1)  # [B, T]

        mean_gate = torch.stack(gate_values).mean()

        return exchanged, agreement, mean_gate
