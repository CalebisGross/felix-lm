"""FelixV2Layer: one unified layer of the v2 architecture.

Each layer combines:
1. Independent transformer processing per stream
2. CentralPost communication (shared hub)
3. Adaptive soft merge (agreement-driven convergence)
"""

import torch
import torch.nn as nn

from felix_lm.rope import build_rope_cache
from felix_lm.transformer_block import TransformerBlock
from felix_lm.v2.adaptive_merge import AdaptiveMergeLayer
from felix_lm.v2.central_post import CentralPostLayer


class FelixV2Layer(nn.Module):
    """One layer: independent stream processing + CentralPost + soft merge."""

    def __init__(
        self,
        d_stream: int,
        d_post: int,
        num_streams: int,
        num_heads: int,
        attention_type: str,
        ffn_mult: int,
        layer_idx: int,
        total_layers: int,
        rope_base: float,
        rope_helical_turns: int,
        rope_depth_alpha: float,
        merge_temp_init: float,
        merge_bias_init: float,
        dropout: float = 0.0,
        shared_stream_weights: bool = False,
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
        self.shared_stream_weights = shared_stream_weights

        # Transformer block(s) for stream processing
        if shared_stream_weights:
            # One shared block applied to all streams
            self.shared_block = TransformerBlock(
                dim=d_stream,
                num_heads=num_heads,
                attention_type=attention_type,
                ffn_mult=ffn_mult,
                dropout=dropout,
            )
        else:
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

        # CentralPost communication
        self.central_post = CentralPostLayer(d_stream, d_post, num_streams)

        # Adaptive merge
        self.adaptive_merge = AdaptiveMergeLayer(d_stream, merge_temp_init, merge_bias_init)

    def forward(
        self,
        streams: list[torch.Tensor],
        central_post: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            streams: list of N tensors, each [B, T, d_stream].
            central_post: [B, T, d_post].

        Returns:
            streams: updated streams.
            central_post: updated CentralPost state.
            agreement: [B, T] inter-stream agreement.
            merge_strength: [B, T] learned merge strength.
        """
        T = streams[0].shape[1]
        head_dim = self.d_stream // self.num_heads

        # Build RoPE cache for this layer
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

        # 1. Transformer processing per stream
        if self.shared_stream_weights:
            processed = [self.shared_block(stream, cos, sin) for stream in streams]
        else:
            processed = [
                block(stream, cos, sin) for block, stream in zip(self.stream_blocks, streams)
            ]

        # 2. CentralPost communication
        processed, central_post = self.central_post(processed, central_post)

        # 3. Adaptive soft merge
        processed, agreement, merge_strength = self.adaptive_merge(processed)

        return processed, central_post, agreement, merge_strength
