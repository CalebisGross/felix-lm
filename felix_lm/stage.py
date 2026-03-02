"""Stage module (Section 3, Architecture overview).

A stage processes S_k independent streams, each through L_k transformer layers.
Streams within a stage do not interact — they process independently
(like Felix agents exploring in parallel).

The overall computation flow per stage:
    for each stream s in 1..S_k:
        for each layer l in 0..L_k-1:
            h_{:,s}^{k,l+1} = TransformerLayer(h_{:,s}^{k,l})
"""

import torch
import torch.nn as nn

from felix_lm.config import FelixConfig, StageConfig
from felix_lm.rope import build_rope_cache
from felix_lm.transformer_block import TransformerBlock


class Stage(nn.Module):
    """A single processing stage: S_k streams x L_k transformer layers.

    Args:
        stage_config: Configuration for this stage.
        global_layer_offset: Starting global layer index (for depth-extended RoPE).
        config: Full model configuration.
    """

    def __init__(self, stage_config: StageConfig, global_layer_offset: int, config: FelixConfig):
        super().__init__()
        self.num_streams = stage_config.num_streams
        self.num_layers = stage_config.num_layers
        self.global_layer_offset = global_layer_offset
        self.total_layers = config.total_layers
        self.helical_turns = config.rope_helical_turns
        self.rope_base = config.rope_base
        self.depth_alpha = config.rope_depth_alpha
        self.head_dim = stage_config.head_dim

        if config.weight_sharing == "shared":
            # All streams share the same layer parameters
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=stage_config.dim,
                        num_heads=stage_config.num_heads,
                        attention_type=stage_config.attention_type,
                        ffn_mult=stage_config.ffn_mult,
                        window_size=stage_config.window_size,
                        dropout=config.dropout,
                    )
                    for _ in range(stage_config.num_layers)
                ]
            )
            self.shared = True
        else:
            # Each stream has independent layer parameters
            self.stream_layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            TransformerBlock(
                                dim=stage_config.dim,
                                num_heads=stage_config.num_heads,
                                attention_type=stage_config.attention_type,
                                ffn_mult=stage_config.ffn_mult,
                                window_size=stage_config.window_size,
                                dropout=config.dropout,
                            )
                            for _ in range(stage_config.num_layers)
                        ]
                    )
                    for _ in range(stage_config.num_streams)
                ]
            )
            self.shared = False

    def forward(self, streams: list[torch.Tensor]) -> list[torch.Tensor]:
        """Process all streams through this stage's layers.

        Args:
            streams: list of S_k tensors, each [B, T, d_k].

        Returns:
            list of S_k processed tensors, each [B, T, d_k].
        """
        T = streams[0].shape[1]
        device = streams[0].device
        output_streams = []

        for s in range(self.num_streams):
            h = streams[s]
            layers = self.layers if self.shared else self.stream_layers[s]

            for ell, layer in enumerate(layers):
                global_l = self.global_layer_offset + ell
                cos, sin = build_rope_cache(
                    seq_len=T,
                    dim=self.head_dim,
                    global_layer_idx=global_l,
                    total_layers=self.total_layers,
                    helical_turns=self.helical_turns,
                    base=self.rope_base,
                    depth_alpha=self.depth_alpha,
                    device=device,
                )
                h = layer(h, cos, sin)

            output_streams.append(h)

        return output_streams
