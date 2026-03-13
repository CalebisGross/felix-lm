"""Felix-LM v2 configuration."""

from dataclasses import dataclass, field


@dataclass
class FelixV2Config:
    """Configuration for Felix-LM v2 Adaptive Convergence Architecture.

    Unlike v1 (fixed stages with hard merges), v2 uses uniform layers with
    continuous adaptive merging driven by inter-stream agreement.
    """

    # Dimensions
    vocab_size: int = 50257
    d_embed: int = 128
    d_stream: int = 64
    d_post: int = 64

    # Architecture
    num_streams: int = 4
    num_layers: int = 0
    num_heads: int = 2  # head_dim = d_stream // num_heads = 32
    num_refine_layers: int = 16
    num_refine_heads: int = 4  # head_dim = d_embed // num_refine_heads = 32
    ffn_mult: int = 4
    attention_type: str = "full_causal"
    # Per-layer attention schedule. If non-empty, overrides attention_type.
    # Must be length num_layers. e.g. ["linear"]*4 + ["sliding_window"]*4 + ["full_causal"]*5
    attention_schedule: list[str] = field(default_factory=list)

    # CentralPost
    cp_read_gated: bool = True
    cp_write_gated: bool = True

    # Adaptive merge
    merge_temp_init: float = 1.0
    merge_bias_init: float = -1.0  # sigmoid(-1) ~ 0.27, starts mostly independent

    # RoPE
    rope_base: float = 10000.0
    rope_helical_turns: int = 2
    rope_depth_alpha: float = 1.0

    # Training
    tie_embeddings: bool = True
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    def get_attention_type(self, layer_idx: int) -> str:
        """Get attention type for a given layer index."""
        if self.attention_schedule:
            return self.attention_schedule[layer_idx]
        return self.attention_type

    @property
    def total_layers(self) -> int:
        return self.num_layers + self.num_refine_layers

    def count_params(self) -> int:
        """Estimate total parameter count."""
        V, de, ds, dp = self.vocab_size, self.d_embed, self.d_stream, self.d_post
        N, L, Lr = self.num_streams, self.num_layers, self.num_refine_layers
        fm = self.ffn_mult

        # Embedding (tied, counted once)
        embed = V * de

        # Stream init: N projections from d_embed -> d_stream
        stream_init = N * (de * ds + ds)

        # Per stream transformer block: attn(4 * ds^2) + ffn(3 * ds * ds*fm)
        block_params = 4 * ds * ds + 3 * ds * (ds * fm)
        # Plus 2 RMSNorm weights
        block_params += 2 * ds
        stream_layers = N * L * block_params

        # CentralPost per layer: N * (read_proj + read_gate + write_proj + write_gate) + norm
        cp_per_layer = N * (dp * ds + ds + 1 + ds * dp + ds + 1) + dp
        central_post = L * cp_per_layer

        # Adaptive merge per layer: temperature + bias + RMSNorm
        adaptive_merge = L * (2 + ds)

        # Stream aggregation weights
        agg = N

        # Output projection: d_stream -> d_embed
        out_proj = ds * de

        # Refinement layers: same as transformer block at d_embed
        refine_block = 4 * de * de + 3 * de * (de * fm) + 2 * de
        refine = Lr * refine_block

        # Output norm
        out_norm = de

        return (
            embed
            + stream_init
            + stream_layers
            + central_post
            + adaptive_merge
            + agg
            + out_proj
            + refine
            + out_norm
        )
