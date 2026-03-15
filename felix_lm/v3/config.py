"""Felix-LM v3 configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FelixV3Config:
    """Configuration for Felix-LM v3 Hub-and-Spoke Architecture.

    The hub is a standard transformer (all the params, all the depth).
    Spokes are lightweight low-rank probes that read/write the hub state.
    """

    # Vocabulary and embedding
    vocab_size: int = 50257
    d_embed: int = 128

    # Hub (backbone transformer)
    num_layers: int = 20
    num_heads: int = 4  # head_dim = d_embed // num_heads = 32
    ffn_mult: int = 4

    # Spokes (lightweight probes)
    num_spokes: int = 4
    spoke_rank: int = 16  # low-rank bottleneck dimension

    # Gate schedule: how spoke feedback scales with depth
    #   "progressive" — early layers get small gates (explore), late layers get large (converge)
    #   "uniform"     — all gates initialized to 0 (sigmoid=0.5)
    #   "none"        — no spokes at all (pure transformer baseline)
    gate_schedule: Literal["progressive", "uniform", "none"] = "progressive"
    # For "progressive": gate_bias linearly interpolated from gate_init_start to gate_init_end
    gate_init_start: float = -2.0  # sigmoid(-2) ~ 0.12, early layers barely affect hub
    gate_init_end: float = 2.0  # sigmoid(2) ~ 0.88, late layers strongly affect hub

    # Embedding projection (adds a linear layer after embedding, like v2's StreamInit)
    embed_proj: bool = False

    # RoPE
    rope_base: float = 10000.0
    rope_helical_turns: int = 2
    rope_depth_alpha: float = 1.0

    # Training
    tie_embeddings: bool = True
    dropout: float = 0.0
    label_smoothing: float = 0.0
    gradient_checkpointing: bool = False

    @property
    def total_layers(self) -> int:
        return self.num_layers

    @property
    def head_dim(self) -> int:
        return self.d_embed // self.num_heads

    def count_params(self) -> int:
        """Estimate total parameter count."""
        V, d = self.vocab_size, self.d_embed
        L, S, r = self.num_layers, self.num_spokes, self.spoke_rank
        fm = self.ffn_mult

        # Embedding (tied, counted once)
        embed = V * d

        # Per transformer block: attn (4 * d^2) + ffn (3 * d * d*fm) + 2 RMSNorm
        block = 4 * d * d + 3 * d * (d * fm) + 2 * d
        backbone = L * block

        # Per spoke layer (if spokes enabled):
        #   RMSNorm(d) + S * (W_down(d,r) + W_up(r,d)) + gate_bias(1)
        if self.gate_schedule != "none":
            spoke_layer = d + S * (d * r + r * d) + 1
            spokes_total = L * spoke_layer
        else:
            spokes_total = 0

        # Embedding projection
        embed_proj = (d * d + d) if self.embed_proj else 0

        # Output norm
        out_norm = d

        return embed + backbone + spokes_total + embed_proj + out_norm
