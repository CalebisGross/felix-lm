"""Configuration system for Felix-LM.

Dataclass-based configs with Felix framework parameter mapping (Section 5)
and parameter count formulas (Appendix B).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class StageConfig:
    """Configuration for a single processing stage."""

    num_streams: int
    dim: int
    num_layers: int
    num_heads: int
    attention_type: Literal["linear", "sliding_window", "full_causal"]
    window_size: int = 64
    ffn_mult: int = 4

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


@dataclass
class FelixConfig:
    """Full Felix-LM configuration.

    All ablation variables from the thesis are explicit flags here.
    """

    # Vocabulary and embedding
    vocab_size: int = 50257
    d_embed: int = 128

    # Stages
    stages: list[StageConfig] = field(default_factory=list)

    # Merge settings
    use_cross_stream_attention: bool = False
    use_orthogonal_merge: bool = False
    merge_gate_bias_init: float = 1.0

    # RoPE settings
    rope_base: float = 10000.0
    rope_depth_alpha: float = 1.0
    rope_helical_turns: int = 2

    # Early exit
    use_early_exit: bool = False
    confidence_threshold: float = 0.8
    entropy_threshold: float = 1.0

    # Deep supervision
    use_deep_supervision: bool = True
    supervision_weights: list[float] | None = None

    # Training
    tie_embeddings: bool = True
    dropout: float = 0.1
    weight_sharing: Literal["shared", "independent"] = "independent"

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def total_layers(self) -> int:
        return sum(s.num_layers for s in self.stages)

    @property
    def final_dim(self) -> int:
        return self.stages[-1].dim

    def get_supervision_weights(self) -> list[float]:
        """Compute lambda_k for deep supervision loss (eq. 24).

        Final stage weighted most: lambda_{K-1} > lambda_k for k < K-1.
        """
        if self.supervision_weights is not None:
            return self.supervision_weights
        K = self.num_stages
        if K == 1:
            return [1.0]
        # Final stage gets 0.6, rest split equally
        early_weight = 0.4 / (K - 1)
        return [early_weight] * (K - 1) + [0.6]

    def count_params(self) -> int:
        """Compute total parameter count per Appendix B (eqs. 34-38)."""
        V = self.vocab_size
        d_e = self.d_embed
        K = self.num_stages

        # N_embed: token embedding + stream initialization projections (eq. 34)
        S_0 = self.stages[0].num_streams
        d_0 = self.stages[0].dim
        n_embed = V * d_e + S_0 * d_0 * d_e + S_0 * d_0  # +bias

        # N_stages: transformer layers per stage (eq. 35, adjusted for SwiGLU)
        # Per layer: attn (4*d^2) + SwiGLU FFN (3*d*ffn_mult*d) + 2 norms (2*d)
        n_stages = 0
        for s in self.stages:
            per_layer = (
                4 * s.dim**2  # Q, K, V, O projections
                + 3 * s.dim * s.ffn_mult * s.dim  # SwiGLU: gate + up + down
                + 2 * s.dim  # 2x RMSNorm
            )
            n_stages += s.num_streams * s.num_layers * per_layer

        # N_merges: merge layers (eq. 36)
        # Gate: 2*d_k -> 2*d_k (weight + bias)
        # Projection: 2*d_k -> d_{k+1} (no bias)
        n_merges = 0
        for k in range(K - 1):
            d_k = self.stages[k].dim
            d_next = self.stages[k + 1].dim
            num_merge_pairs = self.stages[k].num_streams // 2
            per_merge = (
                2 * d_k * 2 * d_k
                + 2 * d_k  # gate weight + bias
                + d_next * 2 * d_k  # projection weight
            )
            n_merges += num_merge_pairs * per_merge

        # N_exits: exit heads (eq. 37)
        # Each: d_k -> d_embed projection (weight + bias) + norm
        n_exits = 0
        for k in range(K - 1):
            d_k = self.stages[k + 1].dim  # operates on merged repr
            n_exits += d_k * d_e + d_e + d_k  # linear + bias + norm

        # N_output: final output head (eq. 38)
        # If tie_embeddings, no extra params (reuses embedding matrix)
        # Otherwise: d_{K-1} -> V
        n_output = 0
        if not self.tie_embeddings:
            n_output = self.final_dim * V
        # Output norm
        n_output += self.final_dim

        return n_embed + n_stages + n_merges + n_exits + n_output

    @classmethod
    def from_felix_params(
        cls,
        top_radius: float = 4.0,
        bottom_radius: float = 1.0,
        turns: int = 2,
        height: int = 13,
        compression_ratio: float = 1.0,
        confidence_threshold: float = 0.8,
        vocab_size: int = 50257,
        d_embed: int = 128,
        base_dim: int = 64,
        base_heads: int = 4,
        **kwargs,
    ) -> FelixConfig:
        """Create config from Felix framework parameters (Section 5 mapping).

        Args:
            top_radius: Felix R_max -> S_0 = 2^ceil(log2(R_max))
            bottom_radius: Felix R_min -> S_{K-1} = 1 (always)
            turns: Felix n -> K = n + 1 stages
            height: Felix H -> L_total proportional to H
            compression_ratio: d_{k+1} / (2 * d_k) at each merge
            confidence_threshold: Felix confidence_threshold -> tau_conf
            base_dim: Dimension of first stage streams
            base_heads: Number of attention heads in first stage
        """
        K = turns + 1
        S_0 = 2 ** math.ceil(math.log2(max(top_radius, 2)))

        # Stream counts halve at each stage (binary merging)
        stream_counts = [max(1, S_0 // (2**k)) for k in range(K)]

        # Dimensions: double at each merge (since we concat pairs then project)
        # Stage 0: base_dim, Stage 1: base_dim * 2, etc. (capped)
        dims = [base_dim]
        for k in range(1, K):
            # After merging 2 streams of dim d_k, we project to d_{k+1}
            # Default: d_{k+1} = d_k * 2 * compression_ratio, capped at 2*base_dim
            next_dim = min(int(dims[-1] * 2 * compression_ratio), base_dim * 4)
            dims.append(next_dim)

        # Distribute layers across stages proportional to height
        # Later stages get slightly more layers (more focused computation)
        total_layers = height
        layers_per_stage = []
        for k in range(K):
            # Weight later stages more
            weight = 1.0 + 0.25 * k
            layers_per_stage.append(weight)
        total_weight = sum(layers_per_stage)
        layers_per_stage = [
            max(1, round(w / total_weight * total_layers)) for w in layers_per_stage
        ]

        # Adjust to match total
        diff = total_layers - sum(layers_per_stage)
        layers_per_stage[-1] += diff

        # Attention types per stage (Section 3.2 table)
        attention_types = []
        for k in range(K):
            if k == 0:
                attention_types.append("linear")
            elif k == K - 1:
                attention_types.append("full_causal")
            else:
                attention_types.append("sliding_window")

        # Build stage configs
        stages = []
        for k in range(K):
            num_heads = max(1, dims[k] // (base_dim // base_heads))
            stages.append(
                StageConfig(
                    num_streams=stream_counts[k],
                    dim=dims[k],
                    num_layers=layers_per_stage[k],
                    num_heads=num_heads,
                    attention_type=attention_types[k],
                )
            )

        return cls(
            vocab_size=vocab_size,
            d_embed=d_embed,
            stages=stages,
            rope_helical_turns=turns,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )


def make_m2_config() -> FelixConfig:
    """M2 MSPM-HETERO: Primary proof-of-concept configuration (~10M params)."""
    return FelixConfig(
        vocab_size=50257,
        d_embed=128,
        stages=[
            StageConfig(
                num_streams=4,
                dim=64,
                num_layers=4,
                num_heads=4,
                attention_type="linear",
            ),
            StageConfig(
                num_streams=2,
                dim=128,
                num_layers=4,
                num_heads=4,
                attention_type="sliding_window",
                window_size=64,
            ),
            StageConfig(
                num_streams=1,
                dim=128,
                num_layers=5,
                num_heads=4,
                attention_type="full_causal",
            ),
        ],
        rope_helical_turns=2,
        use_deep_supervision=True,
        tie_embeddings=True,
    )


def make_m2_fullcausal_config() -> FelixConfig:
    """M2 with full causal attention everywhere (isolates multi-stream vs attention type)."""
    return FelixConfig(
        vocab_size=50257,
        d_embed=128,
        stages=[
            StageConfig(
                num_streams=4,
                dim=64,
                num_layers=4,
                num_heads=4,
                attention_type="full_causal",
            ),
            StageConfig(
                num_streams=2,
                dim=128,
                num_layers=4,
                num_heads=4,
                attention_type="full_causal",
            ),
            StageConfig(
                num_streams=1,
                dim=128,
                num_layers=5,
                num_heads=4,
                attention_type="full_causal",
            ),
        ],
        rope_helical_turns=2,
        use_deep_supervision=True,
        tie_embeddings=True,
    )


def make_m2_nosup_config() -> FelixConfig:
    """M2 with deep supervision disabled (isolates supervision effect)."""
    return FelixConfig(
        vocab_size=50257,
        d_embed=128,
        stages=[
            StageConfig(
                num_streams=4,
                dim=64,
                num_layers=4,
                num_heads=4,
                attention_type="linear",
            ),
            StageConfig(
                num_streams=2,
                dim=128,
                num_layers=4,
                num_heads=4,
                attention_type="sliding_window",
                window_size=64,
            ),
            StageConfig(
                num_streams=1,
                dim=128,
                num_layers=5,
                num_heads=4,
                attention_type="full_causal",
            ),
        ],
        rope_helical_turns=2,
        use_deep_supervision=False,
        tie_embeddings=True,
    )


def make_m2_best_config() -> FelixConfig:
    """M2-BEST: Full causal + no deep supervision (combines best ablation flags)."""
    return FelixConfig(
        vocab_size=50257,
        d_embed=128,
        stages=[
            StageConfig(
                num_streams=4,
                dim=64,
                num_layers=4,
                num_heads=4,
                attention_type="full_causal",
            ),
            StageConfig(
                num_streams=2,
                dim=128,
                num_layers=4,
                num_heads=4,
                attention_type="full_causal",
            ),
            StageConfig(
                num_streams=1,
                dim=128,
                num_layers=5,
                num_heads=4,
                attention_type="full_causal",
            ),
        ],
        rope_helical_turns=2,
        use_deep_supervision=False,
        tie_embeddings=True,
    )


def make_m0_config() -> FelixConfig:
    """M0 UNIFORM: Standard transformer baseline (~10M params)."""
    return FelixConfig(
        vocab_size=50257,
        d_embed=128,
        stages=[
            StageConfig(
                num_streams=1,
                dim=128,
                num_layers=18,
                num_heads=4,
                attention_type="full_causal",
            ),
        ],
        use_deep_supervision=False,
        tie_embeddings=True,
        rope_depth_alpha=0.0,  # standard RoPE for baseline
    )
