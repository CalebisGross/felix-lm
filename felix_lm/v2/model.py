"""Felix-LM v2: Adaptive Convergence Architecture.

Forward pass:
1. Embed tokens, initialize N streams via learned projections
2. Initialize CentralPost to zeros
3. For each of L uniform layers:
    a. Process streams independently through transformer blocks
    b. CentralPost read/write (shared hub communication)
    c. Adaptive soft merge (agreement-driven convergence)
4. Aggregate streams via learned weights
5. Project to embedding dimension
6. Refinement layers at full dimension
7. Output logits via tied embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from felix_lm.embedding import StreamInitialization
from felix_lm.rope import build_rope_cache
from felix_lm.transformer_block import RMSNorm, TransformerBlock
from felix_lm.v2.config import FelixV2Config
from felix_lm.v2.felix_layer import FelixV2Layer


class FelixLMv2(nn.Module):
    """Felix-LM v2 with adaptive convergence."""

    def __init__(self, config: FelixV2Config):
        super().__init__()
        self.config = config

        # 1. Embedding + stream initialization (reused from v1)
        self.stream_init = StreamInitialization(
            config.vocab_size, config.d_embed, config.num_streams, config.d_stream
        )

        # 2. Uniform Felix layers
        self.layers = nn.ModuleList(
            [
                FelixV2Layer(
                    d_stream=config.d_stream,
                    d_post=config.d_post,
                    num_streams=config.num_streams,
                    num_heads=config.num_heads,
                    attention_type=config.attention_type,
                    ffn_mult=config.ffn_mult,
                    layer_idx=i,
                    total_layers=config.total_layers,
                    rope_base=config.rope_base,
                    rope_helical_turns=config.rope_helical_turns,
                    rope_depth_alpha=config.rope_depth_alpha,
                    merge_temp_init=config.merge_temp_init,
                    merge_bias_init=config.merge_bias_init,
                    dropout=config.dropout,
                )
                for i in range(config.num_layers)
            ]
        )

        # 3. Stream aggregation: learned weights over streams
        self.stream_weights = nn.Parameter(torch.ones(config.num_streams))

        # 4. Project d_stream -> d_embed
        self.output_project = nn.Linear(config.d_stream, config.d_embed, bias=False)

        # 5. Refinement layers at d_embed
        self.refine_layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.d_embed,
                    num_heads=config.num_refine_heads,
                    attention_type="full_causal",
                    ffn_mult=config.ffn_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.num_refine_layers)
            ]
        )

        # 6. Output norm
        self.output_norm = RMSNorm(config.d_embed)

        # Tie embedding weights
        if config.tie_embeddings:
            self._logit_scale = config.d_embed**-0.5

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            token_ids: [B, T] input token indices.
            targets: [B, T] target token indices (optional).

        Returns:
            dict with keys: logits, loss (if targets), agreements, merge_strengths.
        """
        B, T = token_ids.shape
        config = self.config

        # Embedding + stream init
        embeddings, streams = self.stream_init(token_ids)

        # Initialize CentralPost to zeros
        central_post = torch.zeros(
            B,
            T,
            config.d_post,
            device=token_ids.device,
            dtype=streams[0].dtype,
        )

        # Process through Felix layers
        agreements = []
        merge_strengths = []
        for layer in self.layers:
            streams, central_post, agreement, strength = layer(streams, central_post)
            agreements.append(agreement.mean().detach())
            merge_strengths.append(strength.mean().detach())

        # Aggregate streams with learned weights
        weights = F.softmax(self.stream_weights, dim=0)  # [N]
        h = sum(w * s for w, s in zip(weights, streams))  # [B, T, d_stream]

        # Project to embedding dimension
        h = self.output_project(h)  # [B, T, d_embed]

        # Refinement layers
        head_dim = config.d_embed // config.num_refine_heads
        for i, refine_layer in enumerate(self.refine_layers):
            global_idx = config.num_layers + i
            cos, sin = build_rope_cache(
                seq_len=T,
                dim=head_dim,
                global_layer_idx=global_idx,
                total_layers=config.total_layers,
                helical_turns=config.rope_helical_turns,
                base=config.rope_base,
                depth_alpha=config.rope_depth_alpha,
                device=token_ids.device,
            )
            h = refine_layer(h, cos, sin)

        # Output logits
        h = self.output_norm(h)
        embed_weight = self.stream_init.token_embedding.weight
        logits = F.linear(h, embed_weight)
        if config.tie_embeddings:
            logits = logits * self._logit_scale

        result = {
            "logits": logits,
            "agreements": agreements,
            "merge_strengths": merge_strengths,
        }

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                targets.view(-1),
                reduction="mean",
            )
            result["loss"] = loss

        return result
