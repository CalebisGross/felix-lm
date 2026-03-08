"""Felix-LM: Top-level model implementing Algorithm 1.

Full forward pass:
1. Embed tokens, initialize S_0 streams via learned projections
2. For each stage k = 0..K-1:
    a. Process streams independently through L_k transformer layers
    b. If k < K-1: merge stream pairs, compute exit logits
3. Final output from the single remaining stream
4. Deep supervision loss combining all exit points + final output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from felix_lm.config import FelixConfig
from felix_lm.embedding import StreamInitialization
from felix_lm.exit_heads import DeepSupervisionLoss, ExitHead, compute_cross_stream_agreement
from felix_lm.merge import MergeLayer
from felix_lm.stage import Stage
from felix_lm.transformer_block import RMSNorm


class FelixLM(nn.Module):
    """Felix-LM: Multi-Stream Progressive Merging for Causal Language Modeling.

    Implements the complete architecture from the design document,
    including deep supervision, optional early exit, and diagnostic hooks.
    """

    def __init__(self, config: FelixConfig):
        super().__init__()
        self.config = config

        # 1. Embedding + Stream Initialization (Section 3.1)
        self.embedding = StreamInitialization(
            vocab_size=config.vocab_size,
            d_embed=config.d_embed,
            num_streams=config.stages[0].num_streams,
            d_stream=config.stages[0].dim,
        )

        # 2. Stages (Section 3.2)
        self.stages = nn.ModuleList()
        global_layer_offset = 0
        for k, sc in enumerate(config.stages):
            self.stages.append(Stage(sc, global_layer_offset, config))
            global_layer_offset += sc.num_layers

        # 3. Merge layers between stages (Section 3.3)
        self.merges = nn.ModuleList()
        for k in range(config.num_stages - 1):
            num_heads = config.stages[k].num_heads
            self.merges.append(
                MergeLayer(
                    d_in=config.stages[k].dim,
                    d_out=config.stages[k + 1].dim,
                    num_heads=num_heads,
                    gate_bias_init=config.merge_gate_bias_init,
                    use_cross_stream_attention=config.use_cross_stream_attention,
                    use_orthogonal_merge=config.use_orthogonal_merge,
                    token_conditional=config.token_conditional_gating,
                    residual=config.residual_merge,
                    bottleneck_ratio=config.merge_bottleneck_ratio,
                    merge_type=config.merge_type,
                    noise_std=config.merge_noise_std,
                    integration_depth=config.merge_integration_depth,
                    dropout=config.dropout,
                )
            )

        # 4. Exit heads for deep supervision (Section 3.5)
        self.exit_heads = nn.ModuleList()
        for k in range(config.num_stages - 1):
            self.exit_heads.append(
                ExitHead(
                    d_stage=config.stages[k + 1].dim,
                    d_embed=config.d_embed,
                )
            )

        # 5. Output head (Section 3.6)
        self.output_norm = RMSNorm(config.final_dim)
        if config.tie_embeddings and config.final_dim != config.d_embed:
            # Project from final stage dim to embedding dim for weight tying
            self.output_project = nn.Linear(config.final_dim, config.d_embed, bias=False)
        if not config.tie_embeddings:
            self.output_proj = nn.Linear(config.final_dim, config.vocab_size, bias=False)

        # 6. Deep supervision loss
        if config.use_deep_supervision:
            self.loss_fn = DeepSupervisionLoss(config.get_supervision_weights())
        else:
            self.loss_fn = None

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list]:
        """Full forward pass (Algorithm 1).

        Args:
            token_ids: [B, T] input token indices.
            targets: [B, T] target token indices (optional, for loss).

        Returns:
            Dict with:
                'logits': [B, T, V] final output logits.
                'loss': scalar (if targets provided).
                'per_stage_losses': list of K floats (if targets provided).
                'stream_agreements': list of agreement scores at each merge.
        """
        embed_weight = self.embedding.token_embedding.weight  # [V, d_embed]

        # Step 1: Embed and initialize streams
        _, streams = self.embedding(token_ids)

        all_logits = []
        stream_agreements = []
        divergence_loss = torch.tensor(0.0, device=token_ids.device)

        # Step 2: Process stages with merges
        for k in range(self.config.num_stages):
            # Process through stage k
            streams = self.stages[k](streams)

            # Merge if not last stage
            if k < self.config.num_stages - 1:
                # Record agreement before merge (for diagnostics)
                agreement = compute_cross_stream_agreement(streams)
                stream_agreements.append(agreement.mean().detach())

                # Stream divergence loss: maximize cosine distance between pairs
                if self.config.stream_divergence_weight > 0 and len(streams) >= 2:
                    for i in range(len(streams)):
                        for j in range(i + 1, len(streams)):
                            cos_sim = F.cosine_similarity(streams[i], streams[j], dim=-1).mean()
                            divergence_loss = divergence_loss + cos_sim
                    n_pairs = len(streams) * (len(streams) - 1) / 2
                    divergence_loss = divergence_loss / n_pairs

                # Stream dropout: randomly zero out streams before merge
                if self.training and self.config.stream_dropout > 0:
                    p = self.config.stream_dropout
                    # Never drop both streams in a pair
                    for i in range(0, len(streams), 2):
                        if torch.rand(1).item() < p:
                            # Drop one of the two streams randomly
                            drop_idx = i + int(torch.rand(1).item() > 0.5)
                            scale = 2.0  # scale survivor to compensate
                            streams[drop_idx] = torch.zeros_like(streams[drop_idx])
                            other = i + 1 - (drop_idx - i)
                            streams[other] = streams[other] * scale

                # Stream permutation: randomly shuffle merge pairings
                if self.training and self.config.stream_permute and len(streams) > 2:
                    perm = torch.randperm(len(streams))
                    streams = [streams[p] for p in perm]

                # Build RoPE for merge integration layers (at boundary between stages)
                merge_cos, merge_sin = None, None
                if self.config.merge_integration_depth > 0:
                    from felix_lm.rope import build_rope_cache

                    T = streams[0].shape[1]
                    # Use the global layer index at the merge boundary
                    merge_layer_idx = sum(s.num_layers for s in self.config.stages[: k + 1])
                    merge_cos, merge_sin = build_rope_cache(
                        seq_len=T,
                        dim=self.config.stages[k + 1].head_dim,
                        global_layer_idx=merge_layer_idx,
                        total_layers=self.config.total_layers,
                        helical_turns=self.config.rope_helical_turns,
                        base=self.config.rope_base,
                        depth_alpha=self.config.rope_depth_alpha,
                        device=streams[0].device,
                    )

                # Merge pairs: (0,1), (2,3), ...
                new_streams = []
                for i in range(0, len(streams), 2):
                    merged = self.merges[k](streams[i], streams[i + 1], merge_cos, merge_sin)
                    new_streams.append(merged)

                # Exit logits from merged representation
                if len(new_streams) > 1:
                    h_exit = torch.stack(new_streams, dim=0).mean(dim=0)
                else:
                    h_exit = new_streams[0]
                exit_logits = self.exit_heads[k](h_exit, embed_weight)
                all_logits.append(exit_logits)

                streams = new_streams

        # Step 3: Final output from single remaining stream
        assert len(streams) == 1, f"Expected 1 stream at end, got {len(streams)}"
        h_final = self.output_norm(streams[0])  # [B, T, d_{K-1}]

        if self.config.tie_embeddings:
            if hasattr(self, "output_project"):
                h_final = self.output_project(h_final)  # [B, T, d_embed]
            logits = F.linear(h_final, embed_weight)  # [B, T, V]
        else:
            logits = self.output_proj(h_final)
        all_logits.append(logits)

        # Step 4: Compute loss
        result: dict[str, torch.Tensor | list] = {
            "logits": logits,
            "stream_agreements": stream_agreements,
        }

        if targets is not None:
            if self.loss_fn is not None:
                loss, per_stage_losses = self.loss_fn(all_logits, targets)
                result["loss"] = loss
                result["per_stage_losses"] = per_stage_losses
            else:
                # No deep supervision: just final stage loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean",
                )
                result["loss"] = loss
                result["per_stage_losses"] = [loss.detach()]

            # Add divergence loss (penalizes stream similarity)
            if self.config.stream_divergence_weight > 0:
                result["loss"] = result["loss"] + (
                    self.config.stream_divergence_weight * divergence_loss
                )

        return result
