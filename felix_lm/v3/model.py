"""Felix-LM v3: Hub-and-Spoke model.

Forward pass:
1. Embed tokens (standard, no stream init)
2. For each of L layers:
    a. TransformerBlock (the hub — full attention + FFN)
    b. SpokeLayer (lightweight probes — read, transform, gate back)
3. RMSNorm -> tied embedding logits with scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from felix_lm.rope import build_rope_cache
from felix_lm.transformer_block import RMSNorm, TransformerBlock
from felix_lm.v3.config import FelixV3Config
from felix_lm.v3.spokes import SpokeLayer


class FelixLMv3(nn.Module):
    """Felix-LM v3 with hub-and-spoke architecture."""

    def __init__(self, config: FelixV3Config):
        super().__init__()
        self.config = config

        # 1. Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.embed_proj = nn.Linear(config.d_embed, config.d_embed) if config.embed_proj else None

        # 2. Transformer backbone (the hub)
        layers = []
        for i in range(config.num_layers):
            if config.attention_pattern == "sssl":
                period = config.sssl_ratio + 1
                is_last = i == config.num_layers - 1
                is_global = (i % period) == config.sssl_ratio or is_last
                attn_type = "full_causal" if is_global else "sliding_window"
                window = config.sssl_window_size
            else:
                attn_type = "full_causal"
                window = 64  # unused
            layers.append(
                TransformerBlock(
                    dim=config.d_embed,
                    num_heads=config.num_heads,
                    attention_type=attn_type,
                    ffn_mult=config.ffn_mult,
                    window_size=window,
                    dropout=config.dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

        # 3. Spoke layers (the agents)
        if config.gate_schedule != "none":
            self.spokes = nn.ModuleList(
                [
                    SpokeLayer(
                        d_model=config.d_embed,
                        num_spokes=config.num_spokes,
                        rank=config.spoke_rank,
                        gate_init=self._gate_init_for_layer(i, config),
                    )
                    for i in range(config.num_layers)
                ]
            )
        else:
            self.spokes = None

        # 4. Per-layer residual lambdas: h = lambda_resid * h + lambda_x0 * x0
        if config.use_residual_lambdas:
            self.lambda_resid = nn.ParameterList(
                [nn.Parameter(torch.ones(1)) for _ in range(config.num_layers)]
            )
            self.lambda_x0 = nn.ParameterList(
                [
                    nn.Parameter(torch.full((1,), config.lambda_x0_init))
                    for _ in range(config.num_layers)
                ]
            )

        # 5. Output
        self.output_norm = RMSNorm(config.d_embed)

        # Tied embeddings with logit scaling
        if config.tie_embeddings:
            self._logit_scale = config.d_embed**-0.5

    @staticmethod
    def _gate_init_for_layer(layer_idx: int, config: FelixV3Config) -> float:
        """Compute gate bias initialization for a given layer."""
        if config.gate_schedule == "uniform":
            return 0.0
        # Progressive: linearly interpolate from start to end
        if config.num_layers == 1:
            return (config.gate_init_start + config.gate_init_end) / 2
        t = layer_idx / (config.num_layers - 1)
        return config.gate_init_start + t * (config.gate_init_end - config.gate_init_start)

    def _checkpointed_forward(
        self,
        layer: TransformerBlock,
        spoke: SpokeLayer | None,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one hub layer + spoke with gradient checkpointing."""

        def run(h_in: torch.Tensor, cos_in: torch.Tensor, sin_in: torch.Tensor):
            h_out = layer(h_in, cos_in, sin_in)
            if spoke is not None:
                h_out, agreement = spoke(h_out)
                return h_out, agreement
            return h_out, torch.tensor(0.0, device=h_in.device)

        result = grad_checkpoint(run, h, cos, sin, use_reentrant=False)
        return result  # type: ignore[return-value]

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
            dict with keys: logits, loss (if targets), agreements, gate_values.
        """
        _, T = token_ids.shape
        config = self.config

        # Embed
        h = self.embedding(token_ids)  # [B, T, d]
        if self.embed_proj is not None:
            h = self.embed_proj(h)

        # Save original embedding for residual lambdas
        x0 = h if config.use_residual_lambdas else None

        # Process through hub layers + spokes
        agreements: list[torch.Tensor] = []
        gate_values: list[torch.Tensor] = []
        use_ckpt = config.gradient_checkpointing and self.training

        for i, layer in enumerate(self.layers):
            # Build RoPE for this layer
            cos, sin = build_rope_cache(
                seq_len=T,
                dim=config.head_dim,
                global_layer_idx=i,
                total_layers=config.total_layers,
                helical_turns=config.rope_helical_turns,
                base=config.rope_base,
                depth_alpha=config.rope_depth_alpha,
                device=token_ids.device,
            )

            spoke = self.spokes[i] if self.spokes is not None else None

            if use_ckpt and spoke is not None:
                h, agreement = self._checkpointed_forward(layer, spoke, h, cos, sin)
                agreements.append(agreement.mean().detach())
                gate_values.append(
                    torch.sigmoid(spoke.gate_bias).detach()  # type: ignore[arg-type]
                )
            elif use_ckpt:
                h, _ = self._checkpointed_forward(layer, None, h, cos, sin)
            else:
                # Hub: standard transformer block
                h = layer(h, cos, sin)

                # Spokes: lightweight probe + gated residual
                if spoke is not None:
                    h, agreement = spoke(h)
                    agreements.append(agreement.mean().detach())
                    gate_values.append(
                        torch.sigmoid(spoke.gate_bias).detach()  # type: ignore[arg-type]
                    )

            # Residual lambdas: mix running hidden state with original embedding
            if x0 is not None:
                h = self.lambda_resid[i] * h + self.lambda_x0[i] * x0

        # Output logits
        h = self.output_norm(h)
        if config.tie_embeddings:
            logits = F.linear(h, self.embedding.weight) * self._logit_scale
        else:
            raise NotImplementedError("Untied embeddings not implemented for v3")

        if config.logit_softcap > 0:
            logits = config.logit_softcap * torch.tanh(logits / config.logit_softcap)

        result = {
            "logits": logits,
            "agreements": agreements,
            "gate_values": gate_values,
        }

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                targets.view(-1),
                reduction="mean",
                label_smoothing=config.label_smoothing,
            )
            result["loss"] = loss

        return result
