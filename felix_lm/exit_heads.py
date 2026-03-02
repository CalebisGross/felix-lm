"""Exit heads and deep supervision (Sections 3.5, 3.5.2).

Exit heads enable:
1. Deep supervision during training (loss at every merge boundary)
2. Confidence-gated early exit at inference time

Early exit criterion (Definition 3.7):
    Token t exits at stage k if:
        Agree_t^{k} > tau_conf  AND  H[p_hat_k(· | x_{<=t})] < tau_ent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from felix_lm.transformer_block import RMSNorm


class ExitHead(nn.Module):
    """Lightweight exit head at a merge boundary (eq. 21).

    Projects from stage dimension to embedding dimension, then reuses
    the embedding matrix for vocabulary logits. This keeps the exit
    heads extremely parameter-efficient.
    """

    def __init__(self, d_stage: int, d_embed: int):
        super().__init__()
        self.norm = RMSNorm(d_stage)
        self.project = nn.Linear(d_stage, d_embed)

    def forward(self, h: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        """Compute exit logits.

        Args:
            h: [B, T, d_stage] representation (merged or mean over streams).
            embed_weight: [V, d_embed] from token embedding (for weight tying).

        Returns:
            logits: [B, T, V].
        """
        h = self.project(self.norm(h))  # [B, T, d_embed]
        return F.linear(h, embed_weight)  # [B, T, V]


def compute_cross_stream_agreement(streams: list[torch.Tensor]) -> torch.Tensor:
    """Cross-stream agreement score (eq. 22).

    Agree_t^{k} = (1 / C(S_k, 2)) * sum_{s < s'} cos(h_{t,s}, h_{t,s'})

    Measures how much independent processing streams have converged.

    Args:
        streams: list of S_k tensors, each [B, T, d_k].

    Returns:
        agreement: [B, T] scores in [-1, 1] (higher = more agreement).
    """
    if len(streams) < 2:
        return torch.ones(streams[0].shape[0], streams[0].shape[1], device=streams[0].device)

    total_sim = torch.zeros(
        streams[0].shape[0], streams[0].shape[1], device=streams[0].device
    )
    count = 0
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            sim = F.cosine_similarity(streams[i], streams[j], dim=-1)
            total_sim = total_sim + sim
            count += 1

    return total_sim / count


class DeepSupervisionLoss(nn.Module):
    """Deep supervision loss combining predictions from all stages (eq. 24).

    L = sum_k lambda_k * L_k

    where L_k = -(1/T) sum_t log p_hat_k(x_{t+1} | x_{<=t})
    """

    def __init__(self, weights: list[float]):
        super().__init__()
        self.weights = weights

    def forward(
        self, all_logits: list[torch.Tensor], targets: torch.Tensor
    ) -> tuple[torch.Tensor, list[float]]:
        """Compute weighted loss across all stages.

        Args:
            all_logits: list of K tensors, each [B, T, V].
            targets: [B, T] ground truth token ids.

        Returns:
            total_loss: scalar, weighted sum of per-stage losses.
            per_stage_losses: list of K floats for diagnostics.
        """
        per_stage_losses = []
        total_loss = torch.tensor(0.0, device=targets.device)

        for logits, weight in zip(all_logits, self.weights):
            loss_k = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="mean",
            )
            per_stage_losses.append(loss_k.item())
            total_loss = total_loss + weight * loss_k

        return total_loss, per_stage_losses
