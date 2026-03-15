"""Spoke layer: lightweight low-rank probes that read/write the hub.

Each spoke projects the hub state to a low-rank bottleneck, applies a
non-linearity, and projects back. The mean of all spoke updates is
gated into the hub as a residual.

Agreement (cross-spoke cosine similarity on the low-rank views) is
computed as a diagnostic signal — it measures how much the spokes
agree on what they see.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from felix_lm.transformer_block import RMSNorm


class SpokeLayer(nn.Module):
    """Lightweight spoke processing: read hub, transform, write back.

    Each spoke s has:
        W_down_s: [d, r] — project to low-rank view
        W_up_s:   [r, d] — project back to hub dimension

    Forward:
        h_norm = RMSNorm(h)
        view_s = SiLU(h_norm @ W_down_s)       for each spoke s
        update_s = view_s @ W_up_s              for each spoke s
        h = h + gate * mean(updates)

    Params per layer: d + S*(d*r + r*d) + 1
    """

    def __init__(
        self,
        d_model: int,
        num_spokes: int,
        rank: int,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_spokes = num_spokes
        self.rank = rank

        self.norm = RMSNorm(d_model)

        # Spoke projections: S pairs of (down, up)
        self.w_down = nn.ModuleList(
            [nn.Linear(d_model, rank, bias=False) for _ in range(num_spokes)]
        )
        self.w_up = nn.ModuleList([nn.Linear(rank, d_model, bias=False) for _ in range(num_spokes)])

        # Initialize up projections near zero so spokes start as near-identity
        for up in self.w_up:
            nn.init.zeros_(up.weight)  # type: ignore[arg-type]

        # Learned scalar gate (per-layer)
        self.gate_bias = nn.Parameter(torch.tensor(gate_init))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process hub state through spokes.

        Args:
            h: Hub hidden state [B, T, d]

        Returns:
            h_updated: Hub state with spoke feedback [B, T, d]
            agreement: Mean pairwise cosine similarity of spoke views [B, T]
        """
        h_norm = self.norm(h)

        # Compute spoke views and updates
        views = []
        updates = []
        for s in range(self.num_spokes):
            view = F.silu(self.w_down[s](h_norm))  # [B, T, r]
            update = self.w_up[s](view)  # [B, T, d]
            views.append(view)
            updates.append(update)

        # Mean update across spokes
        mean_update = torch.stack(updates, dim=0).mean(dim=0)  # [B, T, d]

        # Gated residual
        gate = torch.sigmoid(self.gate_bias)
        h_updated = h + gate * mean_update

        # Agreement: mean pairwise cosine similarity of views (diagnostic only)
        agreement = self._compute_agreement(views)

        return h_updated, agreement

    @torch.no_grad()
    def _compute_agreement(self, views: list[torch.Tensor]) -> torch.Tensor:
        """Mean pairwise cosine similarity across spoke views.

        Args:
            views: List of S tensors, each [B, T, r]

        Returns:
            agreement: [B, T] in [-1, 1]
        """
        if len(views) < 2:
            return torch.ones(views[0].shape[0], views[0].shape[1], device=views[0].device)

        total_sim = torch.zeros(views[0].shape[0], views[0].shape[1], device=views[0].device)
        count = 0
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                sim = F.cosine_similarity(views[i], views[j], dim=-1)
                total_sim = total_sim + sim
                count += 1

        return total_sim / count
