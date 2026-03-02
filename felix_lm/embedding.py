"""Input embedding and stream initialization (Section 3.1).

Given input sequence x = (x_1, ..., x_T):
1. Shared token embedding:  e_t = Embed(x_t) in R^{d_embed}
2. Per-stream projection:   h_{t,s}^{0,0} = W_s^init @ e_t + b_s^init

Stream projections are initialized with orthogonal matrices to encourage
diverse initial representations across streams (Appendix A).
"""

import torch
import torch.nn as nn


class StreamInitialization(nn.Module):
    """Token embedding + per-stream learned projections.

    Each stream receives the shared embedding through its own linear projection,
    producing diverse "views" of the input (analogous to Felix agent roles).
    """

    def __init__(self, vocab_size: int, d_embed: int, num_streams: int, d_stream: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.num_streams = num_streams

        self.stream_projections = nn.ModuleList(
            [nn.Linear(d_embed, d_stream) for _ in range(num_streams)]
        )

        self._init_orthogonal_projections()

    def _init_orthogonal_projections(self):
        """Initialize stream projections for maximum initial diversity.

        Uses nn.init.orthogonal_ on each projection independently.
        This gives diverse (though not strictly orthogonal) initial subspaces.

        Note: Strict orthogonal subspace init (Appendix A, eq. 32-33) requires
        d_embed >= S_0 * d_0. When this doesn't hold, independent orthogonal
        init is a reasonable relaxation.
        """
        for proj in self.stream_projections:
            nn.init.orthogonal_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            token_ids: [B, T] integer token indices.

        Returns:
            e: [B, T, d_embed] shared embeddings (kept for weight tying).
            streams: list of num_streams tensors, each [B, T, d_stream].
        """
        e = self.token_embedding(token_ids)
        streams = [proj(e) for proj in self.stream_projections]
        return e, streams
