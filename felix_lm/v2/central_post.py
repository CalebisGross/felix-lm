"""CentralPost: shared communication hub between streams.

In the Felix multi-agent framework, the CentralPost is the central hub that all
agents report to and read from. This is the neural analog: a persistent shared
state that mediates all inter-stream communication.

Each layer, every stream:
1. Reads from CentralPost (gated additive update to stream)
2. Writes to CentralPost (gated contribution aggregated across streams)

Communication cost is O(N) per layer, not O(N^2) like cross-attention.
"""

import torch
import torch.nn as nn

from felix_lm.transformer_block import RMSNorm


class CentralPostLayer(nn.Module):
    """One layer of CentralPost read/write interaction.

    Read phase:  stream_s += gate_s * proj(central_post)
    Write phase: central_post += sum_s(gate_s * proj(stream_s)), then normalize
    """

    def __init__(self, d_stream: int, d_post: int, num_streams: int):
        super().__init__()
        self.num_streams = num_streams

        # Read: CentralPost -> each stream
        self.read_projs = nn.ModuleList(
            [nn.Linear(d_post, d_stream, bias=False) for _ in range(num_streams)]
        )
        self.read_gates = nn.ModuleList([nn.Linear(d_stream, 1) for _ in range(num_streams)])

        # Write: each stream -> CentralPost
        self.write_projs = nn.ModuleList(
            [nn.Linear(d_stream, d_post, bias=False) for _ in range(num_streams)]
        )
        self.write_gates = nn.ModuleList([nn.Linear(d_stream, 1) for _ in range(num_streams)])

        # Normalize after aggregated write to prevent unbounded growth
        self.norm = RMSNorm(d_post)

    def forward(
        self,
        streams: list[torch.Tensor],
        central_post: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Args:
            streams: list of num_streams tensors, each [B, T, d_stream].
            central_post: [B, T, d_post] shared state.

        Returns:
            updated_streams: list of num_streams tensors, each [B, T, d_stream].
            updated_post: [B, T, d_post] updated shared state.
        """
        # Read phase: each stream reads from CentralPost
        updated_streams = []
        for s in range(self.num_streams):
            gate = torch.sigmoid(self.read_gates[s](streams[s]))  # [B, T, 1]
            updated = streams[s] + gate * self.read_projs[s](central_post)
            updated_streams.append(updated)

        # Write phase: all streams contribute to CentralPost
        new_post = central_post
        for s in range(self.num_streams):
            gate = torch.sigmoid(self.write_gates[s](updated_streams[s]))  # [B, T, 1]
            new_post = new_post + gate * self.write_projs[s](updated_streams[s])
        new_post = self.norm(new_post)

        return updated_streams, new_post
