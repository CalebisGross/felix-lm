"""Gated merge operation (Section 3.3).

The merge is the architectural core of Felix-LM, implementing progressive
convergence. It combines pairs of streams into a single stream:

    m_t = P_k @ (g_k([h_s; h_{s'}]) ⊙ [h_s; h_{s'}])     (eq. 14)

where g_k is a sigmoid gate and P_k is a learned projection.

Via polar decomposition (eq. 16), every merge naturally performs
"rotation followed by compression" — the mathematical structure of
a helical step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from felix_lm.transformer_block import RMSNorm


class GatedMerge(nn.Module):
    """Gated merge operation (Definition 3.2).

    Concatenates two stream representations, applies a learned sigmoid gate,
    then projects to the next stage's dimension.

    Args:
        d_in: Dimension of each input stream (d_k).
        d_out: Dimension of output stream (d_{k+1}).
        gate_bias_init: Initial bias for gate (positive = gates start open).
        use_orthogonal: If True, constrain projection via Cayley parameterization.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        gate_bias_init: float = 1.0,
        use_orthogonal: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Gate: 2*d_k -> 2*d_k
        self.gate = nn.Linear(2 * d_in, 2 * d_in)
        # Initialize gate for good gradient flow (Corollary 4.4)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)

        # Projection: 2*d_k -> d_{k+1}
        self.use_orthogonal = use_orthogonal
        if use_orthogonal:
            # Cayley parameterization: P = (I - A)(I + A)^{-1} @ Pi (eq. 17)
            # A is a learnable skew-symmetric matrix
            self.A = nn.Parameter(torch.zeros(d_out, d_out) * 0.01)
            # Pi is fixed truncation: first d_out rows of I_{2*d_in}
            self.register_buffer("Pi", torch.eye(2 * d_in)[:d_out, :])
        else:
            self.projection = nn.Linear(2 * d_in, d_out, bias=False)

    def forward(self, h_s: torch.Tensor, h_s_prime: torch.Tensor) -> torch.Tensor:
        """Merge two streams into one.

        Args:
            h_s: [B, T, d_k] first stream.
            h_s_prime: [B, T, d_k] second stream.

        Returns:
            [B, T, d_{k+1}] merged representation.
        """
        z = torch.cat([h_s, h_s_prime], dim=-1)  # [B, T, 2*d_k]
        g = torch.sigmoid(self.gate(z))  # [B, T, 2*d_k]
        gated = g * z  # [B, T, 2*d_k]

        if self.use_orthogonal:
            I = torch.eye(self.A.shape[0], device=self.A.device)
            A_skew = self.A - self.A.T  # enforce skew-symmetry
            P_orth = torch.linalg.solve(I + A_skew, I - A_skew) @ self.Pi
            return F.linear(gated, P_orth)
        else:
            return self.projection(gated)

    def get_gate_values(self, h_s: torch.Tensor, h_s_prime: torch.Tensor) -> torch.Tensor:
        """Compute gate values for diagnostics (without full forward pass)."""
        z = torch.cat([h_s, h_s_prime], dim=-1)
        return torch.sigmoid(self.gate(z))


class CrossStreamAttention(nn.Module):
    """Cross-stream attention at merge points (Section 3.3.1).

    Before merging, each stream attends to the other:
        h_hat_{t,s} = h_{t,s} + CrossAttn(h_{t,s}, h_{:,s'})   (eq. 15)

    This allows streams to exchange information before fusion,
    implementing Felix's "structured communication at convergence points."
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, h_query: torch.Tensor, h_kv: torch.Tensor) -> torch.Tensor:
        """h_query attends to h_kv. Returns updated h_query.

        Args:
            h_query: [B, T, D] stream that produces queries.
            h_kv: [B, T, D] stream that provides keys/values.

        Returns:
            [B, T, D] h_query + cross_attention_output.
        """
        B, T, D = h_query.shape
        h_normed = self.norm(h_query)

        q = self.q_proj(h_normed).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal mask still applies (can't attend to future tokens)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return h_query + self.out_proj(out)


class MergeLayer(nn.Module):
    """Complete merge: optional cross-attention + gated projection."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        gate_bias_init: float = 1.0,
        use_cross_stream_attention: bool = False,
        use_orthogonal_merge: bool = False,
    ):
        super().__init__()
        self.cross_attn = (
            CrossStreamAttention(d_in, num_heads)
            if use_cross_stream_attention
            else None
        )
        self.merge = GatedMerge(d_in, d_out, gate_bias_init, use_orthogonal_merge)

    def forward(self, h_s: torch.Tensor, h_s_prime: torch.Tensor) -> torch.Tensor:
        """Merge two streams with optional cross-attention first."""
        if self.cross_attn is not None:
            h_s = self.cross_attn(h_s, h_s_prime)
            h_s_prime = self.cross_attn(h_s_prime, h_s)
        return self.merge(h_s, h_s_prime)
