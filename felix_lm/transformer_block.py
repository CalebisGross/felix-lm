"""Transformer block components (Section 3.2, Equations 10-11).

Pre-norm transformer layer:
    h_tilde = h + Attn(LN(h))
    h_next  = h_tilde + FFN(LN(h_tilde))

Uses RMSNorm [6] and SiLU-gated FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from felix_lm.attention import build_attention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class FeedForward(nn.Module):
    """SiLU-gated FFN (SwiGLU variant).

    gate = SiLU(x @ W_gate)
    up   = x @ W_up
    out  = (gate * up) @ W_down
    """

    def __init__(self, dim: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * ffn_mult
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    """Single pre-norm transformer layer.

    Equations 10-11:
        h_tilde = h + Attn(LN(h))
        h_next  = h_tilde + FFN(LN(h_tilde))

    Supports asymmetric variants:
        attention_type="none" -> FFN-only block (no attention)
        ffn_mult=0            -> Attention-only block (no FFN)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: str,
        ffn_mult: int = 4,
        window_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.has_attn = attention_type != "none"
        self.has_ffn = ffn_mult > 0

        if self.has_attn:
            self.norm1 = RMSNorm(dim)
            self.attn = build_attention(attention_type, dim, num_heads, window_size, dropout)

        if self.has_ffn:
            self.norm2 = RMSNorm(dim)
            self.ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.has_attn:
            x = x + self.attn(self.norm1(x), cos, sin)
        if self.has_ffn:
            x = x + self.ffn(self.norm2(x))
        return x
