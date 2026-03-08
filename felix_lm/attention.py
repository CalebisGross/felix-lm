"""Attention variants for Felix-LM (Section 3.2).

Three attention types, reflecting heterogeneous computation by stage:
- Stage 0:       LinearAttention        O(T)   — broad, cheap context
- Stage 1..K-2:  SlidingWindowAttention  O(Tw)  — local precision
- Stage K-1:     FullCausalAttention     O(T²)  — maximum precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from felix_lm.rope import apply_rope


class FullCausalAttention(nn.Module):
    """Standard causal multi-head attention (Stage K-1).

    Uses F.scaled_dot_product_attention with is_causal=True.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Force math backend — flash/efficient are buggy on MI300X
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class SlidingWindowAttention(nn.Module):
    """Sliding window causal attention (Stages 1..K-2).

    Token i attends to positions [max(0, i-w+1), i] where w is the window size.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def _make_sliding_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Create combined causal + sliding window mask.

        Returns a [T, T] boolean mask where True means ATTEND.
        """
        # Causal: row i can attend to columns 0..i
        # Window: row i can attend to columns max(0, i-w+1)..i
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(T, device=device).unsqueeze(0)
        causal = col_idx <= row_idx
        window = col_idx >= (row_idx - self.window_size + 1)
        return causal & window

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Build sliding window mask
        mask = self._make_sliding_window_mask(T, x.device)
        # Convert to float mask for SDPA: 0.0 for attend, -inf for mask
        attn_mask = torch.where(mask, 0.0, float("-inf"))

        # Force math backend — flash/efficient are buggy on MI300X
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """Causal linear attention with ELU+1 kernel (Stage 0).

    Uses the feature map phi(x) = elu(x) + 1 to avoid materializing
    the T x T attention matrix. Computes attention via running sums:

        S_t = sum_{j<=t} phi(K_j)^T V_j   (accumulated key-value)
        z_t = sum_{j<=t} phi(K_j)          (accumulated normalizer)
        out_t = phi(Q_t) @ S_t / (phi(Q_t) @ z_t)

    O(T) in time and memory per head.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """ELU+1 feature map: phi(x) = elu(x) + 1. Always non-negative."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # Apply RoPE before feature map
        q, k = apply_rope(q, k, cos, sin)

        # Apply feature map
        phi_q = self._feature_map(q)  # [B, H, T, d]
        phi_k = self._feature_map(k)  # [B, H, T, d]

        # Causal linear attention via cumulative sums
        # S_t = cumsum of phi(K)^T V = cumsum of outer(phi_k, v)
        # z_t = cumsum of phi(K)
        kv = torch.einsum("bhti,bhtj->bhtij", phi_k, v)  # [B, H, T, d, d]
        S = kv.cumsum(dim=2)  # [B, H, T, d, d] — running key-value sum
        z = phi_k.cumsum(dim=2)  # [B, H, T, d] — running normalizer

        # out_t = phi(Q_t) @ S_t / (phi(Q_t) @ z_t)
        numerator = torch.einsum("bhti,bhtij->bhtj", phi_q, S)  # [B, H, T, d]
        denominator = torch.einsum("bhti,bhti->bht", phi_q, z)  # [B, H, T]
        denominator = denominator.unsqueeze(-1).clamp(min=1e-6)

        out = numerator / denominator  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


def build_attention(
    attention_type: str, dim: int, num_heads: int, window_size: int = 64, dropout: float = 0.0
) -> nn.Module:
    """Factory function to build the appropriate attention module."""
    if attention_type == "full_causal":
        return FullCausalAttention(dim, num_heads, dropout)
    elif attention_type == "sliding_window":
        return SlidingWindowAttention(dim, num_heads, window_size, dropout)
    elif attention_type == "linear":
        return LinearAttention(dim, num_heads, dropout)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
