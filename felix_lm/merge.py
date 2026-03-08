"""Merge operations (Section 3.3).

The merge is the architectural core of Felix-LM, implementing progressive
convergence. It combines pairs of streams into a single stream:

    m_t = P_k @ (g_k([h_s; h_{s'}]) ⊙ [h_s; h_{s'}])     (eq. 14)

where g_k is a sigmoid gate and P_k is a learned projection.

Via polar decomposition (eq. 16), every merge naturally performs
"rotation followed by compression" — the mathematical structure of
a helical step.

Extended with novel merge algorithms:
- GeometricMerge: multiplicative feature conjunction
- HadamardMerge: fixed orthogonal rotation + learned scaling
- CompetitiveMerge: streams compete, winner routes forward
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from felix_lm.transformer_block import RMSNorm, TransformerBlock


class GatedMerge(nn.Module):
    """Gated merge operation (Definition 3.2).

    Concatenates two stream representations, applies a learned sigmoid gate,
    then projects to the next stage's dimension.

    Args:
        d_in: Dimension of each input stream (d_k).
        d_out: Dimension of output stream (d_{k+1}).
        gate_bias_init: Initial bias for gate (positive = gates start open).
        use_orthogonal: If True, constrain projection via Cayley parameterization.
        token_conditional: If True, use bottleneck MLP for content-dependent gating.
        bottleneck_ratio: If >0, compress through d_out*ratio before expanding.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        gate_bias_init: float = 1.0,
        use_orthogonal: bool = False,
        token_conditional: bool = False,
        residual: bool = False,
        bottleneck_ratio: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.token_conditional = token_conditional
        self.residual = residual
        self.use_bottleneck = bottleneck_ratio > 0

        if token_conditional:
            # Bottleneck MLP gate: 2*d_k -> bottleneck -> 2*d_k
            # Bottleneck = d_in (half of concat dim) for efficiency
            bottleneck = d_in
            self.gate_down = nn.Linear(2 * d_in, bottleneck)
            self.gate_up = nn.Linear(bottleneck, 2 * d_in)
            # Initialize so output starts near gate_bias_init
            nn.init.zeros_(self.gate_down.weight)
            nn.init.zeros_(self.gate_down.bias)
            nn.init.zeros_(self.gate_up.weight)
            nn.init.constant_(self.gate_up.bias, gate_bias_init)
        else:
            # Gate: 2*d_k -> 2*d_k
            self.gate = nn.Linear(2 * d_in, 2 * d_in)
            # Initialize gate for good gradient flow (Corollary 4.4)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_bias_init)

        # Projection: 2*d_k -> d_{k+1}
        self.use_orthogonal = use_orthogonal
        if self.use_bottleneck:
            # Bottleneck autoencoder: 2*d_k -> tiny -> d_{k+1}
            d_neck = max(4, int(d_out * bottleneck_ratio))
            self.proj_down = nn.Linear(2 * d_in, d_neck, bias=False)
            self.proj_up = nn.Linear(d_neck, d_out, bias=False)
        elif use_orthogonal:
            # Cayley parameterization: P = (I - A)(I + A)^{-1} @ Pi (eq. 17)
            # A is a learnable skew-symmetric matrix
            self.A = nn.Parameter(torch.zeros(d_out, d_out) * 0.01)
            # Pi is fixed truncation: first d_out rows of I_{2*d_in}
            self.register_buffer("Pi", torch.eye(2 * d_in)[:d_out, :])
        else:
            self.projection = nn.Linear(2 * d_in, d_out, bias=False)

        # Residual skip connection for the merge
        if residual:
            if d_in == d_out:
                # Same dim: skip is just the mean, no extra params
                self.skip_proj = None
            else:
                # Different dims: learned projection for skip path
                self.skip_proj = nn.Linear(d_in, d_out, bias=False)

    def _compute_gate(self, z: torch.Tensor) -> torch.Tensor:
        """Compute gate values from concatenated stream representations."""
        if self.token_conditional:
            # Bottleneck MLP: compress -> ReLU -> expand -> sigmoid
            h = F.relu(self.gate_down(z))
            return torch.sigmoid(self.gate_up(h))
        else:
            return torch.sigmoid(self.gate(z))

    def forward(self, h_s: torch.Tensor, h_s_prime: torch.Tensor) -> torch.Tensor:
        """Merge two streams into one.

        Args:
            h_s: [B, T, d_k] first stream.
            h_s_prime: [B, T, d_k] second stream.

        Returns:
            [B, T, d_{k+1}] merged representation.
        """
        z = torch.cat([h_s, h_s_prime], dim=-1)  # [B, T, 2*d_k]
        g = self._compute_gate(z)  # [B, T, 2*d_k]
        gated = g * z  # [B, T, 2*d_k]

        if self.use_bottleneck:
            # Compress -> expand: forces distillation of essential information
            compressed = self.proj_down(gated)  # [B, T, d_neck]
            merged = self.proj_up(compressed)  # [B, T, d_out]
        elif self.use_orthogonal:
            eye = torch.eye(self.A.shape[0], device=self.A.device)
            A_skew = self.A - self.A.T  # enforce skew-symmetry
            P_orth = torch.linalg.solve(eye + A_skew, eye - A_skew) @ self.Pi
            merged = F.linear(gated, P_orth)
        else:
            merged = self.projection(gated)

        # Residual: skip = mean of streams (projected if dims differ)
        if self.residual:
            if self.skip_proj is not None:
                skip = (self.skip_proj(h_s) + self.skip_proj(h_s_prime)) / 2
            else:
                skip = (h_s + h_s_prime) / 2
            merged = skip + merged

        return merged

    def get_gate_values(self, h_s: torch.Tensor, h_s_prime: torch.Tensor) -> torch.Tensor:
        """Compute gate values for diagnostics (without full forward pass)."""
        z = torch.cat([h_s, h_s_prime], dim=-1)
        return self._compute_gate(z)


class GeometricMerge(nn.Module):
    """Geometric mean merge: multiplicative feature conjunction.

    Instead of concat→gate→project, independently projects each stream
    then combines via signed geometric mean: sign(a*b) * sqrt(|a*b|).
    Features must be present in BOTH streams to survive.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj_a = nn.Linear(d_in, d_out, bias=False)
        self.proj_b = nn.Linear(d_in, d_out, bias=False)
        self.scale = nn.Parameter(torch.ones(d_out))

    def forward(self, h_s: torch.Tensor, h_s_prime: torch.Tensor) -> torch.Tensor:
        a = self.proj_a(h_s)  # [B, T, d_out]
        b = self.proj_b(h_s_prime)  # [B, T, d_out]
        product = a * b
        # Signed geometric mean: preserves sign, takes sqrt of magnitude
        merged = torch.sign(product) * torch.sqrt(torch.abs(product) + 1e-8)
        return merged * self.scale


def _build_hadamard(n: int) -> torch.Tensor:
    """Build n×n Hadamard matrix via Sylvester construction.

    Requires n to be a power of 2.
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
    return H / math.sqrt(n)  # normalize to orthogonal


class HadamardMerge(nn.Module):
    """Hadamard merge: fixed orthogonal mixing + learned diagonal scaling.

    Concatenates streams, applies a fixed Hadamard rotation (maximally mixes
    all dimensions), then a learned diagonal scale, then truncates to d_out.
    ~99% fewer params than gated merge (only d_out scalars vs full projection).
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        concat_dim = 2 * d_in
        # Pad to next power of 2 for Hadamard
        self.pad_dim = 1 << (concat_dim - 1).bit_length()
        self.register_buffer("H", _build_hadamard(self.pad_dim))
        self.scale = nn.Parameter(torch.ones(d_out) * 0.1)
        self.d_out = d_out
        self.concat_dim = concat_dim

    def forward(self, h_s: torch.Tensor, h_s_prime: torch.Tensor) -> torch.Tensor:
        z = torch.cat([h_s, h_s_prime], dim=-1)  # [B, T, 2*d_in]
        # Pad if needed
        if self.concat_dim < self.pad_dim:
            z = F.pad(z, (0, self.pad_dim - self.concat_dim))
        # Fixed orthogonal rotation
        z = F.linear(z, self.H)  # [B, T, pad_dim]
        # Truncate + scale
        return z[..., : self.d_out] * self.scale  # [B, T, d_out]


class CompetitiveMerge(nn.Module):
    """Competitive merge: streams compete, winner routes forward.

    Each stream is independently projected to d_out, then a learned
    scorer picks the winner per-token. Uses straight-through estimator
    for gradients so both streams learn.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj_a = nn.Linear(d_in, d_out, bias=False)
        self.proj_b = nn.Linear(d_in, d_out, bias=False)
        # Scorer: computes advantage of stream a over b
        self.scorer = nn.Linear(2 * d_in, 1)

    def forward(self, h_s: torch.Tensor, h_s_prime: torch.Tensor) -> torch.Tensor:
        a = self.proj_a(h_s)  # [B, T, d_out]
        b = self.proj_b(h_s_prime)  # [B, T, d_out]

        # Score which stream wins
        z = torch.cat([h_s, h_s_prime], dim=-1)  # [B, T, 2*d_in]
        score = torch.sigmoid(self.scorer(z))  # [B, T, 1]

        # Hard routing with straight-through gradient
        hard_choice = (score > 0.5).float()
        choice = hard_choice + score - score.detach()  # STE

        return choice * a + (1 - choice) * b


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
    """Complete merge: optional cross-attention + configurable merge algorithm."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        gate_bias_init: float = 1.0,
        use_cross_stream_attention: bool = False,
        use_orthogonal_merge: bool = False,
        token_conditional: bool = False,
        residual: bool = False,
        bottleneck_ratio: float = 0.0,
        merge_type: str = "gated",
        noise_std: float = 0.0,
        integration_depth: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.noise_std = noise_std

        self.cross_attn = (
            CrossStreamAttention(d_in, num_heads) if use_cross_stream_attention else None
        )

        if merge_type == "geometric":
            self.merge = GeometricMerge(d_in, d_out)
        elif merge_type == "hadamard":
            self.merge = HadamardMerge(d_in, d_out)
        elif merge_type == "competitive":
            self.merge = CompetitiveMerge(d_in, d_out)
        else:
            self.merge = GatedMerge(
                d_in,
                d_out,
                gate_bias_init,
                use_orthogonal_merge,
                token_conditional,
                residual,
                bottleneck_ratio,
            )

        # Post-merge integration layers: dedicated compute to digest merged output
        if integration_depth > 0:
            out_heads = max(1, d_out // 32)  # head_dim ~32
            self.integration = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=d_out,
                        num_heads=out_heads,
                        attention_type="full_causal",
                        ffn_mult=4,
                        dropout=dropout,
                    )
                    for _ in range(integration_depth)
                ]
            )
        else:
            self.integration = None

    def forward(
        self,
        h_s: torch.Tensor,
        h_s_prime: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge two streams with optional cross-attention and integration."""
        if self.cross_attn is not None:
            h_s = self.cross_attn(h_s, h_s_prime)
            h_s_prime = self.cross_attn(h_s_prime, h_s)
        merged = self.merge(h_s, h_s_prime)
        # Noise injection during training (variational bottleneck)
        if self.training and self.noise_std > 0:
            merged = merged + torch.randn_like(merged) * self.noise_std
        # Post-merge integration
        if self.integration is not None and cos is not None and sin is not None:
            for layer in self.integration:
                merged = layer(merged, cos, sin)
        return merged
