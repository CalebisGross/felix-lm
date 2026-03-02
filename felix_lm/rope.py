"""Depth-Extended Rotary Position Embeddings (Section 3.4).

Standard RoPE encodes token position. We extend it to also encode depth
(layer index), providing the "twist" of the helical trajectory.

Definition 3.4: At global layer index l_global and token position m,
the composite angle for dimension pair (2i, 2i+1) is:

    phi_i(m, l_global) = m * theta_i^pos + l_global * theta_i^depth

where:
    theta_i^pos = beta_pos^{-2i/d}        (standard RoPE frequencies)
    theta_i^depth = alpha_i * 2*pi*n / L_total  (depth frequencies)
"""

import torch


def build_rope_cache(
    seq_len: int,
    dim: int,
    global_layer_idx: int,
    total_layers: int,
    helical_turns: int,
    base: float = 10000.0,
    depth_alpha: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for depth-extended RoPE.

    Args:
        seq_len: Sequence length T.
        dim: Head dimension (must be even).
        global_layer_idx: This layer's index counting across all stages.
        total_layers: L_total = sum of all L_k across stages.
        helical_turns: n, the number of helical turns from Felix config.
        base: RoPE base frequency (beta_pos), typically 10000.
        depth_alpha: Per-dimension scaling for depth coupling.
            alpha=0 recovers standard RoPE, alpha=1 is full helical twist.
        device: Target device.

    Returns:
        (cos_cached, sin_cached) each of shape [seq_len, dim//2].
    """
    half_dim = dim // 2
    i = torch.arange(0, half_dim, device=device, dtype=torch.float32)

    # Standard RoPE frequencies: theta_i^pos = base^{-2i/d}
    theta_pos = base ** (-2.0 * i / dim)

    # Depth frequency: theta_i^depth = alpha * 2*pi*n / L_total
    if total_layers > 0 and depth_alpha > 0:
        theta_depth = depth_alpha * (2.0 * torch.pi * helical_turns / total_layers)
    else:
        theta_depth = 0.0

    # Token positions
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)

    # phi_i(m, l) = m * theta_pos_i + l * theta_depth
    # Shape: [seq_len, half_dim]
    angles = positions.unsqueeze(1) * theta_pos.unsqueeze(0)
    angles = angles + global_layer_idx * theta_depth

    return angles.cos(), angles.sin()


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [B, H, T, D].
        k: Key tensor [B, H, T, D].
        cos: Cosine cache [T, D//2].
        sin: Sine cache [T, D//2].

    Returns:
        Rotated (q, k) with same shapes.
    """
    # Split into pairs: (x0, x1), (x2, x3), ...
    q_r = q.float().reshape(*q.shape[:-1], -1, 2)  # [B, H, T, D//2, 2]
    k_r = k.float().reshape(*k.shape[:-1], -1, 2)

    # Reshape cos/sin for broadcasting: [1, 1, T, D//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation: (x0*cos - x1*sin, x0*sin + x1*cos)
    q_out = torch.stack(
        [q_r[..., 0] * cos - q_r[..., 1] * sin, q_r[..., 0] * sin + q_r[..., 1] * cos],
        dim=-1,
    )
    k_out = torch.stack(
        [k_r[..., 0] * cos - k_r[..., 1] * sin, k_r[..., 0] * sin + k_r[..., 1] * cos],
        dim=-1,
    )

    return q_out.reshape(q.shape).to(q.dtype), k_out.reshape(k.shape).to(k.dtype)
