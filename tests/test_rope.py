"""Tests for depth-extended RoPE (Proposition 3.5)."""

import torch
import pytest

from felix_lm.rope import apply_rope, build_rope_cache


def test_rope_cache_shape():
    cos, sin = build_rope_cache(
        seq_len=32, dim=16, global_layer_idx=0,
        total_layers=13, helical_turns=2,
    )
    assert cos.shape == (32, 8)
    assert sin.shape == (32, 8)


def test_apply_rope_preserves_shape():
    B, H, T, D = 2, 4, 32, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cos, sin = build_rope_cache(T, D, 0, 13, 2)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_alpha_zero_recovers_standard_rope():
    """Setting depth_alpha=0 should give identical results regardless of layer index."""
    T, D = 32, 16
    cos0, sin0 = build_rope_cache(T, D, global_layer_idx=0, total_layers=13,
                                   helical_turns=2, depth_alpha=0.0)
    cos5, sin5 = build_rope_cache(T, D, global_layer_idx=5, total_layers=13,
                                   helical_turns=2, depth_alpha=0.0)
    torch.testing.assert_close(cos0, cos5)
    torch.testing.assert_close(sin0, sin5)


def test_relative_position_property():
    """Proposition 3.5: attention score depends only on relative position and depth.

    Score between (m, l) query and (m', l') key should equal
    score between (m+delta, l+delta_l) query and (m'+delta, l'+delta_l) key
    for the same relative offsets delta_m = m-m', delta_l = l-l'.
    """
    D = 16
    T = 10

    # Case 1: query at (m=5, l=3), key at (m=2, l=1)
    cos3, sin3 = build_rope_cache(T, D, global_layer_idx=3, total_layers=13, helical_turns=2)
    cos1, sin1 = build_rope_cache(T, D, global_layer_idx=1, total_layers=13, helical_turns=2)

    q = torch.randn(1, 1, 1, D)
    k = torch.randn(1, 1, 1, D)

    # Extend to cover position indices we need
    cos3_full, sin3_full = build_rope_cache(T, D, 3, 13, 2)
    cos1_full, sin1_full = build_rope_cache(T, D, 1, 13, 2)

    # Apply RoPE at specific positions
    q1 = q.clone()
    k1 = k.clone()
    q1_rot, _ = apply_rope(q1, q1, cos3_full[5:6].unsqueeze(0), sin3_full[5:6].unsqueeze(0))
    _, k1_rot = apply_rope(k1, k1, cos1_full[2:3].unsqueeze(0), sin1_full[2:3].unsqueeze(0))
    score1 = (q1_rot * k1_rot).sum()

    # Case 2: query at (m=8, l=6), key at (m=5, l=4) — same deltas (3, 2)
    cos6, sin6 = build_rope_cache(T, D, 6, 13, 2)
    cos4, sin4 = build_rope_cache(T, D, 4, 13, 2)

    q2 = q.clone()
    k2 = k.clone()
    q2_rot, _ = apply_rope(q2, q2, cos6[8:9].unsqueeze(0), sin6[8:9].unsqueeze(0))
    _, k2_rot = apply_rope(k2, k2, cos4[5:6].unsqueeze(0), sin4[5:6].unsqueeze(0))
    score2 = (q2_rot * k2_rot).sum()

    torch.testing.assert_close(score1, score2, atol=1e-5, rtol=1e-5)


def test_depth_extension_changes_angles():
    """With depth_alpha > 0, different layers should produce different caches."""
    T, D = 32, 16
    cos0, sin0 = build_rope_cache(T, D, 0, 13, 2, depth_alpha=1.0)
    cos5, sin5 = build_rope_cache(T, D, 5, 13, 2, depth_alpha=1.0)
    assert not torch.allclose(cos0, cos5)
